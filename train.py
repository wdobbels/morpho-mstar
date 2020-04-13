import argparse
import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
# import tensorflow_addons as tfa  # Not supported by default
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--imgs_dir', type=str, default=os.environ['SM_CHANNEL_IMGS'])
    parser.add_argument('--metadata', type=str, default=os.environ['SM_CHANNEL_META'])
    
    return parser.parse_known_args()

def build_model(npix=128):
    """Build CNN that works on images of shape (npix, npix, 1)."""

    f_act = 'relu'
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation=f_act,
                            input_shape=(npix, npix, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (4, 4), activation=f_act))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation=f_act))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation=f_act))
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dense(512, activation=f_act))
    model.add(layers.Dense(1, activation=f_act))

    model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['mse'])
    return model

def get_data(imgs_dir, meta_filename):
    """
    Load images and target data into memory.

    Returns a dictionary with a 'train' and 'val' key.
    Both of the keys contain a tensorflow Dataset.
    [0]: images array, shape (Nset, Npix, Npix, 1)
    [1]: M/L array, shape (Nset,)
    Also returns the number of pixels as an additional output.
    """

    d_data = {}
    print('Reading metadata from', meta_filename)
    meta_exists = os.path.exists(meta_filename)
    print('Meta exists?', meta_exists)
    print('Images folder', imgs_dir)
    print('Contents of base image folder', os.listdir(imgs_dir))
    df_meta = pd.read_csv(meta_filename, sep='\t', index_col=0)
    npix = None
    for dset in ['train', 'val']:
        imgs, names = load_images_npy(imgs_dir, dset)
        y = df_meta.loc[names, 'M/L']
        print(dset, 'images shape:', imgs.shape)
        print(dset, 'y shape:', y.shape)
        if npix is None:
            npix = imgs.shape[1]
        d_data[dset] = tf.data.Dataset.from_tensor_slices((imgs, y))
    return d_data, npix

def _load_images_dir(img_dir, rescale=True):
    """
    Load images (in png or jpg) from directory.
    Returns images (shape ngalaxies, npix, npix, 1), names (shape ngalaxies).
    If rescale is True, the images are transformed to the range [-1, 1], 
    and the dtype is converted to np.float32. If False, the images are
    of dtype np.uint8, in the range [0, 255].
    
    Note: no longer used.
    """

    def valid_extension(filename): return filename[-3:] in ['jpg', 'png']
    filenames = list(filter(valid_extension, os.listdir(img_dir)))
    ngalaxies = len(filenames)
    npix = np.array(Image.open(os.path.join(img_dir, filenames[0]))).shape[0]
    dtype = np.float32 if rescale else np.uint8
    imgs = np.empty((ngalaxies, npix, npix), dtype=dtype)
    names = np.empty((ngalaxies), dtype=np.int64)

    for i, filename in enumerate(filenames):
        img = np.array(Image.open(os.path.join(img_dir, filename)))
        if rescale:
            xmax = 255  # assume xmin = 0, xmax = 255
            img = 2 * img.astype(np.float32) / xmax - 1
        imgs[i, :, :] = img
        # example filename: 1237648722296897660.png
        names[i] = np.int64(filename.split('.')[0])
    imgs = np.expand_dims(imgs, axis=-1)
    return imgs, names

def load_images_npy(img_dir, dset='train'):
    img_filename = os.path.join(img_dir, dset + '_images.npy')
    names_filename = os.path.join(img_dir, dset + '_names.npy')
    imgs = np.load(img_filename)
    if imgs.ndim == 3:  # (ngal, npix, npix)
        imgs = np.expand_dims(imgs, axis=-1)
    names = np.load(names_filename)
    return imgs, names

def augment_data(data, batch_size=32):
    def convert(image, label): return convert_image(image), label
    def augment(image, label): return augment_image(image), label
    aug_data = {}
    # Train: augment
    aug_data['train'] = (data['train']
                            .cache()
                            .shuffle(10000)
                            .map(augment, num_parallel_calls=AUTOTUNE)
                            .batch(batch_size)
                            .prefetch(AUTOTUNE))
    # Val: just convert to right dtype
    aug_data['val'] = (data['val']
                            .map(convert, num_parallel_calls=AUTOTUNE)
                            .batch(batch_size))
    return aug_data

def convert_image(image):
    # Cast and normalize the image to [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def augment_image(image):
    add_pix = 10
    npix = image.shape[1]
    image = convert_image(image)
    # to [0, 2], since it automatically pads with zeros
    image = image + 1
    # Add padding (to translate later)
    image = tf.image.resize_with_crop_or_pad(image, npix + add_pix, npix + add_pix)
    # Rotate all images in batch with same, uniformly chosen angle
#     random_angles = tf.random.uniform(shape = (), minval=-np.pi, maxval=np.pi)
#     image = tfa.image.rotate(image, random_angles)
    # Crop back to needed format
    image = tf.image.random_crop(image, size=[npix, npix, 1]) # Random crop back
    image = image - 1
    return image

def fit_model(model, data, model_dir, nepochs=100):
    nepochs = 100
    callbacks = [tf.keras.callbacks.EarlyStopping(min_delta=1e-4, patience=6,
                                                restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3,
                                                    min_delta=1e-4, verbose=1)]
    model.fit(data['train'], validation_data=data['val'],
              epochs=nepochs, callbacks=callbacks)
    # Save model (unique name by using seconds since Epoch time)
    model_name = str(int(time.time()))
    model_filename = os.path.join(model_dir, model_name + '.h5')
    print('Saving model to', model_filename)
    model.save(model_filename)

if __name__ == '__main__':
    args, unknown = _parse_args()
    data, npix = get_data(args.imgs_dir, os.path.join(args.metadata, 'metadata.tsv'))
    data = augment_data(data)
    model = build_model(npix=npix)
    # args.model_dir points to S3 path, and will fail
    fit_model(model, data, os.environ['SM_MODEL_DIR'])