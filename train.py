import argparse
import os
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--imgs-dir', type=str, default=os.environ['SM_CHANNEL_IMGS'])
    parser.add_argument('--metadata', type=str, default=os.environ['SM_CHANNEL_META'])

    parser.add_argument('--npix', type=int, default=128)

    return parser.parse_known_args()

def _build_model(npix=128):
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

def _get_data(imgs_dir, meta_filename):
    d_data = {'train': {}, 'val': {}}
    for dset, data in d_data.items():
        d_data[dset]['x'] = _load_images(os.path.join(imgs_dir, 'train/'))

def _load_images(img_dir):
    pass

if __name__ == '__main__':
    args, unknown = _parse_args()
    model = _build_model(args.npix)