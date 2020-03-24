"""
Loading image data from different sources.

These functions take a file or directory as input, and return
(images, names).

images has shape (ngalaxies, npix, npix). It can be loaded to
    memory, but this is not necessary (can be memmap).

names is an array of shape (ngalaxies,), with the SDSS ids.
"""
import numpy as np

def load_hdf5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    return f['data'], f['sdss_id']

def load_folder(dirname, rescale=True):
    """
    Load images (in png or jpg) from directory.

    Returns images (shape ngalaxies, npix, npix), names (shape ngalaxies).
    If rescale is True, the images are transformed to the range [-1, 1], 
    and the dtype is converted to np.float32. If False, the images are
    of dtype np.uint8, in the range [0, 255].
    """

    import os
    from PIL import Image

    def valid_extension(filename): return filename[-3:] in ['jpg', 'png']
    filenames = list(filter(valid_extension, os.listdir(dirname)))
    ngalaxies = len(filenames)
    npix = np.array(Image.open(os.path.join(dirname, filenames[0]))).shape[0]
    dtype = np.float32 if rescale else np.uint8
    imgs = np.empty((ngalaxies, npix, npix), dtype=dtype)
    names = np.empty((ngalaxies), dtype=np.int64)

    for i, filename in enumerate(filenames):
        img = np.array(Image.open(os.path.join(dirname, filename)))
        if rescale:
            xmax = 255  # assume xmin = 0, xmax = 255
            img = 2 * img.astype(np.float32) / xmax - 1
        imgs[i, :, :] = img
        # example filename: 1237648722296897660.png
        names[i] = np.int64(filename.split('.')[0])
    return imgs, names