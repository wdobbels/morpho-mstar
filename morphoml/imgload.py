"""
Loading image data from different sources.

These functions take a file or directory as input, and return
(images, names).

images has shape (ngalaxies, npix, npix). It can be loaded to
    memory, but this is not necessary (can be memmap).

names is an array of shape (ngalaxies,), with the SDSS ids.
"""
from pathlib import Path
import numpy as np

def load_npy(dset='train', npix=69, ntrain=10000):
    """
    Loads data from a .npy file. 
    dset can be 'train', 'val', or 'test'.
    The .npy files are stored in './data/images/{npix}pix_npy/'
    These files will be downloaded from AWS if not present.
    For val and test, the .npy files are named: '{dset}_images.npy'
    and '{dset}_names.npy'. For train, the .npy files are named:
    'train_images_{ntrain}.npy' and 'train_names_{ntrain}.npy'.
    """
    
    dirpath = Path(f'./data/images/{npix}pix_npy/')
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    if dset == 'train':
        imgpath = dirpath / f'train_images_{ntrain}.npy'
    else:
        imgpath = dirpath / f'{dset}_images.npy'
    if not imgpath.exists():
        print(f'{dset} images not found, downloading from AWS...')
        import urllib.request
        import json
        import tarfile
        # Get url name
        with open('./data/images/urls.json', 'r') as urlfile:
            urls = json.load(urlfile)
        dset_urlname = f'train{ntrain}' if dset == 'train' else 'valtest'
        url = urls[f'{npix}pix'][dset_urlname]
        filename = f'./data/images/{npix}pix_npy_{dset_urlname}.tar.gz'
        # Download
        urllib.request.urlretrieve(url, filename)
        # Untar
        tar = tarfile.open(filename)
        basedir = './data/images/'
        tar.extractall(basedir)  # extract to untardir (see below)
        tar.close()
        # Move files from untarred dir to final location and clean up
        untardir = Path(f'./data/images/{npix}pix_npy_{dset_urlname}/')
        for npyfile in untardir.iterdir():
            if npyfile.suffix != '.npy':
                npyfile.unlink()
                continue
            npyfile.rename(dirpath / npyfile.name)
        untardir.rmdir()
        Path(filename).unlink()
        # The images should now be at imgpath
        if imgpath.exists():
            print('Succesfully downloaded images!')
        else:
            raise DownloadDataException("Downloaded the files, can not find them locally!")
    imgs = np.load(str(imgpath))
    if dset == 'train':
        namespath = dirpath / f'train_names_{ntrain}.npy'
    else:
        namespath = dirpath / f'{dset}_names.npy'
    if not namespath.exists():
        raise DownloadDataException("Downloaded images, but can not find names!")
    names = np.load(str(namespath))
    return imgs, names

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

class DownloadDataException(Exception):
    pass