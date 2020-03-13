"""
Loading image data from different sources.

These functions take a file or directory as input, and return
(images, names).

images has shape (ngalaxies, ny, nx). It can be loaded to
    memory, but this is not necessary (can be memmap).

names is an array of shape (ngalaxies,), with the SDSS ids.
"""

def load_hdf5(filename):
    import h5py
    f = h5py.File(filename, 'r')
    return f['data'], f['sdss_id']