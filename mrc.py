import os
import numpy as np
import mrcfile

# See ref:
# MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography
# And:
# https://www.ccpem.ac.uk/mrc_format/mrc2014.php

DTYPE_FOR_MODE = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    3: "2h",  # complex number from 2 shorts
    4: np.complex64,
    6: np.uint16,
    12: np.float16,
    16: "3B",
}  # RBG values
MODE_FOR_DTYPE = {vv: kk for kk, vv in DTYPE_FOR_MODE.items()}


class LazyImage:
    """On-the-fly image loading"""

    def __init__(self, fname, shape, dtype, offset):
        self.fname = fname
        self.shape = shape
        self.dtype = dtype
        self.offset = offset

    def get(self):
        with open(self.fname) as f:
            f.seek(self.offset)
            image = np.fromfile(
                f, dtype=self.dtype, count=np.product(self.shape)
            ).reshape(self.shape).transpose()
        return image


def parse_header(fname):
    with mrcfile.open(fname, permissive=True, header_only=True) as mrc:
        return mrc.header
    
    
def get_voxelsize(fname):
    with mrcfile.open(fname, permissive=True, header_only=True) as mrc:
        return mrc.voxel_size
 
def parse_mrc_list(txtfile, lazy=False):
    lines = open(txtfile, "r").readlines()

    def abspath(f):
        if os.path.isabs(f):
            return f
        base = os.path.dirname(os.path.abspath(txtfile))
        return os.path.join(base, f)

    lines = [abspath(x) for x in lines]
    apix = get_voxelsize(lines[0].strip()).x
    if not lazy:
        particles = np.vstack([parse_mrc(x.strip(), is_vol=False, lazy=False)[0] for x in lines])
    else:
        particles = [img for x in lines for img in parse_mrc(x.strip(), is_vol=False, lazy=True)[0]]
    return particles, apix


def parse_mrc(fname, is_vol, lazy=False):
    mrc = mrcfile.mmap(fname,mode='r+',permissive=True)
    # parse the header
    header = mrc.header

    dtype = DTYPE_FOR_MODE[int(header.mode)]
    nz, ny, nx = header.nz, header.ny, header.nx

    # load all in one block
    if not lazy:
        if is_vol:
            array=mrc.data.transpose()
        else:
            array=mrc.data.transpose(0,2,1)
    # or list of LazyImages
    else:
        extbytes = header["nsymbt"]
        start = 1024 + extbytes  # start of image data
        stride = dtype().itemsize * ny * nx
        array = [
            LazyImage(fname, (ny, nx), dtype, start + i * stride) for i in range(nz)
        ]
    mrc.close()
    return array, header


def write(
    fname, array, Apix=1.0, xorg=0.0, yorg=0.0, zorg=0.0, is_vol=None
):
    if is_vol is None:
        is_vol = (
            True if len(set(array.shape)) == 1 else False
        )  # Guess whether data is vol or image stack
    with mrcfile.new(fname,overwrite=True) as mrc:
        if is_vol:
            mrc.set_data(array.transpose())
            mrc.set_volume()
        else:
            mrc.set_data(array.transpose(0,2,1))
            mrc.set_image_stack()
        mrc.voxel_size = Apix
        mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = xorg, yorg, zorg

