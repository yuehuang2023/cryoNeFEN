import numpy as np
import os
from torch.utils import data
import utils
import cryoio
import mrc
import random

log = utils.log


def load_particles(mrcs_txt_star, lazy=False, datadir=None):
    """
    Load particle stack from either a .mrcs file, a .star file, a .txt file containing paths to .mrcs files, or a
    cryosparc particles.cs file.

    lazy (bool): Return numpy array if True, or return list of LazyImages
    datadir (str or None): Base directory overwrite for .star or .cs file parsing
    """
    if mrcs_txt_star.endswith(".txt"):
        particles, apix = mrc.parse_mrc_list(mrcs_txt_star, lazy=lazy)
    elif mrcs_txt_star.endswith(".star"):
        # not exactly sure what the default behavior should be for the data paths if parsing a starfile
        try:
            star = cryoio.Starfile.load(mrcs_txt_star)
            particles = star.get_particles(
                datadir=datadir, lazy=lazy
            )
            apix = star.voxel_size
        except Exception as e:
            if datadir is None:
                datadir = os.path.dirname(
                    mrcs_txt_star
                )  # assume .mrcs files are in the same director as the starfile
                star = cryoio.Starfile.load(mrcs_txt_star)
                particles = star.get_particles(
                    datadir=datadir, lazy=lazy
                )
                apix = star.voxel_size
            else:
                raise RuntimeError(e)
    elif mrcs_txt_star.endswith(".cs"):
        particles, apix = cryoio.csparc_get_particles(mrcs_txt_star, datadir, lazy)
    elif mrcs_txt_star.endswith(".mrc") or mrcs_txt_star.endswith(".mrcs"):
        apix = mrc.get_voxelsize(mrcs_txt_star).x
        particles, _ = mrc.parse_mrc(mrcs_txt_star, lazy=lazy, is_vol=False)
    return particles, apix


class LazyMRCData(data.Dataset):
    """
    Class representing an .mrcs stack file -- images loaded on the fly
    """

    def __init__(
        self,
        mrcfile,
        norm=None,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        window_r=0.85,
        flog=None,
    ):
        log = flog if flog is not None else utils.log
        particles, apix = load_particles(mrcfile, True, datadir=datadir)
        if ind is not None:
            particles = particles[ind]
        N = len(particles)
        ny, nx = particles[0].get().shape
        assert ny == nx, "Images must be square"
        assert (
            ny % 2 == 0
        ), "Image size must be even. Is this a preprocessed dataset? Use the --preprocessed flag if so."
        log("Loaded {} {}x{} images".format(N, ny, nx))
        self.apix = apix
        self.particles = particles
        self.N = N
        self.D = ny  # after symmetrizing HT
        self.invert_data = invert_data
        if norm is None:
            norm = self.estimate_normalization()
        self.norm = norm
        self.window = window_mask(ny, window_r, 0.99) if window else None

    def estimate_normalization(self, n=1000):
        n = min(n, self.N)
        imgs = np.asarray(
            [
                self.particles[i].get()
                for i in random.sample(range(self.N), n)
            ]
        )
        if self.invert_data:
            imgs *= -1
        norm = [np.mean(imgs), np.std(imgs)]
        norm[0] = 0
        log("Normalizing HT by {} +/- {}".format(*norm))
        return norm

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        img = self.particles[index].get()
        if self.window is not None:
            img *= self.window
        if self.invert_data:
            img *= -1
        img = (img - self.norm[0]) / self.norm[1]
        return img, index


def window_mask(D, in_rad, out_rad):
    assert D % 2 == 0
    x0, x1 = np.meshgrid(
        np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
        np.linspace(-1, 1, D, endpoint=False, dtype=np.float32),
    )
    r = (x0**2 + x1**2) ** 0.5
    mask = np.minimum(1.0, np.maximum(0.0, 1 - (r - in_rad) / (out_rad - in_rad)))
    return mask


class MRCData(data.Dataset):
    """
    Class representing an .mrcs stack file
    """

    def __init__(
        self,
        mrcfile,
        norm=None,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        max_threads=16,
        window_r=0.85,
        flog=None,
        use_cupy=False,
    ):
        log = flog if flog is not None else utils.log

        particles, apix = load_particles(mrcfile, False, datadir=datadir)
        if ind is not None:
            particles = particles[ind]
        N, ny, nx = particles.shape
        assert ny == nx, "Images must be square"
        assert (
            ny % 2 == 0
        ), "Image size must be even. Is this a preprocessed dataset? Use the --preprocessed flag if so."
        log("Loaded {} {}x{} images".format(N, ny, nx))

        # Real space window
        if window:
            log(f"Windowing images with radius {window_r}")
            particles *= window_mask(ny, window_r, 0.99)


        if invert_data:
            particles *= -1

        # normalize
        if norm is None:
            norm = [np.mean(particles), np.std(particles)]
            norm[0]=0
        particles = (particles - norm[0]) / norm[1]
        log("Normalized RealSpace by {} +/- {}".format(*norm))

        self.apix = apix
        self.particles = particles
        self.N = N
        self.D = particles.shape[1]  # ny + 1 after symmetrizing HT
        self.norm = norm
        self.use_cupy = use_cupy


    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.particles[index], index

    def get(self, index):
        return self.particles[index]

