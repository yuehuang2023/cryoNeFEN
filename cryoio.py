from collections import OrderedDict
import os
import numpy as np
import mrc
from mrc import LazyImage
import starfile
import ctf
import utils
import torch
import lie_tools


log = utils.log


class Starfile:
    def __init__(self, df, headers=None, data_optics=None, relion31=False):
        if headers:
            assert headers == list(df), f"{headers} != {df.columns}"
            self.headers = headers
        else:
            self.headers = list(df)
        self.df = df
        if 'rlnDetectorPixelSize' in df and 'rlnMagnification' in df:
            self.voxel_size = float(df['rlnDetectorPixelSize'][0])/float(df['rlnMagnification'][0])*1e4 
        elif 'rlnImagePixelSize' in df:
            self.voxel_size = float(df['rlnImagePixelSize'][0])
        else:
            self.voxel_size = None
        self.data_optics = data_optics
        self.relion31 = relion31

    def __len__(self):
        return len(self.df)
        
    @classmethod
    def load(cls, file):
        # detect star file type
        df = starfile.read(file)
        if isinstance(df, OrderedDict):
            return cls(df['particles'], data_optics = df['optics'], relion31 = True)
        else:
            return cls(df)

    def write(self, outstar):
        if self.relion31:
            df = {'optics':self.data_optics, 'particles':self.df}
        else:
            df = self.df
        starfile.write(df, outstar, overwrite=True)

    def get_particles(self, datadir=None, lazy=True):
        """
        Return particles of the starfile

        Input:
            datadir (str): Overwrite base directories of particle .mrcs
                Tries both substituting the base path and prepending to the path
            If lazy=True, returns list of LazyImage instances, else np.array
        """
        # format is index@path_to_mrc
        inds =[]
        mrcs = []
        for particle in self.df["rlnImageName"]:
            ind, mrc_path = particle.split('@')
            inds.append(int(ind)-1)
            if datadir is not None:
                try:
                    path = "{}/{}".format(datadir, os.path.basename(mrc_path))
                    assert os.path.exists(path)
                except AssertionError:
                    path = "{}/{}".format(datadir, mrc_path)
                    assert os.path.exists(path), f"{path} not found"
                mrcs.append(path)
            else:
                assert os.path.exists(mrc_path), f"{mrc_path} not found"
                mrcs.append(mrc_path)
        header = mrc.parse_header(mrcs[0])
        apix = mrc.get_voxelsize(mrcs[0]).x
        D = header['nx']  # image size along one dimension in pixels
        dtype = mrc.DTYPE_FOR_MODE[int(header['mode'])]
        stride = dtype().itemsize * D * D
        dataset = []
        for ii, f in zip(inds, mrcs):
            img = LazyImage(f, (D, D), dtype, 1024 + ii * stride)
            if lazy:
                dataset.append(img)
            else:
                dataset.append(img.get())
        if not lazy:
            dataset = np.array(dataset)
        return dataset


def prefix_paths(mrcs, datadir):
    mrcs1 = ["{}/{}".format(datadir, os.path.basename(x)) for x in mrcs]
    mrcs2 = ["{}/{}".format(datadir, x) for x in mrcs]
    try:
        for path in set(mrcs1):
            assert os.path.exists(path)
        mrcs = mrcs1
    except AssertionError:
        for path in set(mrcs2):
            assert os.path.exists(path), f"{path} not found"
        mrcs = mrcs2
    return mrcs


def csparc_get_particles(csfile, datadir=None, lazy=True):
    metadata = np.load(csfile)
    inds = metadata["blob/idx"]  # 0-based indexing
    mrcs = metadata["blob/path"].astype(str).tolist()
    if mrcs[0].startswith(">"):  # Remove '>' prefix from paths
        mrcs = [x[1:] for x in mrcs]
    if datadir is not None:
        mrcs = prefix_paths(mrcs, datadir)
    else:
        for path in set(mrcs):
            assert os.path.exists(path), f"{path} not found"
    apix = mrc.get_voxelsize(mrcs[0]).x
    D = metadata[0]["blob/shape"][0]
    dtype = np.float32
    stride = np.float32().itemsize * D * D
    dataset = []
    for ii, f in zip(inds, mrcs):
        img = LazyImage(f, (D, D), dtype, 1024 + ii * stride)
        if lazy:
            dataset.append(img)
        else:
            dataset.append(img.get())
 
    dataset = np.array(dataset)
    return dataset, apix


def parse_ctf_csparc(cs, D=None, Apix=None):
    metadata = np.load(cs)
    N = len(metadata)

    # sometimes blob/shape, blob/psize_A are missing from the .cs file
    try:
        D = metadata["blob/shape"][0][0]
        Apix = metadata["blob/psize_A"]
    except ValueError:
        assert D, "Missing image size in .cs file"
        assert Apix, "Missing pixel size in .cs file"

    ctf_params = np.zeros((N, 9), dtype=np.float32)
    ctf_params[:, 0] = D
    ctf_params[:, 1] = Apix
    fields = (
        "ctf/df1_A",
        "ctf/df2_A",
        "ctf/df_angle_rad",
        "ctf/accel_kv",
        "ctf/cs_mm",
        "ctf/amp_contrast",
        "ctf/phase_shift_rad",
    )
    for i, f in enumerate(fields):
        ctf_params[:, i + 2] = metadata[f]
        if f in ("ctf/df_angle_rad", "ctf/phase_shift_rad"):  # convert to degrees
            ctf_params[:, i + 2] *= 180 / np.pi

    ctf.print_ctf_params(ctf_params[0])
    return ctf_params


def parse_ctf_relion(star, **args):
    HEADERS = [
        "rlnDefocusU",
        "rlnDefocusV",
        "rlnDefocusAngle",
        "rlnVoltage",
        "rlnSphericalAberration",
        "rlnAmplitudeContrast",
        "rlnPhaseShift",
    ]
    s = Starfile.load(star)
    N = len(s.df)
    overrides = {}
    if s.relion31:
        assert len(s.data_optics.df) == 1, "Only one optics group supported"
        args.D = int(s.data_optics.df["rlnImageSize"][0])
        args.Apix = float(s.data_optics.df["rlnImagePixelSize"][0])
        overrides[HEADERS[3]] = float(s.data_optics.df[HEADERS[3]][0])
        overrides[HEADERS[4]] = float(s.data_optics.df[HEADERS[4]][0])
        overrides[HEADERS[5]] = float(s.data_optics.df[HEADERS[5]][0])
    else:
        assert args.D is not None, "Must provide image size with -D"
        assert args.Apix is not None, "Must provide pixel size with --Apix"

    # Sometimes CTF parameters are missing from the star file
    if args.kv is not None:
        log(f"Overriding accerlating voltage with {args.kv} kV")
        overrides[HEADERS[3]] = args.kv
    if args.cs is not None:
        log(f"Overriding spherical abberation with {args.cs} mm")
        overrides[HEADERS[4]] = args.cs
    if args.w is not None:
        log(f"Overriding amplitude contrast ratio with {args.w}")
        overrides[HEADERS[5]] = args.w
    if args.ps is not None:
        log(f"Overriding phase shift with {args.ps}")
        overrides[HEADERS[6]] = args.ps

    ctf_params = np.zeros((N, 9))
    ctf_params[:, 0] = args.D
    ctf_params[:, 1] = args.Apix
    for i, header in enumerate(HEADERS):
        ctf_params[:, i + 2] = (
            s.df[header] if header not in overrides else overrides[header]
        )
    ctf.print_ctf_params(ctf_params[0])
    return ctf_params


def parse_pose_csparc(cs):
    raw = np.load(cs)
    RKEY = "alignments3D/pose"
    TKEY = "alignments3D/shift"
    # parse rotations
    log(f"Extracting rotations from {RKEY}")
    rot = np.array([x[RKEY] for x in raw])
    rot = torch.tensor(rot)
    rot = lie_tools.expmap(rot)
    rot = rot.numpy()
    log("Transposing rotation matrix")
    rot = np.array([x.T for x in rot])

    # parse translations
    log(f"Extracting translations from {TKEY}")
    trans = np.array([x[TKEY] for x in raw])
    return rot, trans


def parse_pose_relion(star, Apix=None):
    s = Starfile.load(star)
    # parse rotations
    N = len(s.df)
    euler = np.zeros((N, 3))
    euler[:, 0] = s.df["rlnAngleRot"]
    euler[:, 1] = s.df["rlnAngleTilt"]
    euler[:, 2] = s.df["rlnAnglePsi"]
    log("Euler angles (Rot, Tilt, Psi):")
    log(euler[0])
    log("Converting to rotation matrix:")
    rot = np.asarray([utils.R_from_relion(*x) for x in euler])
    log(rot[0])
    # parse translations
    trans = np.zeros((N, 2))
    if "rlnOriginX" in s.headers and "rlnOriginY" in s.headers:
        # translations in pixels
        trans[:, 0] = s.df["rlnOriginX"]
        trans[:, 1] = s.df["rlnOriginY"]
    elif "rlnOriginXAngst" in s.headers and "rlnOriginYAngst" in s.headers:
        # translation in Angstroms (Relion 3.1)
        assert (
            Apix is not None
        ), "Must provide Apix to convert _rlnOriginXAngst and _rlnOriginYAngst translation units"
        trans[:, 0] = s.df["rlnOriginXAngst"]
        trans[:, 1] = s.df["rlnOriginYAngst"]
        trans /= Apix
    else:
        log(
            "Warning: Neither _rlnOriginX/Y nor _rlnOriginX/YAngst found. Defaulting to 0s."
        )
    log("Translations (pixels):")
    log(trans[0])
    return rot, trans


if __name__ == '__main__':
    star = Starfile.load('/media/sdc/huangyue/empiar-10076/Parameters.star')
    # star.write('/media/sdc/huangyue/empiar-10792/shiny_H2_test.star')
    datadir = '/media/sdc/huangyue/empiar-10076/'
    datasets = star.get_particles(datadir = datadir, lazy=False)[0]
    print(datasets.shape)
    