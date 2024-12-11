import os
from datetime import datetime as dt
import numpy as np
from scipy import ndimage
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mrc
import argparse
import torch
import torch.nn as nn
import pickle
import fft
from collections import OrderedDict
from lattice import Lattice
import itertools
import torch.nn.functional as F
from pytorch3d.transforms import *
import healpy as hp


def add_args(parser):
    parser.add_argument(
        "volumes",
        type=os.path.abspath,
        help="Half-maps directory (.mrc)",
    )
    parser.add_argument(
        "--mask", metavar="mrc", type=os.path.abspath, help="FSC mask (.mrc)"
    )
    parser.add_argument(
        "--Apix", type=float, help="Angstroms per pixel"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        help="Output directory to save model",
    )
    return parser

def calc_fsc(vol1, vol2):
    """
    Helper function to calculate the FSC between two (assumed masked) volumes
    vol1 and vol2 should be maps of the same box size, structured as numpy arrays with ndim=3, i.e. by loading with
    cryodrgn.mrc.parse_mrc
    """
    # load masked volumes in fourier space

    vol1_ft = np.fft.fftshift(np.fft.fftn(vol1))
    vol2_ft = np.fft.fftshift(np.fft.fftn(vol2))

    # define fourier grid and label into shells
    Dx, Dy, Dz = vol1.shape
    x = np.arange(-Dx // 2, Dx // 2)
    y = np.arange(-Dy // 2, Dy // 2) 
    z = np.arange(-Dz // 2, Dz // 2)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    r_max = max(Dx,Dy,Dz) // 2  # sphere inscribed within volume box
    r_step = 1  # int(np.min(r[r>0]))
    bins = np.arange(0, r_max, r_step)
    bin_labels = np.searchsorted(bins, r, side="right")

    # calculate the FSC via labeled shells
    num = ndimage.sum(
        np.real(vol1_ft * np.conjugate(vol2_ft)), labels=bin_labels, index=bins + 1
    )
    den1 = ndimage.sum(np.abs(vol1_ft) ** 2, labels=bin_labels, index=bins + 1)
    den2 = ndimage.sum(np.abs(vol2_ft) ** 2, labels=bin_labels, index=bins + 1)
    fsc = num / np.sqrt(den1 * den2)

    x = bins / Dx  # x axis should be spatial frequency in 1/px
    return x, fsc

def fsc2res(freq, fsc, thresh=0.143, Apix=1):
    idx=np.searchsorted(-fsc,-thresh)
    if idx < freq.shape[0]:
        res = 1/freq[idx] * Apix
    else:
        res = 2 * Apix
    return res

def grid_sample_complex(input, grid, **kwargs):
    output_real = F.grid_sample(input.real, grid, **kwargs)
    output_imag = F.grid_sample(input.imag, grid, **kwargs)
    return torch.complex(output_real, output_imag)

def parse_ccp4(file, return_mask = False):
    vol, header = mrc.parse_mrc(file, is_vol = True)
    full = np.zeros([header.mz, header.my, header.mx])
    mask = np.zeros([header.mz, header.my, header.mx])
    full[header.nzstart:header.nzstart+header.nz, header.nystart:header.nystart+header.ny, header.nxstart:header.nxstart+header.nx] = vol.transpose()
    mask[header.nzstart:header.nzstart+header.nz, header.nystart:header.nystart+header.ny, header.nxstart:header.nxstart+header.nx] = 1.
    if return_mask:
        return full, mask
    else:
        return full

def calc_fslc(file1, file2, mask=None, nside=32, device = 'cuda', batch_size = 64, freq_range = None):
    if device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    B = batch_size
    NPIX = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, torch.arange(NPIX))
    R = euler_angles_to_matrix(torch.stack([torch.zeros_like(theta), theta, torch.pi - phi], -1), 'ZYZ').to(device=device)

    if file1.endswith('.ccp4'):
        vol1 = parse_ccp4(file1)
        vol2 = parse_ccp4(file2)
    elif file1.endswith('.mrc'):
        vol1 = mrc.parse_mrc(file1, is_vol=True)[0]
        vol2 = mrc.parse_mrc(file2, is_vol=True)[0]
    assert vol1.shape == vol2.shape, "The volumes should have the same shape"
    if mask is not None:
        mask = mrc.parse_mrc(mask, is_vol=True)[0]
        volft1 = torch.tensor(np.fft.fftshift(np.fft.fftn(vol1*mask)), device = device)
        volft2 = torch.tensor(np.fft.fftshift(np.fft.fftn(vol2*mask)), device = device)
    else:
        volft1 = torch.tensor(np.fft.fftshift(np.fft.fftn(vol1)), device = device)
        volft2 = torch.tensor(np.fft.fftshift(np.fft.fftn(vol2)), device = device)
    volft1 = fft.symmetrize_3DGPU(volft1)
    volft2 = fft.symmetrize_3DGPU(volft2)
    D = vol1.shape[0]
    lattice = Lattice(D + 1, D + 1, 1., device=device, dtype=torch.float64, endpoint=True)
    coords = F.pad(lattice.coords_2d,[0, 1])
    if freq_range is not None:
        mask_high = lattice.get_circular_mask(freq_range[1]).view(D+1,D+1)
        mask_low = lattice.get_circular_mask(freq_range[0]).view(D+1,D+1)
        freq_mask = (~mask_low) & (mask_high)
    else:
        freq_mask = lattice.get_circular_mask(D//2).view(D+1,D+1)
    fslc = []
    with torch.no_grad():
        for i in range(NPIX//B + 1):
            if i == NPIX//B:
                R_tmp = R[i*B:]
            else:
                R_tmp = R[i*B:i*B+B]
            B_tmp = R_tmp.shape[0]
            if B_tmp == 0:
                continue
            plane = coords[None] @ R_tmp
            slice1 = grid_sample_complex(volft1.permute(0,3,2,1).repeat(B,1,1,1,1), plane.view(B_tmp,1,1,-1,3), mode='nearest', align_corners=True).view(B_tmp, D+1,D+1) * freq_mask[None]
            slice2 = grid_sample_complex(volft2.permute(0,3,2,1).repeat(B,1,1,1,1), plane.view(B_tmp,1,1,-1,3), mode='nearest', align_corners=True).view(B_tmp, D+1,D+1) * freq_mask[None]
            corr = torch.sum(torch.real(slice1 * torch.conj(slice2)), dim = (-1, -2))/(torch.linalg.norm(slice1, dim = (-1, -2)) * torch.linalg.norm(slice2, dim = (-1, -2)))
            fslc.append(corr.cpu().numpy())
    fslc = np.concatenate(fslc, -1)
    return fslc

def main(args):
    if args.outdir is not None:
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir = args.volumes
    dir1 = args.volumes + '/halfA/'
    dir2 = args.volumes + '/halfB/'
    files1 = []
    for file in os.listdir(dir1):
        if file.endswith('.mrc'):
            files1.append(file)
    files1.sort()
    files2 = [] 
    for file in os.listdir(dir2):
        if file.endswith('.mrc'):
            files2.append(file)
    files2.sort()
    assert len(files1) == len(files2), "Must have same number of maps in two halves"
    if args.mask is not None:
        mask, _ = mrc.parse_mrc(args.mask, is_vol=True)
    else:
        mask = None
        log('Warning: make sure the volumes unmasked')
    Apix = mrc.get_voxelsize(dir1 + files1[0]).x
    if np.isnan(Apix):
        Apix = 1.
    if args.Apix is not None:
        Apix = args.Apix
    fscs = []
    count = 0
    for file1, file2 in zip(files1, files2):
        count += 1
        log('Calculating GSFSC for {}th volume'.format(count))
        vol1, _ = mrc.parse_mrc(dir1 + file1, is_vol = True)
        vol2, _ = mrc.parse_mrc(dir2 + file2, is_vol = True)
        if mask is not None:
            freq, fsc = calc_fsc(vol1*mask, vol2*mask)
        else:
            freq, fsc = calc_fsc(vol1, vol2)
        fscs.append(fsc)
    fscs = np.stack(fscs)
    if mask is not None:
        vol_sum = (vol1+vol2)*mask/2
    else:
        vol_sum = (vol1+vol2)/2
    D = vol_sum.shape[0]
    res = fsc2res(freq, fsc, Apix=Apix)
    utils.save_pkl(fscs, outdir + '/fsc.pkl')
    plt.figure(1)
    plt.plot(freq, fsc)
    plt.axhline(0.143, c='k')
    plt.xlim([0, 0.5])
    plt.ylim([0, 1])
    plt.xlabel('Frequency')
    plt.ylabel('GSFSC')  
    plt.xticks(np.linspace(0,0.5,6),['DC']+['{:.1f}'.format(1/ele*Apix) for ele in np.linspace(0,0.5,6)[1:]])     
    plt.title('Resolution: {:.2f} A'.format(res)) 
    plt.savefig(outdir + '/gsfsc.png', bbox_inches='tight')
    plt.figure(2, figsize=(8,3))
    plt.subplot(131)
    plt.imshow(vol_sum[D//2].transpose())
    plt.title('xslice')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(vol_sum[:, D//2].transpose())
    plt.title('yslice')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(vol_sum[:, :, D//2].transpose())
    plt.title('zslice')
    plt.axis('off')
    plt.savefig(outdir + '/volslice.png', bbox_inches='tight')
    log('Save results to {}'.format(outdir))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    log = utils.log
    main(args)
