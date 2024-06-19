import numpy as np
import torch

try:
    import cupy as cp
except ImportError:
    cp = None


def fft2_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    return pp.fft.ifftshift(
        pp.fft.fft2(pp.fft.ifftshift(img))
    )


def ifft2_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    return pp.fft.fftshift(
        pp.fft.ifft2(pp.fft.fftshift(img))
    )



def fft2_centerGPU(img, **kwargs):

    return torch.fft.ifftshift(
        torch.fft.fft2(torch.fft.ifftshift(img), **kwargs)
    )


def fft2_centerGPU(img, **kwargs):

    return torch.fft.ifftshift(
        torch.fft.fft2(torch.fft.ifftshift(img), **kwargs)
    )


def ifft2_centerGPU(img, **kwargs):

    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.fftshift(img), **kwargs)
    )


def fftn_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    return pp.fft.ifftshift(pp.fft.fftn(pp.fft.ifftshift(img)))


def fftn_centerGPU(img, **kwargs):
    pp = np if isinstance(img, np.ndarray) else cp

    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.ifftshift(img, **kwargs), **kwargs), **kwargs)


def ifftn_center(V):
    pp = np if isinstance(V, np.ndarray) else cp

    V = pp.fft.fftshift(V)
    V = pp.fft.ifftn(V)
    V = pp.fft.fftshift(V)
    return V


def ht2_center(img):
    f = fft2_center(img)
    return f.real - f.imag


def ht2_centerGPU(img):
    f = fft2_centerGPU(img)
    return torch.real(f) - torch.imag(f)


def htn_center(img):
    pp = np if isinstance(img, np.ndarray) else cp

    f = pp.fft.fftshift(pp.fft.fftn(pp.fft.fftshift(img)))
    return f.real - f.imag


def iht2_center(img):
    img = fft2_center(img)
    img /= img.shape[-1] * img.shape[-2]
    return img.real - img.imag


def iht2_centerGPU(img):
    img = fft2_centerGPU(img)
    img /= img.shape[-1] * img.shape[-2]
    return torch.real(img) - torch.imag(img)


def ihtn_center(V):
    pp = np if isinstance(V, np.ndarray) else cp

    V = pp.fft.fftshift(V)
    V = pp.fft.fftn(V)
    V = pp.fft.fftshift(V)
    V /= pp.product(V.shape)
    return V.real - V.imag


def symmetrize_ht(ht, preallocated=False):
    pp = np if isinstance(ht, np.ndarray) else cp

    if preallocated:
        D = ht.shape[-1] - 1
        sym_ht = ht
    else:
        if len(ht.shape) == 2:
            ht = ht.reshape(1, *ht.shape)
        assert len(ht.shape) == 3
        D = ht.shape[-1]
        B = ht.shape[0]
        sym_ht = pp.empty((B, D + 1, D + 1), dtype=ht.dtype)
        sym_ht[:, 0:-1, 0:-1] = ht
    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]
    if len(sym_ht) == 1:
        sym_ht = sym_ht[0]
    return sym_ht


def symmetrize_htGPU(ht, preallocated=False):
    if preallocated:
        D = ht.shape[-1] - 1
        sym_ht = ht
    else:
        if len(ht.shape) == 2:
            ht = ht.reshape(1, *ht.shape)
        assert len(ht.shape) == 3
        D = ht.shape[-1]
        B = ht.shape[0]
        sym_ht = torch.empty((B, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
        sym_ht[:, 0:-1, 0:-1] = ht
    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]
    return sym_ht


def symmetrize_3D(ht):
    if len(ht.shape) == 3:
        ht = ht.reshape(1, *ht.shape)
    assert len(ht.shape) == 4
    D = ht.shape[-1]
    B = ht.shape[0]
    sym_ht = np.empty((B, D + 1, D + 1, D + 1), dtype=ht.dtype)
    sym_ht[:, 0:-1, 0:-1, 0:-1] = ht
    assert D % 2 == 0
    sym_ht[:, -1, :, :] = sym_ht[:, 0]  # last row is the first row
    sym_ht[:, :, -1, :] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, :, :, -1] = sym_ht[:, :, :, 0]  # last col is the first col
    sym_ht[:, -1, -1, -1] = sym_ht[:, 0, 0, 0]
    return sym_ht


def symmetrize_3DGPU(ht):
    if len(ht.shape) == 3:
        ht = ht.reshape(1, *ht.shape)
    assert len(ht.shape) == 4
    D = ht.shape[-1]
    B = ht.shape[0]
    sym_ht = torch.empty((B, D + 1, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1, 0:-1] = ht
    assert D % 2 == 0
    sym_ht[:, -1, :, :] = sym_ht[:, 0]  # last row is the first row
    sym_ht[:, :, -1, :] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, :, :, -1] = sym_ht[:, :, :, 0]  # last col is the first col
    sym_ht[:, -1, -1, -1] = sym_ht[:, 0, 0, 0]
    return sym_ht


