"""Lattice object"""

import numpy as np
import torch

import utils

log = utils.log
vlog = utils.vlog


class Lattice:
    def __init__(self, D, Dz=None, extent=0.5, indexing='ij' ,endpoint=False, centered=False, ignore_DC=False, device=None):
        
        if not Dz:
            Dz = D

        coords_x = np.linspace(-extent, extent, D, endpoint=endpoint)
        coords_y = np.linspace(-extent, extent, D, endpoint=endpoint)
        coords_z = np.linspace(-extent, extent, Dz, endpoint=endpoint)

        if centered:
            coords_x += extent/D
            coords_y += extent/D
            coords_z += extent/D
        self.centered = centered

        self.coords_x=coords_x
        self.coords_y=coords_y
        self.coords_z=coords_z

        x0, x1, x2 = np.meshgrid(
            coords_x,
            coords_y,
            coords_z,
            indexing=indexing
        )

        coords = np.stack([x0.ravel(), x1.ravel(), x2.ravel()], -1).astype(
            np.float32
        )

        self.coords = torch.tensor(coords, device=device)

        self.extent = extent
        self.D = D
        self.Dz=Dz
        self.D2 = int(D / 2)

        # todo: center should now just be 0,0; check Lattice.rotate...
        # c = 2/(D-1)*(D/2) -1
        self.center = torch.tensor([0.0, 0.0], device=device)

        x0, x1 = np.meshgrid(
            coords_x,
            coords_y,
            indexing=indexing
        )
        coords_2d = np.stack([x0.ravel(), x1.ravel()], -1).astype(
            np.float32
        )
        self.coords_2d = torch.tensor(coords_2d, device=device)

        self.square_mask = {}
        self.circle_mask = {}

        self.rings_mask = {}
        self.sphere_mask = {}
        self.shell_index = {}

        self.indexing=indexing
        self.ignore_DC = ignore_DC
        self.device = device


    def get_circular_mask(self, R):
        """Return a binary mask for self.coords which restricts coordinates to a centered circular lattice"""
        if R in self.circle_mask:
            return self.circle_mask[R]
        assert (
            2 * R  <= self.D
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        vlog("Using circular lattice with radius {}".format(R))

        r = R / (self.D // 2) * self.extent

        self.circle_mask[R] = self.coords_2d.pow(2).sum(-1) <= r**2
        return self.circle_mask[R]

    def get_rings_mask(self, R, minR=4, ignore_low=True):
        if R in self.rings_mask:
            return self.rings_mask[R]
        assert (
            2 * R  <= self.D
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        vlog("Using circular lattice with radius {}".format(R))

        r = R / (self.D // 2) * self.extent
        x0, x1 = np.meshgrid(self.coords_x, self.coords_y, indexing='ij')
        rr = np.sqrt(x0**2 + x1**2)
        if not ignore_low:
            rings = [rr<=minR/R*r]
            rings[0][self.D//2,self.D//2, 0] = False
        else:
            rings = []
        for i in range(minR, R):
            rings.append((rr>i/R*r) & (rr <= (i+1)/R*r))
        rings=np.stack(rings,-1)
        rings=rings/rings.sum((0,1),keepdims=True).astype(np.float32)
        self.rings_mask[R] = torch.tensor(rings, device=self.device)
        return self.rings_mask[R]
        
    def get_sphere_mask(self, R, soft_edge=0):
        if R in self.sphere_mask:
            return self.sphere_mask[R]
        assert (
            2 * R  <= self.D 
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        vlog("Using sphere lattice with radius {}".format(R))
        # coords=self.coords.view(self.D,self.D,self.Dz,3)
        r = R / (self.D // 2) * self.extent 
        soft_edge = soft_edge / (self.D // 2) * self.extent 
        if soft_edge>0:
            mask_edge = (self.coords.pow(2).sum(-1) <= r**2) & (self.coords.pow(2).sum(-1) > (r-soft_edge)**2)
            r_edge = (torch.sqrt(self.coords.pow(2).sum(-1)[mask_edge])-(r-soft_edge)) /soft_edge * torch.pi
            mask = (self.coords.pow(2).sum(-1) <= (r-soft_edge)**2).to(torch.float32).masked_scatter(mask_edge, 0.5+0.5*torch.cos(r_edge))
        else:
            mask=self.coords.pow(2).sum(-1) <= r**2
        self.sphere_mask[R] = mask
        return mask

