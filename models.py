"""Pytorch models"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lattice
import lie_tools
import utils
import fft
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from scipy.stats import truncnorm
from functorch import vmap

log = utils.log
BOX_OFFSETS = [[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]]

def unparallelize(model):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


class HetOnlyVAE(nn.Module):
    # No pose inference
    def __init__(
        self,
        lattice,  # Lattice object
        qlayers,
        qdim,
        players,
        pdim,
        in_dim,
        zdim=1,
        encode_mode="mlp",
        enc_mask=None,
        enc_type="geom_ft",
        enc_dim=None,
        domain="space",
        activation=nn.ReLU,
        feat_sigma=None,
    ):
        super(HetOnlyVAE, self).__init__()
        self.lattice = lattice
        self.zdim = zdim
        self.in_dim = in_dim
        self.enc_mask = enc_mask
        if encode_mode == "conv":
            self.encoder = ConvEncoder(in_dim, qdim, zdim * 2)
        elif encode_mode == "resid":
            self.encoder = ResidLinearMLP(
                in_dim,
                qlayers,
                qdim,
                zdim * 2,
                activation,  # nlayers  # hidden_dim  # out_dim
            )
        elif encode_mode == "mlp":
            self.encoder = MLP(
                in_dim, qlayers, qdim, zdim * 2, activation  # hidden_dim  # out_dim
            )  # in_dim -> hidden_dim
        else:
            raise RuntimeError("Encoder mode {} not recognized".format(encode_mode))
        self.encode_mode = encode_mode
        self.decoder = get_decoder(
            3 + zdim,
            lattice.D,
            players,
            pdim,
            domain,
            enc_type,
            enc_dim,
            activation,
            feat_sigma,
        )

    @classmethod
    def load(self, config, weights=None, device=None):
        """Instantiate a model from a config.pkl

        Inputs:
            config (str, dict): Path to config.pkl or loaded config.pkl
            weights (str): Path to weights.pkl
            device: torch.device object

        Returns:
            HetOnlyVAE instance, Lattice instance
        """
        cfg = utils.load_pkl(config) if type(config) is str else config
        c = cfg["lattice_args"]
        D=c["D"]
        lat = lattice.Lattice(c["D"], extent=c["extent"], device=device)
        c = cfg["model_args"]
        if c["enc_mask"] > 0:
            mask = lat.get_sphere_mask(c["enc_mask"])
            enc_mask = mask.view(D,D,D).sum(-1)>0
            in_dim = int(enc_mask.sum())
        else:
            assert c["enc_mask"] == -1
            enc_mask = None
            in_dim = lat.D**2
        activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c["activation"]]
        model = HetOnlyVAE(
            lat,
            c["qlayers"],
            c["qdim"],
            c["players"],
            c["pdim"],
            in_dim,
            c["zdim"],
            encode_mode=c["encode_mode"],
            enc_mask=enc_mask,
            enc_type=c["pe_type"],
            enc_dim=c["pe_dim"],
            domain=c["domain"],
            activation=activation,
            feat_sigma=c["feat_sigma"],
        )
        if weights is not None:
            ckpt = torch.load(weights)
            model.load_state_dict(ckpt["model_state_dict"])
        if device is not None:
            model.to(device)
        return model, lat

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, img):
        if self.enc_mask is not None:
            img = img[:, self.enc_mask]
        elif self.encode_mode == 'conv':
            img = img[:,None]
        else:
            img = img.view(img.shape[0], -1)
        z = self.encoder(img)
        return z[:, : self.zdim], z[:, self.zdim :]

    def cat_z(self, coords, z):
        """
        coords: Bx...x3
        z: Bxzdim
        """
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat((coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, mask=None, z=None):
        """
        coords: BxNx3 image coordinates
        z: Bxzdim latent coordinate
        """
        if z is not None:
            coords = self.cat_z(coords, z)
        return self.decoder(coords, mask=mask)

    # Need forward func for DataParallel -- TODO: refactor
    def forward(self, img=None, coords=None, mask=None, encode_only=False, z=None):
        if img is not None:
            z_mu, z_logvar = self.encode(img)
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z_mu = z
            z_logvar = None
        if encode_only:
            return z_mu, z_logvar
        else:
            recon = self.decode(coords, mask=mask, z=z)
            return recon, z_mu, z_logvar


def load_decoder(config, weights=None, device=None):
    """
    Instantiate a decoder model from a config.pkl

    Inputs:
        config (str, dict): Path to config.pkl or loaded config.pkl
        weights (str): Path to weights.pkl
        device: torch.device object

    Returns a decoder model
    """
    cfg = utils.load_pkl(config) if type(config) is str else config
    c = cfg["model_args"]
    D = cfg["lattice_args"]["D"]
    activation = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}[c["activation"]]
    model = get_decoder(
        3,
        D,
        c["layers"],
        c["dim"],
        c["domain"],
        c["pe_type"],
        c["pe_dim"],
        activation,
        c["feat_sigma"],
    )
    if weights is not None:
        ckpt = torch.load(weights)
        model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model.to(device)
    return model


def get_decoder(
    in_dim,
    D,
    layers,
    dim,
    enc_type,
    enc_dim=None,
    activation=nn.ReLU,
    feat_sigma=None,
):
    if enc_type == "none":
        model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
        ResidLinearMLP.eval_volume = PositionalDecoder.eval_volume  # EW FIXME
        return model
    else:
        model = PositionalDecoder 
        return model(
            in_dim,
            D,
            layers,
            dim,
            activation,
            enc_type=enc_type,
            enc_dim=enc_dim,
            feat_sigma=feat_sigma,
        )


class PositionalDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        D,
        nlayers,
        hidden_dim,
        activation,
        dec_type='mlp',
        enc_type="geom_ft",
        enc_dim=None,
        feat_sigma=.5,
        bias=True,
        base_radius = None,
        pos_embed = 0,
        scale_correction = 0,
    ):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 3
        self.zdim = in_dim - 3
        self.D = D
        self.D2 = D // 2
        self.DD = 2 * (D // 2)
        self.enc_dim = self.D2  if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = self.enc_dim *6 + self.zdim
        self.dec_type=dec_type
        self.base_radius = base_radius
        self.scale_correction = False
        self.pos_est = False
        self.rand_freqs = None

        if pos_embed>0:
            self.rot_embed = nn.Embedding(pos_embed, 6)
            self.trans_embed = nn.Embedding(pos_embed, 2)
            # nn.init.uniform_(self.trans_embed.weight, a=-0.0001, b=0.0001)
            self.trans_embed.weight.data.zero_()
            self.pos_est = True
        if scale_correction > 0:
            self.scale = nn.Embedding(scale_correction, 1)
            nn.init.zeros_(self.scale.weight)
            self.background = nn.Embedding(scale_correction, 1)
            nn.init.zeros_(self.background.weight)
            self.scale_correction = True

        if enc_type == "gaussian":
            # We construct 3 * self.enc_dim random vector frequences, to match the original positional encoding:
            # In the positional encoding we produce self.enc_dim features for each of the x,y,z dimensions,
            # whereas in gaussian encoding we produce self.enc_dim features each with random x,y,z components
            #
            # Each of the random feats is the sine/cosine of the dot product of the coordinates with a frequency
            # vector sampled from a gaussian with std of feat_sigma

            rand_freqs = (
                torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * feat_sigma
            )
            # rand_freqs = torch.Tensor(truncnorm.rvs(-np.pi*2/feat_sigma, np.pi*2/feat_sigma, scale = feat_sigma * self.D2, size = (3 * self.enc_dim, 3)))
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs = nn.Parameter(rand_freqs, requires_grad=False) 
            self.feat_sigma = feat_sigma
        else:
            self.rand_feats = None

        if dec_type == 'mlp':
            self.decoder = MLP(self.in_dim, nlayers, hidden_dim, 1, activation, bias=bias)
        elif dec_type == 'resid':
            self.decoder = ResidLinearMLP(self.in_dim, nlayers, hidden_dim, 1, activation, bias=bias)
    
    def positional_encoding_geom(self, coords):
        """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        if self.enc_type == "geom_ft":
            freqs = (
                2 * np.pi * (self.DD / 2) ** (freqs / (self.enc_dim - 1))
            )  # option 1: 2/D to 1
        elif self.enc_type == "geom_full":
            freqs = (
                self.DD
                * np.pi
                * (1.0 / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
            )  # option 2: 2/D to 2pi
        elif self.enc_type == "geom_lowf":
            freqs = self.D2 * (1.0 / self.D2) ** (
                freqs / (self.enc_dim - 1)
            )  # option 3: 2/D*2pi to 2pi
        elif self.enc_type == "geom_nohighf":
            freqs = self.D2 * (2.0 * np.pi / self.D2) ** (
                freqs / (self.enc_dim - 1)
            )  # option 4: 2/D*2pi to 1
        elif self.enc_type == "linear_lowf":
            return self.positional_encoding_linear(coords)
        elif (self.enc_type == "gaussian") or (self.enc_type == "log_uniform"):
            return self.random_fourier_encoding(coords)
        else:
            raise RuntimeError("Encoding type {} not recognized".format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        x = torch.cat([torch.sin(coords * freqs), torch.cos(coords * freqs)], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        return x

    def positional_encoding_linear(self, coords):
        """Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2"""
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float, device=coords.device)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        k = coords[..., 0:3, :] * freqs  # B x 3 x D2
        s = torch.sin(k)  # B x 3 x D2
        c = torch.cos(k)  # B x 3 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def random_fourier_encoding(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        k = (coords[..., None, :] * freqs).sum(-1)  # compute k
        s = torch.sin(k) 
        c = torch.cos(k) 
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        return x
    
    def reinitialize(self):
        if self.rand_freqs is not None:
            if self.symmetry is not None:
                rand_freqs = []
                rand_freqs_tmp = torch.randn((3 * self.enc_dim // len(self.symmetry), 3), dtype=torch.float) * self.feat_sigma
                for i in range(len(self.symmetry)):
                    rand_freqs.append(rand_freqs_tmp @ self.symmetry[i].permute([1, 0]))
                rand_freqs = torch.cat(rand_freqs)
            else:
                rand_freqs = (
                    torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * self.feat_sigma
                )
            # make rand_feats a parameter so it is saved in the checkpoint, but do not perform SGD on it
            self.rand_freqs.data = rand_freqs.to(self.rand_freqs.device)
        for layer in self.decoder.main:
            if isinstance(layer, MyLinear):
                layer.reset_parameters()


    def forward(self, coords, mask=None, ctf=None):
        """Input should be coordinates from [-.5,.5]"""
        # assert (coords[..., 0:3].abs() - 0.5 < 1e-4).all()
        B = coords.shape[0]
        D = self.D
        if self.zdim>0:
            z = coords[...,3:]
        
        if self.zdim > 0:
            output = self.decoder(torch.cat([self.positional_encoding_geom(coords[mask>0]), z], -1))
        else:
            output = self.decoder(self.positional_encoding_geom(coords[mask>0]))
        
        output = torch.zeros([B,D**3],device=coords.device, dtype=output.dtype).masked_scatter(mask>0, output).view(B,D,D,D)
        if ctf is not None:
            output = torch.mean(output, -1)
            output = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(torch.fft.ifftshift(output.float()))*torch.fft.fftshift(ctf)[..., :D//2+1]))

        return output


class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias = True):
        super(ResidLinearMLP, self).__init__()
        layers = [
            ResidLinear(in_dim, hidden_dim, bias = bias)
            if in_dim == hidden_dim
            else MyLinear(in_dim, hidden_dim, bias = bias),
            activation(inplace=True),
        ]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim, bias = bias))
            layers.append(activation(inplace=True))
        layers.append(
            ResidLinear(hidden_dim, out_dim, bias = bias)
            if out_dim == hidden_dim
            else MyLinear(hidden_dim, out_dim, bias = bias)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        flat = x.view(-1, x.shape[-1])
        ret_flat = self.main(flat)
        ret = ret_flat.view(*x.shape[:-1], ret_flat.shape[-1])
        return ret


def half_linear(input, weight, bias):
    # print('half', input.shape, weight.shape)
    if bias is not None:
        return F.linear(input, weight.half(), bias.half())
    else:
        return F.linear(input, weight.half())


def single_linear(input, weight, bias):
    # print('single', input.shape, weight.shape)
    # assert input.shape[0] < 10000

    return F.linear(input, weight, bias)


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


class MyLinear(nn.Linear):
    def forward(self, input):
        if input.dtype == torch.half:
            return half_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight.half(), self.bias.half())
        else:
            return single_linear(
                input, self.weight, self.bias
            )  # F.linear(input, self.weight, self.bias)


class ResidLinear(nn.Module):
    def __init__(self, nin, nout, bias = True):
        super(ResidLinear, self).__init__()
        self.linear = MyLinear(nin, nout, bias = bias)
        # self.linear = nn.utils.weight_norm(MyLinear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation, bias = True):
        super(MLP, self).__init__()
        layers = [MyLinear(in_dim, hidden_dim, bias = bias), activation(inplace = True)]
        for n in range(nlayers):
            layers.append(MyLinear(hidden_dim, hidden_dim, bias = bias))
            layers.append(activation(inplace=True))
        layers.append(MyLinear(hidden_dim, out_dim, bias = bias))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Adapted from soumith DCGAN
class ConvEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(in_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )

    def forward(self, x):
        if x.shape[-1] % 64 != 0:
            pad_size = ((x//64+1)*64 - x)//2
            x = F.pad(x, [pad_size]*4)
        x = x.view(x.shape[0], -1, 64, 64)
        x = self.main(x)
        return x.view(x.size(0), -1)  # flatten


    
