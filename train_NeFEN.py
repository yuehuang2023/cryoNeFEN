import os
import pickle
import sys
from datetime import datetime as dt
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from pose import PoseTracker
import dataset
import ctf
from lattice import Lattice
from models import PositionalDecoder,unparallelize
import utils
import matplotlib.pyplot as plt
import mrc
import fft
import lie_tools
import argparse
import cryoio


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True

log = utils.log
vlog = utils.vlog


def add_args(parser):
    parser.add_argument(
        "particles",
        type=os.path.abspath,
        help="Input particles (.mrcs, .star, .cs, or .txt)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save model",
    )
    parser.add_argument(
        "--poses", type=os.path.abspath, help="Image poses (.pkl)"
    )
    parser.add_argument(
        "--ctf", metavar="pkl", type=os.path.abspath, help="CTF parameters (.pkl)"
    )
    parser.add_argument(
        "--mask", metavar="mrc", type=os.path.abspath, help="Optional mask (.mrc, default: sphere mask)"
    )
    parser.add_argument(
        "--split",         
        choices=("1", "2"),
        help="Split dataset for computing GSFSC"
    )
    parser.add_argument(
        "--load", metavar="WEIGHTS.PKL", help="Initialize training from a checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=1,
        help="Checkpointing interval in N_EPOCHS (default: %(default)s)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval in N_IMGS (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(0, 100000), help="Random seed"
    )

    group = parser.add_argument_group("Dataset loading")
    group.add_argument(
        "--uninvert-data",
        dest="invert_data",
        action="store_false",
        help="Do not invert data sign",
    )
    group.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="Turn off real space windowing of dataset",
    )
    group.add_argument(
        "--window-r",
        type=float,
        default=0.85,
        help="Windowing radius (default: %(default)s)",
    )
    group.add_argument(
        "--ind", type=os.path.abspath, help="Filter particle stack by these indices"
    )
    group.add_argument(
        "--lazy",
        action="store_true",
        help="Lazy loading if full dataset is too large to fit in memory",
    )
    group.add_argument(
        "--datadir",
        type=os.path.abspath,
        help="Path prefix to particle stack if loading relative paths from a .star or .cs file",
    )

    group = parser.add_argument_group("Training parameters")
    group.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: %(default)s)",
    )
    group.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="Minibatch size (default: %(default)s)",
    )
    group.add_argument(
        "--wd",
        type=float,
        default=0,
        help="Weight decay in Adam optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate in Adam optimizer (default: %(default)s)",
    )
    group.add_argument(
        "--norm",
        type=float,
        nargs=2,
        default=[0, 1],
        help="Data normalization as shift, 1/scale (default: 0, 1)",
    )

    group = parser.add_argument_group("Network Architecture")
    group.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Number of nodes in hidden layers (default: %(default)s)",
    )
    group.add_argument(
        "--l-extent",
        type=float,
        default=0.5,
        help="Coordinate lattice size (if not using positional encoding) (default: %(default)s)",
    )
    group.add_argument(
        "--pe-type",
        choices=(
            "geom_ft",
            "geom_full",
            "geom_lowf",
            "geom_nohighf",
            "linear_lowf",
            "gaussian",
            "none",
        ),
        default="geom_ft",
        help="Type of positional encoding (default: %(default)s)",
    )
    group.add_argument(
        "--pe-dim",
        type=int,
        default=32,
        help="Num sinusoid features in positional encoding (default: 32)",
    )
    group.add_argument(
        "--activation",
        choices=("relu", "leaky_relu"),
        default="relu",
        help="Activation (default: %(default)s)",
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


def train_batch(
    model,
    lattice,
    mask,
    ctf_gen,
    y,
    rot,
    trans,
    optim,
    scaler,
):
    optim.zero_grad()
    model.train()
    # Cast operations to mixed precision if using torch.cuda.amp.GradScaler()
    B = y.size(0)
    D = lattice.D
    with torch.no_grad():
        coords = (lattice.coords.repeat(B,1,1) / lattice.extent / 2  + F.pad(trans,[0,1]).unsqueeze(1) / D) @ rot
        if mask is not None:
            mask = F.grid_sample(mask.permute([2,1,0]).repeat(B,1,1,1)[:, None], coords.view(B,D,D,D,3)*2, mode='nearest', align_corners=True).view(B,-1)
        else:
            mask = lattice.get_sphere_mask(D//2).repeat(B, 1)
    with torch.cuda.amp.autocast():
        y_recon = model(coords, mask=mask)
        y_recon = y_recon.mean(-1)
        y_psf = torch.fft.fftshift(torch.fft.irfft2(torch.fft.rfft2(torch.fft.ifftshift(y_recon.float()))*torch.fft.fftshift(ctf_gen)[..., :D//2+1]))
        loss = F.mse_loss(y_psf, y)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    return loss.item()

def eval_vol(
    model,
    lattice,
    mask,
    norm,
    optim,
    epoch,
    Apix=1.,
    flip=False,
    out_weights = None,
    out_vol =None,
):
    assert not model.training
    
    D = lattice.D
    if mask is None:
        mask = lattice.get_sphere_mask(D//2)
    vol_recon = model(lattice.coords[None], mask=mask[None]).squeeze(0)

    if flip:
        vol_recon=vol_recon[:,:,::-1]
    vol_recon=vol_recon*norm[1]+norm[0]
    save_checkpoint(model, optim, epoch, out_weights, vol_recon.cpu().numpy(), out_vol, Apix)


def save_checkpoint(model, optim, epoch, out_weights, vol=None, out_vol=None, Apix=1.):
    """Save model weights, and decoder volumes"""
    # save model weights
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unparallelize(model).state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        },
        out_weights,
    )
    # save vol
    if out_vol is not None and vol is not None:
        mrc.write(out_vol,vol,Apix=Apix,is_vol=True)



def train(data, ctf_params, posetracker, mask, device, ckpt_file=None, outdir=None, args=None):
    t1=dt.now()
    start_epoch = 0
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    checkpoint = args.checkpoint
    log_interval = args.log_interval * batch_size
    Nimg = len(data)
    D = data.D

    Dz = D
    extent = args.l_extent
    mag = 1
    lattice = Lattice(D*mag, Dz*mag, extent, device=device, endpoint=False)
    freq_mask = lattice.get_circular_mask(D//2)

    in_dim=3
    activation=nn.ReLU if args.activation == 'relu' else nn.LeakyReLU
    model=PositionalDecoder(in_dim, D*mag, nlayers=args.layers, hidden_dim=args.dim, enc_type=args.pe_type, enc_dim=args.pe_dim, activation=activation)
    model.to(device)


    if ckpt_file:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    data_generator = DataLoader(
        data, batch_size=batch_size, num_workers=4, pin_memory=True,
        shuffle=True,
    )

    lr=args.lr
    wd=args.wd
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    if ckpt_file:
        optim.load_state_dict(ckpt["optimizer_state_dict"])
    scaler = torch.cuda.amp.GradScaler()

    flog(model)
    flog(
        "{} parameters in model".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    for epoch in range(start_epoch, start_epoch+num_epochs):
        t2 = dt.now()
        loss_accum = 0
        batch_it = 0
        torch.cuda.empty_cache()
        for minibatch in data_generator:
            ind = minibatch[-1]
            y = minibatch[0].to(device)

            B=len(ind) 
            batch_it += B 

            rot, trans = posetracker.get_pose(ind, device=device)
            trans *= mag
            freqs = lattice.coords_2d.unsqueeze(0).expand(
                B, *lattice.coords_2d.shape
            )/ ctf_params[ind, 0].view(B, 1, 1) *mag
            ctf_gen=ctf.compute_ctf(freqs, *torch.split(ctf_params[ind, 1:], 1, 1)).view(B,D*mag,D*mag)
            ctf_gen *= freq_mask.view(1,D,D)
            loss= train_batch(
                model,
                lattice,
                mask,
                ctf_gen,
                y,
                rot,
                trans,
                optim,
                scaler,
            )
            loss_accum += loss * B
            if batch_it % log_interval == 0:
                log(
                    "# [Train Epoch: {}/{}] [{}/{} images] loss={:.6f}".format(
                        epoch + 1, start_epoch+num_epochs, batch_it, Nimg, loss
                    )
                )
        flog(
            "# =====> Epoch: {} Average loss = {:.6f}; Finished in {}".format(
                epoch + 1,
                loss_accum / Nimg,
                dt.now() - t2,
            )
        )
        if  (epoch+1) % checkpoint == 0:
            out_weights = outdir+"weights.{}.pkl".format(epoch)
            if args.split is not None:
                out_vol = outdir + "vol_it{:03d}_half{}.{}.mrc".format(epoch, args.split, D)
            else:
                out_vol = outdir + "vol_it{:03d}.{}.mrc".format(epoch, D)
            with torch.no_grad():
                model.eval()
                eval_vol(
                    model,
                    lattice,
                    mask[:-1,:-1,:-1].flatten() if mask is not None else None,
                    data.norm,
                    optim,
                    epoch,
                    Apix = data.apix,
                    out_weights = out_weights,
                    out_vol = out_vol,
                )

    td = dt.now() - t1
    flog("Finished in {} ({} per epoch)".format(td, td / (num_epochs)))


def main(args):
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.ind is not None:
        ind = utils.load_pkl(args.ind)
    else:
        ind = None

    particles = args.particles
    flog(f"Loading dataset from {particles}")
    datadir = args.datadir

    if args.split is not None:
        assert ind is None, "ERROR: Split dataset with filtered indices"
        if particles.endswith(".cs"):
            raw = np.load(args.particles)
            ind = np.argwhere(raw['alignments3D/split'] == (int(args.split)-1))[:,0].tolist()
        elif particles.endswith(".star"):
            s = cryoio.Starfile.load(args.input)
            ind = np.argwhere(s.df['rlnRandomSubset'] == int(args.split))[:,0].tolist()
        else:
            raise RuntimeError("Particle file doesn't contain split indices")
        flog(f"Split dataset randomly for GSFSC")

    if args.lazy:
        data = dataset.LazyMRCData(
            particles,
            datadir=datadir,
            norm=args.norm,
            window=args.window,
            invert_data=args.invert_data,
            window_r=args.window_r,
            ind=ind,
            flog=flog,
        )
    else:
        data = dataset.MRCData(
            particles,
            datadir=datadir,
            norm=args.norm,
            window=args.window,
            invert_data=args.invert_data,
            window_r=args.window_r,
            ind=ind,
            flog=flog,
        )
    Nimg = data.N
    D = data.D

    if args.ctf is not None:
        ctf_params = ctf.load_ctf_for_training(D, args.ctf)
        flog(f"Loading ctf from {args.ctf}")
    else:
        if particles.endswith(".cs"): # Parse ctf params from a cryoSPARC .cs metafile
            ctf_params = cryoio.parse_ctf_csparc(args.particles, D = D, Apix = data.apix)
        elif particles.endswith(".star"): # Parse ctf params from a Relion .star metafile
            ctf_params = cryoio.parse_ctf_relion(args.particles, D = D, Apix = data.apix)
        else:
            raise RuntimeError("Must provide CTF parameters")
        # Slice out the first column (D)
        ctf_params = ctf_params[:, 1:]
    
    if ind is not None:
        ctf_params = ctf_params[ind]
    ctf_params = torch.tensor(ctf_params, device=device)

    if args.poses is not None:
        pose = args.poses
        posetracker = PoseTracker.load(
            pose, 
            Nimg, 
            D, 
            ind=ind, 
        )
    else:
        if particles.endswith(".cs"): # Parse image poses from a cryoSPARC .cs metafile
            rot, trans = cryoio.parse_pose_csparc(args.particles)
        elif particles.endswith(".star"): # Parse image poses from a Relion .star metafile
            rot, trans = cryoio.parse_pose_relion(args.particles, Apix = data.apix)
        else:
            raise RuntimeError("Must provide estimated poses")
        if ind is not None:
            rot, trans = rot[ind], trans[ind]
        posetracker = PoseTracker(rot, trans, D)

    if args.mask is not None:
        mask, _ = mrc.parse_mrc(args.mask, is_vol=True)
        mask = torch.tensor(mask.copy(), device=device)
        mask = fft.symmetrize_3DGPU(mask).squeeze()
    else:
        mask = None

    if args.load is not None:
        ckpt = args.load
    else:
        ckpt = None

    train(data, ctf_params, posetracker, mask, device=device, ckpt_file=ckpt, outdir=outdir, args=args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    outdir = args.outdir + '/'
    if args.split == '1':
        outdir += 'halfA/'
    elif args.split == '2':
        outdir += 'halfB/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    LOG = f"{outdir}/run.log"

    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")
    def flog(msg):  # HACK: switch to logging module
        return utils.flog(msg, LOG)
    
    flog("Use cuda {}".format(use_cuda))
    if not use_cuda:
        log("WARNING: No GPUs detected")
    flog(" ".join(sys.argv))
    
    main(args)
