# CryoNeFEN: High-resolution reconstruction of cryo-EM structures using neural field network
CryoNeFEN is a neural network based algorithm for cryo-EM reconstruction. In particular, the method models an isotropic representation of 3D structures using neural fields in 3D spatial domain.
# Installation:

```
# clone the repo.
git clone https://github.com/YueHuang2023/cryoNeFEN.git
cd CryoNeFEN

# Make a conda environment.
conda create -n cryonefen python=3.9
conda activate cryonefen

# Install required packages
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install matplotlib 
pip install starfile mrcfile scipy
```
<details>
  <summary> Dependencies (click to expand) </summary>
  
  - pytorch 1.13
  - starfile
  - mrcfile
  - matplotlib
  - scipy

</details>

# How to run? 

<details><summary><code>python train.py -h</code></summary>
  
    usage: train.py [-h] -o OUTDIR [--poses POSES] [--ctf pkl] [--mask mrc] [--split {1,2}] [--load WEIGHTS.PKL] [--checkpoint CHECKPOINT] [--log-interval LOG_INTERVAL] [--seed SEED] [--uninvert-data] [--no-window]
                [--window-r WINDOW_R] [--ind IND] [--lazy] [--datadir DATADIR] [-n NUM_EPOCHS] [-b BATCH_SIZE] [--wd WD] [--lr LR] [--norm NORM NORM] [--layers LAYERS] [--dim DIM] [--l-extent L_EXTENT]
                [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}] [--pe-dim PE_DIM] [--activation {relu,leaky_relu}]
                particles

    positional arguments:
      particles             Input particles (.mrcs, .star, .cs, or .txt)

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTDIR, --outdir OUTDIR
                        Output directory to save model
      --poses POSES         Image poses (.pkl)
      --ctf pkl             CTF parameters (.pkl)
      --mask mrc            Optional mask (.mrc, default: sphere mask)
      --split {1,2}         Split dataset for computing GSFSC
      --load WEIGHTS.PKL    Initialize training from a checkpoint
      --checkpoint CHECKPOINT
                        Checkpointing interval in N_EPOCHS (default: 1)
      --log-interval LOG_INTERVAL
                        Logging interval in N_IMGS (default: 100)
      --seed SEED           Random seed

    Dataset loading:
      --uninvert-data       Do not invert data sign
      --no-window           Turn off real space windowing of dataset
      --window-r WINDOW_R   Windowing radius (default: 0.85)
      --ind IND             Filter particle stack by these indices
      --lazy                Lazy loading if full dataset is too large to fit in memory
      --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file

    Training parameters:
      -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Minibatch size (default: 4)
      --wd WD               Weight decay in Adam optimizer (default: 0)
      --lr LR               Learning rate in Adam optimizer (default: 0.001)
      --norm NORM NORM      Data normalization as shift, 1/scale (default: 0, 1)

    Network Architecture:
      --layers LAYERS       Number of hidden layers (default: 2)
      --dim DIM             Number of nodes in hidden layers (default: 256)
      --l-extent L_EXTENT   Coordinate lattice size (if not using positional encoding) (default: 0.5)
      --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                        Type of positional encoding (default: geom_ft)
      --pe-dim PE_DIM       Num sinusoid features in positional encoding (default: 32)
      --activation {relu,leaky_relu}
                        Activation (default: relu)

</details>

Example usage:

```
python train.py particles.cs --mask mask.mrc --lazy --outdir ./tutorial/
```
# Quickstart: cryo-EM reconstruction
## 1. Preprocess image stack
Perform a **homogeneous refinement** in cryoSPARC software. We will use the poses and CTF parameters from this "consensus reconstruction". 

- In cryoSPARC, 1) import the particles, 2) run an ab initio reconstruction job, and 3) run a homogeneous refinement job, all with default parameters.
- CryoNeFEN extracts image poses from a `.cs` file directly. Copy the path of cryoSPARC's metadata file (`.cs` file) that contains particle poses and CTF parameters.
  
## 2. CryoNeFEN training
When the input image stack (`.cs` file) has been prepared, a cryoNeFEN model can be trained with `python train.py`:

<details><summary><code>python train.py -h</code></summary>
  
    usage: train.py [-h] -o OUTDIR [--poses POSES] [--ctf pkl] [--mask mrc] [--split {1,2}] [--load WEIGHTS.PKL] [--checkpoint CHECKPOINT] [--log-interval LOG_INTERVAL] [--seed SEED] [--uninvert-data] [--no-window]
                [--window-r WINDOW_R] [--ind IND] [--lazy] [--datadir DATADIR] [-n NUM_EPOCHS] [-b BATCH_SIZE] [--wd WD] [--lr LR] [--norm NORM NORM] [--layers LAYERS] [--dim DIM] [--l-extent L_EXTENT]
                [--pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}] [--pe-dim PE_DIM] [--activation {relu,leaky_relu}]
                particles

    positional arguments:
      particles             Input particles (.mrcs, .star, .cs, or .txt)

    optional arguments:
      -h, --help            show this help message and exit
      -o OUTDIR, --outdir OUTDIR
                        Output directory to save model
      --poses POSES         Image poses (.pkl)
      --ctf pkl             CTF parameters (.pkl)
      --mask mrc            Optional mask (.mrc, default: sphere mask)
      --split {1,2}         Split dataset for computing GSFSC
      --load WEIGHTS.PKL    Initialize training from a checkpoint
      --checkpoint CHECKPOINT
                        Checkpointing interval in N_EPOCHS (default: 1)
      --log-interval LOG_INTERVAL
                        Logging interval in N_IMGS (default: 100)
      --seed SEED           Random seed

    Dataset loading:
      --uninvert-data       Do not invert data sign
      --no-window           Turn off real space windowing of dataset
      --window-r WINDOW_R   Windowing radius (default: 0.85)
      --ind IND             Filter particle stack by these indices
      --lazy                Lazy loading if full dataset is too large to fit in memory
      --datadir DATADIR     Path prefix to particle stack if loading relative paths from a .star or .cs file

    Training parameters:
      -n NUM_EPOCHS, --num-epochs NUM_EPOCHS
                        Number of training epochs (default: 20)
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Minibatch size (default: 4)
      --wd WD               Weight decay in Adam optimizer (default: 0)
      --lr LR               Learning rate in Adam optimizer (default: 0.001)
      --norm NORM NORM      Data normalization as shift, 1/scale (default: 0, 1)

    Network Architecture:
      --layers LAYERS       Number of hidden layers (default: 2)
      --dim DIM             Number of nodes in hidden layers (default: 256)
      --l-extent L_EXTENT   Coordinate lattice size (if not using positional encoding) (default: 0.5)
      --pe-type {geom_ft,geom_full,geom_lowf,geom_nohighf,linear_lowf,gaussian,none}
                        Type of positional encoding (default: geom_ft)
      --pe-dim PE_DIM       Num sinusoid features in positional encoding (default: 32)
      --activation {relu,leaky_relu}
                        Activation (default: relu)

</details>

The required arguments are:
- `particles`, an input image stack (`.cs` or other listed file types)
- `-o`, a clean output directory for storing results
  
Additional parameters that are typically set include:
- `-n`, Number of epochs to train
- `-b`, Batchsize of the image stack during the training
- `--lazy`, Lazy loading if full dataset is too large
- `--mask`, Mask for accelerating the training
- `--split`, Split the image stack randomly
- Neural network architecture settings with `--layers`, `--dim`, `--pe-dim`

If the golden standard Fourier shell correlation (GSFSC) is required in further benchmarking, run commands:
```
python train.py {cryoSPARC directory}/xxx_particles.cs  --mask {cryoSPARC directory}/xxx_volume_mask_refine.mrc --lazy --outdir ./tutorial/ --split 1
python train.py {cryoSPARC directory}/xxx_particles.cs  --mask {cryoSPARC directory}/xxx_volume_mask_refine.mrc --lazy --outdir ./tutorial/ --split 2
```
- `{cryoSPARC directory}/xxx_particles.cs` is the `.cs` file processed in step 1.
- `{cryoSPARC directory}/xxx_volume_mask_refine.mrc` is the mask refined by cryoSPARC in step 1.

## 3. CryoNeFEN analysis
Once the model has finished training, the generated density maps are saved in `outdir` for further visualization, and analysis. 

GSFSC of final results can be computed with `python analysis.py`:

<details><summary><code>python analysis.py -h</code></summary>
  
    usage: analysis.py [-h] [--mask mrc] volumes

    positional arguments:
       volumes     Half-maps directory (.mrc)

    optional arguments:
      -h, --help  show this help message and exit
      --mask mrc  FSC mask (.mrc)

</details>

Example usage:
```
python analysis.py ./tutorial/ --mask {cryoSPARC directory}/xxx_volume_mask_refine.mrc
```
Masked FSC curves and reconstructed maps will be plotted. 
