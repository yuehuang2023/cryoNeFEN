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
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install matplotlib
pip install starfile mrcfile
```
<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - pytorch 1.13
  - starfile
  - mrcfile
  - matplotlib

</details>

# How to run? 

```
python train.py -h
```

Example usage:

```
python train.py particles.cs --datadir ./ --mask mask.mrc --lazy --outdir ./tutorial/
```
