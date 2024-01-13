# CryoNeFEN: High-resolution reconstruction of cryo-EM structures using neural field network
CryoNeFEN is a neural network based algorithm for cryo-EM reconstruction. In particular, the method models an isotropic representation of 3D structures using neural fields in 3D spatial domain.
# Installation:

```
git clone https://github.com/YueHuang2023/CryoNeFEN.git
cd CryoNeFEN

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
