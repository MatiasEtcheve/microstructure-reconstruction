# Microstructure reconstruction

## Table of contents

- [`wandb.ai`](#`wandb.ai`)
- [Objective](#objective)
  - [Past work](#past-work)
  - [Current work](#current-work)
- [Github Usage](#github-usage)
- [Usefulness of files](#usefulness-of-files)
  - [Tree of the directory](#tree-of-the-directory)
  - [Usefulness of folders](#usefulness-of-folders)

## `wandb.ai`

To store any important object created in this work, I have been using `wandb.ai`. My repository can be found [here](https://wandb.ai/matiasetcheverry/microstructure-reconstruction?workspace=user-matiasetcheverry).

`wandb.ai` allowed me to store model graphs, code, training checkpoints, images and datasets. It also allows live visualization of any training and its progress. It is very usefull, so we can let any training work and see the live plot of its loss function and any other metrics. As I stored the trainings itself (containing the weights of the model and optimizer), I can retrieve any past training and continue it or analyse it effectively on my computer.

## Objective

The objective of this repository is to determine a relationship between slices of images in a microstructure and its descriptors. In this work, the microstructure is a cube composed of aggregates and cement.

![Alt text](objective.png?raw=true "Objective")

### Past work

Our work is done step by step:

- create random microstructures, store them as `.stl` files.
- use MATLAB to compute the exact descriptors on these microstructures and extract slices
- create datasets containing the sliced images in different format. This dataset can be found on my `wandb.ai` [repository](https://wandb.ai/matiasetcheverry/microstructure-reconstruction?workspace=user-matiasetcheverry)
- determine the descriptors of unseen data through deep learning. I have been using Kaggle to access GPU.

The work we did on determining the descriptors of a microstructure based on a certain number of sliced images was good.

### Current work

Now, we would like to see if the descriptors are a good representation of the microstructure. Some papers have been using descriptors and / or n-points statistic function.

For this work, we would like to be able to reconstruct a microstructure through any latent space, and see if this latent space is closed to our descriptor space.

To do this, we encode the multiple slices of a microstructure into a latent space, and then decode the latent representation, using (Variational) Auto-Encoder.

For now, this work hasn't shown good results.

## Github Usage

Here are a few links to use github, with the terminal:

- <https://gitimmersion.com/index.html>
- <http://up1.github.io/git-guide/index.html>

But you may also use github with the desktop application:

- <https://www.softwaretestinghelp.com/github-desktop-tutorial/>

You may also need to set up a SSH connection to github. Basically, a SSH is a secure protocol that allows you to connect to private repository:

- <https://jdblischak.github.io/2014-09-18-chicago/novice/git/05-sshkeys.html>

## Usefulness of files

### Tree of the directory

The script are currently working with this directories / naming conventions:

```bash
.
├── custom_datasets
│   ├── data_augm.py
│   ├── datasets.py
├── custom_models
│   ├── autoencoders.py
│   ├── cnns.py
│   ├── gans.py
├── MATLAB
│   ├── README.md
│   ├── grain.m
│   ├── import_stl.mlx
│   ├── rev.m
├── predicting
│   ├── autoencoder-predicting.ipynb
│   ├── ...
├── REV1_600
│   ├── REV1_600Meshes
│   │   ├── Spec-1.mat
│   │   ├── Spec-2.mat
│   │   └── ...
│   ├── REV1_600Slices
│   │   ├── 1pics
│   │   │   ├── Spec-1_Imgs
│   │   │   │   ├── *.png
│   │   │   │   ├── *.png
│   │   │   │   └── ...
│   │   │   ├── Spec-2_Imgs
│   │   │   │   ├── *.png
│   │   │   │   ├── *.png
│   │   │   │   └── ...
│   │   │   ├── ...
│   │   ├── 3pics
│   │   └── ...
│   ├── REV1_6003D_model
│   │   ├── Spec-1.STL
│   │   ├── Spec-2.STL
│   │   └── ...
│   └── fabrics.txt
├── tools
│   ├── __init__.py
│   ├── ...
├── training_visualisation
│   ├── colorful-fire-556.ipynb
│   ├── ...
├── README.md <- YOU ARE CURRENTLY HERE
└── ...
```

### Usefulness of folders

Here are the usefulness of every folder

| folder | Usefulness |
|---|---|
| custom_datasets | contains the datasets used in the trainings. Indeed, we can choose how to structure our sliced images, by stacking or concatenating them together |
| custom_models | contains the models used in the trainings, like cnns and autoencoders. There are also not working models like gans. |
| MATLAB | every thing to compute descriptors, sliced images and meshes from REVs stored in `.stl` files. This newly computed data is stored in `REV1_600/`. |
| predicting | notebooks used in the predictions. This is what I daily use for research on Kaggle. |
| REV1_600 | every data we use. It contains sliced images, meshes, descriptors, REV files. This data is very large, so only the images and the descriptors are uploaded on github. The rest is not necessary to the trainings, but is part of a preprocessing step. |
| tools | contains every minor tools used in the trainings. Most of these tools are tools used with `wandb.ai`. `wandb.ai` is the website where the training and validation sets are stored. I also store the code used in the trainings. Some of these tools are also plotting and dataframe manipulation tools. |
| training_visualisation | contains the analysis of every interesting training. Each training is stored in `wandb.ai`. The  filenames of the notebooks are the names of the trainings. |
