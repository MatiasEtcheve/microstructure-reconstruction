# Computation of descriptors

This folder is used to produces data from `.stl` files. Those files contain REV, which are made of ciment and aggregates. All those REVs are stored in `/REV1_600`.

Here is what a REV looks like, opened in MATLAB:
![Alt text](images/rev.png?raw=true "REV")

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Descriptors](#descriptors)
- [Sliced images](#sliced-images)
- [Meshes](#meshes)
- [Installation](#installation)
  - [MATLAB Toolbox](#matlab-toolbox)
  - [`in_polyhedron` module](#-in-polyhedron--module)
- [Usage](#usage)
  - [Usefulness of files](#usefulness-of-files)

## Descriptors

We would like to compute a certain number of statistical descriptors from those REVs.

The descriptors are stored at `REV1_600/fabrics.txt`.

Firstly, we define the orientation matrix: it is the unique 3x3 symmetric matrix, which explains the direction of major axis of irregular shape and is calculated from the unit direction of major axis (the unit direction comes from principal components analysis).

Major unit vector

![equation](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}\overrightarrow{n}&space;=&space;[n_1,&space;n_2,&space;n_3]^T)

Orientation matrix:

![equation](https://latex.codecogs.com/png.image?\inline&space;\dpi{100}\bg{white}[F_{ij}]&space;=&space;\overrightarrow{n}\overrightarrow{n}^T)

Here is a non exhaustive list of the descriptors we compute:

- `nearest_distance`: typical barycenter-barycenter distance between two aggregates
- `invariant`: 3 values which are the invariant of the orientation matrix
- `orientation`: 6 values representing the orientation of the aggregates. These values are the diagonal and upper diagonal coefficient of the 3x3 orientation matrix.
- `aspectratio`: aspect ratio of an aggregate: 2nd largest size divided the 1st largest size
- `size`: scalar, specified as the length of major axis of shape.
- `solidity`: ratio of grain volume to its convex hull volume.
- `roundness`: volume ratio of grain to the surround sphere with major axis length as diameter.
- `volume_fraction`: ratio of volume of aggregate to total volume (Aggregate + Cement)

## Sliced images

We also want sliced images of the REVs. This images are stored at `REV1_600/REV1_600Slices`

We want to have `n` number of sliced images taken along x, y or z axis.

Here is how we take a sliced image along the y axis:
![Alt text](images/slice.png?raw=true "Slice")

## Meshes

We also want meshes from those REVs. For now, those meshes are obtain along a single axis. The meshes are 2D.

The meshes are stored at `REV1_600/REV1_600Meshes` with `.mat` files. These files contain the points and the connectivity list necessary to a 2D mesh.

The connectivity list has 7 columns:

- `elt_id`: id of the current element
- `material_id`: id of the current material: 0 for ciment, 1 for aggregate
- `object_id`: if of the current object: -1 for ciment or the id aggregate
- `upleft_node`
- `downleft_node`
- `downright_node`
- `upright_node`

The points list has 3 columns:

- `point_id`
- `x`: x coordinate of the point
- `y`: y coordinate of the point

## Installation

This script works with the MATLAB version `R2021b`.
You will need several modules to use the package.

### MATLAB Toolbox

You may need to install:

- Statistics and Machine Learning Toolbox
- Phased Array System Toolbox

### `in_polyhedron` module

This module tests if points are inside of triangulated volume. This module allows to know if a 3D point is inside a polyhedron. To install this module, you just need to add this folder in the MATLAB path, at `/MATLAB/`

See <https://fr.mathworks.com/matlabcentral/fileexchange/48041-in_polyhedron> for more information.

## Usage

If you want to run the script, you can run `fabrics.mlx` or `meshes.mlx`.

> ⚠️**WARNING**⚠️: the following functions
>
> - `obj.take_slice_images_along_x`
> - `obj.take_slice_images_along_y`
> - `obj.take_slice_images_along_z`
> - `obj.take_slice_images`
>
> with the argument `save=true` will save and OVERWRITE any images contained in the folder `MATLAB/saved_images` with the same name.

### Usefulness of files

Here are the usefulness of every folder / files

| folder / file | Usefulness |
|---|---|
| rev.m | .m file containing the `rev` class. This class allows to compute the fabrics of each grain contained in the rev and take slice images. |
| grain.m | .m file containing the `grain` class. This class allows to compute the fabrics of a single grain. |
| fabrics.mlx | notebook computing the fabrics of the REVs |
| meshes.mlx | notebook computing the meshes of the REVs |
