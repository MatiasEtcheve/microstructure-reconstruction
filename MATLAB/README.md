# Computation of fabrics

## Installation

This script works with the MATLAB version `R2021b`. 

The folder `in_polyhedron/` contain a module used in the script. This module allows to know if a 3D point is inside a polyhedron. To install this module, you just need to add this folder in the MATLAB path.
See https://fr.mathworks.com/matlabcentral/fileexchange/48041-in_polyhedron for more information.

You may need to install:
* Statistics and Machine Learning Toolbox

The main file is `import_slt.mlx`.

## Usage

If you want to run the script, you can run `import_slt.mlx`.

> ⚠️**WARNING**⚠️: the following functions
> * `obj.take_slice_images_along_x`
> * `obj.take_slice_images_along_y`
> * `obj.take_slice_images_along_z`
> * `obj.take_slice_images`
>
> with the argument `save=true` will save and OVERWRITE any images contained in the folder `MATLAB/saved_images` with the same name.

## Usefulness of files

Here are the usefulness of every folder / files

| folder / file | Usefulness |
|---|---|
| in_polyhedron/ | folder module to know if a 3D point is inside a 3D closed shape. |
| save_images/ | folder containing examples of slice images |
<<<<<<< HEAD
| rev.m | .m file containing the `rev` class. This class allows to compute <br>the fabrics of each grain contained in the rev and take slice images. |
| grain.m | .m file containing the `grain` class. This class allows to compute the <br>fabrics of a single grain. |
=======
| rev.m | .m file containing the 'rev' class. This class allows to compute <br>the fabrics of each grain contained in the rev and take slice images. |
| grain.m | .m file containing the 'grain' class. This class allows to compute the <br>fabrics of a single grain. |
>>>>>>> 9bc13b83599f6031d5043f909cd90c3d0689f103
| import_stl.mlx | notebook which has access to both class. |