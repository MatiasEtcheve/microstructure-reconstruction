import inspect
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from IPython.core.magics.code import extract_symbols

import wandb


def wandb_login():
    IS_KAGGLE = "kaggle_secrets" in sys.modules
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient

        repoPath = Path(
            "/kaggle/input/microstructure-reconstruction/"
        )
        sys.path.append(str(repoPath))

        WANDB_API = UserSecretsClient().get_secret("wandb_api")
    else:
        repoPath = Path("/home/matias/microstructure-reconstruction")
        WANDB_API = os.getenv("WANDB_API")
    wandb.login(key=WANDB_API)
    return repoPath


def new_getfile(object, _old_getfile=inspect.getfile):
    if not inspect.isclass(object):
        return _old_getfile(object)

    # Lookup by parent module (as in current inspect)
    if hasattr(object, "__module__"):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, "__file__"):
            return object_.__file__

    # If parent module is __main__, lookup by methods (NEW)
    for name, member in inspect.getmembers(object):
        if (
            inspect.isfunction(member)
            and object.__qualname__ + "." + member.__name__ == member.__qualname__
        ):
            return inspect.getfile(member)
    else:
        raise TypeError("Source for {!r} not found".format(object))


inspect.getfile = new_getfile


def get_cell_code(obj):
    return "".join(inspect.linecache.getlines(new_getfile(obj)))


def get_class_code(obj):
    cell_code = get_cell_code(obj)
    return extract_symbols(cell_code, obj.__name__)[0][0]


def get_members(obj):
    members = {}
    for i in inspect.getmembers(obj):
        if not i[0].startswith("_") and not inspect.ismethod(i[1]):
            members[i[0]] = i[1]
    return members


def associate_rev_id_to_its_images(
    id: str, path_to_slices: Path, nb_images: int, relative_path: Path = None
):
    """Associates a rev id, like `Spec-5` to its sliced images, given a number of slices per plane and

    Args:
        id (str): id of the rev id
        path_to_slices (Path): path to the slices images
        nb_images (int): number of slices per plane

    Returns:
        List[Path]: list of path where each path points an image
    """
    return [
        str(x.relative_to(relative_path)) if relative_path is not None else str(x)
        for x in path_to_slices.glob(f"{nb_images}p*/{id}_Imgs/*")
    ]


def convert_into_single_entry_df(multi_entry_df):
    nb_images = multi_entry_df.photos.str.len().max()
    assert nb_images == multi_entry_df.photos.str.len().min()
    single_entry_df = pd.concat([multi_entry_df] * nb_images, ignore_index=True)
    single_entry_df["photo"] = single_entry_df.apply(
        func=lambda row: str(row["photos"][row.name % nb_images]),
        axis=1,
    )
    single_entry_df.drop(columns="photos", inplace=True)
    return single_entry_df


def convert_table_to_dataframe(table: wandb.data_types.Table):
    return pd.DataFrame([row for _, row in table.iterrows()], columns=table.columns)


def add_torch_object(artifact: wandb.Artifact, obj, filename: Path):
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.parent / (filename.stem + ".pt")
    filename.mkdir(parents=True, exist_ok=True)
    torch.save(obj, filename)
    artifact.add_file(filename)
    filename.unlink()


def add_pickle_object(artifact: wandb.Artifact, obj, filename: Path):
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.parent / (filename.stem + ".pkl")
    filename.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
    artifact.add_file(filename)
    filename.unlink()
