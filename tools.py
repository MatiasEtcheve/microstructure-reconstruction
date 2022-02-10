import inspect
import os
import pickle
import sys
from ctypes import Union
from pathlib import Path
from typing import List, Union
from xmlrpc.client import _iso8601_format

import numpy as np
import pandas as pd
import torch
from IPython.core.magics.code import extract_symbols
from tqdm import tqdm

import wandb


def set_seed(seed: int):
    """Sets the seed of all random operation in `numpy` and `pytorch`

    Warning:
        `pandas` does not have any fixed seed.

    Args:
        seed (int): seed to set to have reproducible results.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def wandb_login() -> Path:
    """Logs a session in wandb.

    Returns:
        pathlib.Path: path of the repository in the cloud. The path depends whether we are on Colab or Kaggle.
    """
    IS_COLAB = "google.colab" in sys.modules
    IS_KAGGLE = "kaggle_secrets" in sys.modules
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient

        repo_path = Path("/kaggle/input/microstructure-reconstruction/")
        WANDB_API = UserSecretsClient().get_secret("wandb_api")
    if IS_COLAB:
        repo_path = Path("/content/gdrive/MyDrive/microstructure-reconstruction")
        WANDB_API = "3e384d0e21fd4f06a6abc2fdc162b88eadc00994"
    else:
        repo_path = Path("/home/matias/microstructure-reconstruction")
        WANDB_API = os.getenv("WANDB_API")
    wandb.login(key=WANDB_API)
    return repo_path


def new_getfile(object, _old_getfile=inspect.getfile):
    """Working `inspect.getfile` for Notebook"""
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


def get_cell_code(obj: type) -> str:
    """Gets the notebook cell of a given object

    Args:
        obj (type): object to get the cell code

    Returns:
        str: cell code
    """
    return "".join(inspect.linecache.getlines(new_getfile(obj)))


def get_class_code(obj: type) -> str:
    """Gets the notebook code of a given object

    Args:
        obj (type): object to get the class code

    Returns:
        str: code
    """
    cell_code = get_cell_code(obj)
    return extract_symbols(cell_code, obj.__name__)[0][0]


def get_members(instance) -> dict:
    """Gets the members of an instance of any object

    Args:
        instance: instance of an object

    Returns:
        dict: dict whose keys are the name of the members of the instance.
    """
    members = {}
    for i in inspect.getmembers(instance):
        if not i[0].startswith("_") and not inspect.ismethod(i[1]):
            members[i[0]] = i[1]
    return members


def associate_rev_id_to_its_images(
    id: str, path_to_slices: Path, nb_images: int, relative_path: Path = None
) -> List[Path]:
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


def convert_into_single_entry_df(
    multi_entry_df: pd.DataFrame, col_name: str = "photos"
) -> pd.DataFrame:
    """Converts a column of a dataframe whose cells are list, into a longer dataframe where the cells are now values of the list

    Args:
        multi_entry_df (pd.DataFrame): dataframe to modify
        col_name (str): name of the column in the dataframe whose cells are list

    Returns:
        pd.DataFrame: copy of the dataframe, and `dataframe[col_name]` are now values of the list
    """
    if not (multi_entry_df[col_name].apply(type) == list).any():
        return multi_entry_df
    nb_images = multi_entry_df.photos.str.len().max()
    assert nb_images == multi_entry_df.photos.str.len().min()
    single_entry_df = pd.concat([multi_entry_df] * nb_images, ignore_index=True)
    single_entry_df[col_name] = single_entry_df.apply(
        func=lambda row: str(row[col_name][row.name % nb_images]),
        axis=1,
    )
    return single_entry_df


def convert_table_to_dataframe(table: wandb.data_types.Table) -> pd.DataFrame:
    """Convert a wandb.data_types.Table into a pandas.DataFrame

    Args:
        table (wandb.data_types.Table): table to convert

    Returns:
        pd.DataFrame: converted table
    """
    return pd.DataFrame([row for _, row in table.iterrows()], columns=table.columns)


def add_torch_object(
    artifact: Union[wandb.Artifact, wandb.sdk.wandb_run.Run], obj, filename: Path
):
    """Add a torch object to a given artifact, at the given filename

    Args:
        artifact (Union[wandb.Artifact, wandb.Run]): artifact to append the file
        obj: object to save and to add to the artifact
        filename (Path): path to the object in the artifact
    """
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.parent / (filename.stem + ".pt")
    filename.mkdir(parents=True, exist_ok=True)
    if isinstance(artifact, wandb.Artifact):
        torch.save(obj, filename)
        artifact.add_file(filename)
        filename.unlink()
    if isinstance(artifact, wandb.run):
        torch.save(obj, Path(artifact.dir) / filename)


def add_pickle_object(
    artifact: Union[wandb.Artifact, wandb.sdk.wandb_run.Run], obj, filename: Path
):
    """Add a pickleable object to a given artifact, at the given filename

    Args:
        artifact (Union[wandb.Artifact, wandb.Run]): artifact to append the file
        obj: object to save and to add to the artifact
        filename (Path): path to the object in the artifact
    """
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.parent / (filename.stem + ".pkl")
    filename.mkdir(parents=True, exist_ok=True)
    if isinstance(artifact, wandb.sdk.wandb_run.Run):
        filename = Path(artifact.dir) / filename
    with open(filename, "wb") as file:
        pickle.dump(obj, file)
    if isinstance(artifact, wandb.Artifact):
        artifact.add_file(filename)
        filename.unlink()


def get_path_image_along_axis(
    l: List[Union[str, Path]], axis: str = "x"
) -> List[Union[str, Path]]:
    """Keeps only the image paths along a specified axis

    Args:
        l (List[Union[str, Path]]): list whose values are the image paths.
            We suppose the image filename is "y-z[XXX].png" if the image is taken along x axis.
        axis (str, optional): axis to keep. Defaults to "x".

    Returns:
        List[Union[str, Path]]: paths of sliced images taken along `axis`
    """
    return [s for s in l if axis not in Path(s).stem]


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for data, target in tqdm(train_loader, total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss


def compute_outputs(model, device, val_loader):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            if idx == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), dim=0)
                targets = torch.cat((targets, target), dim=0)
    return outputs, targets


def compute_errors(outputs, targets, device, min_val, max_val):
    min_val = torch.FloatTensor(min_val).to(device)
    max_val = torch.FloatTensor(max_val).to(device)
    outputs = outputs.to(device)
    targets = targets.to(device)
    return (outputs - targets) / (targets + (min_val / (max_val - min_val)))


def validate(model, device, val_loader, criterion, min, max):
    model.eval()
    val_loss = 0
    example_images = []
    min = torch.FloatTensor(min).to(device)
    max = torch.FloatTensor(max).to(device)
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            example_images.append(wandb.Image(data[0]))
            error = (output - target) / (target + (min / (max - min)))
            if idx == 0:
                errors = error
            else:
                errors = torch.cat((errors, error), dim=0)
    return val_loss, example_images, errors.max()
