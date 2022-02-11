import os
import pickle
import sys
from ctypes import Union
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import wandb


def login():
    """Logs a session in wandb.

    Returns:
        pathlib.Path: path of the repository in the cloud. The path depends whether we are on Colab or Kaggle.
    """
    IS_COLAB = "google.colab" in sys.modules
    IS_KAGGLE = "kaggle_secrets" in sys.modules
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient

        WANDB_API = UserSecretsClient().get_secret("wandb_api")
    elif IS_COLAB:
        WANDB_API = "3e384d0e21fd4f06a6abc2fdc162b88eadc00994"
    else:
        WANDB_API = os.getenv("WANDB_API")
    wandb.login(key=WANDB_API)


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
    if isinstance(artifact, wandb.sdk.wandb_run.Run):
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


def add_writeable_object(
    artifact: Union[wandb.Artifact, wandb.sdk.wandb_run.Run], obj, filename: Path
):
    """Add a writeable object to a given artifact, at the given filename

    Args:
        artifact (Union[wandb.Artifact, wandb.Run]): artifact to append the file
        obj: object to save and to add to the artifact
        filename (Path): path to the object in the artifact
    """
    filename = Path(filename)
    if filename.suffix == "":
        filename = filename.parent / (filename.stem + ".txt")
    filename.mkdir(parents=True, exist_ok=True)
    if isinstance(artifact, wandb.sdk.wandb_run.Run):
        filename = Path(artifact.dir) / filename
    with open(filename, "w") as file:
        file.write(obj)
    if isinstance(artifact, wandb.Artifact):
        artifact.add_file(filename)
        filename.unlink()
