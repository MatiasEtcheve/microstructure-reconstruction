import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import wandb


@dataclass
class Config(object):
    pass


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


def upload_files(run_name: str, paths: Union[str, Path, List]):
    """Uploads file in a run, after finishing it

    Args:
        run_name (str): run id to update
        paths (Union[str, Path, List]): paths of the files to update
    """
    login()
    api = wandb.Api()
    run = api.run(f"matiasetcheverry/microstructure-reconstruction/{run_name}")
    if isinstance(paths, Path) or isinstance(paths, str):
        run.upload_file(paths)
    if isinstance(paths, list):
        for p in paths:
            run.upload_file(p)
    run.finish()


def get_training(
    run_name: str,
    model_dict_path: Union[Path, str] = "model_dict.pt",
    model_script_path: Union[Path, str] = "model_script.txt",
) -> Tuple:
    """Retrieves a training from a name of a run

    Args:
        run_name (str): id of the run
        model_dict_path (Union[Path, str], optional): path to the model weights. Defaults to "model_dict.pt".
        model_script_path (Union[Path, str], optional): path to the model script. Defaults to "model_script.txt".

    Returns:
        Tuple: class name and script of the model
    """
    login()
    api = wandb.Api()
    run = api.run(f"matiasetcheverry/microstructure-reconstruction/{run_name}")
    model_dict = run.file(model_dict_path)
    model_dict = model_dict.download(root="tmp/", replace=True)
    model_script = run.file(model_script_path)
    model_script = model_script.download(root="tmp/", replace=True)
    class_name = re.findall(r"(?<=class ).[a-zA-Z0-9_.-]*", model_script.read())[0]
    model_script.seek(0)
    return class_name, model_script


def add_parameter(run, dict):
    for key, value in dict.items():
        if key == "group":
            setattr(run, key, value)
        else:
            run.config[key] = value


def delete_parameter(run, l):
    for key in l:
        run.config.pop(key, None)


def modify_run(run_name, action, dict):
    login()
    api = wandb.Api()
    run = api.run(f"matiasetcheverry/microstructure-reconstruction/{run_name}")
    if action == "add":
        add_parameter(run, dict)
    if action == "delete":
        delete_parameter(run, dict)
    run.update()


def modify_runs(runs):
    assert all([len(i) == 3 for i in runs])
    for run in runs:
        modify_run(*run)
