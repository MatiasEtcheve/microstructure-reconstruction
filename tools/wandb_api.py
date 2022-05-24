import io
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler


def login():
    """Logs a session in wandb."""
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


def get_training(
    run_name: str,
    model_dict_path: Union[Path, str] = "model_dict.pt",
    model_script_path: Union[Path, str] = "model_script.txt",
) -> Tuple[str, io.TextIOWrapper]:
    """Retrieves a training from a name of a run

    Args:
        run_name (str): id of the run
        model_dict_path (Union[Path, str], optional): path to the model weights. Defaults to "model_dict.pt".
        model_script_path (Union[Path, str], optional): path to the model script. Defaults to "model_script.txt".

    Returns:
        Tuple[str, io.TextIOWrapper]: class name and script of the model
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


def add_parameter(run: wandb.apis.public.Run, dict: Dict):
    """Add parameters to the wandb run.
    If there are any existing parameter whose name is in `dict`, this parameter will be overwriten.

    The parameters are stored in `dict`. The keys are the field to add.
    If the field to add is `group`, then `run.group` is modified.
    Else, `run.config` is modified.

    Args:
        run (wandb.apis.public.Run): wandb run to add parameters to
        dict (Dict): dict containing the parameters name and their values to add
    """
    for key, value in dict.items():
        if key == "group":
            setattr(run, key, value)
        else:
            run.config[key] = value


def delete_parameter(run: wandb.apis.public.Run, l: List[str]):
    """Deletes parameters in the wandb run.

    Args:
        run (wandb.apis.public.Run): run to modify.
        l (List[str]): list of keys to delete in the `run.config`.
    """
    for key in l:
        run.config.pop(key, None)


def modify_run(run_name: str, action: str, values: Union[List, Dict]):
    """Modify a run.
    We can either overwrite (or add) or deleter parameters in the wandb run.

    Args:
        run_name (str): run name
        action (str): action to perform, one of `["overwrite", "delete"]`
        values (Union[List, Dict]): values to perform action with

    Raises:
        ValueError: if the `action` is `overwrite`, the `values` must be a dictionnary of values
        ValueError: if the `action` is `delete`, the `values` must be a list of values
        NotImplementedError: `action` must be in ["overwrite", "delete"]
    """
    login()
    api = wandb.Api()
    run = api.run(f"matiasetcheverry/microstructure-reconstruction/{run_name}")
    if action == "overwrite":
        if not isinstance(values, dict):
            raise ValueError(
                "When deleting values in a run, the values must be stored in a list."
            )
        add_parameter(run, values)
    elif action == "delete":
        if not isinstance(values, list):
            raise ValueError(
                "When deleting values in a run, the values must be stored in a list."
            )
        delete_parameter(run, values)
    else:
        raise NotImplementedError('Action must be in ["overwrite", "delete"]')
    run.update()


def modify_runs(parameters: List[List]):
    """Modify several runs at once.

    Example:
        Let's say I want to modify several runs::

            modify_runs(
                [
                    ["2mnovjbb", "overwrite", {"architecture": "pretrained VGG"}],
                    ["2xkr5576", "overwrite", {"job_type": "train"}],
                    ["xbflutgy", "overwrite", {"group": "NWidth Network"}],
                    ["usewudy8", "delete", ["architecture", "learning_rate"]
                ]
            )

    Args:
        parameters (List[List]): list whose elements are composed of:
            * run name
            * action, which is one of `["overwrite", "delete"]`
            * values to overwrite or to delete

    Raises:
        ValueError: All elements of parameters must be list of 3 items.
    """
    if not all([len(i) == 3 for i in parameters]):
        raise ValueError("All elements of parameters must be list of 3 items.")
    for param in parameters:
        modify_run(*param)


def fetch_train_test_df(
    alias: Optional[str] = "latest",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the train and test dataframes from wandb artifacts.

    Notes:
        All the paths in the dataframe will be given relatively to the repository path.

    Args:
        alias (Optional[str], optional): aliases of the artifacts. Defaults to "latest".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: tuple containing the train and test dataframes.
    """
    repo_path = Path(__file__).resolve().parents[1]
    training_data_at = wandb.Api().artifact(
        "matiasetcheverry/microstructure-reconstruction/train_df:" + alias
    )
    test_data_at = wandb.Api().artifact(
        "matiasetcheverry/microstructure-reconstruction/test_df:" + alias
    )
    training_data_at.download()
    test_data_at.download()
    train_df = convert_table_to_dataframe(training_data_at.get("fabrics"))
    train_df["photos"] = train_df["photos"].apply(
        func=lambda photo_paths: [str(repo_path / Path(x)) for x in photo_paths]
    )
    test_df = convert_table_to_dataframe(test_data_at.get("fabrics"))
    test_df["photos"] = test_df["photos"].apply(
        func=lambda photo_paths: [str(repo_path / Path(x)) for x in photo_paths]
    )
    return train_df, test_df


def normalize(
    dataframes=List[pd.DataFrame], feature_range=(0, 1)
) -> Tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame]:
    """Normalizes a list of dataframes. Each dataframe is supposed to have the same features, in the same columns.

    Notes:
        It normalizes only float columns.

    Args:
        dataframes (List[pd.DataFrame]): list of dataframes to scale.
        feature_range (tuple, optional): range of the MinMaxScaler. Defaults to (0, 1).

    Returns:
        Tuple[MinMaxScaler, List[pd.DataFrame]]: Tuple containing:
            * the MinMaxScaler
            * the list of scaled dataframes.
    """
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    scaler = MinMaxScaler(feature_range=feature_range)

    columns_to_scale = []
    for idx, col in enumerate(dataframes[0].columns):
        if dataframes[0][col].dtype.kind in "biufc":
            columns_to_scale.append(idx)

    for df in dataframes:
        scaler.partial_fit(df.iloc[:, columns_to_scale])

    scaled_dfs = [deepcopy(dp) for dp in dataframes]
    for index, df in enumerate(dataframes):
        scaled_dfs[index].iloc[:, columns_to_scale] = scaler.transform(
            df.iloc[:, columns_to_scale]
        )
    return scaler, scaled_dfs
