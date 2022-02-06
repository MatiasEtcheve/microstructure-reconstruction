from dataclasses import dataclass, fields, asdict
import pickle
from unittest.util import _MAX_LENGTH
from attr import attributes
from . import database_api
from datetime import datetime
from uuid import uuid4
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from typing import Union, List
import numpy as numpy
from copy import deepcopy
import pandas as pd
from pathlib import Path
from pprint import pprint
from typing import List, Union, Optional, Callable
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms, utils
import torch.nn as nn
import os
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import matplotlib
import torch.optim as optim
from tqdm import tqdm
import inspect
from . import tools


@dataclass
class SavedObject:
    def __repr__(self):
        s = ""
        for field in fields(Training):
            value = self.__getattribute__(field.name)
            s += "{:>20}  {}\n".format(
                field.name, value if value is not None else "None"
            )
        return s

    def save(self, db_name, overwrite=False):
        # either the model is not already saved in the db
        if self.id == 0:
            self.id = database_api.create(self.as_dict(), db=db_name)
            return self.id
        # either the object is saved and we update it
        elif overwrite:
            database_api.update(self.id, self.as_dict(), db=db_name)
            return self.id
        print(f"Object not saved as {self.id} is already in {db_name}")


@dataclass
class Training(SavedObject):
    # Saved parameters
    id: int = 0
    datetime: str = str(datetime.now())
    comment: str = ""
    training_dataset_id: int = None
    checkpoint: dict = None
    metrics_id: str = str(uuid4())
    epoch: int = 0

    def __post_init__(self):
        if self.checkpoint is None:
            self.checkpoint = {"first": 0, "best": 0, "last": 0}

    @classmethod
    def from_id(cls, id):
        return cls(id=id, **database_api.retrieve(id, db="trainings"))

    def __repr__(self):
        return super().__repr__()

    def as_dict(self):
        return {
            "datetime": self.datetime,
            "comment": self.comment,
            "training_dataset_id": self.training_dataset_id,
            "checkpoint": self.checkpoint,
            "metrics_id": self.metrics_id,
            "epoch": self.epoch,
        }

    def update_checkpoint(self, cp_name, id, save=False):
        self.checkpoint[cp_name] = id
        if save:
            self.save()

    def increment_epoch(self, save=False):
        self.epoch += 1
        if save:
            self.save()

    def save(self):
        assert self.comment != ""
        return super(Training, self).save("trainings", overwrite=True)


@dataclass
class Dataset(SavedObject):
    # Saved parameters
    id: int = 0
    datetime: str = str(datetime.now())
    comment: str = ""
    # Saved files
    script: str = ""
    dataset: pd.DataFrame = None
    dataframe: pd.DataFrame = None

    def __post_init__(self):
        if self.id == 0:
            if self.dataset is not None and self.script == "":
                self.script = tools.get_class_code(type(self.dataset))

    @classmethod
    def from_id(cls, id):
        folder_path = Path(__file__).resolve().parent / "datasets" / str(id)
        dataset = [x for x in folder_path.glob("*.pt") if x.is_file()]
        pd_dataframe = [x for x in folder_path.glob("*.zip") if x.is_file()]
        np_dataframe = [x for x in folder_path.glob("*.npy") if x.is_file()]
        script = [x for x in folder_path.glob("*.pkl") if x.is_file()]
        attributes = {}
        if dataset:
            attributes["dataset"] = torch.load(dataset[0])
        if pd_dataframe:
            attributes["dataframe"] = pd.read_pickle(pd_dataframe[0])
        if np_dataframe:
            attributes["dataframe"] = np.load(np_dataframe[0])
        if script:
            attributes["script"] = pickle.load(open(script[0], "rb"))
        return cls(id=id, **database_api.retrieve(id, db="datasets"), **attributes)

    def as_dict(self):
        return {
            "datetime": self.datetime,
            "comment": self.comment,
        }

    def save(self):
        assert self.comment != ""
        assert self.dataset is not None or self.dataframe is not None
        super(Dataset, self).save("datasets", overwrite=True)
        folder_path = Path(__file__).resolve().parent / "datasets" / str(self.id)
        folder_path.mkdir(parents=True, exist_ok=True)
        if self.dataset is not None:
            torch.save(self.dataset, folder_path / "dataset.pt")
        if self.dataframe is not None:
            if isinstance(self.dataframe, np.ndarray):
                np.save(folder_path / "dataframe.npy", self.dataframe)
            else:
                self.dataframe.to_pickle(folder_path / "dataframe.zip")
        if self.script is not None:
            with open(folder_path / "script.pkl", "wb") as file:
                pickle.dump(self.script, file)
        return self.id


@dataclass
class Model(SavedObject):
    # Saved parameters
    id: int = 0
    comment: str = ""
    epoch: int = 0
    loss: float = np.inf
    datetime: str = str(datetime.now())
    # Saved files
    script: str = ""
    model_dict: dict = None
    optimizer_dict: dict = None
    loss_type: str = None

    model: int = None
    optimizer: int = None

    def __post_init__(self):
        if self.id == 0 and self.model is not None:
            if self.script == "":
                self.script = tools.get_class_code(type(self.model))
            if self.model_dict is None:
                self.model_dict = deepcopy(self.model.state_dict())
            if self.optimizer_dict is None:
                self.optimizer_dict = deepcopy(self.optimizer.state_dict())

    def as_dict(self):
        return {
            "datetime": self.datetime,
            "comment": self.comment,
            "epoch": self.epoch,
            "loss": self.loss,
        }

    @classmethod
    def from_id(cls, id):
        folder_path = Path(__file__).resolve().parent / "models" / str(id)
        checkpoint = [x for x in folder_path.glob("*.pt") if x.is_file()]
        script = [x for x in folder_path.glob("*.pkl") if x.is_file()]
        attributes = {}
        if checkpoint:
            checkpoint = torch.load(checkpoint[0])
            attributes["model_dict"] = checkpoint["model_state_dict"]
            attributes["optimizer_dict"] = checkpoint["optimizer_state_dict"]
            attributes["loss_type"] = checkpoint["loss_type"]
        if script:
            attributes["script"] = pickle.load(open(script[0], "rb"))
        return cls(id=id, **database_api.retrieve(id, db="models"), **attributes)

    def save(self):
        assert self.comment != "", "There is not comment"
        assert self.loss != np.inf, "Loss is not defined"
        assert self.model_dict is not None, "Model weights are not defined"
        assert self.optimizer_dict is not None, "Optimizer weights are not defined"
        assert self.loss_type is not None, "Loss type is not defined"
        super(Model, self).save("models", overwrite=True)
        folder_path = Path(__file__).resolve().parent / "models" / str(self.id)
        folder_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model_dict,
                "optimizer_state_dict": self.optimizer_dict,
                "loss_type": self.loss_type,
            },
            folder_path / "model.pt",
        )
        if self.script is not None:
            with open(folder_path / "script.pkl", "wb") as file:
                pickle.dump(self.script, file)
        return self.id


@dataclass
class Metrics:
    id: str = None
    dataframe: pd.DataFrame = None

    @classmethod
    def from_id(cls, id):
        folder_path = Path(__file__).resolve().parent / "metrics" / str(id)
        folder_path.mkdir(parents=True, exist_ok=True)
        pd_dataframe = [x for x in folder_path.glob("*.zip") if x.is_file()]
        if pd_dataframe:
            dataframe = pd.read_pickle(pd_dataframe[0]).to_dict(orient="list")
        else:
            dataframe = {}
        return cls(id=id, dataframe=dataframe)

    def to_dict(self):
        if isinstance(self.dataframe, dict):
            return self.dataframe
        else:
            return self.dataframe.to_dict(orient="list")

    def to_dataframe(self):
        max_length = np.max([len(n) for n in self.dataframe.values()])
        for key in self.dataframe.keys():
            gap = max_length - len(self.dataframe[key])
            self.dataframe[key] += [np.nan for i in range(gap)]
        if isinstance(self.dataframe, dict):
            return pd.DataFrame.from_dict(self.dataframe)
        else:
            return self.dataframe

    def append(self, col_name, arr):
        assert col_name not in self.dataframe.keys()
        self.dataframe[col_name] = arr

    def append_to_existing(self, col_name, arr):
        if col_name in self.dataframe.keys():
            self.dataframe[col_name] += arr
        else:
            self.dataframe[col_name] = arr

    def get(self, col_name, col_name2=None):
        assert col_name in self.dataframe.keys()
        if col_name2 is None:
            return self.dataframe[col_name]
        assert col_name2 in self.dataframe.keys()
        return self.dataframe[col_name], self.dataframe[col_name2]

    def save(self):
        folder_path = Path(__file__).resolve().parent / "metrics" / str(self.id)
        folder_path.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_pickle(folder_path / "dataframe.zip")
        return self.id


# topLevelFolder = Path(__file__).resolve().parents[1] / "REV1_600"
# path_to_fabrics = topLevelFolder / "fabrics.txt"
# fabrics_df = pd.read_csv(path_to_fabrics)
# transform = transforms.Compose(
#     [
#         transforms.CenterCrop(207),
#         transforms.Resize((24, 24)),
#         transforms.ToTensor(),
#         transforms.GaussianBlur(kernel_size=3, sigma=0.5),
#     ]
# )
# train_dataset = a.SinglePhotoDataset(
#     fabrics_df, normalization=True, transform=transform
# )


# print(inspect.getsourcelines(type(train_dataset)))
# dt = Dataset()
# print(dt)
# tr = Training()
# tr.comment = "whatever the fuck 3"
# tr.training_dataset_id = 1
# tr.update_checkpoint("first", 1)
# id = tr.save()
# print(id)
