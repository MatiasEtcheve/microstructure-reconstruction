from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from tqdm import tqdm

from . import inspect_code


class SaveOutput:
    """Class to save in the forward hook"""

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class ScriptCheckpoint(pl.callbacks.Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath

    def on_pretrain_routine_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        super().on_pretrain_routine_start(trainer, pl_module)
        filename_model = Path(self.dirpath) / "model_script.txt"
        with open(filename_model, "w") as file:
            file.write(inspect_code.get_class_code(type(pl_module)))
        filename_datamodule = Path(self.dirpath) / "datamodule_script.txt"
        with open(filename_datamodule, "w") as file:
            file.write(inspect_code.get_class_code(type(trainer.datamodule)))


def train(model, device, train_loader, optimizer, criterion) -> List:
    """Train the model for 1 epoch

    Args:
        model: model to train
        device: device of the training
        train_loader: training dataloader
        optimizer: optimizer of the training
        criterion: loss of the model

    Returns:
        List: list of training loss
    """
    model.train()
    train_loss = []
    for data, target in tqdm(train_loader, total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return train_loss


def validate(model, device, val_loader, criterion, scaler=None) -> Tuple:
    """Validate the model on the validation dataloader

    Args:
        model: model to evaluate
        device: device of the evaluation
        val_loader: validation dataloader
        criterion: loss of the model
        scaler: Scaler of the data. Defaults to None.

    Returns:
        Tuple: list of validation loss, example images and error
    """
    model.eval()
    val_loss = []
    example_images = []
    error = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss.append(criterion(output, target).item())
            if scaler is not None:
                output = scaler.inverse_transform(output.cpu().numpy())
                target = scaler.inverse_transform(target.cpu().numpy())
                example_images.append(wandb.Image(data[0]))
                error = max(error, ((output - target) / target).max())
            else:
                error = np.inf
    return val_loss, example_images, error


def compute_predictions(model, device, val_loader) -> Tuple:
    """Compute predictions of the model

    Args:
        model: model to evaluate
        device: device of the evaluation
        val_loader: validation dataloader

    Returns:
        Tuple: prediction and targets
    """
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            prediction = model(data)
            if idx == 0:
                predictions = prediction
                targets = target
            else:
                predictions = torch.cat((predictions, prediction), dim=0)
                targets = torch.cat((targets, target), dim=0)
    return predictions, targets


def compute_errors(predictions, targets, device, scaler=None):
    """Compute errors of the model from predictions and targets

    Args:
        predictions: predictions to compute errors from
        targets: targets to compute errors from
        device: device of the evaluation
        scaler: Scaler of the data. Defaults to None.

    Returns:
        errors
    """
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.to(device))
        targets = scaler.inverse_transform(targets.to(device))
    return (predictions - targets) / targets
