from pathlib import Path
from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from torch import nn

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
    """Script checkpoint. This saves the code of different snippets:
    * the model
    * the data module used in the Trainer
    * the encoder / decoder if the model has such attributes
    * the generator / discriminator if the model has such attributes
    """

    def __init__(self, dirpath: Path):
        """Constructor

        Args:
            dirpath (Path): path to save the snippets
        """
        super().__init__()
        self.dirpath = dirpath
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Saves the snippets at the beginning of the training.

        Args:
            trainer (pl.Trainer): trainer of `pytorch lightning`
            pl_module (pl.LightningModule): `pytorch lightning` module (which is often the model itself)
        """
        super().on_fit_start(trainer, pl_module)

        filename_model = Path(self.dirpath) / "model_script.txt"
        with open(filename_model, "w") as file:
            file.write(inspect_code.get_class_code(type(pl_module)))
        filename_datamodule = Path(self.dirpath) / "datamodule_script.txt"
        with open(filename_datamodule, "w") as file:
            file.write(inspect_code.get_class_code(type(trainer.datamodule)))

        for attr in ["generator", "discriminator", "encoder", "decoder"]:
            if hasattr(pl_module, attr):
                filename_model = Path(self.dirpath) / f"{attr}_script.txt"
                with open(filename_model, "w") as file:
                    file.write(
                        inspect_code.get_class_code(type(getattr(pl_module, attr)))
                    )


class GeneratedImagesCallback(pl.callbacks.Callback):
    """Callback to log images generated from an `inputs`.
    The concerned model can be an autoencoder, ie `inputs` is a set of images.
    The concerned model can be a GAN, ie `inputs` is a random vector."""

    def __init__(
        self, inputs: Union[torch.Tensor, np.ndarray], log_every_n_epochs: int = 10
    ):
        """Constructor

        Args:
            inputs (Union[torch.Tensor, np.ndarray]): inputs to encode and decode.
            log_every_n_epochs (int, optional): frequence of the log. Defaults to 10.
        """
        super().__init__()
        self.inputs = inputs
        self.log_every_n_epochs = log_every_n_epochs
        self.current_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        """Encode and decode `self.inputs` at the beginning of each epoch.

        Args:
            trainer (pl.Trainer): trainer of `pytorch lightning`
            pl_module (pl.LightningModule): `pytorch lightning` module (which is often the model itself)
        """

        if self.current_epoch % self.log_every_n_epochs == 0:
            sample_outputs = pl_module(self.inputs)
            image = wandb.Image(
                sample_outputs,
                caption=f"First batch generated at epoch {self.current_epoch}",
            )
            trainer.logger.experiment.log({"generated_images": image})
        self.current_epoch += 1


def compute_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Computes metrics between predictions and targets

    Args:
        predictions (Union[torch.Tensor, np.ndarray]): tensor of predictions in regression
        targets (Union[torch.Tensor, np.ndarray]): tensor of targets (descriptors) in regression

    Returns:
        Dict[str, float]: dict whose keys are metric's names and whose values are the metrics.
    """
    predictions = torch.FloatTensor(predictions)
    targets = torch.FloatTensor(targets)

    return {
        "cosine_similarity": torchmetrics.CosineSimilarity(reduction="mean")(
            predictions, targets
        ),
        "r2_score": torchmetrics.R2Score(num_outputs=predictions.shape[1])(
            predictions, targets
        ),
        "smape": torchmetrics.SymmetricMeanAbsolutePercentageError()(
            predictions, targets
        ),
        "mape": torchmetrics.MeanAbsolutePercentageError()(predictions, targets),
        "mae": torchmetrics.MeanAbsoluteError()(predictions, targets),
        "mse": torchmetrics.MeanSquaredError()(predictions, targets),
    }


def metrics_str(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> str:
    """Produces a metric string between predictions and targets.

    Args:
        predictions (Union[torch.Tensor, np.ndarray]): tensor of predictions in regression
        targets (Union[torch.Tensor, np.ndarray]): tensor of targets (descriptors) in regression

    Returns:
        str: metric string
    """
    metrics = compute_metrics(predictions, targets)
    metrics_str = ""
    for metric_name, value in metrics.items():
        metrics_str += f"{metric_name}: {value}\n"
    metrics_str += "_______________________________________________________________"
    return metrics_str
