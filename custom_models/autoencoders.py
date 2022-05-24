from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder of an autoencoder network.

    Attributes:
        model (nn.Sequential): encoder model until the flattening layer
        output (nn.Sequential): encoder model from the flattening layer to the linear layer to the latent space
    """

    def __init__(
        self,
        config: Dict,
    ):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `input_width` of all the images
                * `batch_size` of the inputs
                * `latent_size` of the latent space
        """
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=2, padding=0),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
        )
        _, length = self.model(
            torch.rand(
                (
                    config["batch_size"],
                    1,
                    config["input_width"],
                    config["input_width"],
                )
            )
        ).shape
        self.output = nn.Sequential(
            nn.Linear(
                length,
                config["latent_size"],
            ),
        )

    def forward(self, img):
        x = self.model(img)
        return self.output(x)


class Decoder(nn.Module):
    """Decoder of an autoencoder network.

    Attributes:
        linear (nn.Sequential): decoder model from the latent representation to the next linear layer
        model (nn.Sequential): decoder model
    """

    def __init__(self, config: Dict):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `latent_size` of the latent space
        """
        super(Decoder, self).__init__()
        self.linear = nn.Linear(config["latent_size"], 128)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.Upsample(scale_factor=(2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.Upsample(scale_factor=(2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(
                64,
                128,
                kernel_size=5,
                stride=2,
                padding=0,
            ),
            nn.Upsample(scale_factor=(2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(
                128,
                1,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.linear(z)
        z = torch.reshape(z, (-1, 32, 2, 2))
        img = self.model(z)
        return img


class Autoencoder(pl.LightningModule):
    """Autoencoder model.

    Attributes:
        encoder (Union[Encoder, nn.Module]): encoder model
        decoder (Union[Decoder, nn.Module]): decoder model
    """

    def __init__(
        self,
        config: Dict,
    ):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `input_width` of all the images
                * `batch_size` of the inputs
                * `latent_size` of the latent space
                * `learning_rate` of the Adam optimizer
                * `weight_decay` of the Adam optimizer
        """
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        return optimizer

    def compute_metrics(self, img):
        encoding = self.encoder(img)
        fake_img = self.decoder(encoding)
        mae_encoding = torch.mean(torch.abs(encoding))
        metrics = {
            "loss": nn.MSELoss(reduction="mean")(fake_img, img),
            "mse": nn.MSELoss(reduction="mean")(fake_img, img),
            "mae_encoding": mae_encoding,
            "bce": nn.BCELoss(reduction="mean")(fake_img, img),
        }
        return metrics

    def training_step(self, batch, batch_idx):
        img, _ = batch
        metrics = {
            "train_" + metric_name: metric_value
            for metric_name, metric_value in self.compute_metrics(img).items()
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return metrics["train_loss"]

    def validation_step(self, batch, batch_idx):
        img, _ = batch
        metrics = {
            "val_" + metric_name: metric_value
            for metric_name, metric_value in self.compute_metrics(img).items()
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def training_epoch_end(self, outputs):
        self.config["epochs"] += 1
