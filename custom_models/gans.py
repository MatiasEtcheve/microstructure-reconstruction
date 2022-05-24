import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

# WARNING:
# This module was stopped during development. I hadn't fully understood how a gan worked,
# until I figured out that I was needing Autoencoders instead.
# All the classes don't work as a gan would do.


class Generator(nn.Module):
    """Generator of an GAN network.

    Attributes:
        linear_layer (nn.Sequential): linear layer that maps the latent uniform vector to another dimension space
        model (nn.Sequential): generator model from the "another dimension space" to the images
    """

    def __init__(self, config: Dict):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `latent_size` of the latent space
                * `nb_image_per_axis` tells the depth of the produced tensor.
        """
        super(Generator, self).__init__()

        filters = [400, 512, 256, 64, 32, config["nb_image_per_axis"]]

        self.linear_layer = nn.Sequential(
            nn.Linear(config["latent_size"], filters[0]),
            nn.ReLU(),
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                filters[0],
                filters[0],
                stride=2,
                kernel_size=3,
                padding=0,
            ),
            nn.Conv2d(
                filters[0],
                filters[1],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                filters[1],
                filters[1],
                stride=2,
                kernel_size=3,
                padding=0,
            ),
            nn.Conv2d(
                filters[1],
                filters[2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                filters[2],
                filters[2],
                stride=2,
                kernel_size=3,
                padding=0,
            ),
            nn.Conv2d(
                filters[2],
                filters[3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(
                filters[3],
                filters[3],
                stride=2,
                kernel_size=4,
                padding=0,
            ),
            nn.Conv2d(
                filters[3],
                filters[4],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(filters[4]),
            nn.ConvTranspose2d(
                filters[4],
                filters[4],
                stride=2,
                kernel_size=4,
                padding=1,
            ),
            nn.Conv2d(
                filters[4],
                filters[5],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(filters[5]),
            nn.ReLU(),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.linear_layer(z)
        z = torch.unsqueeze(torch.unsqueeze(z, -1), -1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    """Discriminator of an GAN network.

    Attributes:
        model (nn.Sequential): discriminator model until the flattening layer
        output (nn.Sequential): last layer outputting a probability
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
                * `nb_image_per_axis` tells the depth of the produced tensor.
        """
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                3 * config["nb_image_per_axis"], 512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 126, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(126),
            nn.LeakyReLU(),
            nn.Conv2d(126, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        _, length = self.model(
            torch.rand(
                (
                    config["batch_size"],
                    3 * config["nb_image_per_axis"],
                    config["input_width"],
                    config["input_width"],
                )
            )
        ).shape
        self.output = nn.Sequential(
            nn.Linear(
                length,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, img):
        x = self.model(img)
        return self.output(x)


class WGANGP(pl.LightningModule):
    """Gan implemented with Wasserstein distance and gradient penalty.

    Attributes:
        generator (Union[Encoder, nn.Module]): generator model
        discriminator (Union[Decoder, nn.Module]): discriminator model
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
                * `nb_image_per_axis` tells the depth of the produced tensor.
                * `lambda_gp` the proportion of the gradient penalty in the loss
                * `learning_rate_generator` the learning rate of the generator
                * `learning_rate_discriminator` the learning rate of the discriminator
                * `n_critic` tells how many times we train the discriminator for every training iteration of the generator
                * `beta1` and `beta2` parameters of the Adam optimizer
        """
        super().__init__()
        self.config = config

        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def forward(self, z):
        return self.generator(z)

    def compute_gradient_penalty(
        self, real_samples: torch.Tensor, fake_samples: torch.Tensor
    ) -> float:
        """Computes the gradient penalty between real and fake samples.

        Args:
            real_samples (torch.Tensor): the real samples taken in the dataset
            fake_samples (torch.Tensor): the fake samples generated by the generator

        Returns:
            float: the gradient penalty
        """

        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
            self.device
        )
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def adversarial_loss(self, y_hat, y):
        return nn.BCELoss()(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Training step of the bodel.

        According to the `optimizer_idx`. We either train the generator or the discriminator.

        Args:
            batch: batch containing the images and descriptors.
            batch_idx: index of the batch
            optimizer_idx: index of the optimizer. Either 0 for the generator or 1 for the discriminator

        Returns:
            Dict["loss", float]: dict whose unique item is "loss" and the corresponding value
        """
        imgs, descriptors = batch

        # train generator
        if optimizer_idx == 0:
            fake_imgs = self(descriptors)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            g_loss = -torch.mean(self.discriminator(fake_imgs))
            metrics = {"g_loss": g_loss}
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
            )
            return {"loss": g_loss}

        # train discriminator
        elif optimizer_idx == 1:
            fake_imgs = self(descriptors)
            real_validity = self.discriminator(imgs)
            fake_validity = self.discriminator(fake_imgs)

            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = self.adversarial_loss(real_validity, valid)

            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = self.adversarial_loss(fake_validity, fake)

            gradient_penalty = self.compute_gradient_penalty(imgs.data, fake_imgs.data)
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + self.config["lambda_gp"] * gradient_penalty
            )
            metrics = {
                "real_discriminator": real_loss,
                "fake_discriminator": fake_loss,
                "penalty_discriminator": self.config["lambda_gp"] * gradient_penalty,
                "d_loss": d_loss,
            }
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
            )
            return {"loss": d_loss}

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config["learning_rate_generator"],
            betas=(self.config["beta1"], self.config["beta2"]),
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["learning_rate_discriminator"],
            betas=(self.config["beta1"], self.config["beta2"]),
        )
        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": self.config["n_critic"]},
        )
