from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as pretrained_models
from sklearn.preprocessing import MinMaxScaler


class BaseModel(pl.LightningModule):
    """Base model. Instantiates basic functions:
    * training, validation and predict step
    * configures criterion, optimizer, and metrics
    """

    def __init__(self, config: Dict):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `epochs`: current epoch
                * `loss_type`: type of the loss, eg `torchmetrics.MeanAbsoluteError`
                * `learning_rate` of the optimizer
                * `weight_decay` of the optimizer
                * `optimizer_type` of the optimizer
        """
        super().__init__()

        self.config = config

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = {name: metric(y, y_hat) for name, metric in self.metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        if isinstance(batch, tuple):
            return self(batch[0])
        return self(batch)

    def training_epoch_end(self, outputs):
        self.config["epochs"] += 1

    def configure_criterion(self):
        """Configures the criterion / loss"""
        self.criterion = nn.L1Loss()
        self.config["loss_type"] = type(self.criterion)

    def configure_metrics(self, num_outputs: Optional[int] = 25):
        """Configure the metrics to run in the validation step.

        Args:
            num_outputs (Optional[int], optional): Some metrics requier the number of features (=descriptors to predict).
                Defaults to 25.
        """
        self.metrics = {
            "val_loss": self.criterion.to(self.config["device"]),
            "mae": torchmetrics.MeanAbsoluteError().to(self.config["device"]),
            "mape": torchmetrics.MeanAbsolutePercentageError().to(
                self.config["device"]
            ),
            "smape": torchmetrics.SymmetricMeanAbsolutePercentageError().to(
                self.config["device"]
            ),
            "r2_score": torchmetrics.R2Score(num_outputs=num_outputs).to(
                self.config["device"]
            ),
            "cosine_similarity": torchmetrics.CosineSimilarity(reduction="mean").to(
                self.config["device"]
            ),
        }

    def configure_optimizers(self) -> torch.optim:
        """Configures the optimizer in the pytorch lightning module.
        WARNING: compared to other `self.configure_` methods, this one is required by pytorch lightning.

        Returns:
            torch.optim: optimizer used in the model.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.config["optimizer_type"] = type(optimizer)
        return optimizer


class VGG11(BaseModel):
    """Manually generated VGG11 model. Each layer is randomly initiated.

    Attributes:
        conv_layers (nn.Module): convolutional layers.
        linear_layers (nn.Module): fully connected linear layers.

    Notes:
        * This model works less well than a pre-trained VGG model.
        * The input of this model is a batch of images of size `(N, 1, config["input_width"], config["input_width"])`
        * This model predict 23 features.
    """

    def __init__(self, config: Dict, scaler: Optional[MinMaxScaler] = None):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `input_width` of all the images
                * `learning_rate` of the optimizer
                * `weight_decay` of the optimizer
            scaler (Optional[MinMaxScaler], optional): scaler for the features. Defaults to None.
        """
        super().__init__(config)

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

    def configure_model(self):
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        input_fc = int((self.config["input_width"] / (2**5)) ** 2 * 512)
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_fc, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=512, out_features=23),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.linear_layers(x)
        return x


class PreTrainedVGG(BaseModel):
    """Pretrained VGG16 model. This model includes batch norm layers.

    Attributes:
        layers (nn.Module): convolutional layers of the VGG16 model. Does not include the fully connected linear layers.
        linear_layers (nn.Module): fully connected linear layers.

    Notes:
        * This model works well.
        * The input of this model is a batch of images of size
            `(N, 3, config["input_width"], config["nb_image_per_axis"]*config["input_width"])`.
            See `custom_datasets.datasets.NWidthStackedPhotosDataset` for the inputs.
        * This model predict 25 features.
    """

    def __init__(self, config: Dict, scaler: Optional[MinMaxScaler] = None):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `input_width` of all the images
                * `learning_rate` of the optimizer
                * `weight_decay` of the optimizer
                * `total_layers`: number of layers to take from the pretrained VGG16 model.
                * `fixed_layers`: number of layers to freeze in the pretrained VGG16 model.
                    `fixed_layers` must be lower than `total_layers`
            scaler (Optional[MinMaxScaler], optional): scaler for the features. Defaults to None.
        """
        super().__init__(config)

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

    def configure_model(self):
        """Configure the `self.layers` and `self.linear_layers`.

        Raises:
            ValueError: if the number of frozen layer is greater than the number of total layers.
        """
        if self.config["total_layers"] < self.config["fixed_layers"]:
            raise ValueError(
                "The number of fixed layers must be lower than the number of total layers"
            )
        vgg = pretrained_models.vgg16_bn(pretrained=True)
        self.layers = nn.Sequential(
            *(list(vgg.features.children())[: self.config["total_layers"]])
        )
        for idx, child in enumerate(self.layers.children()):
            if idx < self.config["fixed_layers"] and isinstance(child, nn.Conv2d):
                for param in child.parameters():
                    param.requires_grad = False

        nb_channels, height, width = (
            self.layers(
                torch.rand(
                    (
                        1,
                        3,
                        self.config["input_width"],
                        self.config["nb_image_per_axis"] * self.config["input_width"],
                    )
                )
            )
            .squeeze()
            .shape
        )
        input_fc = int(height * width * nb_channels)
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_fc, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=512, out_features=25),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.linear_layers(x)
        return x


class PreTrainedResnet(BaseModel):
    """Pretrained Resnet18 model.

    Attributes:
        layers (nn.Module): convolutional layers of the Resnet18 model. Does not include the fully connected linear layers.
        linear_layers (nn.Module): fully connected linear layers.

    Notes:
        * This model doesn't work well.
        * The input of this model is a batch of images of size
            `(N, 3, config["input_width"], config["nb_image_per_axis"]*config["input_width"])`.
            See `custom_datasets.datasets.NWidthStackedPhotosDataset` for the inputs.
        * This model predict 25 features.
    """

    def __init__(self, config: Dict, scaler: Optional[MinMaxScaler] = None):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `input_width` of all the images
                * `nb_image_per_axis`: the number of sliced images per axis
                * `learning_rate` of the optimizer
                * `weight_decay` of the optimizer
                * `total_layers`: number of layers to take from the pretrained VGG16 model.
                * `fixed_layers`: number of layers to freeze in the pretrained VGG16 model.
                    `fixed_layers` must be lower than `total_layers`
            scaler (Optional[MinMaxScaler], optional): scaler for the features. Defaults to None.
        """
        super().__init__(config)

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

    def freeze_sequence(self, sequence, current_layer: List[int]):
        """Recursively freeze the layers of the Resnet18 model.

        The Resnet18 model is made of block which itself is made of Conv layers, batch layers...
        The function only freezes the conv layers.

        Args:
            sequence : Current layer to freeze (or not)
            current_layer (List[int]): current level of layers. This make sure to only freeze the first `config["fixed_layers"]` layers.
        """
        if (
            isinstance(sequence, nn.Sequential)
            or isinstance(sequence, pretrained_models.resnet.BasicBlock)
            or isinstance(sequence, pretrained_models.resnet.Bottleneck)
        ):
            for child in sequence.children():
                self.freeze_sequence(child, current_layer)
        if current_layer[0] < self.config["fixed_layers"] and isinstance(
            sequence, nn.Conv2d
        ):
            for param in sequence.parameters():
                param.requires_grad = False
        current_layer[0] += 1

    def configure_model(self):
        """Configure the `self.layers` and `self.linear_layers`.

        Raises:
            ValueError: if the number of frozen layer is greater than the number of total layers.
        """
        if self.config["total_layers"] < self.config["fixed_layers"]:
            raise ValueError(
                "The number of fixed layers must be lower than the number of total layers"
            )
        resnet = pretrained_models.resnet18(pretrained=True)
        self.layers = nn.Sequential(*(list(resnet.children())[:-1]))
        self.freeze_sequence(self.layers, [0])

        nb_channels, height, width = (
            self.layers(
                torch.rand(
                    (
                        1,
                        3,
                        self.config["input_width"],
                        self.config["nb_image_per_axis"] * self.config["input_width"],
                    )
                )
            )
            .squeeze()
            .shape
        )

        input_fc = int(height * width * nb_channels)
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=input_fc, out_features=25),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.linear_layers(x)
        return x


class PreTrainedEfficientNet(BaseModel):
    """Pretrained EfficientNet model.

    Attributes:
        layers (nn.Module): convolutional layers of the EfficientNet model. Does not include the fully connected linear layers.
        linear_layers (nn.Module): fully connected linear layers.

    Notes:
        * This model doesn't work well.
        * The input of this model is a batch of images of size
            `(N, 3, config["input_width"], config["nb_image_per_axis"]*config["input_width"])`.
            See `custom_datasets.datasets.NWidthStackedPhotosDataset` for the inputs.
        * This model predict 25 features.
    """

    def __init__(self, config: Dict, scaler: Optional[MinMaxScaler] = None):
        """Constructor

        Args:
            config (Dict): dict specifying the configuration:
                * `input_width` of all the images
                * `nb_image_per_axis`: the number of sliced images per axis
                * `learning_rate` of the optimizer
                * `weight_decay` of the optimizer
                * `total_layers`: number of layers to take from the pretrained VGG16 model.
                * `fixed_layers`: number of layers to freeze in the pretrained VGG16 model.
                    `fixed_layers` must be lower than `total_layers`
            scaler (Optional[MinMaxScaler], optional): scaler for the features. Defaults to None.
        """
        super().__init__(config)

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

    def freeze_sequence(self, sequence, current_layer):
        """Recursively freeze the layers of the Resnet18 model.

        The EfficientNet model is made of block which itself is made of Conv layers, batch layers...
        The function only freezes the conv layers.

        Args:
            sequence : Current layer to freeze (or not)
            current_layer (List[int]): current level of layers. This make sure to only freeze the first `config["fixed_layers"]` layers.
        """
        if (
            isinstance(sequence, nn.Sequential)
            or isinstance(sequence, pretrained_models.efficientnet.ConvNormActivation)
            or isinstance(sequence, pretrained_models.efficientnet.MBConv)
            or isinstance(sequence, pretrained_models.efficientnet.SqueezeExcitation)
        ):
            for child in sequence.children():
                self.freeze_sequence(child, current_layer)
        if current_layer[0] < self.config["fixed_layers"] and isinstance(
            sequence, nn.Conv2d
        ):
            for param in sequence.parameters():
                param.requires_grad = False
        current_layer[0] += 1

    def configure_model(self):
        """Configure the `self.layers` and `self.linear_layers`.

        Raises:
            ValueError: if the number of frozen layer is greater than the number of total layers.
        """
        if self.config["total_layers"] < self.config["fixed_layers"]:
            raise ValueError(
                "The number of fixed layers must be lower than the number of total layers"
            )
        effnet = pretrained_models.efficientnet_b0(pretrained=True)
        self.layers = nn.Sequential(*(list(effnet.features.children())[:-1]))
        self.freeze_sequence(self.layers, [0], "")

        nb_channels, height, width = (
            self.layers(
                torch.rand(
                    (
                        1,
                        3,
                        self.config["input_width"],
                        self.config["nb_image_per_axis"] * self.config["input_width"],
                    )
                )
            )
            .squeeze()
            .shape
        )

        input_fc = int(height * width * nb_channels)
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=input_fc, out_features=25),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.linear_layers(x)
        return x
