from pprint import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as pretrained_models


class BaseModel(pl.LightningModule):
    def __init__(self, config, scaler=None):
        super().__init__()

        # self.config = config
        # self.config["model_type"] = type(self)
        # self.scaler = scaler

        # self.configure_model()
        # self.configure_criterion()
        # self.configure_metrics()

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
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        if isinstance(batch, tuple):
            return self(batch[0])
        return self(batch)

    def training_epoch_end(self, outputs):
        self.config["epochs"] += 1

    def configure_criterion(self):
        self.criterion = nn.L1Loss()
        self.config["loss_type"] = type(self.criterion)

    def configure_metrics(self):
        self.metrics = {
            "val_loss": self.criterion.to(self.config["device"]),
            "mae": torchmetrics.MeanAbsoluteError().to(self.config["device"]),
            "mape": torchmetrics.MeanAbsolutePercentageError().to(
                self.config["device"]
            ),
            "smape": torchmetrics.SymmetricMeanAbsolutePercentageError().to(
                self.config["device"]
            ),
            "r2_score": torchmetrics.R2Score(num_outputs=23).to(self.config["device"]),
            "cosine_similarity": torchmetrics.CosineSimilarity(reduction="mean").to(
                self.config["device"]
            ),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        self.config["optimizer_type"] = type(optimizer)
        return optimizer


class VGG11(BaseModel):
    def __init__(self, config, scaler=None):
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
        input_fc = int((self.config["input_width"] / (2 ** 5)) ** 2 * 512)
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


class PreTrainedVGG(models.BaseModel):
    def __init__(self, config, scaler=None):
        super().__init__(config)

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

    def configure_model(self):
        assert self.config["total_layers"] >= self.config["fixed_layers"]
        vgg = pretrained_models.vgg16_bn(pretrained=True)
        self.layers = nn.Sequential(
            *(list(vgg.features.children())[: self.config["total_layers"]])
        )
        for idx, child in enumerate(self.layers.children()):
            if idx < self.config["fixed_layers"] and isinstance(child, nn.Conv2d):
                for param in child.parameters():
                    param.requires_grad = False
        #             else:
        #                 reset_parameters = getattr(child, "reset_parameters", None)
        #                 if callable(reset_parameters):
        #                     child.reset_parameters()
        nb_channels, width, a = (
            self.layers(
                torch.rand(
                    (1, 3, self.config["input_width"], self.config["input_width"])
                )
            )
            .squeeze()
            .shape
        )

        input_fc = int(width ** 2 * nb_channels)
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = {name: metric(y, y_hat) for name, metric in self.metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return metrics


class PreTrainedResnet(BaseModel):
    def __init__(self, config, scaler=None):
        super().__init__(config)

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

    def freeze_sequence(self, sequence, current_layer, str):
        print(str, current_layer[0], type(sequence))
        if (
            isinstance(sequence, nn.Sequential)
            or isinstance(sequence, pretrained_models.resnet.BasicBlock)
            or isinstance(sequence, pretrained_models.resnet.Bottleneck)
        ):
            for idx, child in enumerate(sequence.children()):
                self.freeze_sequence(child, current_layer, str + "\t")
        if current_layer[0] < self.config["fixed_layers"] and isinstance(
            sequence, nn.Conv2d
        ):
            for param in sequence.parameters():
                param.requires_grad = False
        current_layer[0] += 1

    def configure_model(self):
        assert self.config["total_layers"] >= self.config["fixed_layers"]
        resnet = pretrained_models.resnet18(pretrained=True)
        self.layers = nn.Sequential(*(list(resnet.children())[:-1]))
        self.freeze_sequence(self.layers, [0], "")

        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=512),
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
