from pprint import pprint

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models


class BaseModel(pl.LightningModule):
    def __init__(self, config, scaler=None):
        super().__init__()

        self.config = config
        self.config["model_type"] = type(self)
        self.scaler = scaler

        self.configure_model()
        self.configure_criterion()
        self.configure_metrics()

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
        super().__init__()

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


class PreTrainedResnet(BaseModel):
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
        resnet = models.resnet18(pretrained=True)
        self.layers = nn.Sequential(*(list(resnet.children())[:-1]))
        print(self.layers)
        for idx, child in enumerate(self.layers.children()):
            print(type(child))
            if isinstance(child, nn.Sequential):
                for mini_idx, mini_child in enumerate(child.children()):
                    print(f"\t{type(mini_child)}")
                    if isinstance(mini_child, models.resnet.BasicBlock):
                        for megamini_idx, megamini_child in enumerate(
                            mini_child.children()
                        ):
                            print(f"\t\t{type(megamini_child)}")
        #     if idx < self.config["fixed_layers"]:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #     else:
        #         reset_parameters = getattr(child, "reset_parameters", None)
        #         if callable(reset_parameters):
        #             child.reset_parameters()

        self.linear_layers = nn.Sequential(
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
        # x = x.view(x.size(0), -1)
        # x = self.linear_layers(x)
        return x


config = {}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
config["job_type"] = run.job_type if "run" in locals() else "test"
config["train_val_split"] = 0.7
config["seed"] = 42
config["batch_size"] = 64
config["learning_rate"] = 0.00001
config["device"] = device
config["momentum"] = 0.9
config["architecture"] = "pretrained VGG"
config["input_width"] = 64
config["weight_decay"] = 0.00005
config["epochs"] = 0
config["frac_sample"] = 1
config["frac_noise"] = 0.1
config["nb_image_per_axis"] = 1
config["total_layers"] = 1000
config["fixed_layers"] = 0
config["log_wandb"] = False
model = PreTrainedResnet(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"[INFO]: {total_params:,} total parameters.")
print(model(torch.rand((1, 3, config["input_width"], config["input_width"]))).shape)
