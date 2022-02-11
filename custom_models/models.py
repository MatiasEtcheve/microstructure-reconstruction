import torch
import torch.nn as nn
import torchvision.models as models


class VGG11(nn.Module):
    def __init__(self, input_channel, input_width, output_size):
        super(VGG11, self).__init__()
        self.in_channels = input_channel
        self.output_size = output_size
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        input_fc = int((input_width / (2 ** 4)) ** 2 * 512)
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_fc, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=self.output_size),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class PreTrainedVGG(nn.Module):
    def __init__(self, input_width, output_size, total_layers=16, fixed_layers=4):
        super(PreTrainedVGG, self).__init__()
        assert total_layers >= fixed_layers
        vgg = models.vgg16(pretrained=True)
        self.layers = nn.Sequential(*(list(vgg.features.children())[:total_layers]))
        for idx, child in enumerate(self.layers.children()):
            if idx < fixed_layers:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                reset_parameters = getattr(child, "reset_parameters", None)
                if callable(reset_parameters):
                    child.reset_parameters()
        nb_channels, width, _ = (
            self.layers(torch.rand((1, 3, input_width, input_width))).squeeze().shape
        )
        # self.layers.add_module(
        #     str(total_layers), nn.MaxPool2d(kernel_size=width, stride=1)
        # )
        # self.layers.add_module(str(total_layers + 1), nn.Flatten())
        input_fc = int(width ** 2 * nb_channels)
        # fully connected linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=input_fc, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(in_features=4096, out_features=output_size),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
