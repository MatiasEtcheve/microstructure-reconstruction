from typing import List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

print("yoyo")


class SinglePhotoDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        normalization: Union[bool, List[float]] = True,
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
    ):
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)

        self.images = df.pop("photo").to_numpy()
        self.labels = df.to_numpy()

        self._normalize(normalization)

        if not any(
            [isinstance(tr, transforms.ToTensor) for tr in transform.transforms]
        ):
            transform.transforms.append(transforms.ToTensor())

        self.transform = transform

    def _normalize(self, normalization):
        if isinstance(normalization, bool):
            if normalization:
                self.max = np.max(self.labels, axis=0)
                self.min = np.min(self.labels, axis=0)

                self.labels = (self.labels - self.min) / (self.max - self.min)

            else:
                self.max, self.min = None, None

        if isinstance(normalization, list):
            assert len(normalization) == 2
            assert all(
                [
                    isinstance(x, float) or isinstance(x, np.ndarray)
                    for x in normalization
                ]
            )

            self.max = normalization[0]
            self.min = normalization[1]

            self.labels = (self.labels - self.min) / (self.max - self.min)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        if isinstance(idx, list):
            image = [Image.open(path) for path in img_path]
        else:
            image = Image.open(img_path).convert("RGB").convert("1")
            if self.transform:
                image = self.transform(image)

        label = torch.Tensor(self.labels[idx, :])
        return image, label
