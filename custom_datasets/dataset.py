from typing import List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tools import dataframe_reformat
from torchvision import transforms

"""
This file contains useful dataset to work with. We imagine we work with data as follow:

+---+------+-------------------+-----+----------------+---------------+----------------------+---------------------------------------------+
|   | id   | orientation_0_std | ... | roundness_mean | roundness_std | volume_fraction_mean | photos                                      |
+---+------+-------------------+-----+----------------+---------------+----------------------+---------------------------------------------+
| 0 | REV1 | 0.324161          | ... | 0.097545       | 0.210755      | 0.086931             | ["REV1/x-y(1).png", ..., "REV1/x-y(N).png", |
|   |      |                   |     |                |               |                      |  "REV1/x-z(1).png", ...,                    |
|   |      |                   |     |                |               |                      |                     ..., "REV1/y-z(N).png"] |
+---+------+-------------------+-----+----------------+---------------+----------------------+---------------------------------------------+
| 1 | REV0 | 0.373141          | ... | 0.030363       | 0.297964      | 0.101607             | ["REV2/x-y(1).png", ..., "REV2/x-y(2).png", |
|   |      |                   |     |                |               |                      |  "REV2/x-z(1).png", ...,                    |
|   |      |                   |     |                |               |                      |                     ..., "REV2/y-z(N).png"] |
+---+------+-------------------+-----+----------------+---------------+----------------------+---------------------------------------------+

We aim at building different datasets according to our needs.
"""


class SinglePhotoDataset(torch.utils.data.Dataset):
    """Dataset whose inputs are sliced images from a plane, not matter which. From a dataframe of 600 revs of N images,
    this dataset can contain 600N inputs:

    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    |      | orientation_0_std | ... | roundness_mean | roundness_std | volume_fraction_mean | photo             |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    | 0    | 0.324161          | ... | 0.097545       | 0.210755      | 0.086931             | "REV1/x-y(1).png" |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    | 1    | 0.324161          | ... | 0.097545       | 0.210755      | 0.086931             | "REV1/x-y(2).png" |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    | ...                                                                                                        |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    | N    | 0.373141          | ... | 0.030363       | 0.297964      | 0.101607             | "REV2/x-y(1).png" |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    | ...                                                                                                        |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+
    | 600N | 0.373141          | ... | 0.030363       | 0.297964      | 0.101607             | "REV2/x-y(N).png" |
    +------+-------------------+-----+----------------+---------------+----------------------+-------------------+

    Important:
        Pros:
            * This type of dataset is easy to compute. As we work with grayscale images, each input has size (1, W, H)
            * We have more data. In the original ML model, for m revs and n images per slice on each rev, we had m inputs. We now have 3nm inputs.
            * Computation is easier and thus faster. We are using 2D convolutions instead of 2.5D or 3D convolutions.
        Cons:
            * Looses the spatial dependance: 2 images from the same rev are not related anymore
    """

    def __init__(
        self,
        df: pd.DataFrame,
        col_name_photos: str = "photos",
        normalization: Union[bool, List[float]] = True,
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
    ):
        """Constructor

        Args:
            df (pd.DataFrame): original dataframe
            col_name_photos (str): name of the column containing the photos
            normalization (Union[bool, List[float]], optional): normalization parameters
                If a list is provided, the first item is the min and the second item is the max value to use to normalize the data.
                If a bool is provided, it precises whether to normalize or not.
                Defaults to True.
            transform (transforms.Compose, optional): Basic transformation to do on images. Defaults to transforms.Compose([transforms.ToTensor()]).
        """
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)
        df = dataframe_reformat.convert_into_single_entry_df(
            df, col_name=col_name_photos
        )

        self.images = df.pop(col_name_photos).to_numpy()
        self.labels = df.to_numpy()

        if not any(
            [isinstance(tr, transforms.ToTensor) for tr in transform.transforms]
        ):
            transform.transforms.append(transforms.ToTensor())

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Method for dataset[idx]

        Args:
            idx (int): index of the dataset to fetch

        Returns:
            tuple: first item is the feature (image) of size `(W, H)`, the second item is the target (fabric descriptors)
        """
        img_path = self.images[idx]
        if isinstance(idx, list):
            image = [Image.open(path) for path in img_path]
        else:
            image = Image.open(img_path).convert("RGB").convert("1")
            if self.transform:
                image = self.transform(image)

        label = torch.Tensor(self.labels[idx, :])
        return image, label


class NChannelPhotosDataset(SinglePhotoDataset):
    """Dataset whose inputs are images of size `(C, W, H)`. On each channel, there is a white and black image corresponding to a sliced image.

    Attributes:
        nb_input_photos_per_plane (int): number of sliced images per plane in the input. For instance,
            if we chose `nb_input_photos_per_plane=2`, the input will have a size `(nb_input_photos_per_plane*3, W, H)`:
                * the first 2 images are sliced images from x plane
                * the 2 next images are sliced images from y plane
                * the 2 final images are sliced images from z plane
    """

    def __init__(
        self,
        df: pd.DataFrame,
        nb_input_photos_per_plane: int = 1,
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
    ):
        """Constructor

        Args:
            df (pd.DataFrame): original dataframe to be used
            nb_input_photos_per_plane (int): Number of sliced images to use. Defaults to 1.
            normalization (Union[bool, List[float]], optional): normalization parameters
                If a list is provided, the first item is the min and the second item is the max value to use to normalize the data.
                If a bool is provided, it precises whether to normalize or not.
                Defaults to True.
            transform (transforms.Compose, optional): Basic transformation to do on images. Defaults to transforms.Compose([transforms.ToTensor()]).
        """
        df = dataframe_reformat.convert_into_n_entry_df(
            df, col_name="photos", nb_input_photos_per_plane=nb_input_photos_per_plane
        )
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)
        self.images = df.pop("photos").to_numpy()
        self.labels = df.to_numpy()
        if not any(
            [isinstance(tr, transforms.ToTensor) for tr in transform.transforms]
        ):
            transform.transforms.append(transforms.ToTensor())

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Method for dataset[idx]

        Args:
            idx (int): index of the dataset to fetch

        Returns:
            tuple: first item is the feature (image) of size `(C, W, H)`. Each channel corresponds to an image.
                The second item is the target (fabric descriptors)
        """
        img_paths = self.images[idx]
        images = [
            Image.open(img_path).convert("RGB").convert("1") for img_path in img_paths
        ]
        if self.transform:
            images = torch.cat([self.transform(image) for image in images])

        label = torch.Tensor(self.labels[idx, :])
        return images, label
