from typing import List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tools import dataframe_reformat
from torchvision import transforms

from .data_augm import RotateImgOnX, RotateImgOnY, RotateImgOnZ, RotateLabel

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
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        noise: int = 0,
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
        self.label_transform = None
        self.std = np.std(self.labels, axis=0)
        if noise != 0:
            self.label_transform = transforms.Lambda(
                lambda x: x + torch.Tensor(x.shape).uniform_(-1, 1) * self.std * noise
            )

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

        if self.label_transform is not None:
            label = self.label_transform(torch.Tensor(self.labels[idx, :]))
        else:
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
        order: str = "xyz",
        noise: int = 0,
        proba_rotating=0,
        proba_axis={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
        proba_angle={90: 0.5, -90: 0.5},
        mode="replace",
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
        self.nb_input_photos_per_plane = nb_input_photos_per_plane
        self.noise = noise
        self.proba_rotating = proba_rotating
        self.proba_axis = proba_axis
        self.proba_angle = proba_angle
        self.mode = mode
        df = dataframe_reformat.convert_into_n_entry_df(
            df,
            col_name="photos",
            nb_input_photos_per_plane=self.nb_input_photos_per_plane,
            order=order,
        )
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)
        self.images = df.pop("photos").to_numpy()
        self.labels = torch.Tensor(df.to_numpy())
        self.image_transform = transform
        # self.std = np.std(self.labels, axis=0)
        # if noise != 0:
        #     self.label_transform = transforms.Lambda(
        #         lambda x: x + torch.Tensor(x.shape).uniform_(-1, 1) * self.std * noise
        #     )

    def fetch_img_aug(self, angle, axis):
        if axis == 0:
            return RotateImgOnX(angle, self.nb_input_photos_per_plane)
        if axis == 1:
            return RotateImgOnY(angle, self.nb_input_photos_per_plane)
        if axis == 2:
            return RotateImgOnZ(angle, self.nb_input_photos_per_plane)
        raise NotImplemented("")

    def data_augmente_transformations(
        self,
        shape,
    ):
        indexes = np.random.binomial(1, self.proba_rotating, size=shape)
        axis = np.random.choice(
            list(self.proba_axis.keys()), size=shape, p=list(self.proba_axis.values())
        )
        angle = torch.Tensor(
            np.random.choice(
                list(self.proba_angle.keys()),
                size=shape,
                p=list(self.proba_angle.values()),
            )
        )
        image_transformations = [
            None if idx == 0 else self.fetch_img_aug(angle, axis)
            for idx, angle, axis in zip(indexes, angle, axis)
        ]
        label_transformations = [
            None if idx == 0 else RotateLabel(angle, axis)
            for idx, angle, axis in zip(indexes, angle, axis)
        ]
        return image_transformations, label_transformations

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
        labels = self.labels[idx, :]
        if isinstance(img_paths[0], str):
            img_paths = [img_paths]
            labels = [labels]
        aug_image_transforms, label_transforms = self.data_augmente_transformations(
            len(img_paths)
        )
        print(aug_image_transforms[0])
        images = [
            [
                transforms.ToTensor()(Image.open(img).convert("RGB").convert("1"))
                for img in img_path
            ]
            for img_path in img_paths
        ]

        if self.image_transform is not None:
            images = [self.image_transform(torch.cat(image)) for image in images]

        images = [
            aug_image_transform(image) if aug_image_transform is not None else image
            for aug_image_transform, image in zip(aug_image_transforms, images)
        ]
        labels = [
            label_transform(label) if label_transform is not None else label
            for label_transform, label in zip(label_transforms, labels)
        ]
        return torch.stack(images), torch.stack(labels)


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
        order: str = "xyz",
        noise: int = 0,
        proba_rotating=0,
        proba_axis={0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
        proba_angle={90: 0.5, -90: 0.5},
        mode="replace",
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
        self.nb_input_photos_per_plane = nb_input_photos_per_plane
        self.noise = noise
        self.proba_rotating = proba_rotating
        self.proba_axis = proba_axis
        self.proba_angle = proba_angle
        self.mode = mode
        df = dataframe_reformat.convert_into_n_entry_df(
            df,
            col_name="photos",
            nb_input_photos_per_plane=self.nb_input_photos_per_plane,
            order=order,
        )
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)
        self.images = df.pop("photos").to_numpy()
        self.labels = torch.Tensor(df.to_numpy())
        self.image_transform = transform
        # self.std = np.std(self.labels, axis=0)
        # if noise != 0:
        #     self.label_transform = transforms.Lambda(
        #         lambda x: x + torch.Tensor(x.shape).uniform_(-1, 1) * self.std * noise
        #     )

    def fetch_img_aug(self, angle, axis):
        if axis == 0:
            return RotateImgOnX(angle, self.nb_input_photos_per_plane)
        if axis == 1:
            return RotateImgOnY(angle, self.nb_input_photos_per_plane)
        if axis == 2:
            return RotateImgOnZ(angle, self.nb_input_photos_per_plane)
        raise NotImplemented("")

    def data_augmente_transformations(
        self,
        shape,
    ):
        nb_rotation = 3
        indexes = np.random.binomial(1, self.proba_rotating, size=shape)
        axis = np.random.choice(
            list(self.proba_axis.keys()),
            size=(shape, nb_rotation),
            p=list(self.proba_axis.values()),
        )
        angle = torch.Tensor(
            np.random.choice(
                list(self.proba_angle.keys()),
                size=(shape, nb_rotation),
                p=list(self.proba_angle.values()),
            )
        )
        image_transformations = [
            None
            if idx == 0
            else transforms.Compose(
                [self.fetch_img_aug(angle[i], axis[i]) for i in range(nb_rotation)]
            )
            for idx, angle, axis in zip(indexes, angle, axis)
        ]
        label_transformations = [
            None
            if idx == 0
            else transforms.Compose(
                [RotateLabel(angle[i], axis[i]) for i in range(nb_rotation)]
            )
            for idx, angle, axis in zip(indexes, angle, axis)
        ]
        return image_transformations, label_transformations

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
        labels = self.labels[idx, :]
        if isinstance(img_paths[0], str):
            img_paths = [img_paths]
            labels = [labels]
        aug_image_transforms, label_transforms = self.data_augmente_transformations(
            len(img_paths)
        )
        images = [
            [
                transforms.ToTensor()(Image.open(img).convert("RGB").convert("1"))
                for img in img_path
            ]
            for img_path in img_paths
        ]

        if self.image_transform is not None:
            images = [self.image_transform(torch.cat(image)) for image in images]

        images = [
            aug_image_transform(image) if aug_image_transform is not None else image
            for aug_image_transform, image in zip(aug_image_transforms, images)
        ]
        labels = [
            label_transform(label) if label_transform is not None else label
            for label_transform, label in zip(label_transforms, labels)
        ]
        if len(images) == 1:
            return images[0], labels[0]
        return torch.stack(images), torch.stack(labels)
