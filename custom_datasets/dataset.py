from typing import Callable, List, Union

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
    | 600N | 0.871309          | ... | 0.031973       | 0.891398      | 0.983910             | "REV600/x-y(N).png" |
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
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        width: int = 64,
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
        self.width = width
        self.noise = noise
        df = dataframe_reformat.convert_into_single_entry_df(df, col_name="photos")
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)
        self.images = np.stack(df.pop("photos"))
        self.labels = torch.Tensor(df.to_numpy())
        self.image_transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        """Method for dataset[idx]

        Args:
            idx (int): index of the dataset to fetch

        Returns:
            tuple: first item is the feature (image) of size `(W, H)`, the second item is the target (fabric descriptors)
        """
        img_paths = self.images[idx]
        labels = self.labels[idx, :]

        if len(img_paths.shape) == 0:
            img_paths = np.expand_dims(img_paths, -1)
            labels = torch.unsqueeze(labels, -1)

        images = [
            transforms.Resize((self.width, self.width))(
                transforms.ToTensor()(Image.open(path).convert("RGB").convert("1"))
            )
            for path in img_paths
        ]
        if self.image_transform is not None:
            images = [self.image_transform(image) for image in images]
        return torch.cat(images, axis=0), labels


class NChannelPhotosDataset(SinglePhotoDataset):
    """Dataset whose inputs are images of size `(C, W, H)`. On each channel, there is a white and black image corresponding to a sliced image.

    Attributes:
        nb_image_per_axis (int): number of sliced images per plane in the input. For instance,
            if we chose `nb_image_per_axis=2`, the input will have a size `(nb_image_per_axis*3, W, H)`:
                * the first 2 images are sliced images from x plane
                * the 2 next images are sliced images from y plane
                * the 2 final images are sliced images from z plane
    """

    def __init__(
        self,
        df: pd.DataFrame,
        nb_image_per_axis: int = 1,
        transform: transforms.Compose = transforms.Compose([]),
        width: int = 64,
        order: str = "xyz",
        noise: int = 0,
        proba_rotating: float = 0,
        proba_axis: dict = {0: 1 / 3, 1: 1 / 3, 2: 1 / 3},
        proba_angle: dict = {90: 0.5, -90: 0.5},
        mode: str = "replace",
        nb_input_photos_per_plane=None,
    ):
        """Constructor

        Args:
            df (pd.DataFrame): original dataframe to be used
            nb_image_per_axis (int): Number of sliced images to use. Defaults to 1.
            transform (transforms.Compose, optional): Basic transformation to do on images. Defaults to transforms.Compose([transforms.ToTensor()]).
            width (int): width of the photos. Defaults to 64.
            order (str): order of the stacked channels. Defaults to "xyz".
            noise (int): noise added to the lables. Defaults to 0.
            proba_rotating (float): probability of rotating a rev, in data augmentation. Defaults to 0.
            proba_axis (dict): dictionnary whose keys are {0, 1, 2} and the values are probabilites of a rotating around 0, 1 or 2nd axis. Defaults to {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}.
            proba_angle (dict): dictionnary whose keys are {-90, 90} and the values are probabilites of a rotating around of -90 or 90°. Defaults to {90: 0.5, -90: 0.5}.
            mode (str): Whether to do the data augmentation on the fly (`replace`) or appending in the dataset (`append`). Defaults to replace.
        """

        self.width = width
        self.nb_image_per_axis = (
            nb_image_per_axis
            if nb_input_photos_per_plane is None
            else nb_input_photos_per_plane
        )
        self.noise = noise
        self.proba_rotating = proba_rotating
        self.proba_axis = proba_axis
        self.proba_angle = proba_angle
        self.mode = mode
        df = dataframe_reformat.convert_into_n_entry_df(
            df,
            col_name="photos",
            nb_image_per_axis=self.nb_image_per_axis,
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

    def fetch_img_aug(self, angle: int, axis: int) -> Callable:
        """Fetch the data augmentation rotation function.

        Args:
            angle (int): angle of the rotation
            axis (int): axis of the rotation

        Raises:
            NotImplemented: if the axis is not in `[0, 1, 2]`

        Returns:
            Callable: function/class which can be called to augment the inputs
        """
        if axis == 0:
            return RotateImgOnX(angle, self.nb_image_per_axis)
        if axis == 1:
            return RotateImgOnY(angle, self.nb_image_per_axis)
        if axis == 2:
            return RotateImgOnZ(angle, self.nb_image_per_axis)
        raise NotImplementedError("")

    def data_augmente_transformations(
        self,
        length: int,
    ) -> List[Callable]:
        """Creates `length` image and label transformations.

        Args:
            length (int): size of the transformations.

        Returns:
            List[Callable]: list of callable of length `length`
        """
        nb_rotation = 3
        indexes = np.random.binomial(1, self.proba_rotating, size=length)
        axis = np.random.choice(
            list(self.proba_axis.keys()),
            size=(length, nb_rotation),
            p=list(self.proba_axis.values()),
        )
        angle = torch.Tensor(
            np.random.choice(
                list(self.proba_angle.keys()),
                size=(length, nb_rotation),
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
        """Computes the length of the dataset

        Returns:
            int: length of the dataset
        """
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
                transforms.Resize((self.width, self.width))(
                    transforms.ToTensor()(Image.open(img).convert("RGB").convert("1"))
                )
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


class NWidthStackedPhotosDataset(SinglePhotoDataset):
    """Dataset whose inputs are images of size `(C, W, H)`. On each channel, there is a white and black image corresponding to a sliced image.

    Attributes:
        nb_image_per_axis (int): number of sliced images per plane in the input. For instance,
            if we chose `nb_image_per_axis=2`, the input will have a size `(nb_image_per_axis*3, W, H)`:
                * the first 2 images are sliced images from x plane
                * the 2 next images are sliced images from y plane
                * the 2 final images are sliced images from z plane
    """

    def __init__(
        self,
        df: pd.DataFrame,
        nb_image_per_axis: int = 1,
        transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),
        width: int = 64,
        order: str = "xyz",
        noise: int = 0,
    ):
        """Constructor

        Args:
            df (pd.DataFrame): original dataframe to be used
            nb_image_per_axis (int): Number of sliced images to use. Defaults to 1.
            transform (transforms.Compose, optional): Basic transformation to do on images. Defaults to transforms.Compose([transforms.ToTensor()]).
            width (int): width of the photos. Defaults to 64.
            order (str): order of the stacked channels. Defaults to "xyz".
            noise (int): noise added to the lables. Defaults to 0.
            proba_rotating (float): probability of rotating a rev, in data augmentation. Defaults to 0.
            proba_axis (dict): dictionnary whose keys are {0, 1, 2} and the values are probabilites of a rotating around 0, 1 or 2nd axis. Defaults to {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}.
            proba_angle (dict): dictionnary whose keys are {-90, 90} and the values are probabilites of a rotating around of -90 or 90°. Defaults to {90: 0.5, -90: 0.5}.
            mode (str): Whether to do the data augmentation on the fly (`replace`) or appending in the dataset (`append`). Defaults to replace.
        """
        self.width = width
        self.nb_image_per_axis = nb_image_per_axis
        self.noise = noise
        df = dataframe_reformat.convert_into_n_width_df(
            df,
            col_name="photos",
            nb_image_per_axis=self.nb_image_per_axis,
            order=order,
        )
        df.reset_index(drop=True, inplace=True)
        df = df.drop(columns=["id"], inplace=False)
        self.images = np.stack(df.pop("photos"))
        self.labels = torch.Tensor(df.to_numpy())
        self.image_transform = transform

    def __len__(self):
        """Computes the length of the dataset

        Returns:
            int: length of the dataset
        """
        return len(self.labels)

    def _load_images(self, axis: int, img_paths: List[List[str]]) -> List[torch.Tensor]:
        """Load images along axis

        Args:
            axis (int): One of `0`, `1` and `2`.
            img_paths (List[List[List[str]]]): List whose elements are images of a rev.
                This element is a list of length 3 composed a of a list of path to images. We have:
                    * img_paths[0] = images of the 1st REV
                        ** img_paths[0][0] = image paths of x-axis of the 1st REV
                        ** img_paths[0][1] = image paths of y-axis of the 1st REV
                        ** img_paths[0][2] = image paths of z-axis of the 1st REV
                    * img_paths[1] = images of the 2nd REV
                    * ...

        Returns:
            List[torch.Tensor]: List whose elements are tensors of a rev.
                This is element is a tensor of shape `(3, 64, n*64)`, because we concatenate the `n` images of the same axis in the width.
        """
        images = [
            [
                transforms.Resize((self.width, self.width))(
                    transforms.ToTensor()(Image.open(img).convert("RGB").convert("1"))
                )
                for img in img_path[axis]
            ]
            for img_path in img_paths
        ]

        if self.image_transform is not None:
            images = [[self.image_transform(img) for img in image] for image in images]
            images = [torch.cat(x_image, axis=2) for x_image in images]
        return images

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
        if len(img_paths.shape) == 2:
            img_paths = np.expand_dims(img_paths, 0)
            labels = torch.unsqueeze(labels, 0)

        x_images = self._load_images(0, img_paths)
        y_images = self._load_images(1, img_paths)
        z_images = self._load_images(2, img_paths)

        images = [
            torch.cat([x_image, y_image, z_image], axis=0)
            for x_image, y_image, z_image in zip(x_images, y_images, z_images)
        ]
        if len(images) == 1:
            return images[0], labels[0]
        return torch.stack(images), labels


class NWidthConcatPhotosDataset(NWidthStackedPhotosDataset):
    """Dataset whose inputs are images of size `(C, W, H)`. On each channel, there is a white and black image corresponding to a sliced image.

    Attributes:
        nb_image_per_axis (int): number of sliced images per plane in the input. For instance,
            if we chose `nb_image_per_axis=2`, the input will have a size `(nb_image_per_axis*3, W, H)`:
                * the first 2 images are sliced images from x plane
                * the 2 next images are sliced images from y plane
                * the 2 final images are sliced images from z plane
    """

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
        if len(img_paths.shape) == 2:
            img_paths = np.expand_dims(img_paths, 0)
            labels = torch.unsqueeze(labels, 0)

        x_images = self._load_images(0, img_paths)
        y_images = self._load_images(1, img_paths)
        z_images = self._load_images(2, img_paths)

        images = [
            torch.cat([x_image, y_image, z_image], axis=1)
            for x_image, y_image, z_image in zip(x_images, y_images, z_images)
        ]
        if len(images) == 1:
            return images[0], labels[0]
        return torch.stack(images), labels
