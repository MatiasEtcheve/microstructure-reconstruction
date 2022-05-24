from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF


class RotateLabel(object):
    """Preprocessing function in `torchvision.transforms` to rotate the descriptors.

    The descriptors must be a vector. Only the orientation descriptors are going to be rotated.
    The orientation descriptors are the 6 off diagonal coefficients of the orientation matrix.

    The descriptors must be as follows::
        descriptors[self.index:self.index + 6] = orientation_descriptors.mean()
        descriptors[self.index + 6:self.index + 12] = orientation_descriptors.std()
    """

    def __init__(self, angle: int, axis: Union[int, str], index: Optional[int] = 0):
        """Constructor.

        Args:
            angle (int): angle to rotate the label.
            axis (Union[int, str]): axis of the rotation.
            index (Optional[int], optional): first index of the orientation descriptors. Defaults to 0.

        Raises:
            ValueError: if axis is not in [0, 1, 2] or in ["x", "y", "z"]
        """
        if axis not in [0, 1, 2, "x", "y", "z"]:
            raise ValueError('Axis must be in [0, 1, 2] or in ["x", "y", "z"]')
        self.rad_angle = angle * np.pi / 180
        self.angle = angle
        if isinstance(axis, str):
            axis = ["x", "y", "z"].index(axis)
        self.axis = axis
        self.index = index

    @property
    def Rx(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Rotation matrix on the x axis
        """
        return torch.FloatTensor(
            [
                [1, 0, 0],
                [0, torch.cos(self.rad_angle), -torch.sin(self.rad_angle)],
                [0, torch.sin(self.rad_angle), torch.cos(self.rad_angle)],
            ]
        )

    @property
    def Ry(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Rotation matrix on the y axis
        """
        return torch.FloatTensor(
            [
                [torch.cos(self.rad_angle), 0, torch.sin(self.rad_angle)],
                [0, 1, 0],
                [-torch.sin(self.rad_angle), 0, torch.cos(self.rad_angle)],
            ]
        )

    @property
    def Rz(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Rotation matrix on the z axis
        """
        return torch.FloatTensor(
            [
                [torch.cos(self.rad_angle), -torch.sin(self.rad_angle), 0],
                [torch.sin(self.rad_angle), torch.cos(self.rad_angle), 0],
                [0, 0, 1],
            ]
        )

    def rotate_orientation_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Rotate a tensor on the `self.axis` axis with a `self.angle` angle.

        Args:
            vector (torch.Tensor): vector to rotate

        Returns:
            torch.Tensor: rotated vector
        """
        rotation_matrix = [self.Rx, self.Ry, self.Rz][self.axis]
        vector = torch.matmul(rotation_matrix, vector)
        vector[self.axis] *= -(
            torch.div(torch.abs(self.angle), 90, rounding_mode="trunc") % 2 * 2 - 1
        )
        return vector

    def __call__(
        self,
        vector,
    ) -> torch.Tensor:
        """Rotate a vector of descriptors

        Only the orientation descriptors are going to be rotated.
        The orientation descriptors are the 6 off diagonal coefficients of the orientation matrix.

        The descriptors must be as follows::
            descriptors[self.index:self.index + 6] = orientation_descriptors.mean()
            descriptors[self.index + 6:self.index + 12] = orientation_descriptors.std()

        Args:
            vector (torch.Tensor): vector of descriptors to rotate.

        Returns:
            torch.Tensor: vector of rotated descriptors.
        """
        vector[self.index : self.index + 3] = torch.abs(
            self.rotate_orientation_vector(vector[self.index : self.index + 3])
        )
        vector[self.index + 3 : self.index + 6] = self.rotate_orientation_vector(
            vector[self.index + 3 : self.index + 6]
        )
        vector[self.index + 6 : self.index + 9] = torch.abs(
            self.rotate_orientation_vector(vector[self.index + 6 : self.index + 9])
        )
        vector[self.index + 9 : self.index + 12] = torch.abs(
            self.rotate_orientation_vector(vector[self.index + 9 : self.index + 12])
        )
        return vector


class RotateImgOn(object):
    """Base class to rotate the sliced images."""

    def __init__(self, angle: Optional[int] = 90, nb_input_photos_per_plane: int = 1):
        """Constructor.

        Args:
            angle (Optional[int], optional): angle of the rotation. Defaults to 90.
            nb_input_photos_per_plane (Optional[int], optional): Number of sliced images per plane. Defaults to 1.

        Raises:
            ValueError: Angle must be in [-90, 90].
        """
        if not angle in [-90, 90]:
            raise ValueError("Angle must be in [-90, 90].")
        self.angle = angle
        self.nb_input_photos_per_plane = nb_input_photos_per_plane

    def split_images_along_axis(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the input images into x, y, and z images.

        Args:
            images (torch.Tensor): images to split, along the first dimension.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: tuple of x, y and z images.
        """
        images_x = images[: self.nb_input_photos_per_plane]
        images_y = images[
            self.nb_input_photos_per_plane : 2 * self.nb_input_photos_per_plane
        ]
        images_z = images[2 * self.nb_input_photos_per_plane :]
        return images_x, images_y, images_z


class RotateImgOnX(RotateImgOn):
    """Rotates the x images"""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images_x, images_y, images_z = self.split_images_along_axis(images)
        images_x = torch.cat(
            [TF.rotate(image_x.unsqueeze(0), -int(self.angle)) for image_x in images_x]
        )
        if self.angle == 90:
            images_y = torch.flip(images_y, [0])
            images_z = torch.cat(
                [TF.vflip(image_z.unsqueeze(0)) for image_z in images_z]
            )

        else:
            images_z = torch.flip(images_z, [0])
            images_y = torch.cat(
                [TF.vflip(image_y.unsqueeze(0)) for image_y in images_y]
            )
        return torch.cat([images_x, images_z, images_y])


class RotateImgOnY(RotateImgOn):
    """Rotates the y images"""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images_x, images_y, images_z = self.split_images_along_axis(images)
        images_y = torch.cat(
            [TF.rotate(image_y.unsqueeze(0), int(self.angle)) for image_y in images_y]
        )
        if self.angle == -90:
            images_x = torch.flip(images_x, [0])
            images_z = torch.cat(
                [
                    TF.rotate(image_z.unsqueeze(0), int(self.angle))
                    for image_z in images_z
                ]
            )

        else:
            images_z = torch.cat(
                [
                    TF.rotate(TF.hflip(image_z.unsqueeze(0)), int(self.angle))
                    for image_z in torch.flip(images_z, [0])
                ]
            )
            images_x = torch.cat(
                [
                    TF.rotate(image_x.unsqueeze(0), int(self.angle))
                    for image_x in images_x
                ]
            )
        return torch.cat([images_z, images_y, images_x])


class RotateImgOnZ(RotateImgOn):
    """Rotates the z images"""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        images_x, images_y, images_z = self.split_images_along_axis(images)
        images_z = torch.cat(
            [TF.rotate(image_z.unsqueeze(0), -int(self.angle)) for image_z in images_z]
        )
        if self.angle == 90:
            images_x = torch.flip(images_x, [0])
            images_y = torch.cat(
                [TF.hflip(image_y.unsqueeze(0)) for image_y in images_y]
            )

        else:
            images_y = torch.flip(images_y, [0])
            images_x = torch.cat(
                [TF.vflip(image_x.unsqueeze(0)) for image_x in images_x]
            )

        return torch.cat([images_y, images_x, images_z])
