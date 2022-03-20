from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF


class RotateLabel(object):
    def __init__(self, angle, axis):
        self.rad_angle = angle * np.pi / 180
        self.angle = angle
        if isinstance(axis, str):
            axis = ["x", "y", "z"].index(axis)
        self.axis = axis

    @property
    def Rx(self):
        return torch.FloatTensor(
            [
                [1, 0, 0],
                [0, torch.cos(self.rad_angle), -torch.sin(self.rad_angle)],
                [0, torch.sin(self.rad_angle), torch.cos(self.rad_angle)],
            ]
        )

    @property
    def Ry(self):
        return torch.FloatTensor(
            [
                [torch.cos(self.rad_angle), 0, torch.sin(self.rad_angle)],
                [0, 1, 0],
                [-torch.sin(self.rad_angle), 0, torch.cos(self.rad_angle)],
            ]
        )

    @property
    def Rz(self):
        return torch.FloatTensor(
            [
                [torch.cos(self.rad_angle), -torch.sin(self.rad_angle), 0],
                [torch.sin(self.rad_angle), torch.cos(self.rad_angle), 0],
                [0, 0, 1],
            ]
        )

    def rotate_orientation_vector(self, vector):
        rotation_matrix = [self.Rx, self.Ry, self.Rz][self.axis]
        vector = torch.matmul(rotation_matrix, vector)
        vector[self.axis] *= -(
            torch.div(torch.abs(self.angle), 90, rounding_mode="trunc") % 2 * 2 - 1
        )
        return vector

    def __call__(self, vector):
        vector[0:3] = torch.abs(self.rotate_orientation_vector(vector[0:3]))
        vector[3:6] = self.rotate_orientation_vector(vector[3:6])
        vector[6:9] = torch.abs(self.rotate_orientation_vector(vector[6:9]))
        vector[9:12] = torch.abs(self.rotate_orientation_vector(vector[9:12]))
        return vector


class RotateImgOn(object):
    def __init__(self, angle=90, nb_input_photos_per_plane=1):
        assert angle in [-90, 90]
        self.angle = angle
        self.nb_input_photos_per_plane = nb_input_photos_per_plane

    def split_images_along_axis(self, images):
        images_x = images[: self.nb_input_photos_per_plane]
        images_y = images[
            self.nb_input_photos_per_plane : 2 * self.nb_input_photos_per_plane
        ]
        images_z = images[2 * self.nb_input_photos_per_plane :]
        return images_x, images_y, images_z


class RotateImgOnX(RotateImgOn):
    def __call__(self, images):
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
    def __call__(self, images):
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
    def __call__(self, images):
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
