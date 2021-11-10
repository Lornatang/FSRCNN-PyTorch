# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Realize the function of dataset preparation."""
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

import imgproc

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot         (str): Training data set address.
        image_size       (int): High resolution image size.
        upscale_factor   (int): Magnification.
        mode             (str): Data set loading method, the training data set is for data enhancement,
                                and the verification data set is not for data enhancement.
    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        # Get the index of all images in the high-resolution folder and low-resolution folder under the data set address.
        # Note: The high and low resolution file index should be corresponding.
        lr_image_size = (image_size // upscale_factor, image_size // upscale_factor)
        hr_image_size = (image_size, image_size)
        self.filenames = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]
        # Low-resolution images and high-resolution images have different processing methods.
        if mode == "train":
            self.hr_transforms = transforms.RandomResizedCrop(hr_image_size)
        else:
            self.hr_transforms = transforms.CenterCrop(hr_image_size)
        self.lr_transforms = transforms.Resize(lr_image_size, interpolation=IMode.BICUBIC)

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        hr_image_data = Image.open(self.filenames[batch_index])

        # Transform image
        hr_image_data = self.hr_transforms(hr_image_data)
        lr_image_data = self.lr_transforms(hr_image_data)

        # RGB convert YCbCr
        lr_ycbcr_image_data = lr_image_data.convert("YCbCr")
        hr_ycbcr_image_data = hr_image_data.convert("YCbCr")

        # Only extract the image data of the Y channel
        lr_y_image_data = lr_ycbcr_image_data.split()[0]
        hr_y_image_data = hr_ycbcr_image_data.split()[0]

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor_data = imgproc.image2tensor(lr_y_image_data, range_norm=False, half=False)
        hr_tensor_data = imgproc.image2tensor(hr_y_image_data, range_norm=False, half=False)

        return lr_tensor_data, hr_tensor_data

    def __len__(self) -> int:
        return len(self.filenames)
