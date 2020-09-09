# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
import glob

import h5py
import numpy as np
from PIL import Image

from fsrcnn_pytorch import convert_rgb_to_y


def train(dataroot, data_file, scale_factor, augment=False):
    r"""Create training data file.

    Args:
        dataroot (str): Source image directory.
        data_file (str): Where the dataset file is saved.
        scale_factor (int): Image magnification scale.
        augment (bool): Whether to perform data augment operation.

    Returns:
        training data file

    """
    h5_file = h5py.File(data_file, "w")

    lr_patches = []
    hr_patches = []

    if scale_factor == 2:
        patch_size = 10
    elif scale_factor == 3:
        patch_size = 7
    elif scale_factor == 4:
        patch_size = 6
    else:
        raise Exception("Scale Error", args.scale_factor)

    for filename in sorted(glob.glob(f"{dataroot}/*")):
        hr = Image.open(filename).convert("RGB")
        hr_images = []

        if augment:
            # s: Image scaling ratio
            # r: Image random rotation
            for s in [1.0, 0.9, 0.8, 0.7, 0.6]:
                for r in [0, 90, 180, 270]:
                    new_width = hr.width * s
                    new_height = hr.height * s
                    new_image = hr.resize((int(new_width), int(new_height)),
                                          resample=Image.BICUBIC)
                    new_image = new_image.rotate(r, expand=True)
                    hr_images.append(new_image)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            hr_width = (hr.width // scale_factor) * scale_factor
            hr_height = (hr.height // scale_factor) * scale_factor
            hr = hr.resize((int(hr_width), int(hr_height)),
                           resample=Image.BICUBIC)

            lr_width = hr.width // scale_factor
            lr_height = hr_height // scale_factor
            lr = hr.resize((int(lr_width), int(lr_height)),
                           resample=Image.BICUBIC)

            hr = np.array(hr).astype(np.float32)
            hr = convert_rgb_to_y(hr)

            lr = np.array(lr).astype(np.float32)
            lr = convert_rgb_to_y(lr)

            for i in range(0, lr.shape[0] - patch_size + 1, scale_factor):
                for j in range(0, lr.shape[1] - patch_size + 1, scale_factor):
                    lr_patches.append(lr[i:i + patch_size, j:j + patch_size])
                    hr_patches.append(hr[
                                      i * scale_factor:i * scale_factor + patch_size * scale_factor,
                                      j * scale_factor:j * scale_factor + patch_size * scale_factor])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)

    h5_file.create_dataset("lr", data=lr_patches)
    h5_file.create_dataset("hr", data=hr_patches)

    h5_file.close()


def eval(dataroot, data_file, scale_factor):
    r"""Create Eval/Test data file.

    Args:
        dataroot (str): Source image directory.
        data_file (str): Where the dataset file is saved.
        scale_factor (int): Image magnification scale.

    Returns:
        Valid data file.

    """
    h5_file = h5py.File(data_file, "w")

    lr_group = h5_file.create_group("lr")
    hr_group = h5_file.create_group("hr")

    for i, image_path in enumerate(sorted(glob.glob(f"{dataroot}/*"))):
        hr = Image.open(image_path).convert("RGB")

        hr_width = (hr.width // scale_factor) * scale_factor
        hr_height = (hr.height // scale_factor) * scale_factor
        hr = hr.resize((int(hr_width), int(hr_height)), resample=Image.BICUBIC)

        lr_width = (hr.width // scale_factor) * scale_factor
        lr_height = (hr.height // scale_factor) * scale_factor

        lr = hr.resize((int(lr_width), int(lr_height)), resample=Image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        hr = convert_rgb_to_y(hr)

        lr = np.array(lr).astype(np.float32)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if not args.eval:
        train(args.dataroot, args.data_file, args.scale_factor, args.augment)
    else:
        eval(args.dataroot, args.data_file, args.scale_factor)
