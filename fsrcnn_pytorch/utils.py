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
import numpy as np
import torch


def convert_rgb_to_y(img, dim_order="hwc"):
    r"""Convert RGB image to Y-axis format image

    Args:
        img (PIL.Image.open): RGB image.
        dim_order (str): Image format.

    Returns:
        Y-axis image.

    """
    if dim_order == "hwc":
        return 16. + (
                64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[
            ..., 2]) / 256.
    else:
        return 16. + (
                64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order="hwc"):
    r"""Convert HWC image to Y-axis format image

    Args:
        img (PIL.Image.open): RGB image.
        dim_order (str): Image format.

    Returns:
        Y-axis image.

    """
    if dim_order == "hwc":
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[
            ..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 *
                     img[..., 2]) / 256.
        cr = 128. + (
                112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[
            ..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[
            2]) / 256.
        cr = 128. + (
                112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def convert_ycbcr_to_rgb(img, dim_order="hwc"):
    r"""Convert Y-axis image to HWC format image

    Args:
        img (PIL.Image.open): RGB image.
        dim_order (str): Image format.

    Returns:
        H*W*C image.

    """
    if dim_order == "hwc":
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[
            ..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[
            ..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[
            ..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[
            2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])


def preprocess(img, device):
    r"""Preprocess the input RGB image.

    Args:
        img (PIL.Image.open): RGB image.
        device (torch.device): Run on CPU or GPU.

    Returns:
        Image format and y-axis image after preprocessing.

    """
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr
