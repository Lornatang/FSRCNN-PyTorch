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

import numpy as np
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

from fsrcnn_pytorch import FSRCNN
from fsrcnn_pytorch import select_device

parser = argparse.ArgumentParser(description="Accelerating the Super-Resolution Convolutional Neural Network.")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="Test low resolution image name. "
                         "(default:`./assets/baby.png`)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--weights", type=str, default="./weights/FSRCNN_4x.pth",
                    help="Generator model name. (default:`./weights/FSRCNN_4x.pth`)")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")

args = parser.parse_args()
print(args)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=1)

# Construct SRCNN model.
model = FSRCNN(upscale_factor=args.upscale_factor).to(device)

# Load state dicts
model.load_state_dict(torch.load(args.weights, map_location=device))

# Open image
image = Image.open(args.file).convert("YCbCr")
y, cb, cr = image.split()

preprocess = transforms.ToTensor()
lr = preprocess(y).view(1, -1, y.size[1], y.size[0])
lr = lr.to(device)
with torch.no_grad():
    sr = model(lr)
sr = sr.cpu()
out_image_y = sr[0].detach().numpy()
out_image_y *= 255.0
out_image_y = out_image_y.clip(0, 255)
out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

out_img_cb = cb.resize(out_image_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_image_y.size, Image.BICUBIC)
out_img = Image.merge("YCbCr", [out_image_y, out_img_cb, out_img_cr]).convert("RGB")
# before converting the result in RGB
out_img.save(f"fsrcnn_{args.upscale_factor}x.png")
