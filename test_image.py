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

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from sewar.full_ref import msssim
from sewar.full_ref import sam
from sewar.full_ref import vifp

from fsrcnn_pytorch import FSRCNN
from fsrcnn_pytorch import cal_mse
from fsrcnn_pytorch import cal_niqe
from fsrcnn_pytorch import cal_psnr
from fsrcnn_pytorch import cal_rmse
from fsrcnn_pytorch import cal_ssim
from fsrcnn_pytorch import convert_ycbcr_to_rgb
from fsrcnn_pytorch import preprocess

parser = argparse.ArgumentParser(description="Fast Super Resolution CNN.")
parser.add_argument("--file", type=str, default="./assets/baby.png",
                    help="The file address where the image needs "
                         "to be processed. (default: `./assets/baby.png`).")
parser.add_argument("--weights", type=str, default="weights/fsrcnn_X4.pth",
                    required=True, help="Generator model name.  "
                                        "(default:`weights/fsrcnn_X4.pth`)")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Image scaling ratio. (default: `4`).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = FSRCNN(scale_factor=args.scale_factor).to(device)
model.load_state_dict(torch.load(args.weights))

model.eval()

image = Image.open(args.file).convert("RGB")

image_width = (image.width // args.scale) * args.scale
image_height = (image.height // args.scale) * args.scale

hr = image.resize((image_width, image_height),
                  resample=Image.BICUBIC)
lr = hr.resize((hr.width // args.scale, hr.height // args.scale),
               resample=Image.BICUBIC)
bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale),
                    resample=Image.BICUBIC)

lr, _ = preprocess(lr, device)
hr, _ = preprocess(hr, device)
_, ycbcr = preprocess(bicubic, device)

with torch.no_grad():
    preds = model(lr).clamp(0.0, 1.0)

preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose(
    [1, 2, 0])
output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(
    np.uint8)
output = Image.fromarray(output)
output.save("result.png")

# Evaluate performance
src_img = cv2.imread("result.png")
dst_img = cv2.imread(args.file)
dst_img = cv2.resize(dst_img, (src_img.shape[1], src_img.shape[0]),
                     interpolation=cv2.INTER_CUBIC)

mse_value = cal_mse(src_img, dst_img)
rmse_value = cal_rmse(src_img, dst_img)
psnr_value = cal_psnr(src_img, dst_img)
ssim_value = cal_ssim(src_img, dst_img)
ms_ssim_value = msssim(src_img, dst_img)
niqe_value = cal_niqe("result.png")
sam_value = sam(src_img, dst_img)
vif_value = vifp(src_img, dst_img)

print(f"MSE: {mse_value:.2f}\n"
      f"RMSE: {rmse_value:.2f}\n"
      f"PSNR: {psnr_value:.2f}\n"
      f"SSIM: {ssim_value:.4f}\n"
      f"MS-SSIM: {ms_ssim_value:.4f}\n"
      f"NIQE: {niqe_value:.2f}\n"
      f"SAM: {sam_value:.4f}\n"
      f"VIF: {vif_value:.4f}")
