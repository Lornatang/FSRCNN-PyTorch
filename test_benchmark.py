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
import math
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from fsrcnn_pytorch import FSRCNN
from fsrcnn_pytorch import ValDataset

parser = argparse.ArgumentParser(description="Fast Super Resolution CNN.")
parser.add_argument("--data-file", type=str,
                    default="./data/test/Set5_X4.h5",
                    help="Path to data datasets. "
                         "(default:`./data/test/Set5_X4.h5`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--scale-factor", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--outf", default="./results",
                    help="folder to output images. (default:`./results`).")
parser.add_argument("--weights", type=str, default="weights/fsrcnn_X4.pth",
                    required=True, help="Generator model name.  "
                                        "(default:`weights/fsrcnn_X4.pth`)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, "
          "so you should probably run with --cuda")

dataset = ValDataset(args.eval_file)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False, pin_memory=True,
                                         num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

model = FSRCNN(scale_factor=args.scale_factor).to(device)
model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.MSELoss().to(device)

best_psnr = 0.

# Test
avg_psnr = 0
with torch.no_grad():
    for data in dataloader:
        inputs = data[0].to(device)
        target = data[1].to(device)

        output = model(inputs).clamp(0.0, 1.0)
        loss = criterion(output, target)
        psnr = 10 * math.log10(1 / loss.item())
        avg_psnr += psnr
    print(f"Average PSNR: {avg_psnr / len(dataloader):.2f} dB.")
