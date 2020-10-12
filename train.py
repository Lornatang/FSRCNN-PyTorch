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
"""It is mainly used to train models."""
import argparse
import csv
import os
import random

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda import amp
from tqdm import tqdm

from fsrcnn_pytorch import DatasetFromFolder
from fsrcnn_pytorch import FSRCNN
from fsrcnn_pytorch import init_torch_seeds
from fsrcnn_pytorch import load_checkpoint
from fsrcnn_pytorch import select_device

parser = argparse.ArgumentParser(description="Accelerating the Super-Resolution Convolutional Neural Network.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("--iters", default=1e8, type=int, metavar="N",
                    help="Number of total epochs to run. "
                         "According to the 1e8 iters in the original paper."
                         "(default:1e8)")
parser.add_argument("-b", "--batch-size", default=16, type=int,
                    metavar="N",
                    help="mini-batch size (default: 16), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate. (default:0.001)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--resume", action="store_true",
                    help="Path to latest checkpoint for FSRCNN.")
parser.add_argument("--manualSeed", type=int, default=10000,
                    help="Seed for initializing training. (default:10000)")
parser.add_argument("--device", default="",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``).")

args = parser.parse_args()
print(args)

try:
    os.makedirs("weights")
except OSError:
    pass

# Set random initialization seed, easy to reproduce.
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
init_torch_seeds(args.manualSeed)

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=args.batch_size)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                            target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

# Construct FSRCNN model.
model = FSRCNN(upscale_factor=args.upscale_factor).to(device)

# We use Adam instead of SGD like in the paper, because it's faster
optimizer = optim.Adam([
    {"params": model.features.parameters()},
    {"params": model.shrink.parameters()},
    {"params": model.map.parameters()},
    {"params": model.expand.parameters()},
    {"params": model.deconv.parameters(), "lr": args.lr * 0.1}
], lr=args.lr)

# Loading PSNR pre training model
if args.resume:
    args.start_epoch = load_checkpoint(model, optimizer, f"./weights/FSRCNN_{args.upscale_factor}x_checkpoint.pth")

# Define loss
criterion = nn.MSELoss().to(device)

# From the total number of iterations, how many training datasets are needed
epochs = int(args.iters // len(dataloader))
print("[*] Start training model based on MSE loss.")
print(f"[*] Generator pre-training for {epochs} epochs.")

# Writer train PSNR model log.
if args.start_epoch == 0:
    with open(f"FSRCNN_{args.upscale_factor}x_Loss.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "MSE Loss"])

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()
# Start write training log
writer = SummaryWriter("logs")
print("Run `tensorboard --logdir=./logs` view training log.")

for epoch in range(args.start_epoch, epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    avg_loss = 0.
    for iteration, (input, target) in progress_bar:
        optimizer.zero_grad()

        lr, hr = input.to(device), target.to(device)

        # Runs the forward pass with autocasting.
        with amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)

        # Scales loss.  Calls backward() on scaled loss to
        # create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same dtype autocast chose
        # for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of
        # the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs,
        # optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        avg_loss += loss.item()

        progress_bar.set_description(f"[{epoch + 1}/{epochs}][{iteration + 1}/{len(dataloader)}] "
                                     f"MSE: {loss.item():.4f}")

    # The model is saved every 1 epoch.
    torch.save({"epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "state_dict": model.state_dict()
                }, f"./weights/FSRCNN_{args.upscale_factor}x_checkpoint.pth")

    # Writer training log
    with open(f"FSRCNN_{args.upscale_factor}x_Loss.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, avg_loss / len(dataloader)])

torch.save(model.state_dict(), f"./weights/FSRCNN_{args.upscale_factor}x.pth")
print(f"[*] Training FSRCNN model done! Saving model weight to `./weights/FSRCNN_{args.upscale_factor}x.pth`.")
