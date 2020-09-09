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
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data.dataloader import DataLoader

from fsrcnn_pytorch import FSRCNN
from fsrcnn_pytorch import TrainDataset
from fsrcnn_pytorch import ValDataset

parser = argparse.ArgumentParser(description="Fast Super Resolution CNN.")
parser.add_argument("--train-file", type=str,
                    default="./data/train/BSDS300_X4.h5",
                    help="Path to train datasets. "
                         "(default:`./data/train/BSDS300_X4.h5`)")
parser.add_argument("--eval-file", type=str,
                    default="./data/val/Set5_X4.h5",
                    help="Path to eval datasets. "
                         "(default:`./data/val/Set5_X4.h5`)")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                    help="Number of data loading workers. (default:0)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="Number of total epochs to run. (default:200)")
parser.add_argument("-b", "--batch-size", default=64, type=int,
                    metavar="N",
                    help="mini-batch size (default: 64), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel.")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate. (default:0.001)")
parser.add_argument("--scale-factor", type=int, default=4,
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--outf", default="./outputs",
                    help="folder to output images. (default:`./outputs`).")
parser.add_argument("--weights", default="",
                    help="Path to weights (to continue training).")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs("weights")
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

train_dataset = TrainDataset(args.train_file)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True,
                                               num_workers=int(args.workers))
val_dataset = ValDataset(args.eval_file)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False, pin_memory=True,
                                             num_workers=int(args.workers))

device = torch.device("cuda:0" if args.cuda else "cpu")

model = FSRCNN(scale_factor=args.scale_factor).to(device)

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam([
    {"params": model.first_part.parameters()},
    {"params": model.mid_part.parameters()},
    {"params": model.last_part.parameters(), "lr": args.lr * 0.1}
], lr=args.lr)

best_psnr = 0.

# Creates a GradScaler once at the beginning of training.
scaler = amp.GradScaler()

for epoch in range(0, args.epochs):
    model.train()

    # Train
    epoch_loss = 0
    for data in train_dataloader:
        optimizer.zero_grad()

        inputs = data[0].to(device)
        target = data[1].to(device)

        # Runs the forward pass with autocasting.
        with amp.autocast():
            output = model(inputs)
            loss = criterion(output, target)

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

        epoch_loss += loss.item()

    print(f"Epoch {epoch}. "
          f"Training loss: {epoch_loss / len(train_dataloader):.6f}")

    # Test
    avg_psnr = 0
    with torch.no_grad():
        for data in val_dataloader:
            inputs = data[0].to(device)
            target = data[1].to(device)

            output = model(inputs).clamp(0.0, 1.0)
            loss = criterion(output, target)
            psnr = 10 * math.log10(1 / loss.item())
            avg_psnr += psnr
        print(f"Average PSNR: {avg_psnr / len(val_dataloader):.2f} dB.")

    # Save model
    torch.save(model.state_dict(), f"weights/model_{epoch}.pth")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(),
                   f"weights/fsrcnn_X{args.scale_factor}.pth")
