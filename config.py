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
"""Realize the parameter configuration function of dataset, model, training and verification code."""
import os

import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from model import FSRCNN

# ==============================================================================
# General configuration
# ==============================================================================
torch.manual_seed(0)
upscale_factor = 2
device = torch.device("cuda:0")
cudnn.benchmark = True
mode = "train"
exp_name = "exp000"

# ==============================================================================
# Training configuration
# ==============================================================================
if mode == "train":
    # Dataset.
    train_dir = f"data/T91_General100/X{upscale_factor}/train"
    valid_dir = f"data/T91_General100/X{upscale_factor}/valid"
    batch_size = 16

    # Model.
    model = FSRCNN(upscale_factor).to(device)

    # Resuming training.
    start_epoch = 0
    resume = False
    resume_weight = ""

    # Total number of epochs.
    epochs = 307

    # Loss function.
    criterion = nn.MSELoss().to(device)

    # Optimizer.
    optimizer = optim.SGD([{"params": model.feature_extraction.parameters()},
                           {"params": model.shrink.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.expand.parameters()},
                           {"params": model.deconv.parameters(), "lr": 0.0001}], 0.001)

    # Training log.
    writer = SummaryWriter(os.path.join("samples", "logs", exp_name))

    # Additional variables.
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)

# ==============================================================================
# Verify configuration
# ==============================================================================
if mode == "valid":
    # Test data address.
    lr_dir = f"data/Set5/LRbicx2"
    sr_dir = f"results/test/{exp_name}"
    hr_dir = f"data/Set5/GTmod12"

    # Load sr model.
    model = FSRCNN(upscale_factor).to(device)
    model.load_state_dict(torch.load(f"results/{exp_name}/best.pth", map_location=device))

    # Additional variables.
    exp_dir = os.path.join("results", "test", exp_name)