#!/bin/bash

echo "Start downloading pre training model..."
wget https://github.com/Lornatang/FSRCNN-PyTorch/releases/download/1.2/fsrcnn_2x.pth
wget https://github.com/Lornatang/FSRCNN-PyTorch/releases/download/1.2/fsrcnn_3x.pth
wget https://github.com/Lornatang/FSRCNN-PyTorch/releases/download/1.2/fsrcnn_4x.pth
echo "All pre training models have been downloaded!"
