# FSRCNN-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367).

### Table of contents
1. [About Accelerating the Super-Resolution Convolutional Neural Network](#about-accelerating-the-super-resolution-convolutional-neural-network)
2. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights)
    * [Download dataset](#download-dataset)
3. [Test](#test)
4. [Train](#train-eg-div2k)
    * [Example](#example-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Accelerating the Super-Resolution Convolutional Neural Network

If you're new to FSRCNN, here's an abstract straight from the paper:

As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (SRCNN) has demonstrated superior performance to the previous hand-crafted models either in speed and restoration quality. However, the high computational cost still hinders it from practical usage that demands real-time performance (24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure for faster and better SR. We re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution layer at the end of the network, then the mapping is learned directly from the original low-resolution image (without interpolation) to the high-resolution one. Second, we reformulate the mapping layer by shrinking the input feature dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers. The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance. A corresponding transfer strategy is also proposed for fast training and testing across different upscaling factors.

### Installation

#### Clone and install requirements

```bash
git clone https://github.com/Lornatang/FSRCNN-PyTorch.git
cd FSRCNN-PyTorch/
pip install -r requirements.txt
```

#### Download pretrained weights

```bash
cd weights/
bash download_weights.sh
```

#### Download dataset

```bash
cd data/
bash download_dataset.sh
```

### Test

Evaluate the overall performance of the network.
```bash
usage: test.py [-h] [--dataroot DATAROOT] [--weights WEIGHTS] [--cuda]
               [--scale-factor {2,3,4}] [--outf OUTF]
               [--manualSeed MANUALSEED]

Fast Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   The directory address where the image needs to be
                        processed. (default: `./data/Set5/val`).
  --weights WEIGHTS     Generator model name. (default:`weights/fsrcnn_X4.pth`)
  --cuda                Enables cuda
  --scale-factor {2,3,4}
                        Image scaling ratio. (default: `4`).
  --outf OUTF           folder to output images. (default:`./results`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)

# Example
python test.py --dataroot ./data/Set5/val --weights ./weights/srcnn_X4.pth --scale-factor 4 --cuda
```

Evaluate the benchmark of validation data set in the network
```bash
usage: test_benchmark.py [-h] [--data-file DATA_FILE] [-j N]
                         [--scale-factor SCALE_FACTOR] [--cuda] [--outf OUTF]
                         --weights WEIGHTS [--manualSeed MANUALSEED]

Fast Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --data-file DATA_FILE
                        Path to data datasets.
                        (default:`./data/test/Set5_X4.h5`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --scale-factor SCALE_FACTOR
                        Low to high resolution scaling factor. (default:4).
  --cuda                Enables cuda
  --outf OUTF           folder to output images. (default:`./results`).
  --weights WEIGHTS     Generator model name.
                        (default:`weights/fsrcnn_X4.pth`)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)
# Example
python test_benchmark.py --data-file ./data/test/Set5_X4.h5 --weights ./weights/fsrcnn_X4.pth --scale-factor 4 --cuda
```

Test single picture
```bash
usage: test_image.py [-h] [--file FILE] [--weights WEIGHTS] [--cuda]
                     [--scale-factor {2,3,4}] [--manualSeed MANUALSEED]

Fast Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           The file address where the image needs to be
                        processed. (default: `./assets/baby.png`).
  --weights WEIGHTS     Generator model name. (default:`weights/fsrcnn_X4.pth`)
  --cuda                Enables cuda
  --scale-factor {2,3,4}
                        Image scaling ratio. (default: `4`).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)

# Example
python test_image.py --file ./assets/baby.png --weights ./weights/fsrcnn_X4.pth --scale-factor 4 --cuda
```

### Train (e.g DIV2K)

```bash
usage: train.py [-h] [--train-file TRAIN_FILE] [--eval-file EVAL_FILE] [-j N]
                [--epochs N] [-b N] [--lr LR] [--scale-factor SCALE_FACTOR]
                [-p N] [--cuda] [--outf OUTF] [--weights WEIGHTS]
                [--manualSeed MANUALSEED]

PyTorch Super Resolution CNN.

optional arguments:
  -h, --help            show this help message and exit
  --train-file TRAIN_FILE
                        Path to train datasets.
                        (default:`./data/train/BSDS300_X4.h5`)
  --eval-file EVAL_FILE
                        Path to eval datasets.
                        (default:`./data/val/Set5_X4.h5`)
  -j N, --workers N     Number of data loading workers. (default:0)
  --epochs N            Number of total epochs to run. (default:200)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.001)
  --scale-factor SCALE_FACTOR
                        Low to high resolution scaling factor. (default:4).
  -p N, --print-freq N  Print frequency. (default:5)
  --cuda                Enables cuda
  --outf OUTF           folder to output images. (default:`./outputs`).
  --weights WEIGHTS     Path to weights (to continue training).
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:none)
```

#### Example (e.g T91)

```bash
python train.py --train-file ./data/train/T91_X4.h5 --eval-file ./data/val/Set5_X4.h5 --scale-factor 4 --cuda
```

If you want to load weights that you've trained before, run the following command.

```bash
python train.py --train-file ./data/train/T91_X4.h5 --eval-file ./data/val/Set5_X4.h5 --scale-factor 4 --weights ./weights/model_epoch_100.pth --cuda
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Accelerating the Super-Resolution Convolutional Neural Network
_Chao Dong, Chen Change Loy, Xiaoou Tang_ <br>

**Abstract** <br>
As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (SRCNN) has demonstrated superior performance to the previous hand-crafted models either in speed and restoration quality. However, the high computational cost still hinders it from practical usage that demands real-time performance (24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure for faster and better SR. We re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution layer at the end of the network, then the mapping is learned directly from the original low-resolution image (without interpolation) to the high-resolution one. Second, we reformulate the mapping layer by shrinking the input feature dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers. The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance. A corresponding transfer strategy is also proposed for fast training and testing across different upscaling factors.

[[Paper]](https://arxiv.org/pdf/1608.00367)

```
@misc{dong2016accelerating,
    title={Accelerating the Super-Resolution Convolutional Neural Network},
    author={Chao Dong and Chen Change Loy and Xiaoou Tang},
    year={2016},
    eprint={1608.00367},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
}
```
