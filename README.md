# FSRCNN-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation of [Accelerating the Super-Resolution Convolutional Neural Network](https://arxiv.org/abs/1608.00367v1).

## Table of contents

- [FSRCNN-PyTorch](#fsrcnn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Accelerating the Super-Resolution Convolutional Neural Network](#about-accelerating-the-super-resolution-convolutional-neural-network)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Accelerating the Super-Resolution Convolutional Neural Network](#accelerating-the-super-resolution-convolutional-neural-network)

## About Accelerating the Super-Resolution Convolutional Neural Network

If you're new to FSRCNN, here's an abstract straight from the paper:

As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (
SRCNN) has demonstrated superior performance to the previous hand-crafted models either in speed and restoration quality. However, the high
computational cost still hinders it from practical usage that demands real-time performance (
24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure for faster and better SR. We
re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution layer at the end of the network, then the mapping is
learned directly from the original low-resolution image (without interpolation) to the high-resolution one. Second, we reformulate the mapping layer
by shrinking the input feature dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers.
The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present the parameter settings that
can achieve real-time performance on a generic CPU while still maintaining good performance. A corresponding transfer strategy is also proposed for
fast training and testing across different upscaling factors.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `mode` change Set to valid mode.
- line 67: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 29: `upscale_factor` change to the magnification you need to enlarge.
- line 31: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 47: `start_epoch` change number of training iterations in the previous round.
- line 48: `resume` change weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1608.00367v1.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |
|:-------:|:-----:|:----------------:|
|  Set5   |   2   | 36.94(**37.09**) |
|  Set5   |   3   | 33.06(**33.06**) |
|  Set5   |   4   | 30.55(**30.66**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

## Credit

### Accelerating the Super-Resolution Convolutional Neural Network

_Chao Dong, Chen Change Loy, Xiaoou Tang_ <br>

**Abstract** <br>
As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (
SRCNN) has demonstrated superior performance to the previous hand-crafted models either in speed and restoration quality. However, the high
computational cost still hinders it from practical usage that demands real-time performance (
24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure for faster and better SR. We
re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution layer at the end of the network, then the mapping is
learned directly from the original low-resolution image (without interpolation) to the high-resolution one. Second, we reformulate the mapping layer
by shrinking the input feature dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers.
The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present the parameter settings that
can achieve real-time performance on a generic CPU while still maintaining good performance. A corresponding transfer strategy is also proposed for
fast training and testing across different upscaling factors.

[[Paper]](https://arxiv.org/pdf/1608.00367v1.pdf) [[Author's implements(Caffe)]](https://drive.google.com/open?id=0B7tU5Pj1dfCMWjhhaE1HR3dqcGs)

```bibtex
@article{DBLP:journals/corr/DongLT16,
  author    = {Chao Dong and
               Chen Change Loy and
               Xiaoou Tang},
  title     = {Accelerating the Super-Resolution Convolutional Neural Network},
  journal   = {CoRR},
  volume    = {abs/1608.00367},
  year      = {2016},
  url       = {http://arxiv.org/abs/1608.00367},
  eprinttype = {arXiv},
  eprint    = {1608.00367},
  timestamp = {Mon, 13 Aug 2018 16:47:56 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/DongLT16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
