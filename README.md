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
        - [Download train dataset](#download-train-dataset)
        - [Download val dataset](#download-val-dataset)
    - [Test](#test)
    - [Train](#train)
    - [Result](#result)
    - [Credit](#credit)
        - [Accelerating the Super-Resolution Convolutional Neural Network](#accelerating-the-super-resolution-convolutional-neural-network)

## About Accelerating the Super-Resolution Convolutional Neural Network

If you're new to FSRCNN, here's an abstract straight from the paper:

As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (
SRCNN) has demonstrated superior performance to the previous hand-crafted models either in speed and restoration
quality. However, the high computational cost still hinders it from practical usage that demands real-time performance (
24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure
for faster and better SR. We re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution
layer at the end of the network, then the mapping is learned directly from the original low-resolution image (without
interpolation) to the high-resolution one. Second, we reformulate the mapping layer by shrinking the input feature
dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers.
The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present
the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance.
A corresponding transfer strategy is also proposed for fast training and testing across different upscaling factors.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/1-Cp0UVqSLBvW-gNV_Xvw5hlj1Ph7f6Oc?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1mXzvJeHQtSQxmhbBHQYcnA) access:`llot`

## Download datasets

### Download train dataset

- [Google Driver](https://drive.google.com/drive/folders/1iSmgWI7uU3vsHnlE1oOe59CCees0yncU?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/11X1WQSurtDJ9rNa8lF8NvQ) access: `llot`

### Download val dataset

Set5 dataset:

- [Google Driver](https://drive.google.com/file/d/1GJZztdiJ6oBmJe9Ntyyos_psMzM8KY4P/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1_B97Ga6thSi5h43Wuqyw0Q) access:`llot`

Set14 dataset:

- [Google Driver](https://drive.google.com/file/d/14bxrGB3Nej8vBqxLoqerGX2dhChQKJoa/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1wy_kf4Kkj2nSkgRUkaLzVA) access:`llot`

Bsd100 dataset:

- [Google Driver](https://drive.google.com/file/d/1RTlPATPBCfUufJspgTik5KUEzAuVcyFF/view?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1Ig8t3_G4Nzhl8MvPAvdzFA) access:`llot`

## Test

Modify the contents of the file as follows.

- `config.py` line 32 `mode="train"` change to `mode="valid"`.
- `config.py` line 83 `model.load_state_dict(torch.load(f"results/{exp_name}/best.pth", map_location=device))` change to `model.load_state_dict(torch.load("<YOUR-WEIGHTS-PATH>", map_location=device))`.
- Run `python validate.py`.

## Train

Modify the contents of the file as follows.

- `config.py` line 32 `mode="valid"` change to `mode="train"`.
- Run `python train.py`.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- `config.py` line 32 `mode="valid"` change to `mode="train"`.
- `config.py` line 48 `start_epoch=0` change to `start_epoch=<RESUME-EPOCH>`.
- `config.py` line 59 `resume=False` change to `resume=True`.
- `config.py` line 50 `resume_weight=""` change to `resume_weight="<YOUR-RESUME-WIGHTS-PATH>"`.
- Run `python train.py`.

## Result

Source of original paper results: https://arxiv.org/pdf/1608.00367v1.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |        SSIM        |
| :-----: | :---: | :--------------: | :----------------: |
|  Set5   |   2   | 37.00(**36.76**) | 0.9558(**0.9564**) |
|  Set5   |   3   | 33.16(**32.46**) | 0.9140(**0.9051**) |
|  Set5   |   4   | 30.71(**30.37**) | 0.8657(**0.8589**) |
|  Set14  |   2   | 32.63(**32.33**) | 0.9088(**0.9089**) |
|  Set14  |   3   | 29.43(**28.85**) | 0.8242(**0.8181**) |
|  Set14  |   4   | 27.59(**27.20**) | 0.7535(**0.7507**) |

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png"/></span>

## Credit

### Accelerating the Super-Resolution Convolutional Neural Network

_Chao Dong, Chen Change Loy, Xiaoou Tang_ <br>

**Abstract** <br>
As a successful deep model applied in image super-resolution (SR), the Super-Resolution Convolutional Neural Network (
SRCNN) has demonstrated superior performance to the previous hand-crafted models either in speed and restoration
quality. However, the high computational cost still hinders it from practical usage that demands real-time performance (
24 fps). In this paper, we aim at accelerating the current SRCNN, and propose a compact hourglass-shape CNN structure
for faster and better SR. We re-design the SRCNN structure mainly in three aspects. First, we introduce a deconvolution
layer at the end of the network, then the mapping is learned directly from the original low-resolution image (without
interpolation) to the high-resolution one. Second, we reformulate the mapping layer by shrinking the input feature
dimension before mapping and expanding back afterwards. Third, we adopt smaller filter sizes but more mapping layers.
The proposed model achieves a speed up of more than 40 times with even superior restoration quality. Further, we present
the parameter settings that can achieve real-time performance on a generic CPU while still maintaining good performance.
A corresponding transfer strategy is also proposed for fast training and testing across different upscaling factors.

[[Paper]](https://arxiv.org/pdf/1608.00367v1.pdf)

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
