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
import math
import torch.nn as nn
from torch import Tensor
from torch.cuda import amp


class FSRCNN(nn.Module):
    r"""Implement the model file in FSRCNN.
    `"Accelerating the Super-Resolution Convolutional Neural Network" <https://arxiv.org/pdf/1608.00367.pdf>`_
    """

    def __init__(self, upscale_factor, init_weights=True):
        super(FSRCNN, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU()
        )

        # Channel shrinking
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        # Channel mapping
        self.map = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )

        # Channel expanding
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        # Deconvolution
        self.deconv = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=9, stride=upscale_factor,
                                         padding=4, output_padding=upscale_factor - 1)

        if init_weights:
            self._initialize_weights()

    # The filter weights of each layer are initialized by drawing randomly from
    # a Gaussian distribution with a zero mean and a standard deviation
    # of 0.001 (and a deviation of 0).
    def _initialize_weights(self):
        for m in self.features:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data,
                                mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        for m in self.shrink:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data,
                                mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        for m in self.map:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data,
                                mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        for m in self.expand:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data,
                                mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)

    @amp.autocast()
    def forward(self, input: Tensor) -> Tensor:
        out = self.features(input)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)

        return out
