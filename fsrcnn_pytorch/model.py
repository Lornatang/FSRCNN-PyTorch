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
import torch.nn as nn
from torch.cuda import amp


class FSRCNN(nn.Module):
    def __init__(self, num_channels, scale_factor):
        super(FSRCNN, self).__init__()

        # Feature extraction
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.PReLU())

        self.mid_part = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=12, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),

            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),

            nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=64, out_channels=num_channels, kernel_size=9,
                                            stride=scale_factor, padding=3, output_padding=1)

    @amp.autocast()
    def forward(self, x):
        out = self.first_part(x)
        out = self.mid_part(out)
        out = self.last_part(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.0001)
                if m.bias is not None:
                    m.bias.data.zero_()
