#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)
# Reference (https://github.com/r9y9/wavenet_vocoder)
# Reference (https://github.com/jik876/hifi-gan)

"""Multi-fusion modules."""

import math
import torch
import torch.nn as nn
from AudioDec.layers.conv_layer import Conv1d1x1
from AudioDec.models.vocoder.modules.residual_block import HiFiGANResidualBlock


class MultiReceptiveField(nn.Module):
    """Multi-receptive field module in HiFiGAN."""

    def __init__(
        self,
        channels=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        groups=1,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        assert len(resblock_kernel_sizes) == len(resblock_dilations)
        super().__init__()
        self.num_blocks = len(resblock_kernel_sizes)

        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks += [
                HiFiGANResidualBlock(
                    kernel_size=resblock_kernel_sizes[i],
                    channels=channels,
                    dilations=resblock_dilations[i],
                    groups=groups,
                    bias=bias,
                    use_additional_convs=use_additional_convs,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
    
    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        cs = 0.0  # initialize
        for i in range(self.num_blocks):
            cs += self.blocks[i](c)
        c = cs / self.num_blocks

        return c
    
    def inference(self, c):
        cs = 0.0  # initialize
        for i in range(self.num_blocks):
            cs += self.blocks[i].inference(c)
        c = cs / self.num_blocks
        
        return c
        

class MultiGroupConv1d(HiFiGANResidualBlock):
    """Multi-group convolution module."""

    def __init__(
        self,
        channels=512,
        resblock_kernel_sizes=(3),
        resblock_dilations=[(1, 3, 5)],
        groups=3,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
    ):
        assert len(resblock_kernel_sizes) == len(resblock_dilations) == 1
        super(MultiGroupConv1d, self).__init__(
            kernel_size=resblock_kernel_sizes[0],
            channels=channels*groups,
            dilations=resblock_dilations[0],
            groups=groups,
            bias=bias,
            use_additional_convs=use_additional_convs,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
        )
        self.groups = groups
        self.conv_out = Conv1d1x1(
            in_channels=channels*groups, 
            out_channels=channels,
            bias=False,
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        x = x.repeat(1, self.groups, 1) # (B, n*C, T)
        for idx in range(self.num_layer):
            xt = self.convs1[idx](self.activation(x))
            if self.use_additional_convs:
                xt = self.convs2[idx](self.activation(xt))
            x = xt + x
        x = self.conv_out(x) # (B, C, T)
        return x
    
    def inference(self, x):
        x = x.repeat(1, self.groups, 1) # (B, n*C, T)
        for idx in range(self.num_layer):
            xt = self.convs1[idx].inference(self.activation(x))
            if self.use_additional_convs:
                xt = self.convs2[idx].inference(self.activation(xt))
            x = xt + x
        x = self.conv_out(x) # (B, C, T)
        return x