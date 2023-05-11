#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://ieeexplore.ieee.org/document/9625818)

"""Decoder modules."""

import torch
import inspect

from layers.conv_layer import NonCausalConv1d, NonCausalConvTranspose1d
from layers.conv_layer import CausalConv1d, CausalConvTranspose1d
from models.autoencoder.modules.residual_unit import NonCausalResidualUnit
from models.autoencoder.modules.residual_unit import CausalResidualUnit
from models.utils import check_mode


class DecoderBlock(torch.nn.Module):
    """ Decoder block (upsampling) """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilations=(1, 3, 9),
        bias=True,
        mode='causal',
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            ResidualUnit = NonCausalResidualUnit
            ConvTranspose1d = NonCausalConvTranspose1d
        elif self.mode == 'causal':
            ResidualUnit = CausalResidualUnit
            ConvTranspose1d = CausalConvTranspose1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.conv = ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2 * stride),
            stride=stride,
            bias=bias,
        )

        self.res_units = torch.nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            self.res_units += [
                ResidualUnit(out_channels, out_channels, dilation=dilation)]
        self.num_res = len(self.res_units)
        
    def forward(self, x):
        x = self.conv(x)
        for idx in range(self.num_res):
            x = self.res_units[idx](x)
        return x
    
    def inference(self, x):
        check_mode(self.mode, inspect.stack()[0][3])
        x = self.conv.inference(x)
        for idx in range(self.num_res):
            x = self.res_units[idx].inference(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self,
        code_dim, 
        output_channels,
        decode_channels,
        channel_ratios=(16, 8, 4, 2),
        strides=(5, 5, 4, 3),
        kernel_size=7,
        bias=True,
        mode='causal',
    ):
        super().__init__()
        assert len(channel_ratios) == len(strides)
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({self.mode}) is not supported!")

        self.conv1 = Conv1d(
            in_channels=code_dim, 
            out_channels=(decode_channels * channel_ratios[0]), 
            kernel_size=kernel_size, 
            stride=1, 
            bias=False)

        self.conv_blocks = torch.nn.ModuleList()
        for idx, stride in enumerate(strides):
            in_channels = decode_channels * channel_ratios[idx]
            if idx < (len(channel_ratios)-1):
                out_channels = decode_channels * channel_ratios[idx+1]
            else:
                out_channels = decode_channels
            self.conv_blocks += [
                DecoderBlock(in_channels, out_channels, stride, bias=bias, mode=self.mode)]
        self.num_blocks = len(self.conv_blocks)

        self.conv2 = Conv1d(out_channels, output_channels, kernel_size, 1, bias=False)

    def forward(self, z):
        x = self.conv1(z)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i](x)
        x = self.conv2(x)
        return x
    
    def decode(self, z):
        check_mode(self.mode, inspect.stack()[0][3])
        x = self.conv1.inference(z)
        for i in range(self.num_blocks):
            x = self.conv_blocks[i].inference(x)
        x = self.conv2.inference(x)
        return x