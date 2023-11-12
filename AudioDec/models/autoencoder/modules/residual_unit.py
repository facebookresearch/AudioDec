#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://ieeexplore.ieee.org/document/9625818)

"""Residual Units."""

import torch
import torch.nn as nn

from AudioDec.layers.conv_layer import Conv1d1x1, NonCausalConv1d, CausalConv1d
from AudioDec.layers.activation_function import get_activation

class NonCausalResidualUnit(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=7,
        dilation=1,
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={}, 
    ):
        super().__init__()
        self.activation = get_activation(nonlinear_activation, nonlinear_activation_params)
        self.conv1 = NonCausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
        self.conv2 = Conv1d1x1(out_channels, out_channels, bias)

    def forward(self, x):
        y = self.conv1(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y


class CausalResidualUnit(NonCausalResidualUnit):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=7,
        dilation=1,
        bias=False,
        nonlinear_activation="ELU",
        nonlinear_activation_params={}, 
    ):
        super(CausalResidualUnit, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation, 
            bias=bias,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params, 
        )
        self.conv1 = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
        )
    
    def inference(self, x):
        y = self.conv1.inference(self.activation(x))
        y = self.conv2(self.activation(y))
        return x + y