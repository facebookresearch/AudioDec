#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Convolution layers."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def int2tuple(variable, length):
    if isinstance(variable, int):
        return (variable,)*length
    else:
        assert len(variable) == length, f"The length of {variable} is not {length}!"
        return variable


class Conv1d1x1(nn.Conv1d):
    """1x1 Conv1d."""

    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, bias=bias)


class NonCausalConv1d(nn.Module):
    """1D noncausal convloution w/ 2-sides padding."""

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=-1, 
            dilation=1,
            groups=1,
            bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x
    

class NonCausalConvTranspose1d(nn.Module):
    """1D noncausal transpose convloution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=-1,
        output_padding=-1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if padding < 0:
            padding = (stride+1) // 2
        if output_padding < 0:
            output_padding = 1 if stride % 2 else 0
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C', T').
        """
        x = self.deconv(x)
        return x


class CausalConv1d(NonCausalConv1d):
    """1D causal convloution w/ 1-side padding."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1,
        bias=True,
        pad_buffer=None,
    ):
        super(CausalConv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0,
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.pad_length = (kernel_size - 1) * dilation
        if pad_buffer is None:
            pad_buffer = torch.zeros(1, in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)

    def forward(self, x):
        pad = nn.ConstantPad1d((self.pad_length, 0), 0.0)
        x = pad(x)
        return self.conv(x)
    
    def inference(self, x):
        x = torch.cat((self.pad_buffer, x), -1)
        self.pad_buffer = x[:, :, -self.pad_length:]
        return self.conv(x)
    
    def reset_buffer(self):
        self.pad_buffer.zero_()

        
class CausalConvTranspose1d(NonCausalConvTranspose1d):
    """1D causal transpose convloution."""

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        bias=True,
        pad_buffer=None,
    ):
        super(CausalConvTranspose1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
            bias=bias,
        )
        self.stride = stride
        self.pad_length = (math.ceil(kernel_size/stride) - 1)
        if pad_buffer is None:
            pad_buffer = torch.zeros(1, in_channels, self.pad_length)
        self.register_buffer("pad_buffer", pad_buffer)

    def forward(self, x):
        pad = nn.ReplicationPad1d((self.pad_length, 0))
        x = pad(x)
        return self.deconv(x)[:, :, self.stride : -self.stride]
    
    def inference(self, x):
        x = torch.cat((self.pad_buffer, x), -1)
        self.pad_buffer = x[:, :, -self.pad_length:]
        return self.deconv(x)[:, :, self.stride : -self.stride]
    
    def reset_buffer(self):
        self.pad_buffer.zero_()


class NonCausalConv2d(nn.Module):
    """2D noncausal convloution w/ 4-sides padding."""

    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=-1, 
            dilation=1,
            groups=1,
            bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int2tuple(kernel_size, 2)
        self.dilation = int2tuple(dilation, 2)
        if isinstance(padding, int) and padding < 0:
            padding_0 = (self.kernel_size[0] - 1) // 2 * self.dilation[0]
            padding_1 = (self.kernel_size[1] - 1) // 2 * self.dilation[1]
            padding = (padding_0, padding_1)
        
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Float tensor variable with the shape  (B, C, T).
        Returns:
            Tensor: Float tensor variable with the shape (B, C, T).
        """
        x = self.conv(x)
        return x