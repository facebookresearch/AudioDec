#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Projector modules."""

import torch
import inspect

from AudioDec.layers.conv_layer import NonCausalConv1d
from AudioDec.layers.conv_layer import CausalConv1d
from AudioDec.models.utils import check_mode


class Projector(torch.nn.Module):
    def __init__(self,
        input_channels,
        code_dim, 
        kernel_size=3,
        stride=1,
        bias=False,
        mode='causal',
        model='conv1d',
    ):
        super().__init__()
        self.mode = mode
        if self.mode == 'noncausal':
            Conv1d = NonCausalConv1d
        elif self.mode == 'causal':
            Conv1d = CausalConv1d
        else:
            raise NotImplementedError(f"Mode ({mode}) is not supported!")

        if model == 'conv1d':
            self.project = Conv1d(input_channels, code_dim, kernel_size=kernel_size, stride=stride, bias=bias)
        elif model == 'conv1d_bn':
            self.project = torch.nn.Sequential(
                Conv1d(input_channels, code_dim, kernel_size=kernel_size, stride=stride, bias=bias),
                torch.nn.BatchNorm1d(code_dim)
            )
        else:
            raise NotImplementedError(f"Model ({model}) is not supported!")
        
    def forward(self, x): 
        return self.project(x)
    
    def encode(self, x):
        check_mode(self.mode, inspect.stack()[0][3])
        return self.project.inference(x)


