#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Activation functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(nonlinear_activation, nonlinear_activation_params={}):
    if hasattr(nn, nonlinear_activation):
        return getattr(nn, nonlinear_activation)(**nonlinear_activation_params)
    else:
        raise NotImplementedError(f"Activation {nonlinear_activation} is not supported!")