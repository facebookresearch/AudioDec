#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Waveform-based loss modules."""

import torch


class WaveformShapeLoss(torch.nn.Module):
    """Waveform shape loss."""

    def __init__(self, winlen):
        super().__init__()
        self.loss = torch.nn.L1Loss()
        self.winlen = winlen
        self.maxpool = torch.nn.MaxPool1d(self.winlen)

    def forward(self, y_hat, y):
        """Calculate L1 loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: L1 loss value.

        """
        ys = self.maxpool(torch.abs(y))
        ys_hat = self.maxpool(torch.abs(y_hat))
        loss = self.loss(ys_hat, ys)
        return loss


class MultiWindowShapeLoss(torch.nn.Module):
    """Multi-window-lengthe waveform shape loss."""

    def __init__(
        self,
        winlen=[300, 200, 100],
    ):
        """Initialize Multi window shape loss module.

        Args:
            winlen (list): List of window lengths.

        """
        super(MultiWindowShapeLoss, self).__init__()
        self.shape_losses = torch.nn.ModuleList()
        for wl in winlen:
            self.shape_losses += [WaveformShapeLoss(wl)]

    def forward(self, y_hat, y):
        """Calculate L1 loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, 1, T).
            y (Tensor): Groundtruth single tensor (B, 1, T).

        Returns:
            Tensor: L2 loss value.

        """
        loss = 0.0
        for f in self.shape_losses:
            loss += f(y_hat, y)
        loss /= len(self.shape_losses)

        return loss