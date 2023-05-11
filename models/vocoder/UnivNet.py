#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)
# Reference (https://github.com/chomeyama/SiFiGAN)

"""HiFi-GAN Modules. (Causal)"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vocoder.modules.discriminator import UnivNetMultiResolutionSpectralDiscriminator
from models.vocoder.modules.discriminator import HiFiGANMultiPeriodDiscriminator


class Discriminator(nn.Module):
    """UnivNet multi-resolution spectrogram + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-resolution spectrogram discriminator related
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        spectral_discriminator_params={
            "channels": 32,
            "kernel_sizes": [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
            "strides": [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
        },
        # Multi-period discriminator related
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
        flat_channel=False,
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            fft_sizes (list): FFT sizes for each spectral discriminator.
            hop_sizes (list): Hop sizes for each spectral discriminator.
            win_lengths (list): Window lengths for each spectral discriminator.
            window (stt): Name of window function.
            sperctral_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.
            flat_channel (bool):set true to flat multi-channel input to one-channel with multi-batch

        """
        super().__init__()
        self.flat_channel = flat_channel
        self.mrsd = UnivNetMultiResolutionSpectralDiscriminator(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            discriminator_params=spectral_discriminator_params,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, C, T).

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        (batch, channel, time) = x.size()
        if channel != 1 and self.flat_channel:
            x = x.reshape(batch * channel, 1, time)
        mrsd_outs = self.mrsd(x)
        mpd_outs = self.mpd(x)
        return mrsd_outs + mpd_outs
