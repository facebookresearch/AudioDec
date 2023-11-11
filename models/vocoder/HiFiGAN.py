#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)
# Reference (https://github.com/jik876/hifi-gan)

"""HiFi-GAN Modules. (Causal)"""

import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from AudioDec.layers.conv_layer import CausalConv1d, CausalConvTranspose1d
from AudioDec.models.vocoder.modules.multi_fusion import MultiReceptiveField, MultiGroupConv1d
from AudioDec.models.vocoder.modules.discriminator import HiFiGANMultiScaleDiscriminator
from AudioDec.models.vocoder.modules.discriminator import HiFiGANMultiPeriodDiscriminator

class Generator(nn.Module):
    """HiFiGAN causal generator module."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        groups=1,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        stats=None,
    ):
        """Initialize HiFiGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (int): Number of hidden representation channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            upsample_scales (list): List of upsampling scales.
            upsample_kernel_sizes (list): List of kernel sizes for upsampling layers.
            resblock_kernel_sizes (list): List of kernel sizes for residual blocks.
            resblock_dilations (list): List of dilation list for residual blocks.
            groups (int): Number of groups of residual conv
            bias (bool): Whether to add bias parameter in convolution layers.
            use_additional_convs (bool): Whether to use additional conv layers in residual blocks.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            stats (str): File name of the statistic file

        """
        super().__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, "Kernel size must be odd number."
        assert len(upsample_scales) == len(upsample_kernel_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        # Group conv or MRF
        if (len(resblock_dilations) == len(resblock_kernel_sizes) == 1) and (groups > 1):
            multi_fusion = MultiGroupConv1d
        else:
            multi_fusion = MultiReceptiveField
        
        # define modules
        self.num_upsamples = len(upsample_kernel_sizes)
        self.input_conv = CausalConv1d(
            in_channels,
            channels,
            kernel_size,
            stride=1,
        )
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.activation_upsamples = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        for i in range(len(upsample_kernel_sizes)):
            assert upsample_kernel_sizes[i] == 2 * upsample_scales[i]
            self.upsamples += [
                CausalConvTranspose1d(
                    channels // (2 ** i),
                    channels // (2 ** (i + 1)),
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_scales[i],
                )
            ]
            self.blocks += [
                multi_fusion(
                    channels=channels // (2 ** (i + 1)),
                    resblock_kernel_sizes=resblock_kernel_sizes,
                    resblock_dilations=resblock_dilations,
                    groups=groups,
                    bias=bias,
                    use_additional_convs=use_additional_convs,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                )
            ]
        self.activation_output1 = nn.LeakyReLU()
        self.activation_output2 = nn.Tanh()
        self.output_conv = CausalConv1d(
            channels // (2 ** (i + 1)),
            out_channels,
            kernel_size,
            stride=1,
        )

        # load stats
        if stats is not None:
            self.register_stats(stats)
            self.norm = True
        else:
            self.norm = False
        logging.info(f"Input normalization: {self.norm}")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    
    def forward(self, c):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        if self.norm:
            c = (c.transpose(2, 1) - self.mean) / self.scale
            c = c.transpose(2, 1)
        c = self.input_conv(c)
        for i in range(self.num_upsamples):
            c = self.upsamples[i](self.activation_upsamples(c))
            c = self.blocks[i](c)
        c = self.output_conv(self.activation_output1(c))
        c = self.activation_output2(c)

        return c
    

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows the official implementation manner.
        https://github.com/jik876/hifi-gan/blob/master/models.py

        """

        def _reset_parameters(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(
                m, nn.ConvTranspose1d
            ):
                nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


    def register_stats(self, stats):
        """Register stats for de-normalization as buffer.

        Args:
            stats (str): Path of statistics file (".npy" or ".h5").

        """
        assert stats.endswith(".h5") or stats.endswith(".npy")
        assert os.path.exists(stats), f"Stats {stats} does not exist!"
        mean = np.load(stats)[0].reshape(-1)
        scale = np.load(stats)[1].reshape(-1)
        self.register_buffer("mean", torch.from_numpy(mean).float())
        self.register_buffer("scale", torch.from_numpy(scale).float())
        logging.info("Successfully registered stats as buffer.")


class StreamGenerator(Generator):
    """HiFiGAN streaming generator."""

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        groups=1,
        bias=True,
        use_additional_convs=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        stats=None,
    ):

        super(StreamGenerator, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            kernel_size=kernel_size,
            upsample_scales=upsample_scales,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilations=resblock_dilations,
            groups=groups,
            bias=bias,
            use_additional_convs=use_additional_convs,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            use_weight_norm=use_weight_norm,
            stats=stats,
        )
        self.reset_buffer()

    
    def initial_decoder(self, c):
        self.decode(c)
    

    def decode(self, c):
        c = self.decode_norm(c)
        c = self.decode_input(c.transpose(2, 1))
        c = self.decode_upsample(c)
        c = self.decode_output(c)
        return c
    

    def decode_norm(self, c):
        if self.norm:
            c = (c - self.mean) / self.scale
        return c 


    def decode_input(self, c):
        c = self.input_conv.inference(c)
        return c


    def decode_upsample(self, c):        
        for i in range(self.num_upsamples):
            c = self.upsamples[i].inference(self.activation_upsamples(c))
            c = self.blocks[i].inference(c)
        return c


    def decode_output(self, c):
        c = self.output_conv.inference(self.activation_output1(c))
        return self.activation_output2(c)
    

    def reset_buffer(self):
        """Apply weight normalization module from all layers."""

        def _reset_buffer(m):
            if isinstance(m, CausalConv1d) or isinstance(m, CausalConvTranspose1d):
                m.reset_buffer()
        self.apply(_reset_buffer)


class Discriminator(nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales=3,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=True,
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
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
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
        if channel != 1:
            x = x.reshape(batch * channel, 1, time)
        msd_outs = self.msd(x)
        mpd_outs = self.mpd(x)
        return msd_outs + mpd_outs
