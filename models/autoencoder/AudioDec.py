#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""AudioDec model."""

import torch
import inspect

from layers.conv_layer import CausalConv1d, CausalConvTranspose1d
from models.autoencoder.modules.encoder import Encoder
from models.autoencoder.modules.decoder import Decoder
from models.autoencoder.modules.projector import Projector
from models.autoencoder.modules.quantizer import Quantizer
from models.utils import check_mode


### GENERATOR ###
class Generator(torch.nn.Module):
    """AudioDec generator."""

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        encode_channels=32,
        decode_channels=32,
        code_dim=64,
        codebook_num=8,
        codebook_size=1024,
        bias=True,
        enc_ratios=(2, 4, 8, 16),
        dec_ratios=(16, 8, 4, 2),
        enc_strides=(3, 4, 5, 5),
        dec_strides=(5, 5, 4, 3),
        mode='causal',
        codec='audiodec',
        projector='conv1d',
        quantier='residual_vq',
    ):
        super().__init__()
        if codec == 'audiodec':
            encoder = Encoder
            decoder = Decoder
        else:
            raise NotImplementedError(f"Codec ({codec}) is not supported!")
        self.mode = mode
        self.input_channels = input_channels

        self.encoder = encoder(
            input_channels=input_channels,
            encode_channels=encode_channels,
            channel_ratios=enc_ratios,
            strides=enc_strides,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )

        self.decoder = decoder(
            code_dim=code_dim,
            output_channels=output_channels,
            decode_channels=decode_channels,
            channel_ratios=dec_ratios,
            strides=dec_strides,
            kernel_size=7,
            bias=bias,
            mode=self.mode,
        )

        self.projector = Projector(
            input_channels=self.encoder.out_channels,
            code_dim=code_dim,
            kernel_size=3,
            stride=1,
            bias=False,
            mode=self.mode,
            model=projector,
        )

        self.quantizer = Quantizer(
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size,
            model=quantier,
        )

    def forward(self, x):
        x = self.encoder(x)
        z = self.projector(x)
        zq, vqloss, perplexity = self.quantizer(z)
        y = self.decoder(zq)
        return y, zq, z, vqloss, perplexity


# STREAMING
class StreamGenerator(Generator):
    """AudioDec streaming generator."""

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        encode_channels=32,
        decode_channels=32,
        code_dim=64,
        codebook_num=8,
        codebook_size=1024,
        bias=True,
        enc_ratios=(2, 4, 8, 16),
        dec_ratios=(16, 8, 4, 2),
        enc_strides=(3, 4, 5, 5),
        dec_strides=(5, 5, 4, 3),
        mode='causal',
        codec='audiodec',
        projector='conv1d',
        quantier='residual_vq',
    ):
        super(StreamGenerator, self).__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            encode_channels=encode_channels,
            decode_channels=decode_channels,
            code_dim=code_dim,
            codebook_num=codebook_num,
            codebook_size=codebook_size,
            bias=bias,
            enc_ratios=enc_ratios,
            dec_ratios=dec_ratios,
            enc_strides=enc_strides,
            dec_strides=dec_strides,
            mode=mode,
            codec=codec,
            projector=projector,
            quantier=quantier,
        )
        check_mode(mode, "AudioDec Streamer")
        self.reset_buffer()


    def initial_encoder(self, receptive_length, device):
        self.quantizer.initial()
        z = self.encode(torch.zeros(1, self.input_channels, receptive_length).to(device))
        idx = self.quantize(z)
        zq = self.lookup(idx)
        return zq


    def initial_decoder(self, zq):
        self.decode(zq)


    def encode(self, x):
        x = self.encoder.encode(x)
        z = self.projector.encode(x)
        return z


    def quantize(self, z):
        zq, idx = self.quantizer.encode(z)
        return idx


    def lookup(self, idx):
        return self.quantizer.decode(idx)


    def decode(self, zq):
        return self.decoder.decode(zq.transpose(2, 1))


    def reset_buffer(self):
        """Apply weight normalization module from all layers."""

        def _reset_buffer(m):
            if isinstance(m, CausalConv1d) or isinstance(m, CausalConvTranspose1d):
                m.reset_buffer()
        self.apply(_reset_buffer)
