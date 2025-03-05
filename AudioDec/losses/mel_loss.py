#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Mel-spectrogram loss modules."""

import librosa
import torch
import torch.nn.functional as F


class MelSpectrogram(torch.nn.Module):
    """Calculate Mel-spectrogram."""

    def __init__(
        self,
        fs=22050,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann_window",
        num_mels=80,
        fmin=80,
        fmax=7600,
        center=True,
        normalized=False,
        onesided=True,
        eps=1e-10,
        log_base=10.0,
    ):
        """Initialize MelSpectrogram module."""
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        if win_length is not None:
            self.win_length = win_length
        else:
            self.win_length = fft_size
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.register_buffer("window", getattr(torch, window)(self.win_length))
        self.eps = eps

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        melmat = librosa.filters.mel(
            sr=fs,
            n_fft=fft_size,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

        self.log_base = log_base
        if self.log_base is None:
            self.log = torch.log
        elif self.log_base == 2.0:
            self.log = torch.log2
        elif self.log_base == 10.0:
            self.log = torch.log10
        else:
            raise ValueError(f"log_base: {log_base} is not supported.")


    def forward(self, x):
        """Calculate Mel-spectrogram.

        Args:
            x (Tensor): Input waveform tensor (B, T) or (B, C, T).

        Returns:
            Tensor: Mel-spectrogram (B, #mels, #frames).

        """
        if x.dim() == 3:
            # (B, C, T) -> (B*C, T)
            x = x.reshape(-1, x.size(2))

        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length, self.window, return_complex=True)
        x_power = x_stft.real ** 2 + x_stft.imag ** 2
        x_amp = torch.sqrt(torch.clamp(x_power, min=self.eps)).transpose(2, 1) # (B, D, T') -> (B, T', D)
        x_mel = torch.matmul(x_amp, self.melmat)
        x_mel = torch.clamp(x_mel, min=self.eps)

        return self.log(x_mel).transpose(1, 2) # (B, D, T')


class MultiMelSpectrogramLoss(torch.nn.Module):
    """Multi resolution Mel-spectrogram loss."""

    def __init__(
        self,
        fs=22050,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        num_mels=80,
        fmin=80,
        fmax=7600,
        center=True,
        normalized=False,
        onesided=True,
        eps=1e-10,
        log_base=10.0,
    ):
        """Initialize Mel-spectrogram loss."""
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.mel_transfers = torch.nn.ModuleList()
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            self.mel_transfers += [
                MelSpectrogram(
                    fs=fs,
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_length=win_length,
                    window=window,
                    num_mels=num_mels,
                    fmin=fmin,
                    fmax=fmax,
                    center=center,
                    normalized=normalized,
                    onesided=onesided,
                    eps=eps,
                    log_base=log_base,
                )
            ]


    def forward(self, y_hat, y):
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated single tensor (B, C, T).
            y (Tensor): Groundtruth single tensor (B, C, T).

        Returns:
            Tensor: Mel-spectrogram loss value.

        """
        mel_loss = 0.0
        for f in self.mel_transfers:
            mel_loss += F.l1_loss(f(y_hat), f(y))
        mel_loss /= len(self.mel_transfers)

        return mel_loss