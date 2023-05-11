#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Customized collater modules for Pytorch DataLoader."""

import torch
import numpy as np


class CollaterAudio(object):
    """Customized collater for loading single audio."""

    def __init__(
        self,
        batch_length=9600,
    ):
        """
        Args:
            batch_length (int): The length of audio signal batch.

        """
        self.batch_length = batch_length


    def __call__(self, batch):
        # filter short batch
        xs = [b for b in batch if len(b) > self.batch_length]
        
        # random cut
        starts, ends = self._random_segment(xs)
        x_batch = self._cut(xs, starts, ends)
        
        return x_batch
    

    def _random_segment(self, xs):
        x_lengths = [len(x) for x in xs]
        start_offsets = np.array(
            [
                np.random.randint(0, xl - self.batch_length)
                for xl in x_lengths
            ]
        )
        starts = start_offsets
        ends = starts + self.batch_length
        return starts, ends
    

    def _cut(self, xs, starts, ends):
        x_batch = np.array([x[start:end] for x, start, end in zip(xs, starts, ends)])
        x_batch = torch.tensor(x_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T)
        return x_batch


class CollaterAudioPair(CollaterAudio):
    """Customized collater for loading audio pair."""

    def __init__(
        self,
        batch_length=9600,
    ):
        super().__init__(
            batch_length=batch_length
        )


    def __call__(self, batch):
        batch = [
            b for b in batch if (len(b[0]) > self.batch_length) and (len(b[0]) == len(b[1]))
        ]
        assert len(batch) > 0, f"No qualified audio pairs.!"
        xs, ns = [b[0] for b in batch], [b[1] for b in batch]

        # random cut
        starts, ends = self._random_segment(xs)
        x_batch = self._cut(xs, starts, ends)
        n_batch = self._cut(ns, starts, ends)
        
        return n_batch, x_batch # (input, output)


