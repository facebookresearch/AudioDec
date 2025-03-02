#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/lucidrains/vector-quantize-pytorch/)

"""Vector quantizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """Vector quantization w/ exponential moving averages (EMA)"""

    def __init__(
        self,
        dim,
        codebook_size,
        decay = 0.8,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
    ):
        super().__init__()
        n_embed = self.default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)
    
    def exists(self,val):
        return val is not None

    def default(self, val, d):
        return val if self.exists(val) else d
    
    def ema_inplace(self, moving_avg, new, decay):
        moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    
    def laplace_smoothing(self, x, n_categories, eps=1e-5):
        return (x + eps) / (x.sum() + n_categories * eps)
    
    def forward(self, input):
        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            self.ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = self.laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        loss = F.mse_loss(quantize.detach(), input) * self.commitment
        quantize = input + (quantize - input).detach()

        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantize, loss, perplexity
    
    def forward_index(self, input):
        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))
        quantize = input + (quantize - input).detach()

        return quantize, embed_ind


class ResidualVQ(nn.Module):
    """ Residual VQ following algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_out = 0.
        residual = x
        all_losses = []
        all_perplexities = []
        for layer in self.layers:
            quantized, loss, perplexity = layer(residual)
            # Issue: https://github.com/lucidrains/vector-quantize-pytorch/issues/33
            # We found considering only the 1st layer VQ's graident results in better performance
            #residual = residual - quantized.detach() # considering all layers' graidents
            residual = residual - quantized # considering only the first layer's graident 
            quantized_out = quantized_out + quantized
            all_losses.append(loss)
            all_perplexities.append(perplexity)
        all_losses, all_perplexities = map(torch.stack, (all_losses, all_perplexities))
        return quantized_out, all_losses, all_perplexities

    def forward_index(self, x, flatten_idx=False):
        quantized_out = 0.
        residual = x
        all_indices = []
        for i, layer in enumerate(self.layers):
            quantized, indices = layer.forward_index(residual)
            #residual = residual - quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            if flatten_idx:
                indices += (self.codebook_size * i)
            all_indices.append(indices)
        all_indices= torch.stack(all_indices)
        return quantized_out, all_indices.squeeze(1)
    
    def initial(self):
        self.codebook = []
        for layer in self.layers:
            self.codebook.append(layer.codebook)
        self.codebook_size = self.codebook[0].size(0)
        self.codebook = torch.stack(self.codebook)
        self.codebook = self.codebook.reshape(-1, self.codebook.size(-1))
    
    def lookup(self, indices):
        quantized_out = F.embedding(indices, self.codebook) # Num x T x C
        return  torch.sum(quantized_out, dim=0,keepdim=True)
