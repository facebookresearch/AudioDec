#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from AudioDec.layers.vq_module import ResidualVQ


class Quantizer(torch.nn.Module):
    def __init__(self,
        code_dim,
        codebook_num,
        codebook_size,
        model='residual_vq',
        ):
        super().__init__()
        # speech
        if model == 'residual_vq':
            self.codebook = ResidualVQ(dim=code_dim, num_quantizers=codebook_num, codebook_size=codebook_size)
        else:
            raise NotImplementedError(f"Model ({model}) is not supported!")

    def initial(self):
        self.codebook.initial()    
    
    def forward(self, z):
        zq, vqloss, perplexity = self.codebook(z.transpose(2, 1))
        zq = zq.transpose(2, 1)        
        return zq, vqloss, perplexity
    
    def inference(self, z):  
        zq, indices = self.codebook.forward_index(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, indices
    
    def encode(self, z):  
        zq, indices = self.codebook.forward_index(z.transpose(2, 1), flatten_idx=True)
        return zq, indices
    
    def decode(self, indices):  
        z = self.codebook.lookup(indices)
        return z
