#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Training flow of symmetric codec."""

import logging
import torch
from trainer.trainerGAN import TrainerVQGAN


class Trainer(TrainerVQGAN):
    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        device=torch.device("cpu"),
    ):
        super(Trainer, self).__init__(
           steps=steps,
           epochs=epochs,
           data_loader=data_loader,
           model=model,
           criterion=criterion,
           optimizer=optimizer,
           scheduler=scheduler,
           config=config,
           device=device,
        )
        # fix quantizer
        for parameter in self.model["generator"].quantizer.parameters():
            parameter.requires_grad = False
        # fix decoder
        for parameter in self.model["generator"].decoder.parameters():
            parameter.requires_grad = False
        logging.info("Quantizer, codebook, and decoder are fixed")


    def _train_step(self, batch):
        """Single step of training."""
        mode = 'train'
        x_n, x_c = batch
        x_n = x_n.to(self.device)
        x_c = x_c.to(self.device)
        
        # fix codebook
        self.model["generator"].quantizer.codebook.eval()
        
        # initialize generator loss
        gen_loss = 0.0

        # main genertor operation
        y_nc, zq, z, vqloss, perplexity = self.model["generator"](x_n)

        # perplexity info
        self._perplexity(perplexity, mode=mode)

        # vq loss
        gen_loss += self._vq_loss(vqloss, mode=mode)
        
        # metric loss
        gen_loss += self._metric_loss(y_nc, x_c, mode=mode)

        # update generator
        self._record_loss('generator_loss', gen_loss, mode=mode)
        self._update_generator(gen_loss)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()


    @torch.no_grad()
    def _eval_step(self, batch):
        """Single step of evaluation."""
        mode = 'eval'
        x_n, x_c = batch
        x_n = x_n.to(self.device)
        x_c = x_c.to(self.device)
        
        # initialize generator loss
        gen_loss = 0.0

        # main genertor operation
        y_nc, zq, z, vqloss, perplexity = self.model["generator"](x_n)

        # perplexity info
        self._perplexity(perplexity, mode=mode)

        # vq_loss
        gen_loss += self._vq_loss(vqloss, mode=mode)
        
        # metric loss
        gen_loss += self._metric_loss(y_nc, x_c, mode=mode)

        # generator loss
        self._record_loss('generator_loss', gen_loss, mode=mode)

        

       

