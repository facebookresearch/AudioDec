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
        self.fix_encoder = False
        self.paradigm = config.get('paradigm', 'efficient') 
        self.generator_start = config.get('start_steps', {}).get('generator', 0)
        self.discriminator_start = config.get('start_steps', {}).get('discriminator', 200000)


    def _train_step(self, batch):
        """Single step of training."""
        mode = 'train'
        x = batch
        x = x.to(self.device)

        # check generator step
        if self.steps < self.generator_start:
            self.generator_train = False
        else:
            self.generator_train = True
            
        # check discriminator step
        if self.steps < self.discriminator_start:
            self.discriminator_train = False
        else:
            self.discriminator_train = True
            if (not self.fix_encoder) and (self.paradigm == 'efficient'):
                # fix encoder, quantizer, and codebook
                for parameter in self.model["generator"].encoder.parameters():
                    parameter.requires_grad = False
                for parameter in self.model["generator"].projector.parameters():
                    parameter.requires_grad = False
                for parameter in self.model["generator"].quantizer.parameters():
                    parameter.requires_grad = False
                self.fix_encoder = True
                logging.info("Encoder, projector, quantizer, and codebook are fixed")
        
        # check codebook updating
        if self.fix_encoder:
            self.model["generator"].quantizer.codebook.eval()

        #######################
        #      Generator      #
        #######################
        if self.generator_train:
            # initialize generator loss
            gen_loss = 0.0

            # main genertor operation
            y_, zq, z, vqloss, perplexity = self.model["generator"](x)

            # perplexity info
            self._perplexity(perplexity, mode=mode)

            # vq loss
            gen_loss += self._vq_loss(vqloss, mode=mode)
            
            # metric loss
            gen_loss += self._metric_loss(y_, x, mode=mode)
            
            # adversarial loss
            if self.discriminator_train:
                p_ = self.model["discriminator"](y_)
                if self.config["use_feat_match_loss"]:
                    with torch.no_grad():
                        p = self.model["discriminator"](x)
                else:
                    p = None
                gen_loss += self._adv_loss(p_, p, mode=mode)

            # update generator
            self._record_loss('generator_loss', gen_loss, mode=mode)
            self._update_generator(gen_loss)

        #######################
        #    Discriminator    #
        #######################
        if self.discriminator_train:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_, _, _, _, _ = self.model["generator"](x)
            
            p = self.model["discriminator"](x)
            p_ = self.model["discriminator"](y_.detach())

            # discriminator loss & update discriminator
            self._update_discriminator(self._dis_loss(p_, p, mode=mode))

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()


    @torch.no_grad()
    def _eval_step(self, batch):
        """Single step of evaluation."""
        mode = 'eval'
        x = batch
        x = x.to(self.device)
        
        # initialize generator loss
        gen_loss = 0.0

        # main genertor operation
        y_, zq, z, vqloss, perplexity = self.model["generator"](x)

        # perplexity info
        self._perplexity(perplexity, mode=mode)

        # vq_loss
        gen_loss += self._vq_loss(vqloss, mode=mode)
        
        # metric loss
        gen_loss += self._metric_loss(y_, x, mode=mode)

        if self.discriminator_train:
            # adversarial loss
            p_ = self.model["discriminator"](y_)
            p = self.model["discriminator"](x)
            gen_loss += self._adv_loss(p_, p, mode=mode)

            # discriminator loss
            self._dis_loss(p_, p, mode=mode)

        # generator loss
        self._record_loss('generator_loss', gen_loss, mode=mode)

        

       

