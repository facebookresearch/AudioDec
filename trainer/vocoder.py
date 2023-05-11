#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Training flow of GAN-based vocoder."""

import logging
import torch
from trainer.trainerGAN import TrainerGAN


class Trainer(TrainerGAN):
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
        self.fix_analyzer = False
        self.generator_start = config.get("generator_train_start_steps", 0)
        self.discriminator_start = config.get("discriminator_train_start_steps", 0)


    def _train_step(self, batch):
        """Train model one step."""
        mode = 'train'
        x = batch
        x = x.to(self.device)

        # fix analyzer
        if not self.fix_analyzer:    
            for parameter in self.model["analyzer"].parameters():
                parameter.requires_grad = False
            self.fix_analyzer = True
            logging.info("Analyzer is fixed!")
        self.model["analyzer"].eval()

        #######################
        #      Generator      #
        #######################
        if self.steps > self.generator_start:
            # initialize generator loss
            gen_loss = 0.0

            # main genertor operation
            e = self.model["analyzer"].encoder(x)
            z = self.model["analyzer"].projector(e)
            zq, _, _ = self.model["analyzer"].quantizer(z)
            y_ = self.model["generator"](zq)

            # metric loss
            gen_loss += self._metric_loss(y_, x, mode=mode)

            # adversarial loss
            if self.steps > self.discriminator_start:
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
        if self.steps > self.discriminator_start:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                e = self.model["analyzer"].encoder(x)
                z = self.model["analyzer"].projector(e)
                zq, _, _ = self.model["analyzer"].quantizer(z)
                y_ = self.model["generator"](zq)
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
        e = self.model["analyzer"].encoder(x)
        z = self.model["analyzer"].projector(e)
        zq, _, _ = self.model["analyzer"].quantizer(z)
        y_ = self.model["generator"](zq)

        # metric loss
        gen_loss += self._metric_loss(y_, x, mode=mode)

        # adversarial loss & feature matching loss
        if self.steps > self.discriminator_start:
            p_ = self.model["discriminator"](y_)
            if self.config["use_feat_match_loss"]:
                p = self.model["discriminator"](x)
            else:
                p = None
            gen_loss += self._adv_loss(p_, p, mode=mode)

            # discriminator loss
            self._dis_loss(p_, p, mode=mode)

        # generator loss
        self._record_loss('generator_loss', gen_loss, mode=mode)

