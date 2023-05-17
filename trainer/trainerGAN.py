#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Template GAN training flow."""

import logging
import os
import abc
import torch

from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import tqdm


class TrainerGAN(abc.ABC):
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
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.train_max_steps = config.get("train_max_steps", 0)

    
    @abc.abstractmethod
    def _train_step(self, batch):
        """Single step of training."""
        pass
        

    @abc.abstractmethod
    def _eval_step(self, batch):
        """Single step of evaluation."""
        pass


    def run(self):
        """Run training."""
        self.finish_train = False
        self.tqdm = tqdm(
            initial=self.steps, total=self.train_max_steps, desc="[train]"
        )
        while True:
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")


    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "generator": self.model["generator"].state_dict(),
            "discriminator": self.model["discriminator"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)


    def load_checkpoint(self, checkpoint_path, strict=True, load_only_params=False, load_discriminator=True):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
            load_discriminator (bool): Whether to load optimizer and scheduler of the discriminators.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["generator"].load_state_dict(
            state_dict["model"]["generator"], strict=strict)
        self.model["discriminator"].load_state_dict(
            state_dict["model"]["discriminator"], strict=strict)
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"])
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"])
            if load_discriminator:
                self.optimizer["discriminator"].load_state_dict(
                    state_dict["optimizer"]["discriminator"])
                self.scheduler["discriminator"].load_state_dict(
                    state_dict["scheduler"]["discriminator"])
        

    def _train_epoch(self):
        """One epoch of training."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        if train_steps_per_epoch > 200:
            logging.info(
                f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                f"({self.train_steps_per_epoch} steps per epoch)."
            )


    def _eval_epoch(self):
        """One epoch of evaluation."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

        logging.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()


    def _metric_loss(self, predict_y, natural_y, mode='train'):
        """Metric losses."""
        metric_loss=0.0

        # mel spectrogram loss
        if self.config.get('use_mel_loss', False):
            mel_loss = self.criterion["mel"](predict_y, natural_y)
            mel_loss *= self.config["lambda_mel_loss"]
            self._record_loss('mel_loss', mel_loss, mode=mode)
            metric_loss += mel_loss
        
        # multi-resolution sfft loss
        if self.config.get('use_stft_loss', False):
            sc_loss, mag_loss = self.criterion["stft"](predict_y, natural_y)
            sc_loss *= self.config["lambda_stft_loss"]
            mag_loss *= self.config["lambda_stft_loss"]
            self._record_loss('spectral_convergence_loss', sc_loss, mode=mode)
            self._record_loss('log_stft_magnitude_loss', mag_loss, mode=mode)
            metric_loss += (sc_loss + mag_loss)

        # waveform shape loss
        if self.config.get("use_shape_loss", False):
            shape_loss = self.criterion["shape"](predict_y, natural_y)
            shape_loss *= self.config["lambda_shape_loss"]
            self._record_loss('shape_loss', shape_loss, mode=mode)
            metric_loss += shape_loss
        
        return metric_loss
    

    def _adv_loss(self, predict_p, natural_p=None, mode='train'):
        """Adversarial loss."""
        adv_loss = self.criterion["gen_adv"](predict_p)

        # feature matching loss
        if natural_p is not None:
            fm_loss = self.criterion["feat_match"](predict_p, natural_p)
            self._record_loss('feature_matching_loss', fm_loss, mode=mode)
            adv_loss += self.config["lambda_feat_match"] * fm_loss

        adv_loss *= self.config["lambda_adv"]
        self._record_loss('adversarial_loss', adv_loss, mode=mode)

        return adv_loss
    

    def _dis_loss(self, predict_p, natural_p, mode='train'):
        """Discriminator loss."""
        real_loss, fake_loss = self.criterion["dis_adv"](predict_p, natural_p)
        dis_loss = real_loss + fake_loss
        self._record_loss('real_loss', real_loss, mode=mode)
        self._record_loss('fake_loss', fake_loss, mode=mode)
        self._record_loss('discriminator_loss', dis_loss, mode=mode)

        return dis_loss


    def _update_generator(self, gen_loss):
        """Update generator."""
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"],
            )
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()  
    

    def _update_discriminator(self, dis_loss):
        """Update discriminator."""
        self.optimizer["discriminator"].zero_grad()
        dis_loss.backward()
        if self.config["discriminator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["discriminator"].parameters(),
                self.config["discriminator_grad_norm"],
            )
        self.optimizer["discriminator"].step()
        self.scheduler["discriminator"].step()
    

    def _record_loss(self, name, loss, mode='train'):
        """Record loss."""
        if torch.is_tensor(loss):
            loss = loss.item()

        if mode == 'train':
            self.total_train_loss[f"train/{name}"] += loss
        elif mode == 'eval':
            self.total_eval_loss[f"eval/{name}"] += loss
        else:
            raise NotImplementedError(f"Mode ({mode}) is not supported!")


    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)


    def _check_save_interval(self):
        if self.steps and (self.steps % self.config["save_interval_steps"] == 0):
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")


    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()


    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)


    def _check_train_finish(self):
        if self.steps >= self.train_max_steps:
            self.finish_train = True
        else:
            self.finish_train = False
        return self.finish_train


class TrainerVQGAN(TrainerGAN):
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

        super(TrainerVQGAN, self).__init__(
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


    # perplexity info
    def _perplexity(self, perplexity, label=None, mode='train'):
        if label:
            name = f"{mode}/ppl_{label}"
        else:
            name = f"{mode}/ppl"
        if torch.numel(perplexity) > 1:
            perplexity = perplexity.tolist()
            for idx, ppl in enumerate(perplexity):
                self._record_loss(f"{name}_{idx}", ppl, mode=mode)
        else:
            self._record_loss(name, perplexity, mode=mode)


    # vq loss
    def _vq_loss(self, vqloss, label=None, mode='train'):
        if label:
            name = f"{mode}/vqloss_{label}"
        else:
            name = f"{mode}/vqloss"
        vqloss = torch.sum(vqloss)
        vqloss *= self.config["lambda_vq_loss"]
        self._record_loss(name, vqloss, mode=mode)

        return vqloss
    
