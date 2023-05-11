#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""Training stage template."""

import os
import abc
import sys
import yaml
import random
import logging
import torch
import numpy as np

from bin.utils import load_config


class TrainGAN(abc.ABC):
    def __init__(
        self,
        args,
    ):
        # set logger
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        # Fix seed and make backends deterministic
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logging.info(f"device: cpu")
        else:
            self.device = torch.device('cuda')
            logging.info(f"device: gpu")
            torch.cuda.manual_seed_all(args.seed)
            if args.disable_cudnn == "False":
                torch.backends.cudnn.benchmark = True
        
        # initialize config
        with open(args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.config.update(vars(args))

        # initialize model folder
        expdir = os.path.join(args.exp_root, args.tag)
        os.makedirs(expdir, exist_ok=True)
        self.config["outdir"] = expdir

        # save config
        with open(os.path.join(expdir, "config.yml"), "w") as f:
            yaml.dump(self.config, f, Dumper=yaml.Dumper)
        for key, value in self.config.items():
            logging.info(f"[TrainGAN] {key} = {value}")
        
        # initialize attribute
        self.resume = args.resume
        self.data_loader = None
        self.model = {}
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.trainer = None

        # initialize batch_length
        self.batch_length = self.config['batch_length']


    @abc.abstractmethod    
    def initialize_data_loader(self):
        pass
    
    
    @abc.abstractmethod
    def define_model(self):
        pass
    
    
    @abc.abstractmethod
    def define_trainer(self):
        pass


    @abc.abstractmethod
    def initialize_model(self):
        pass
    

    @abc.abstractmethod
    def define_criterion(self):
        pass


    def run(self):
        try:
            logging.info(f"The current training step: {self.trainer.steps}")
            self.trainer.train_max_steps = self.config["train_max_steps"]
            if not self.trainer._check_train_finish():
                self.trainer.run()
            if self.config.get("adv_train_max_steps", False) and self.config.get("adv_batch_length", False):
                self.batch_length = self.config['adv_batch_length']
                logging.info(f"Reload dataloader for adversarial training.")                
                self.initialize_data_loader()
                self.trainer.data_loader = self.data_loader
                self.trainer.train_max_steps = self.config["adv_train_max_steps"]
                self.trainer.run()
        finally:
            self.trainer.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.trainer.steps}steps.pkl")
            )
            logging.info(f"Successfully saved checkpoint @ {self.trainer.steps}steps.")
    
    
    def _show_setting(self):
        logging.info(self.model['generator'])
        logging.info(self.model['discriminator'])
        logging.info(self.optimizer['generator'])
        logging.info(self.optimizer['discriminator'])
        logging.info(self.scheduler['generator'])
        logging.info(self.scheduler['discriminator'])
        for criterion_ in self.criterion.values():
            logging.info(criterion_)
    

    def _load_config(self, checkpoint, config_name='config.yml'):
        return load_config(checkpoint, config_name)
        
