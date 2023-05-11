#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

import os
import sys
import yaml
import torch
import logging
import argparse
import numpy as np
import soundfile as sf

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from dataloader import SingleDataset
from models.autoencoder.AudioDec import Generator as generator_audiodec


class StatisticMain(object):
    def __init__(self, args,):
        # set logger
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        # device
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
            logging.info(f"device: cpu")
        else:
            self.device = torch.device('cuda')
            logging.info(f"device: gpu")
        
        # initialize config
        with open(args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # initialize attribute
        self.stats_path = self.config['stats']
        self.analyzer_checkpoint = self.config['analyzer']
        self.analyzer_config = self._load_config(self.analyzer_checkpoint)
        self.model_type = self.analyzer_config.get('model_type', 'symAudioDec')
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True) 


    def _load_config(self, checkpoint, config_name='config.yml'):
        dirname = os.path.dirname(checkpoint)
        config_path = os.path.join(dirname, config_name)
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        return config


    def load_dataset(self, subset, subset_num):
        audio_path = os.path.join(
            self.config['data']['path'], 
            self.config['data']['subset'][subset],
        )
        assert os.path.exists(audio_path), f"{audio_path} does not exist!"
        self.dataset = SingleDataset(
            files=audio_path,
            query="*.wav",
            load_fn=sf.read,
            return_utt_id=False,
            subset_num=subset_num,
        )
        logging.info(f"The number of {subset} audio files = {len(self.dataset)}.")
    
        
    def load_analyzer(self):
        if self.model_type in ['symAudioDec', 'symAudioDecUniv']:
            analyzer = generator_audiodec
        else:     
            raise NotImplementedError(f"Analyzer {self.model_type} is not supported!")
        self.analyzer = analyzer(**self.analyzer_config['generator_params'])
        self.analyzer.load_state_dict(
            torch.load(self.analyzer_checkpoint, map_location='cpu')['model']['generator'])
        self.analyzer = self.analyzer.eval().to(self.device)
        logging.info(f"Loaded Analyzer from {self.analyzer_checkpoint}.")


    def audio_analysis(self, audio):
        x = torch.tensor(audio, dtype=torch.float).to(self.device)
        x = x.transpose(1, 0).unsqueeze(0) # (T, C) -> (1, C, T)
        x = self.analyzer.encoder(x)
        z = self.analyzer.projector(x)
        zq, _, _ = self.analyzer.quantizer(z)
        return zq.squeeze(0).transpose(1, 0).cpu().numpy() # (T', C)

    
    def run(self):
        with torch.no_grad(), tqdm(self.dataset, desc="[statistic]") as pbar:
            scaler = StandardScaler()
            for idx, x in enumerate(pbar, 1):
                zq = self.audio_analysis(x)
                scaler.partial_fit(zq)
            stats = np.stack([scaler.mean_, scaler.scale_], axis=0)
            np.save(
                    self.stats_path,
                    stats.astype(np.float32),
                    allow_pickle=False,
            )   
        logging.info(f"Finished statistical calculation of {idx} utterances.")


def main():
    """Run feature extraction process."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--subset_num", type=int, default=-1)
    args = parser.parse_args()

    # initial statistic_main
    statistic_main = StatisticMain(args=args) 

    # load dataset
    statistic_main.load_dataset(args.subset, args.subset_num)

    # load analyzer
    statistic_main.load_analyzer()
    
    # run testing
    statistic_main.run()


if __name__ == "__main__":
    main()
