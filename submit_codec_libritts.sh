#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

#SBATCH -p xxx
#SBATCH -N 1                       # number of nodes
#SBATCH -n 1                       # number of nodes
#SBATCH -c 16                      # number of cores
#SBATCH --mem 64g
#SBATCH -t 7-00:00
#SBATCH --job-name=AD_libritts
#SBATCH --output=/mnt/home/slurmlogs/libritts/codec/AudioDec_libritts_24000.out  # STDOUT
#SBATCH --error=/mnt/home/slurmlogs/libritts/codec/AudioDec_libritts_24000.err   # STDERR
#SBATCH --gres=gpu:1

### useage ###
# (autoencoder training from scratch): sbatch/bash submit_codec_libritts.sh --start 0 --stop 0 
# (statistics extraction): sbatch/bash submit_codec_libritts.sh --start 1 --stop 1  
# (vocoder training from scratch): sbatch/bash submit_codec_libritts.sh --start 2 --stop 2  
# (autoencoder testing): sbatch/bash submit_codec_libritts.sh --start 3 --stop 3 (--encoder_checkpoint xxx --decoder_checkpoint xxx --subset xxx --subset_num xxx) 
# (autoencoder+vocoder testing): sbatch/bash submit_codec_libritts.sh --start 4 --stop 4 (--encoder_checkpoint xxx --decoder_checkpoint xxx --subset xxx --subset_num xxx) 

### LibriTTS ###
autoencoder=autoencoder/symAD_libritts_24000_hop300
statistic=statistic/symAD_libritts_24000_hop300_clean
vocoder=vocoder/AudioDec_v1_symAD_libritts_24000_hop300_clean

start=-1 # stage to start
stop=100 # stage to stop
# stage 0: training autoencoder from scratch
# stage 1: statistics extraction
# stage 2: training vocoder from scratch
# stage 3: testing (symAE)
# stage 4: testing (AE + Vocoder)
resumepoint=500000
encoder_checkpoint=500000
decoder_checkpoint=500000
exp=exp # Folder of models
disable_cudnn=False
statistic_subset=train
test_subset=test
subset_num=-1

. ./parse_options.sh || exit 1;

# AutoEncoder training
if [ "${start}" -le 0 ] && [ "${stop}" -ge 0 ]; then
    echo "AutoEncoder Training"
    config_name="config/${autoencoder}.yaml"
    echo "Configuration file="$config_name
    python codecTrain.py \
    -c ${config_name} \
    --tag ${autoencoder} \
    --exp_root ${exp} \
    --disable_cudnn ${disable_cudnn} 
fi

# Statistics extraction
if [ "${start}" -le 1 ] && [ "${stop}" -ge 1 ]; then
    echo "Statistics Extraction"
    config_name="config/${statistic}.yaml"
    echo "Configuration file="$config_name
    python codecStatistic.py \
    -c ${config_name} \
    --subset ${statistic_subset} \
    --subset_num ${subset_num}
fi

# Vocoder training
if [ "${start}" -le 2 ] && [ "${stop}" -ge 2 ]; then
    echo "Vocoder Training"
    config_name="config/${vocoder}.yaml"
    echo "Configuration file="$config_name
    python codecTrain.py \
    -c ${config_name} \
    --tag ${vocoder} \
    --exp_root ${exp} \
    --disable_cudnn ${disable_cudnn} 
fi

# Testing (AutoEncoder)
if [ "${start}" -le 3 ] && [ "${stop}" -ge 3 ]; then
    echo "Testing (AutoEncoder)"
    python codecTest.py \
    --subset ${test_subset} \
    --encoder exp/${autoencoder}/checkpoint-${encoder_checkpoint}steps.pkl \
    --decoder exp/${autoencoder}/checkpoint-$((${encoder_checkpoint}+${decoder_checkpoint}))steps.pkl \
    --output_dir output/${autoencoder}
fi

# Testing (AutoEncoder + Vocoder)
if [ "${start}" -le 4 ] && [ "${stop}" -ge 4 ]; then
    echo "Testing (AutoEncoder + Vocoder)"
    python codecTest.py \
    --subset ${test_subset} \
    --encoder exp/${autoencoder}/checkpoint-${encoder_checkpoint}steps.pkl \
    --decoder exp/${vocoder}/checkpoint-${decoder_checkpoint}steps.pkl \
    --output_dir output/${vocoder}
fi


