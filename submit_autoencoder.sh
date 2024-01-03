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
#SBATCH --job-name=ae_vctk
#SBATCH --output=/mnt/home/slurmlogs/vctk/autoencoder/symAD_vctk_48000_hop300.out  # STDOUT
#SBATCH --error=/mnt/home/slurmlogs/vctk/autoencoder/symAD_vctk_48000_hop300.err   # STDERR
#SBATCH --gres=gpu:1

### useage ###
# (training from scratch): sbatch/bash submit_autoencoder.sh --stage 0 (--tag_name xxx)
# (resume training): sbatch/bash submit_autoencoder.sh --stage 1 (--tag_name xxx --resumepoint xxx)
# (testing): sbatch/bash submit_autoencoder.sh --stage 2 (--tag_name xxx --encoder_checkpoint xxx --decoder_checkpoint xxx --subset xxx --subset_num xxx)

### VCTK ###
tag_name="autoencoder/symAD_vctk_48000_hop300"
#tag_name="autoencoder/symADuniv_vctk_48000_hop300"
#tag_name="autoencoder/symAAD_snake_vctk_48000_hop300"
#tag_name="autoencoder/symAD_c16_vctk_48000_hop320"

### LibriTTS ###
#tag_name="autoencoder/symAD_libritts_24000_hop300"

stage=0
# stage 0: training
# stage 1: resuming training from previous saved checkpoint
# stage 2: testing
resumepoint=200000
encoder_checkpoint=200000
decoder_checkpoint=700000
#resumepoint=500000
#encoder_checkpoint=500000
#decoder_checkpoint=1000000
exp=exp # Folder of models
disable_cudnn=False
subset="test"
subset_num=-1

. ./parse_options.sh || exit 1;

config_name="config/${tag_name}.yaml"
echo "Configuration file="$config_name

# stage 0
if echo ${stage} | grep -q 0; then
    echo "Training from scratch"
    python codecTrain.py \
    -c ${config_name} \
    --tag ${tag_name} \
    --exp_root ${exp} \
    --disable_cudnn ${disable_cudnn} 
fi

# stage 1
if echo ${stage} | grep -q 1; then
    resume=exp/${tag_name}/checkpoint-${resumepoint}steps.pkl
    echo "Resume from ${resume}"
    python codecTrain.py \
    -c ${config_name} \
    --tag ${tag_name} \
    --resume ${resume} \
    --disable_cudnn ${disable_cudnn} 
fi

# stage 2
if echo ${stage} | grep -q 2; then
    echo "Testing"
    python codecTest.py \
    --subset ${subset} \
    --subset_num ${subset_num} \
    --encoder exp/${tag_name}/checkpoint-${encoder_checkpoint}steps.pkl \
    --decoder exp/${tag_name}/checkpoint-${decoder_checkpoint}steps.pkl \
    --output_dir output/${tag_name}
fi