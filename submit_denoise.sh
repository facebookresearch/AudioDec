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
#SBATCH -t 1-00:00
#SBATCH --job-name=vctk_denoise
#SBATCH --output=/mnt/home/slurmlogs/vctk/denoise/symAD_vctk_48000_hop300.out  # STDOUT
#SBATCH --error=/mnt/home/slurmlogs/vctk/denoise/symAD_vctk_48000_hop300.err   # STDERR
#SBATCH --gres=gpu:1

### useage ###
# (training from scratch): sbatch/bash submit_denoise.sh --stage 0 (--tag_name xxx)
# (resume training): sbatch/bash submit_denoise.sh --stage 1 (--tag_name xxx --resumepoint xxx)
# (testing): sbatch/bash submit_denoise.sh --stage 2 (--tag_name xxx --encoder_checkpoint xxx --decoder_checkpoint xxx --subset xxx --subset_num xxx)

### VCTK ###
encoder="denoise/symAD_vctk_48000_hop300"
decoder="vocoder/AudioDec_v1_symAD_vctk_48000_hop300_clean"

stage=0
# stage 0: training
# stage 1: resuming training from previous saved checkpoint
# stage 2: testing
resumepoint=200000
encoder_checkpoint=200000
decoder_checkpoint=500000
exp=exp # Folder of models
disable_cudnn=False
subset="noisy_test"

. ./parse_options.sh || exit 1;

# stage 0
if echo ${stage} | grep -q 0; then
    echo "Denoising Training"
    config_name="config/${encoder}.yaml"
    echo "Configuration file="$config_name
    python codecTrain.py \
    -c ${config_name} \
    --tag ${encoder} \
    --exp_root ${exp} \
    --disable_cudnn ${disable_cudnn} 
fi

# stage 1
if echo ${stage} | grep -q 1; then
    resume=exp/${encoder}/checkpoint-${resumepoint}steps.pkl
    echo "Resume from ${resume}"
    config_name="config/${encoder}.yaml"
    python codecTrain.py -c ${config_name} --tag ${encoder} --resume ${resume} \
    --disable_cudnn ${disable_cudnn} 
fi

# stage 2
if echo ${stage} | grep -q 2; then
    echo "Denoising Testing"
    python codecTest.py --subset ${subset} \
    --encoder exp/${encoder}/checkpoint-${encoder_checkpoint}steps.pkl \
    --decoder exp/${decoder}/checkpoint-${decoder_checkpoint}steps.pkl \
    --output_dir output/${encoder}
fi