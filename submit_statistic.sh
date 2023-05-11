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
#SBATCH -c 40                      # number of cores
#SBATCH --mem 124g
#SBATCH -t 1-00:00
#SBATCH --job-name=extraction
#SBATCH --output=/mnt/home/slurmlogs/vctk/extraction/symAD_vctk.out  # STDOUT
#SBATCH --error=/mnt/home/slurmlogs/vctk/extraction/symAD_vctk.err   # STDERR

### useage ###
# (extract statistics): sbatch/bash submit_statistic.sh 

tag_name=statistic/symAD_vctk_48000_hop300_clean
#tag_name=statistic/symADuniv_vctk_48000_hop300_clean
#tag_name=statistic/symAD_libritts_24000_hop300_clean

subset=train
subset_num=-1

. ./parse_options.sh || exit 1;

# Statistic
config_name="config/${tag_name}.yaml"
echo "Configuration file="$config_name
python codecStatistic.py -c ${config_name} --subset ${subset} --subset_num ${subset_num}