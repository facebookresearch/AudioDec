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
import fnmatch
import logging
import numpy as np


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
        Args:
            root_dir (str): Root root_dir to find.
            query (str): Query to find.
            include_root_dir (bool): If False, root_dir name is not included.
        Returns:
            list: List of found filenames.
    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def load_files(data_path, query="*.wav", num_core=40):
    # sort all files    
    file_list = sorted(find_files(data_path, query))
    logging.info(f"The number of {os.path.basename(data_path)} files = {len(file_list)}.")
    # divide
    if num_core < len(file_list):
        file_lists = np.array_split(file_list, num_core)
        file_lists = [f_list.tolist() for f_list in file_lists]
    else:
        file_lists = [file_list]
    return file_lists