#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Utility modules."""

import os
import yaml


def load_config(checkpoint, config_name='config.yml'):
    dirname = os.path.dirname(checkpoint)
    config_path = os.path.join(dirname, config_name)
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config
    