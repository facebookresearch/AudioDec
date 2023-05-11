#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Utility modules."""

def check_mode(mode, method):
    stream_modes = ['causal']
    assert mode in stream_modes, f"Mode {mode} does not support {method}!"
    