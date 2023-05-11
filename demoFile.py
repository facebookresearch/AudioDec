#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### useage ###
# (run w/ gpu): python demoFile.py --model libritts_v1 -i xxx.wav -o ooo.wav
# (run w/ cpu): python demoFile.py --cuda -1 --model libritts_sym -i xxx.wav -o ooo.wav

import os
import torch
import argparse
import numpy as np
import soundfile as sf
from utils.audiodec import AudioDec, assign_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="libritts_v1")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument('--cuda', type=int, default=0 )
    parser.add_argument('--num_threads', type=int, default=4)
    args = parser.parse_args()

    # device assignment
    if args.cuda < 0:
        tx_device = f'cpu'
        rx_device = f'cpu'
    else:
        tx_device = f'cuda:{args.cuda}'
        rx_device = f'cuda:{args.cuda}'
    torch.set_num_threads(args.num_threads)

    # model assignment
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(args.model)

    # AudioDec initinalize
    print("AudioDec initinalizing!")
    audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

    with torch.no_grad():
        if os.path.exists(args.input):
            data, fs = sf.read(args.input, always_2d=True)
        else:
            raise ValueError(f'Input file {args.input} does not exist!')
        assert fs == sample_rate, f"data ({fs}Hz) is not matched to model ({sample_rate}Hz)!"
        x = np.expand_dims(data.transpose(1, 0), axis=1) # (T, C) -> (C, 1, T)
        x = torch.tensor(x, dtype=torch.float).to(tx_device)
        print("Encode/Decode...")
        z = audiodec.tx_encoder.encode(x)
        idx = audiodec.tx_encoder.quantize(z)
        zq = audiodec.rx_encoder.lookup(idx)
        y = audiodec.decoder.decode(zq)[:, :, :x.size(-1)]
        y = y.squeeze(1).transpose(1, 0).cpu().numpy() # T x C
        sf.write(
            args.output,
            y,
            fs,
            "PCM_16",
        )
        print(f"Output {args.output}!")

    



if __name__ == "__main__":
    main()
