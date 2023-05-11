#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### useage ###
# (run w/ gpu): python dempStream.py --tx_cuda 1 --rx_cuda 2 --model libritts_v1 --input_device x --output_device o 
# (run w/ cpu): python dempStream.py --tx_cuda -1 --rx_cuda -1 --model libritts_sym --input_device x --output_device o 

import torch
import argparse
from utils.audiodec import AudioDec, AudioDecStreamer, assign_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="libritts_sym")
    parser.add_argument("-i", "--input", type=str, default="input.wav")
    parser.add_argument("-o", "--output", type=str, default="output.wav")
    parser.add_argument('--tx_cuda', type=int, default=-1 )
    parser.add_argument('--rx_cuda', type=int, default=-1 )
    parser.add_argument('--input_device', type=int, default=1)
    parser.add_argument('--output_device', type=int, default=4)
    parser.add_argument('--frame_size', type=int, default=1200)
    parser.add_argument('--num_threads', type=int, default=4)
    args = parser.parse_args()

    # device assignment
    if args.tx_cuda < 0:
        tx_device = f'cpu'
    else:
        tx_device = f'cuda:{args.tx_cuda}'
    if args.rx_cuda < 0:
        rx_device = f'cpu'
    else:
        rx_device = f'cuda:{args.rx_cuda}'
    torch.set_num_threads(args.num_threads)

    # model assignment
    sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(args.model)

    # AudioDec initinalize
    print("AudioDec initinalizing!")
    audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
    audiodec.load_transmitter(encoder_checkpoint)
    audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

    # Streamer initinalize
    print("Streamer initinalizing!")
    streamer = AudioDecStreamer(
        input_device=args.input_device,
        output_device=args.output_device,
        frame_size=args.frame_size,
        sample_rate=sample_rate,
        tx_encoder=audiodec.tx_encoder,
        tx_device=tx_device,
        rx_encoder=audiodec.rx_encoder,
        decoder=audiodec.decoder,
        rx_device=rx_device,
    )

    streamer.enable_filedump(
        input_stream_file=args.input,
        output_stream_file=args.output,
    )

    # run
    print("Ready to run!")
    latency="low"
    # TODO this is responsible for ~100ms latency, seems to be driver dependent. latency=0 works on Mac but not on Windows
    streamer.run(latency)


if __name__ == "__main__":
    main()
