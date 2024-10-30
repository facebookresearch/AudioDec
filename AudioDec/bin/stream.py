#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import abc
import yaml
import time
import queue
import threading
import torch
import torchaudio
import numpy as np

from typing import Union


class AudioCodec(abc.ABC):
    def __init__(
        self,
        tx_device: str = "cpu",
        rx_device: str = "cpu",
        receptive_length: int = 8192,
    ):
        self.tx_device = tx_device
        self.rx_device = rx_device
        self.receptive_length = receptive_length
        self.tx_encoder = None
        self.rx_encoder = None
        self.decoder = None


    @abc.abstractmethod
    def _load_encoder(self, checkpoint):
        pass


    @abc.abstractmethod
    def _load_decoder(self, checkpoint):
        pass


    def _load_config(self, checkpoint, config_name='config.yml'):
        dirname = os.path.dirname(checkpoint)
        config_path = os.path.join(dirname, config_name)
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        return config


    def load_transmitter(self, encoder_checkpoint):
        # load transmitter model(s)
        assert os.path.exists(encoder_checkpoint), f'{encoder_checkpoint} does not exist!'
        self.tx_encoder = self._load_encoder(encoder_checkpoint)
        self.tx_encoder.eval().to(self.tx_device)
        self.tx_encoder.initial_encoder(self.receptive_length, self.tx_device)
        print("Load tx_encoder: %s" % (encoder_checkpoint))


    def load_receiver(self, encoder_checkpoint, decoder_checkpoint):
        # load receiver model(s)
        assert os.path.exists(encoder_checkpoint), f'{encoder_checkpoint} does not exist!'
        self.rx_encoder = self._load_encoder(encoder_checkpoint)
        self.rx_encoder.eval().to(self.rx_device)
        zq = self.rx_encoder.initial_encoder(self.receptive_length, self.rx_device)
        print("Load rx_encoder: %s" % (encoder_checkpoint))

        assert os.path.exists(decoder_checkpoint), f'{decoder_checkpoint} does not exist!'
        self.decoder = self._load_decoder(decoder_checkpoint)
        self.decoder.eval().to(self.rx_device)
        self.decoder.initial_decoder(zq)
        print("Load decoder: %s" % (decoder_checkpoint))


class AudioCodecStreamer(abc.ABC):
    """
    Streams audio from an input microphone to headpones/speakers.
    For each model that can be optionally provided (encoder, decoder), the input audio is processed by the forward call of these models.

    Main functions (see function definition for detailed documentation):
    * __init__
    * enable_filedump
    * set_tx_rx_poses
    * run

    Example usage:

        streamer = AudioCodecStreamer(
            input_device=1,
            output_device=4,
            frame_size=512,
            encoder=my_encoder_network,
            tx_device="cuda:0",
            decoder=my_decoder_network,
            rx_device="cuda:1",
        )

        streamer.enable_filedump(input_stream_file="input.wav", output_stream_file="output.wav")

        streamer.run()
    """
    def __init__(
        self,
        input_device: Union[str, int],
        output_device: Union[str, int],
        input_channels: int = 1,
        output_channels: int = 1,
        frame_size: int = 512,
        sample_rate: int = 48000,
        gain: int = 1.0,
        max_latency: float = 0.1,
        # Transmitter params
        tx_encoder = None,
        tx_device: str = "cpu",
        # Receiver params
        rx_encoder = None,
        decoder = None,
        rx_device: str = "cpu",
    ):
        """
        Sounddevice parameters

        :param input_device:    int or str, name or index of the input device.
                                To get a list of all input devices call python3 -m sounddevice.
        :param output_device:   int or str, name or index of the output device.
                                To get a list of all output devices call python3 -m sounddevice.
        :param input_channels:  number of input channels, usually 1 but might be multiple microphones as well
        :param output_channels: number of output channels, usually 2 for binaural audio
        :param frame_size:      number of audio samples in a frame
        :param sample_rate:     sample rate of the audio signal
        :param gain:            linear factor to scale the input audio by
        :param max_latency:     maximal accepted latency in seconds before frames get dropped

        #######

        Transmitter parameters

        :param tx_encoder:      encoder network in the transimtter side
                                Is an instance of torch.nn.Module and must be fully initialized and loaded.
                                Must have a forward function that expects a batch x input_channels x frame_size tensor as input.
                                Default: None (input tensor is forwarded to decoder without change)
        :param tx_device:       device on transmitter (cpu, cuda:0, cuda:1, ...)

        #######

        Receiver parameters

        :param rx_encoder:      encoder network in the receiver side
                                Is an instance of torch.nn.Module and must be fully initialized and loaded.
                                Must have a forward function that expects a batch x input_channels x frame_size tensor as input.
                                Default: None (input tensor is forwarded to decoder without change)

        :param decoder:         decoder network
                                Is an instance of torch.nn.Module and must be fully initialized and loaded.
                                Must have a forward function that expects a tensor of the shape produced by the encoder.
                                Default: None (input tensor is forwarded to binauralizer without change)
        :param rx_device:       device on receiver (cpu, cuda:0, cuda:1, ...)
        """
        self.input_device = input_device
        self.output_device = output_device
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.gain = gain
        self.max_latency = max_latency

        # encoder
        self.tx_encoder = tx_encoder
        self.tx_device = tx_device
        print(f'Encoder device: {tx_device}')

        # decoder
        self.rx_encoder = rx_encoder
        self.decoder = decoder
        self.rx_device = rx_device
        print(f'Decoder device: {rx_device}')

        # queues for encoder, decoder, and output
        self.encoder_queue = queue.Queue()
        self.decoder_queue = queue.Queue()
        self.output_queue = queue.Queue()

        # file dump if requested
        self.input_dump = []
        self.output_dump = []
        self.input_dump_filename = None
        self.output_dump_filename = None

        # streaming statistics
        self.frame_drops = 0
        self.n_frames = 0
        self.encoder_times = []
        self.decoder_times = []
        self.latency_queue = queue.Queue()
        self.latencies = []

    @abc.abstractmethod
    def _encode(self, x):
        pass


    @abc.abstractmethod
    def _decode(self, x):
        pass
 
    def _run_encoder(self):
        while threading.main_thread().is_alive():
            try:
                x = self.encoder_queue.get(timeout=1)
            except:
                continue
            start = time.time()
            x = x.to(self.tx_device)
            with torch.no_grad():
                if self.tx_encoder is not None:
                    x = self._encode(x)
            self.encoder_times.append(time.time() - start)
            self.decoder_queue.put(x)


    def _run_decoder(self):
        while threading.main_thread().is_alive():
            try:
                x = self.decoder_queue.get(timeout=1)
            except:
                continue
            start = time.time()
            x = x.to(self.rx_device)
            with torch.no_grad():
                if (self.rx_encoder is not None) and (self.decoder is not None):
                    x = self._decode(x)
            self.decoder_times.append(time.time() - start)
            self.output_queue.put(x)


    def _process(self, data):
        data = data * self.gain
        input_data = torch.from_numpy(data).transpose(1, 0).contiguous()  # channels x frame_size

        if self.input_dump_filename is not None:
            self.input_dump.append(input_data)

        # add batch dimension
        input_data = input_data.unsqueeze(0)

        # process data
        self.encoder_queue.put(input_data)
        self.latency_queue.put(time.time())
        try:
            output_data = self.output_queue.get_nowait()
            latency = time.time() - self.latency_queue.get_nowait()
            self.latencies.append(latency)
            # clear queues if latency get too high; this will lead to frame drops
            if latency > self.max_latency:
                self.encoder_queue.queue.clear()
                self.decoder_queue.queue.clear()
                self.output_queue.queue.clear()
                while not self.latency_queue.empty():
                    self.frame_drops += 1
                    self.latency_queue.get_nowait()
        except queue.Empty:
            output_data = torch.zeros(1, self.output_channels, self.frame_size)
        output_data = output_data.squeeze(0).detach().cpu()

        self.n_frames += 1

        if self.output_dump_filename is not None:
            self.output_dump.append(output_data)

        data = output_data.transpose(1, 0).contiguous().numpy()

        return data

    def _callback(self, indata, outdata, frames, _time, status):
        if status:
            print(status)
        outdata[:] = self._process(indata)

    def _exit(self):
        # dump data to file if required
        if self.input_dump_filename is not None:
            audio = torch.clamp(torch.cat(self.input_dump, dim=-1), min=-1, max=1)
            torchaudio.save(self.input_dump_filename, audio, self.sample_rate)

        if self.output_dump_filename is not None:
            audio = torch.clamp(torch.cat(self.output_dump, dim=-1), min=-1, max=1)
            torchaudio.save(self.output_dump_filename, audio, self.sample_rate)

        # compute statistics
        with threading.Lock():
            encoder_mean = np.mean(np.array(self.encoder_times) * 1000.0)
            encoder_std = np.std(np.array(self.encoder_times) * 1000.0)
            decoder_mean = np.mean(np.array(self.decoder_times) * 1000.0)
            decoder_std = np.std(np.array(self.decoder_times) * 1000.0)
            latency_mean = np.mean(np.array(self.latencies) * 1000.0)
            latency_std = np.std(np.array(self.latencies) * 1000.0)
        frame_drops_ratio = self.frame_drops / self.n_frames

        # print statistics
        print('#' * 80)
        print(f"encoder processing time (ms):      {encoder_mean:.2f} +- {encoder_std:.2f}")
        print(f"decoder processing time (ms):      {decoder_mean:.2f} +- {decoder_std:.2f}")
        print(f"system latency (ms):               {latency_mean:.2f} +- {latency_std:.2f}")
        print(f"frame drops:                       {self.frame_drops} ({frame_drops_ratio * 100:.2f}%)")
        print('#' * 80)


    def enable_filedump(self, input_stream_file: str = None, output_stream_file: str = None):
        """
        dumps input/output audio to file if input/output filenames are specified
        call this function before run()
        :param input_stream_file:   name of the file to dump input audio to
        :param output_stream_file:  name of the file to dump output audio to
        at least one of the files needs to be specified
        """
        if input_stream_file is None and output_stream_file is None:
            raise Exception("At least one of input_stream_file and output_stream_file must be specified.")

        if input_stream_file is not None:
            if not input_stream_file[-4:] == ".wav":
                input_stream_file += ".wav"
            self.input_dump_filename = input_stream_file

        if output_stream_file is not None:
            if not output_stream_file[-4:] == ".wav":
                output_stream_file += ".wav"
            self.output_dump_filename = output_stream_file


    def run(self, latency):
        """
        start streaming from the input device and forward the processed audio to the output device
        prints statistics about mean processing time, standard deviation of each processing pass, and percentage of buffer underflows
        """

        # start encoder and decoder threads
        encoder_thread = threading.Thread(target=self._run_encoder, daemon=True)
        encoder_thread.start()
        decoder_thread = threading.Thread(target=self._run_decoder, daemon=True)
        decoder_thread.start()

        try:
            # import device
            import sounddevice as sd
            with sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.sample_rate,
                blocksize=self.frame_size,
                dtype=np.float32,
                latency=latency,
                channels=(self.input_channels, self.output_channels),
                callback=self._callback
            ):
                print('### starting stream [press Return to quit] ###')
                input()
                self._exit()
        except KeyboardInterrupt:
            self._exit()
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
