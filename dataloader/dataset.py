#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Reference (https://github.com/kan-bayashi/ParallelWaveGAN/)

"""PyTorch compatible dataset modules."""

import os
import soundfile as sf
from torch.utils.data import Dataset
from dataloader.utils import find_files


class SingleDataset(Dataset):
    def __init__(
        self,
        files,
        query="*.wav",
        load_fn=sf.read,
        return_utt_id=False,
        subset_num=-1,
    ):
        self.return_utt_id = return_utt_id
        self.load_fn = load_fn
        self.subset_num = subset_num

        self.filenames = self._load_list(files, query)
        self.utt_ids = self._load_ids(self.filenames)


    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        data = self._data(idx)
                
        if self.return_utt_id:
            items = utt_id, data
        else:
            items = data

        return items


    def __len__(self):
        return len(self.filenames)
    
    
    def _read_list(self, listfile):
        filenames = []
        with open(listfile) as f:
            for line in f:
                line = line.strip()
                if len(line):
                    filenames.append(line)
        return filenames
    

    def _load_list(self, files, query):
        if isinstance(files, list):
            filenames = files
        else:
            if os.path.isdir(files):
                filenames = sorted(find_files(files, query))
            elif os.path.isfile(files):
                filenames = sorted(self._read_list(files))
            else:
                raise ValueError(f"{files} is not a list / existing folder or file!")
            
        if self.subset_num > 0:
            filenames = filenames[:self.subset_num]
        assert len(filenames) != 0, f"File list in empty!"
        return filenames
    
    
    def _load_ids(self, filenames):
        utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in filenames
        ]
        return utt_ids
    

    def _data(self, idx):
        return self._load_data(self.filenames[idx], self.load_fn)
    

    def _load_data(self, filename, load_fn):
        if load_fn == sf.read:
            data, _ = load_fn(filename, always_2d=True) # (T, C)
        else:
            data = load_fn(filename)
        return data


class MultiDataset(SingleDataset):
    def __init__(
        self,
        multi_files,
        queries,
        load_fns,
        return_utt_id=False,
        subset_num=-1,
    ):
        errmsg = f"multi_files({len(multi_files)}), queries({len(queries)}), and load_fns({len(load_fns)}) are length mismatched!"
        assert len(multi_files) == len(queries) == len(load_fns), errmsg
        super(MultiDataset, self).__init__(
            files=multi_files,
            query=queries,
            load_fn=load_fns,
            return_utt_id=return_utt_id,
            subset_num=subset_num,
        )
        self._check_length(self.filenames)
    

    def _load_list(self, multi_files, queries):
        multi_filenames = []
        if isinstance(multi_files, list):
            for files, query in zip(multi_files, queries):
                multi_filenames.append(super()._load_list(files, query))
        else:
            raise ValueError(f"{multi_files} should be a list!")
        
        return multi_filenames
    
    
    def _load_ids(self, multi_filenames):
        return super()._load_ids(multi_filenames[0])


    def _data(self, idx):
        filenames = [
            f[idx] for f in self.filenames
        ]
        data = []
        for filename, load_fn in zip(filenames, self.load_fn):
            data.append(self._load_data(filename, load_fn))
        return data
    

    def _check_length(self, multi_filenames):
        errmsg = f"Not all lists have the same number of files!"
        self.file_num = len(multi_filenames[0])
        assert all(len(x)==self.file_num for x in multi_filenames), errmsg
        
    
    def __len__(self):
        return self.file_num

