import logging
import glob
from multiprocessing.sharedctypes import Value
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import torchaudio
import json
import os
from math import floor
import sys
import time
import copy
import math
import logging


np.random.seed(123)
random.seed(123)


# load used subset of metadata
def extract_data(params):
    with open(params.metadata_labeled_path) as f:
        metadata_labeled = json.load(f)

    if params.use_unlabeled_data:
        with open(params.metadata_unlabeled_path) as f:
            metadata_unlabeled = json.load(f)
        n_unlabeled_samples = len(metadata_unlabeled)

    # shuffle data and removed unused files
    random.shuffle(metadata_labeled)
    keep = math.ceil(params.use_frac * len(metadata_labeled))
    metadata_labeled = metadata_labeled[0:keep]

    # construct raga label lookup table.
    raga2label = get_raga2label(params)

    # remove ragas unused ragas from metadata dictionary
    metadata_labeled_final = remove_unused_ragas(metadata_labeled, raga2label)

    return metadata_labeled_final, raga2label


def get_raga2label(params):
    with open(params.num_files_per_raga_path) as f:
        num_files_per_raga = json.load(f)
    raga2label = {}
    for i, raga in enumerate(num_files_per_raga.keys()):
        raga2label[raga] = (
            i  # assign every raga to a unique number from 0 to self.num_classes
        )
        if i == params.num_classes - 1:
            break
    return raga2label


def remove_unused_ragas(metadata_labeled, raga2label):
    temp = copy.deepcopy(metadata_labeled)
    for i, entry in enumerate(metadata_labeled):
        raga = entry["filename"].split("/")[0]
        if (
            raga not in raga2label.keys()
        ):  # this raga is not in the top self.params.num_classes ragas
            temp.remove(entry)

    return temp


class RagaDataset(Dataset):
    def __init__(self, params, metadata_labeled, raga2label):
        self.params = params
        self.metadata_labeled = metadata_labeled
        self.raga2label = raga2label
        self.n_labeled_samples = len(self.metadata_labeled)
        self.transform_dict = {}
        self.count = 0

        if params.local_rank == 0:
            print(
                "Begin training using ",
                self.__len__(),
                " audio samples of ",
                self.params.clip_length,
                " seconds each.",
            )
            print("Total number of ragas specified: ", self.params.num_classes)

    def construct_label(self, raga, label_smoothing=False):
        # construct one hot encoding vector for raga
        raga_index = self.raga2label[raga]
        label = torch.zeros((self.params.num_classes,), dtype=torch.float32)
        label[raga_index] = 1
        return label

    def normalize(self, audio):

        return (audio - torch.mean(audio, dim=1, keepdim=True)) / (
            torch.std(audio, dim=1, keepdim=True) + 1e-5
        )

    def pad_audio(self, audio):
        pad = (0, self.params.sample_rate * self.params.clip_length - audio.shape[1])
        return torch.nn.functional.pad(audio, pad=pad, value=0)

    def __len__(self):
        return len(self.metadata_labeled)

    def __getitem__(self, idx):
        # get metadata
        file_info = self.metadata_labeled[idx]

        # sample offset uniformly
        rng = max(0, file_info["duration"] - self.params.clip_length)
        if rng == 0:
            rng = file_info["duration"]

        seconds_offset = np.random.randint(floor(rng))

        # open audio file
        audio_clip, sample_rate = torchaudio.load(
            filepath=os.path.join(self.params.labeled_data_dir, file_info["filename"]),
            frame_offset=seconds_offset * file_info["sample_rate"],
            num_frames=self.params.clip_length * file_info["sample_rate"],
            normalize=True,
        )

        if audio_clip.shape[0] != 2:
            audio_clip = audio_clip.repeat(2, 1)

        # keep stereo
        # audio_clip = audio_clip.mean(dim=0, keepdim=True)
        # add transform to dictionary
        if sample_rate not in self.transform_dict.keys():
            self.transform_dict[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.params.sample_rate
            )

        # load cached transform
        resample = self.transform_dict[sample_rate]
        audio_clip = resample(audio_clip)

        if self.params.normalize:
            audio_clip = self.normalize(audio_clip)

        if audio_clip.size()[1] < self.params.sample_rate * self.params.clip_length:
            # pad audio with zeros if it's not long enough
            audio_clip = self.pad_audio(audio_clip)

        raga = file_info["filename"].split("/")[0]

        # construct label
        label = self.construct_label(raga)

        assert not torch.any(torch.isnan(audio_clip))
        assert not torch.any(torch.isnan(label))
        assert audio_clip.shape[1] == self.params.sample_rate * self.params.clip_length
        return audio_clip, label
