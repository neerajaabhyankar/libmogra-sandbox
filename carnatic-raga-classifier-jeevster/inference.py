import logging
import torch
import json
import random
import numpy as np
from dataloader import extract_data
import torchaudio
from models import (
    BaseRagaClassifier,
    ResNetRagaClassifier,
    Wav2VecTransformer,
    count_parameters,
)
from collections import OrderedDict


NUM_CLASSES = 150


np.random.seed(123)
random.seed(123)


class Evaluator:

    def __init__(self, params):
        self.params = params
        self.device = self.params.device
        # get raga to label mapping
        with open("raga2label0.json", "r") as fp:
            self.raga2label = json.load(fp)
        # _, self.raga2label = extract_data(self.params)
        self.raga_list = list(self.raga2label.keys())
        self.label_list = list(self.raga2label.values())

        # initialize model
        if params.model == "base":
            self.model = BaseRagaClassifier(params).to(self.device)
        elif params.model == "resnet":
            self.model = ResNetRagaClassifier(params).to(self.device)
        elif params.model == "wav2vec":
            self.model = Wav2VecTransformer(params).to(self.device)
        else:
            logging.error("Model must be either 'base', 'resnet', or 'wav2vec'")

        # load best model
        self.restore_checkpoint("ckpts/best_ckpt.tar")
        self.model.eval()

    def normalize(self, audio):
        return (audio - torch.mean(audio, dim=1, keepdim=True)) / (
            torch.std(audio, dim=1, keepdim=True) + 1e-5
        )

    def pad_audio(self, audio, sample_rate, clip_length):
        pad = (0, sample_rate * clip_length - audio.shape[1])
        return torch.nn.functional.pad(audio, pad=pad, value=0)

    def inference(self, k, audio):
        # open audio file
        audio_clip, sample_rate = audio

        # repeat mono channel to get stereo if necessary
        if len(audio_clip.shape) == 1:
            audio_clip = (
                torch.tensor(audio_clip).unsqueeze(0).repeat(2, 1).to(torch.float32)
            )
        else:
            audio_clip = torch.tensor(audio_clip).T.to(torch.float32)

        # resample audio clip
        resample = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=self.params.sample_rate
        )
        audio_clip = resample(audio_clip)

        # normalize the audio clip
        audio_clip = self.normalize(audio_clip)

        # pad audio with zeros if it's not long enough
        if audio_clip.size()[1] < self.params.sample_rate * self.params.clip_length:
            audio_clip = self.pad_audio(audio_clip, self.params.sample_rate, self.params.clip_lenght)

        assert not torch.any(torch.isnan(audio_clip))
        audio_clip = audio_clip.to(self.device)

        with torch.no_grad():
            length = audio_clip.shape[1]
            train_length = self.params.sample_rate * self.params.clip_length

            pred_probs = torch.zeros((NUM_CLASSES,)).to(self.device)

            # loop over clip_length segments and perform inference
            num_clips = int(np.floor(length / train_length))
            for i in range(num_clips):

                clip = audio_clip[
                    :, i * train_length : (i + 1) * train_length
                ].unsqueeze(0)

                # perform forward pass through model
                pred_distribution = self.model(clip).reshape(-1, NUM_CLASSES)
                pred_probs += (
                    1
                    / num_clips
                    * (
                        torch.exp(pred_distribution)
                        / torch.exp(pred_distribution).sum(axis=1, keepdim=True)
                    )[0]
                )

        pred_probs, labels = pred_probs.sort(descending=True)
        pred_probs_topk = pred_probs[:k]
        pred_ragas_topk = [
            self.raga_list[self.label_list.index(label)] for label in labels[:k]
        ]
        d = dict(zip(pred_ragas_topk, pred_probs_topk))
        return {k: v.item() for k, v in d.items()}

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except:
            # loading DDP checkpoint into non-DDP model
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model_state"].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            self.model.load_state_dict(new_state_dict)

        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]
