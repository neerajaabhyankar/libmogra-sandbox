import json
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import librosa
from IPython.display import Audio as ipy_audio


def read_pitch(pitch_file, verbose=False) -> (List[List[float]], float):
    """Annotated data"""
    pitch_annotations = []
    with open(pitch_file) as pf:
        for line in pf:
            pitch_annotations.append(line.split())
    pitch_annotations = np.array(pitch_annotations).astype(float)

    ar = 1 / np.mean(np.diff(pitch_annotations[:, 0]))
    print(f"{ar} annotations per second")

    return pitch_annotations, ar


def infer_pitch(audio_file) -> List[List[float]]:
    """Return a list of [times, hz]"""
    raise NotImplemented


class PitchValidator:
    def __init__(self, audio_array: List[float], sampling_rate: float) -> None:
        self.sr = sampling_rate
        self.y = audio_array
        self.pitch_annotations = None
        self.tonic = None

    def play_sample(self, start_time, end_time):
        if start_time > end_time:
            print("start_time > end_time")
            return None
        y_sample = self.y[start_time * self.sr : end_time * self.sr]
        print(f"audio sample from {start_time}-{end_time}")
        return ipy_audio(data=y_sample, rate=self.sr)

    def set_annotation(
        self, pitch_annotations: List[List[float]], annotation_rate: float
    ) -> None:
        self.ar = int(annotation_rate)
        self.pitch_annotations = pitch_annotations
        if self.tonic is not None:
            print("pitch annotations have been normalized")
            self.normalized = True
            self.pitch_annotations = self.pitch_annotations / self.tonic

    def set_tonic(self, tonic: float):
        self.tonic = tonic
        if self.pitch_annotations is not None:
            print("pitch annotations have been normalized")
            self.normalized = True
            self.pitch_annotations = self.pitch_annotations / self.tonic

    def validate_annotations(self, start_time, end_time, downsample_factor=25):
        """Generate an audio clip and play it with the OG"""
        if self.pitch_annotations is None:
            print("please set pitch annotations first")
            return None

        if start_time > end_time:
            print("start_time > end_time")
            return None

        p_sample = self.pitch_annotations[start_time * self.ar : end_time * self.ar][
            :, 1
        ]
        # downsample so as to make clearer tones
        p_sample_d = p_sample[::downsample_factor]

        # create a waveform
        y_from_p_sample_d = []
        for anno in p_sample_d:
            tone = librosa.tone(
                2 * anno, sr=self.sr, length=downsample_factor * self.sr / self.ar
            )
            y_from_p_sample_d += tone.tolist()

        print(f"pitch annotations' sample from {start_time}-{end_time}")
        return ipy_audio(data=y_from_p_sample_d, rate=self.sr)

    def plot_annotations_hist(self):
        """Plot an interactive time-histogram
        TODO: Also do a matra-normed-histogram
        """
        if self.pitch_annotations is None:
            print("please set pitch annotations first")
            return None

        nz_annotations = self.pitch_annotations[
            np.where(self.pitch_annotations[:, 1] != 0), 1
        ][0]
        h = np.histogram(nz_annotations, bins=700)
        counts = h[0]
        fro = (h[1][:-1] + h[1][1:]) / 2

        plt.figure()
        if not self.normalized:
            print("please normalize frequencies by setting the tonic")

        fig = px.line(
            pd.DataFrame({"fratio": fro, "count": counts}),
            x="fratio",
            y="count",
            log_x=True,
            width=600,
            height=200,
        )
        fig.update_xaxes(range=[np.log10(0.6), np.log10(2.6)], type="log")
        fig.update_xaxes(minor=dict(showgrid=True, nticks=10))
        fig.update_traces(
            hovertemplate="frequency(ratio wrt root): %{x}<br>occurence: %{y}"
        )
        fig.update_layout(
            margin=dict(l=10, r=20, t=30, b=10),
        )

        fig.show()
