"""torch Datasets. One per input shape, all reading through `common.audio.clip_tensor`.

Each item is a dict, so a model can take what it needs and ignore the rest:

    input_values   the audio or spectrogram, already the right shape for its architecture
    labels         int class id
    tonic          (15,) the tonic conditioning vector -- present always, used only by
                   models built with `tonic_mode="condition"`
    index          position in the clip list, so predictions can be matched back to clips
    side           (d,) a per-clip vector computed outside the network -- present only when
                   the dataset was given one (Stage 5's melody histogram)

The tonic appears in an item in *two* ways and they are independent: `input_values` may have
been pitch-normalised (`tonic="normalise"`), and `tonic` may be read by the model (FiLM).
Running both at once is a valid experiment; running neither is the baseline.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from . import audio
from . import tonic as tonic_mod


class _Base(Dataset):
    def __init__(self, clips, tonic="none", separate=None, seconds=audio.DEFAULT_SECONDS,
                 length_policy="fixed", tonic_override=None, train=False, side=None):
        """tonic_override : {video: tonic_hz}, for the shuffled-tonic control.
        side            : {clip_id: vector}, attached to every item as "side"."""
        self.clips = list(clips)
        self.tonic = tonic
        self.separate = separate
        self.seconds = seconds
        self.length_policy = length_policy
        self.tonic_override = tonic_override or {}
        self.train = train
        self.side = side

    def __len__(self):
        return len(self.clips)

    def tonic_hz(self, clip):
        return self.tonic_override.get(clip.video, clip.tonic_hz)

    def _wrap(self, x, i):
        c = self.clips[i]
        item = {
            "input_values": x,
            "labels": torch.tensor(c.label, dtype=torch.long),
            "tonic": torch.from_numpy(tonic_mod.conditioning(self.tonic_hz(c))),
            "index": torch.tensor(i, dtype=torch.long),
        }
        if self.side is not None:
            item["side"] = torch.from_numpy(np.asarray(self.side[c.clip_id],
                                                       dtype=np.float32))
        return item


class WaveformDataset(_Base):
    """1-D audio for distilHuBERT (`channels=1`) or the jeevster ResNet (`channels=2`).

    Normalisation differs by architecture and is not cosmetic: HuBERT's feature extractor
    expects zero-mean unit-variance audio, and jeevster's ResNet was trained on
    per-channel-standardised stereo. Getting this wrong costs several points and looks like
    a modelling failure.
    """

    def __init__(self, clips, sr, channels=1, gain_jitter_db=0.0, **kw):
        super().__init__(clips, **kw)
        self.sr = sr
        self.channels = channels
        self.gain_jitter_db = gain_jitter_db

    def __getitem__(self, i):
        c = self.clips[i]
        y = audio.clip_tensor(c, self.sr, tonic=self.tonic, separate=self.separate,
                              seconds=self.seconds, length_policy=self.length_policy,
                              tonic_hz=self.tonic_hz(c))
        if self.train and self.gain_jitter_db:
            y = y * float(10.0 ** (np.random.uniform(-1, 1) * self.gain_jitter_db / 20.0))
        x = torch.from_numpy(np.ascontiguousarray(y))[None]
        if self.channels > 1:
            x = x.repeat(self.channels, 1)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)
        if self.channels == 1:
            x = x.squeeze(0)          # HuBERT wants (T,), not (1, T)
        return self._wrap(x, i)


class CQTDataset(_Base):
    """(1, n_bins, n_frames) log-CQT for the C architecture.

    `tonic="normalise"` here does not resample anything -- it selects the **Sa-anchored**
    CQT, whose fmin is the clip's own tonic, so bin 0 is Sa exactly. That is a different and
    strictly better mechanism than shifting a waveform: no resampling, no duration change,
    no interpolation error, and the invariance is structural rather than learned.
    """

    def __init__(self, clips, time_frames=None, freq_shift_bins=0, **kw):
        super().__init__(clips, **kw)
        self.time_frames = time_frames
        self.freq_shift_bins = freq_shift_bins
        if self.separate not in (None, "none") and self.tonic == "normalise":
            pass  # both cached variants exist; nothing special to do

    def __getitem__(self, i):
        c = self.clips[i]
        C = audio.cached_cqt(c, tonic="anchor" if self.tonic == "normalise" else "none",
                             separate=self.separate, tonic_hz=self.tonic_hz(c))
        if self.time_frames:
            C = C[:, :self.time_frames] if C.shape[1] >= self.time_frames else np.pad(
                C, ((0, 0), (0, self.time_frames - C.shape[1])), constant_values=C.min())
        if self.train and self.freq_shift_bins:
            # small pitch jitter, *below* a semitone at 36 bins/octave: it must not move a
            # swar to a different swar, only model tuning drift between performances
            C = np.roll(C, int(np.random.randint(-self.freq_shift_bins,
                                                 self.freq_shift_bins + 1)), axis=0)
        x = torch.from_numpy(np.ascontiguousarray((C + 80.0) / 80.0))[None].float()
        return self._wrap(x, i)


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def loader(dataset, batch_size, shuffle, num_workers=0):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate,
                      persistent_workers=bool(num_workers), drop_last=False)
