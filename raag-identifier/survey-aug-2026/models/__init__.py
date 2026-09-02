"""The three architectures, behind one registry so scripts and sweeps are arch-agnostic.

    model = models.build("cqt", tonic_mode="condition", aux_occupancy=True)
    groups = models.param_groups("cqt", model, lr=1e-3)

Each module exposes `build(...)` and `param_groups(model, ...)`; everything they have in
common (FiLM conditioning, the classifier head, the auxiliary occupancy head) lives in
`heads.py`, so an experiment applies to all three without being written three times.
"""

from . import cqtnet, hubert, resnet1d

ARCHS = {
    "hubert": hubert,      # D -- distilHuBERT, 16 kHz waveform
    "resnet1d": resnet1d,  # R -- jeevster Carnatic ResNet, 8 kHz stereo waveform
    "cqt": cqtnet,         # C -- 2-D ResNet over a Sa-anchored CQT
}


def build(arch, **kw):
    return ARCHS[arch].build(**kw)


def build_backbone(arch, **kw):
    """The feature extractor alone, for heads other than the default classifier
    (see `models.dbhead`, which scores against the libmogra templates instead)."""
    return ARCHS[arch].backbone(**kw)


def param_groups(arch, model, **kw):
    return ARCHS[arch].param_groups(model, **kw)
