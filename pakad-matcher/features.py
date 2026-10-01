"""Features for a candidate, built from what the annotations say actually decides a verdict.

The matcher's own cost turned out to rank Neeraja's "yes" above her "no" at chance within a
phrase (mean per-phrase AUC ~0.50 over 168 labels). Her comments say why, and each of these
features is one of her reasons:

    "the S is coming from the drone", "P from tanpura not voice"   -> salience_rel, salience_min
    "too fast to be counted, given the context"                    -> tempo_ratio
    "this is n S g m", "it's ,N r G M d P"                         -> held_extra
    "this is speech!", "audience laughing"                         -> salience_rel
"""

import numpy as np

import _bootstrap  # noqa: F401
import config as C
import fullaudio
import matcher


def _local(arr, f0, f1, hop):
    lo = max(0, f0 - int(C.LOCAL_CONTEXT_S / hop))
    hi = min(len(arr), f1 + int(C.LOCAL_CONTEXT_S / hop))
    return arr[lo:hi]


def candidate_features(video, cand, swars, octaves):
    """cand: a matcher.Candidate on this video's contour."""
    ctr = fullaudio.contour(video)
    sal = fullaudio.salience(video)
    n = min(len(ctr.cents), len(sal))
    seg = slice(cand.f0, cand.f1 + 1)
    cents, path = ctr.cents[seg], cand.path
    voiced = ~np.isnan(cents)
    note_frames = path >= 0

    # --- salience: is this the voice, or the drone / a stray / the audience?
    local_sal = _local(sal[:n], cand.f0, cand.f1, ctr.hop)
    ref = np.median(local_sal[local_sal > 0]) if (local_sal > 0).any() else 1.0
    cand_sal = sal[seg][: len(path)]
    sal_rel = float(np.mean(cand_sal[note_frames & voiced]) / ref) if (note_frames & voiced).any() else 0.0
    per_note = [float(np.mean(cand_sal[(path == k) & voiced]) / ref)
                for k in range(len(swars)) if ((path == k) & voiced).any()]
    sal_min = float(min(per_note)) if per_note else 0.0

    # --- tempo: fast is fine in a taan, not in an alap. Compare with the neighbourhood.
    local_cents = _local(ctr.cents, cand.f0, cand.f1, ctr.hop)
    held = matcher._held(local_cents, ctr.hop, C.MATCH)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[False, held, False]))[::2]) if held.any() else []
    starts = np.flatnonzero(~held[:-1] & held[1:]) + 1
    ends = np.flatnonzero(held[:-1] & ~held[1:]) + 1
    lens = (ends[:len(starts)] - starts[:len(ends)]) * ctr.hop if len(starts) and len(ends) else []
    local_note_s = float(np.median(lens)) if len(lens) else 0.25
    per_note_s = (cand.t1 - cand.t0) / max(1, len(swars))
    tempo_ratio = float(per_note_s / local_note_s) if local_note_s > 0 else 1.0

    # --- a held note inside the match that the phrase does not account for
    held_c = matcher._held(cents, ctr.hop, C.MATCH)
    orn = path == -1
    held_extra = int(np.sum((~held_c[:-1] & held_c[1:]) & orn[1:]))

    return dict(
        pitch_cost=cand.pitch_cost, orn_frac=cand.orn_frac,
        gap_frac=float(np.mean(~voiced)), leaps=float(cand.leaps),
        register=float(matcher._register_error(cents, path, swars, octaves)) if octaves else 0.0,
        salience_rel=sal_rel, salience_min=sal_min,
        tempo_ratio=float(np.clip(tempo_ratio, 0, 5)), held_extra=float(held_extra),
    )
