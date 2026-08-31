"""Frame-level f0 -> discrete note events.

A "note" is a maximal run of voiced frames whose pitch stays within `tol_cents`
of the running segment mean. Segments shorter than `min_note_dur` (the 0.2s
floor the user asked for) get absorbed into whichever neighboring segment is
closer in pitch, so slow meend/gamak slides collapse into ~0.2s steps instead
of fragmenting into one segment per frame.
"""

from dataclasses import dataclass


@dataclass
class Note:
    t_start: float
    t_end: float
    cents_relative: float

    @property
    def chroma_cents(self) -> float:
        return self.cents_relative % 1200.0


def _raw_segments(cents, voiced_mask, hop_seconds, tol_cents):
    """Greedy stable-pitch-band segmentation within each contiguous voiced run."""
    segments = []  # list of [start_idx, end_idx_inclusive, [cents...]]
    cur = None
    for i, (c, v) in enumerate(zip(cents, voiced_mask)):
        if not v:
            cur = None
            continue
        if cur is None:
            cur = [i, i, [c]]
            segments.append(cur)
            continue
        running_mean = sum(cur[2]) / len(cur[2])
        if abs(c - running_mean) <= tol_cents and i == cur[1] + 1:
            cur[1] = i
            cur[2].append(c)
        else:
            cur = [i, i, [c]]
            segments.append(cur)
    return [
        {
            "t_start": s * hop_seconds,
            "t_end": (e + 1) * hop_seconds,
            "cents_relative": sum(vals) / len(vals),
        }
        for s, e, vals in segments
    ]


def _merge_short_segments(segments, min_note_dur):
    """Repeatedly fold segments shorter than min_note_dur into their nearest-pitch neighbor."""
    segments = list(segments)
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            dur = seg["t_end"] - seg["t_start"]
            if dur >= min_note_dur:
                continue
            prev_seg = segments[i - 1] if i > 0 else None
            next_seg = segments[i + 1] if i + 1 < len(segments) else None
            # Only merge with a temporally-adjacent segment (no gap in between).
            prev_adjacent = prev_seg is not None and prev_seg["t_end"] == seg["t_start"]
            next_adjacent = next_seg is not None and next_seg["t_start"] == seg["t_end"]
            if not prev_adjacent and not next_adjacent:
                continue  # isolated short voiced blip between unvoiced gaps; leave it
            candidates = []
            if prev_adjacent:
                candidates.append(("prev", i - 1, abs(prev_seg["cents_relative"] - seg["cents_relative"])))
            if next_adjacent:
                candidates.append(("next", i + 1, abs(next_seg["cents_relative"] - seg["cents_relative"])))
            which, j, _ = min(candidates, key=lambda t: t[2])
            target = segments[j]
            lo, hi = sorted([i, j])
            merged_dur_a = seg["t_end"] - seg["t_start"]
            merged_dur_b = target["t_end"] - target["t_start"]
            total = merged_dur_a + merged_dur_b
            merged_cents = (
                seg["cents_relative"] * merged_dur_a + target["cents_relative"] * merged_dur_b
            ) / total
            merged = {
                "t_start": min(seg["t_start"], target["t_start"]),
                "t_end": max(seg["t_end"], target["t_end"]),
                "cents_relative": merged_cents,
            }
            segments = segments[:lo] + [merged] + segments[hi + 1 :]
            changed = True
            break
    return segments


def segment_notes(cents, voiced_mask, hop_seconds, tol_cents=50.0, min_note_dur=0.2):
    """cents, voiced_mask: equal-length 1D sequences (np.ndarray or list).

    Returns a list of Note(t_start, t_end, cents_relative).
    """
    raw = _raw_segments(cents, voiced_mask, hop_seconds, tol_cents)
    merged = _merge_short_segments(raw, min_note_dur)
    return [Note(**seg) for seg in merged]
