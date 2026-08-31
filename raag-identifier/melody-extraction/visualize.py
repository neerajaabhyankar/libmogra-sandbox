"""Plotting for relative-pitch note trajectories. Nothing here writes to disk."""

import matplotlib.pyplot as plt

SWARA_LABELS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]

# Stable per-tracker colours, so a method keeps its colour across figures.
METHOD_COLORS = {"pyin": "C0", "crepe": "C1", "praat": "C2", "tony": "C3"}
_FALLBACK_COLORS = ["C4", "C5", "C6", "C7", "C8", "C9"]

QUANTIZED_ALPHA = 0.35

# Plots never scale beyond this window. Octave-error strays down at M(-2) would
# otherwise stretch the y-axis across five octaves and squash the actual melody
# into a band a few pixels tall. Within the window the scale still auto-fits the
# notes, so a narrow melody stays zoomed in.
PLOT_RANGE_CENTS = (-800, 2100)  # G(-1) .. D(+1)


def _swara_label(cents):
    semitone = int(round(cents / 100))
    octave = semitone // 12
    idx = semitone % 12
    return SWARA_LABELS[idx] if octave == 0 else f"{SWARA_LABELS[idx]}({octave:+d})"


def _color_for(label, index):
    key = (label or "").lower()
    for method, color in METHOD_COLORS.items():
        if method in key:
            return color
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def _draw_notes(ax, notes, key, color, label, alpha=1.0, linewidth=2):
    """One horizontal dash per note, faint risers across note boundaries."""
    prev_end_t, prev_end_y = None, None
    for n in notes:
        y = key(n)
        ax.plot([n.t_start, n.t_end], [y, y], color=color, linewidth=linewidth,
                alpha=alpha, label=label, solid_capstyle="butt")
        label = None  # only the first segment goes in the legend
        if prev_end_t is not None and prev_end_t == n.t_start:
            ax.plot([prev_end_t, n.t_start], [prev_end_y, y], color=color,
                    linewidth=1, alpha=alpha * 0.5)
        prev_end_t, prev_end_y = n.t_end, y


def _pitch_range(all_notes):
    """(y_lo, y_hi) padded out to whole semitones, clamped to PLOT_RANGE_CENTS."""
    clip_lo, clip_hi = PLOT_RANGE_CENTS
    cents = [n.cents_relative for notes in all_notes for n in notes
             if clip_lo <= n.cents_relative <= clip_hi]
    if not cents:
        return clip_lo, clip_hi  # nothing in range; show the whole window
    y_lo = max((int(min(cents) // 100) - 1) * 100, clip_lo)
    y_hi = min((int(max(cents) // 100) + 2) * 100, clip_hi)
    return y_lo, y_hi


def _yticks(y_lo, y_hi):
    """Label every semitone when that fits; thin out to Sa/Ma or Sa when it doesn't.

    A stray octave-error note can stretch the range across five octaves, at which
    point a tick per semitone is 60 unreadable labels rather than a useful grid.
    """
    span = y_hi - y_lo
    step = 100 if span <= 3000 else 600 if span <= 6000 else 1200
    first = -(-y_lo // step) * step  # round up to the first multiple of step
    return list(range(int(first), y_hi + 1, step))


def draw_relative_pitch(ax, notes, title=None, tonic_hz=None, color="C0",
                        label=None, pitch_range=None, xlabel=True, legend=True):
    """Draw one tracker's trajectory into a caller-supplied axis.

    Both readings of the same notes are overlaid on the one axis:
      solid, full-strength — each note's raw mean pitch in cents vs the tonic
      thick, translucent   — the same notes snapped to the nearest 100-cent semitone

    Neither is octave-folded, so S(+1) sits a full 1200 cents above S. Pass
    `pitch_range` from `_pitch_range()` over several trackers to share one y-scale.
    """
    y_lo, y_hi = pitch_range or _pitch_range([notes])
    ticks = _yticks(y_lo, y_hi)

    for c in ticks:
        ax.axhline(c, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)  # tonic
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks(ticks)
    ax.set_yticklabels([_swara_label(c) for c in ticks], fontsize="small")

    name = label or "pitch"
    _draw_notes(ax, notes, key=lambda n: round(n.cents_relative / 100) * 100,
                color=color, label=f"{name} — quantized", alpha=QUANTIZED_ALPHA, linewidth=6)
    _draw_notes(ax, notes, key=lambda n: n.cents_relative,
                color=color, label=f"{name} — raw", linewidth=2)

    ax.set_ylabel("swara / cents vs tonic")
    if xlabel:
        ax.set_xlabel("time (s)")

    hidden = sum(1 for n in notes if not y_lo <= n.cents_relative <= y_hi)
    if hidden:
        ax.text(0.995, 0.03, f"{hidden} note{'s' if hidden > 1 else ''} off-scale",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize="x-small", color="gray")
    if legend:
        ax.legend(loc="upper right", fontsize="small", framealpha=0.9)

    suffix = f"  [tonic = {tonic_hz:.1f} Hz]" if tonic_hz else ""
    if title or suffix:
        ax.set_title((title or "Relative pitch trajectory") + suffix)


def plot_relative_pitch(notes, title=None, tonic_hz=None, show=True):
    """notes: list of note_segmentation.Note. One tracker, one panel."""
    fig, ax = plt.subplots(figsize=(12, 5))
    draw_relative_pitch(ax, notes, title=title, tonic_hz=tonic_hz,
                        color=_color_for(title, 0), label=title or "pitch")
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def draw_relative_pitch_multi(axes, results, pitch_range=None):
    """Draw one tracker per axis, into caller-supplied axes.

    `axes` must be at least as long as `results`, ordered top-to-bottom. All panels
    share one y-scale (computed across every tracker unless `pitch_range` is given)
    so the trackers can be read off against each other. Split out from
    `plot_relative_pitch_multi` so a caller building a bigger figure — e.g.
    `freq_histogram.plot_relative_pitch_with_histograms` — can stack these panels
    above rows of its own.
    """
    if len(axes) < len(results):
        raise ValueError(f"need at least {len(results)} axes, got {len(axes)}")

    n = len(results)
    pitch_range = pitch_range or _pitch_range([notes for _label, notes in results])

    for i, (label, notes) in enumerate(results):
        draw_relative_pitch(
            axes[i], notes,
            title=None,  # the legend already names the tracker
            color=_color_for(label, i),
            label=label,
            pitch_range=pitch_range,
            xlabel=(i == n - 1),
        )
    return pitch_range


def plot_relative_pitch_multi(results, tonic_hz=None, show=True):
    """results: list of (label, notes) — one panel per tracker, stacked top-to-bottom
    in the order given, sharing one x-axis and one y-scale so the trackers can be
    read off against each other.
    """
    if not results:
        raise ValueError("plot_relative_pitch_multi() needs at least one (label, notes) pair")

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.6 * n), sharex=True, sharey=True,
                             squeeze=False)
    draw_relative_pitch_multi(axes[:, 0], results)

    suffix = f"  [tonic = {tonic_hz:.1f} Hz]" if tonic_hz else ""
    fig.suptitle("Relative pitch trajectory" + suffix)
    fig.tight_layout()

    if show:
        plt.show()
    return fig
