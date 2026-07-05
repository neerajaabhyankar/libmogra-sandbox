"""Plotting for relative-pitch note trajectories. Nothing here writes to disk."""

import matplotlib.pyplot as plt

SWARA_LABELS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]


def _swara_label(cents):
    semitone = int(round(cents / 100))
    octave = semitone // 12
    idx = semitone % 12
    return SWARA_LABELS[idx] if octave == 0 else f"{SWARA_LABELS[idx]}({octave:+d})"


def _draw_notes(ax, notes, key, color):
    prev_end_t, prev_end_y = None, None
    for n in notes:
        y = key(n)
        ax.plot([n.t_start, n.t_end], [y, y], color=color, linewidth=2)
        if prev_end_t is not None and prev_end_t == n.t_start:
            ax.plot([prev_end_t, n.t_start], [prev_end_y, y], color=color, linewidth=1, alpha=0.5)
        prev_end_t, prev_end_y = n.t_end, y


def plot_relative_pitch(notes, title=None, tonic_hz=None, show=True):
    """notes: list of note_segmentation.Note.

    Two stacked panels with a shared x-axis, both unfolded (no octave wrapping):
      top    — continuous pitch in cents relative to tonic
      bottom — pitch snapped to nearest 100-cent semitone (swara), still unfolded
    """
    if notes:
        all_cents = [n.cents_relative for n in notes]
        y_min, y_max = min(all_cents), max(all_cents)
    else:
        y_min, y_max = -200, 1400

    y_lo = (int(y_min // 100) - 1) * 100
    y_hi = (int(y_max // 100) + 2) * 100
    grid_ticks = list(range(y_lo, y_hi + 100, 100))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax in (ax_top, ax_bot):
        for c in grid_ticks:
            ax.axhline(c, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(grid_ticks)
        ax.set_yticklabels([_swara_label(c) for c in grid_ticks])

    _draw_notes(ax_top, notes, key=lambda n: n.cents_relative, color="C0")
    ax_top.set_ylabel("pitch (cents vs tonic)")

    _draw_notes(ax_bot, notes, key=lambda n: round(n.cents_relative / 100) * 100, color="C1")
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("swara (unfolded)")

    suffix = f"  [tonic = {tonic_hz:.1f} Hz]" if tonic_hz else ""
    ax_top.set_title((title or "Relative pitch trajectory") + suffix)
    fig.tight_layout()

    if show:
        plt.show()
    return fig
