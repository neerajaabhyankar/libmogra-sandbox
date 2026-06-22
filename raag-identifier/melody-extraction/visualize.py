"""Plotting for relative-pitch note trajectories. Nothing here writes to disk."""

import matplotlib.pyplot as plt

SWARA_LABELS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]


def plot_relative_pitch(notes, title=None, use_chroma=False, tonic_hz=None, show=True):
    """notes: list of note_segmentation.Note.

    Draws a piano-roll-style step plot: each note as a horizontal segment at its
    pitch, connected vertically to its temporal neighbor, with gaps left blank
    wherever the underlying audio was unvoiced (i.e., no note covers that time).
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    for i in range(12):
        ax.axhline(i * 100.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    prev_end_t, prev_end_y = None, None
    for n in notes:
        y = n.chroma_cents if use_chroma else n.cents_relative
        ax.plot([n.t_start, n.t_end], [y, y], color="C0", linewidth=2)
        if prev_end_t is not None and prev_end_t == n.t_start:
            ax.plot([prev_end_t, n.t_start], [prev_end_y, y], color="C0", linewidth=1, alpha=0.5)
        prev_end_t, prev_end_y = n.t_end, y

    ax.set_xlabel("time (s)")
    ax.set_ylabel("chroma (cents, mod 1200)" if use_chroma else "relative pitch (cents vs tonic)")
    if use_chroma:
        ax.set_ylim(0, 1200)
        ax.set_yticks([i * 100 for i in range(12)])
        ax.set_yticklabels(SWARA_LABELS)
    suffix = f" (tonic={tonic_hz:.1f} Hz)" if tonic_hz else ""
    ax.set_title((title or "Relative pitch trajectory") + suffix)
    fig.tight_layout()

    if show:
        plt.show()
    return fig
