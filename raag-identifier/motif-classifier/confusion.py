"""Confusion matrices on the test split, one PNG per method.

50 classes over 92 test clips means the matrix is *sparse by construction* — at most 92 of
2500 cells are non-zero. So the design leans on that: a single-hue sequential ramp for the
counts (magnitude, one hue, light -> dark), empty cells left as bare surface, and the
diagonal outlined so correct predictions read at a glance rather than having to be traced
along the axis. Row-normalised shading would be misleading here (most rows hold one or two
clips), so cells carry raw counts.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

RESULTS = Path(__file__).resolve().parent / "results"

# Sequential ramp, one hue, light -> dark. Surface is the lightest step, so a zero cell is
# indistinguishable from the page and only real predictions carry ink.
SURFACE = "#FFFFFF"
RAMP = ["#FFFFFF", "#D6E4F5", "#9CC0E8", "#5B93D4", "#2A6AB8", "#123F73"]
INK = "#1A1D21"
MUTED = "#6B7280"
GRID = "#E5E7EB"
DIAG = "#B4531F"  # correct-prediction outline; warm, so it never reads as another count


def plot_confusion(rows, labels, title, subtitle, out_path, annotate=True):
    idx = {r: i for i, r in enumerate(labels)}
    n = len(labels)
    m = np.zeros((n, n), dtype=int)
    for r in rows:
        if r["true"] in idx and r["pred"] in idx:
            m[idx[r["true"]], idx[r["pred"]]] += 1

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    vmax = max(m.max(), 1)

    fig, ax = plt.subplots(figsize=(15, 15.6), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    # recessive cell grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for i in range(n):  # outline the diagonal: identity, not magnitude
        ax.add_patch(
            Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=DIAG,
                      linewidth=1.4 if m[i, i] else 0.7, alpha=1.0 if m[i, i] else 0.35)
        )

    if annotate:
        for i in range(n):
            for j in range(n):
                if m[i, j]:
                    ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=6.5,
                            color=SURFACE if m[i, j] > vmax * 0.55 else INK)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=MUTED)
    ax.set_yticklabels(labels, fontsize=7, color=MUTED)
    ax.set_xlabel("predicted", fontsize=10, color=MUTED, labelpad=10)
    ax.set_ylabel("true", fontsize=10, color=MUTED, labelpad=10)
    for s in ax.spines.values():
        s.set_color(GRID)

    fig.suptitle(title, fontsize=14, color=INK, x=0.5, y=0.965, ha="center")
    ax.set_title(subtitle, fontsize=9.5, color=MUTED, pad=14)
    fig.tight_layout(rect=(0.02, 0.045, 1, 0.955))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
