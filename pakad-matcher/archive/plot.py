"""Contour plots of candidates (aligned path drawn on the pitch track) and audio snippets."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import _bootstrap  # noqa: F401
import config as C
from utils import raagdb

SWAR = raagdb.SWAR_NAMES
S = C.STYLE
plt.rcParams.update({"font.family": S["font"], "font.size": 8, "axes.edgecolor": S["grid"],
                     "xtick.color": S["ink2"], "ytick.color": S["ink2"],
                     "figure.facecolor": S["surface"], "axes.facecolor": S["surface"]})


def _swar_label(y):
    s, o = int(round(y / 100)) % 12, int(np.floor(round(y / 100) / 12))
    return {-1: ",", 0: "", 1: "`"}.get(o, "") + SWAR[s]


def plot_candidate(ax, ctr, cand, phrase, scale):
    pad = int(C.PLOT_PAD_S / ctr.hop)
    a, b = max(0, cand.f0 - pad), min(len(ctr.cents), cand.f1 + pad + 1)
    t, c = ctr.times[a:b], ctr.cents[a:b].copy()
    # fold everything so the candidate sits near the middle octave
    core_med = np.nanmedian(ctr.cents[cand.f0:cand.f1 + 1])
    c = c - 1200 * np.round((core_med - 600) / 1200)
    seg = slice(cand.f0 - a, cand.f1 - a + 1)
    lo, hi = np.nanmin(c[seg]) - 250, np.nanmax(c[seg]) + 250

    ticks = [y for y in range(int(lo // 100) * 100, int(hi) + 1, 100) if int(round(y / 100)) % 12 in scale]
    ax.set_yticks(ticks, [_swar_label(y) for y in ticks])
    ax.grid(axis="y", color=S["grid"], lw=0.8)
    ax.tick_params(length=0, pad=4)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)

    ax.axvspan(cand.t0, cand.t1, color=S["band"], alpha=S["band_alpha"], lw=0)
    ax.plot(t, c, color=S["context"], lw=1.2)
    ts, cs, kinds = t[seg], c[seg], cand.path
    names = phrase.text.split()
    for k in np.unique(kinds):
        m = np.where(kinds == k, cs, np.nan)
        ax.plot(ts, m, color=S["orn"] if k < 0 else S["note"], lw=2.4 if k >= 0 else 1.6,
                solid_capstyle="round")
    # direct-label each note at its segment's midpoint
    for k in range(len(names)):
        idx = np.flatnonzero(kinds == k)
        if len(idx):
            mid = idx[len(idx) // 2]
            y = np.nanmax(cs[idx]) if not np.isnan(cs[idx]).all() else np.nan
            ax.annotate(names[k], (ts[mid], y), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8, color=S["ink"], fontweight="bold")
    ax.set_ylim(lo, hi)
    ax.set_xlim(t[0], t[-1])
    ax.set_title(f"{ctr.clip_id.split('/')[-1].removesuffix('.mp3')}   {cand.t0:.1f}–{cand.t1:.1f} s",
                 loc="left", fontsize=8.5, color=S["ink"])
    ax.set_title(f"cost {cand.cost:.2f}   pitch {cand.pitch_cost:.2f}   ornament {cand.orn_frac:.2f}",
                 loc="right", fontsize=8, color=S["ink2"])


def grid(items, phrase, scale, path):
    """items: [(contour, candidate)] -> one PNG, one row per candidate."""
    fig, axes = plt.subplots(len(items), 1, figsize=(S["width"], S["row_h"] * len(items) + 0.6),
                             squeeze=False)
    for ax, (ctr, cand) in zip(axes[:, 0], items):
        plot_candidate(ax, ctr, cand, phrase, scale)
    axes[-1, 0].set_xlabel("time (s)", color=S["ink2"])
    fig.suptitle(f"{phrase.id}   {phrase.text}", x=0.01, ha="left", fontsize=11, color=S["ink"],
                 fontweight="bold")
    fig.legend(handles=[Line2D([], [], color=S["note"], lw=2.4, label="phrase note"),
                        Line2D([], [], color=S["orn"], lw=1.6, label="ornament / transit"),
                        Line2D([], [], color=S["context"], lw=1.2, label="context")],
               loc="upper right", ncol=3, frameon=False, fontsize=8, labelcolor=S["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()), h_pad=1.2)
    fig.savefig(path, dpi=S["dpi"])
    plt.close(fig)


def snippet(clip, t0, t1, path, pad=C.PLOT_PAD_S):
    import librosa
    import soundfile as sf
    y, sr = librosa.load(clip.path, sr=None, mono=True, offset=max(0, t0 - pad), duration=t1 - t0 + 2 * pad)
    sf.write(path, y, sr)


def s2_summary(rows, path, focus):
    """Two panels, shared phrase order: AUC vs legal raags | AUC vs shuffles. Gate + chance lines."""
    key = lambda r: -1.0 if np.isnan(float(r["auc_legal"])) else float(r["auc_legal"])
    rows = sorted([r for r in rows if r["raag"] in focus], key=key)
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(S["width"], 0.26 * len(rows) + 1.3), sharey=True)
    panels = [("auc_legal", "own raag vs raags where the phrase is playable", C.S2_GATE_AUC_NULL),
              ("auc_shuffle", "phrase vs its re-orderings (own raag)", C.S2_GATE_AUC_SHUFFLE)]
    for ax, (key, title, gate) in zip(axes, panels):
        x = np.array([float(r[key]) for r in rows])
        ok = np.array([r["passes"] == "True" for r in rows])
        ax.axvline(0.5, color=S["context"], lw=1)
        ax.axvline(gate, color=S["ink2"], lw=1, ls=(0, (1, 2)))
        ax.text(gate, len(rows) - 0.3, f"gate {gate}", fontsize=7, color=S["ink2"], ha="left", va="bottom")
        ax.hlines(y, 0.5, x, color=S["grid"], lw=1.5)
        ax.scatter(x[~ok], y[~ok], s=36, color=S["context"], zorder=3, edgecolor=S["surface"], lw=1.5)
        ax.scatter(x[ok], y[ok], s=36, color=S["note"], zorder=3, edgecolor=S["surface"], lw=1.5)
        for yi in y[np.isnan(x)]:
            ax.text(0.505, yi, "no other raag has these swars", fontsize=7, color=S["ink2"], va="center",
                    bbox=dict(fc=S["surface"], ec="none", pad=0.5))
        ax.set_xlim(0.3, 1.0)
        ax.set_title(title, loc="left", fontsize=8.5, color=S["ink"])
        ax.grid(axis="x", color=S["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("AUC  (0.5 = no signal)", color=S["ink2"])
    axes[0].set_yticks(y, [f"{r['phrase_id']}  {r['phrase']}" for r in rows])
    for lab, r in zip(axes[0].get_yticklabels(), rows):
        lab.set_fontweight("bold" if r["passes"] == "True" else "normal")
        lab.set_color(S["ink"] if r["passes"] == "True" else S["ink2"])
    fig.suptitle("S2 — does the matcher find the phrase, or just the raag's swars?", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=S["ink"])
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=S["note"], label="passes both gates"),
                        Line2D([], [], marker="o", ls="", color=S["context"], label="fails")],
               loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=S["ink2"])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    fig.savefig(path, dpi=S["dpi"])
    plt.close(fig)


def s2_overview(rows, path, focus):
    """Every kept phrase: AUC vs legal raags (x) against AUC vs shuffles (y); gates as a quadrant."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    x = np.array([float(r["auc_legal"]) for r in rows])
    yv = np.array([float(r["auc_shuffle"]) for r in rows])
    f = np.array([r["raag"] in focus for r in rows])
    ax.axvline(C.S2_GATE_AUC_NULL, color=S["ink2"], lw=1, ls=(0, (1, 2)))
    ax.axhline(C.S2_GATE_AUC_SHUFFLE, color=S["ink2"], lw=1, ls=(0, (1, 2)))
    ax.scatter(x[~f], yv[~f], s=30, color=S["context"], edgecolor=S["surface"], lw=1.5, label="other raags")
    ax.scatter(x[f], yv[f], s=36, color=S["note"], edgecolor=S["surface"], lw=1.5, label="focus raags")
    ax.text(0.99, 0.99, "passes", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=S["ink2"])
    ax.set_xlabel("AUC: own raag vs raags where playable", color=S["ink2"])
    ax.set_ylabel("AUC: phrase vs its re-orderings", color=S["ink2"])
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.grid(color=S["grid"], lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"All {len(rows)} kept phrases", loc="left", fontsize=10, fontweight="bold", color=S["ink"])
    ax.legend(frameon=False, fontsize=8, labelcolor=S["ink2"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=S["dpi"])
    plt.close(fig)
