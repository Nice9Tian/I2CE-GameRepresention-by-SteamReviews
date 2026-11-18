# -*- coding: utf-8 -*-
"""Figure 1: what the sentences of Steam reviews are about, from the BERTopic topic table.

    python figures/paper_plots/fig1_composition.py [--data DIR] [--out DIR]

Reads the BERTopic topic table (data/fig1_bertopic_topics.md, the count of every topic) and the
126-topic to 7-category mapping (data/fig1_topic_buckets.json), computes each category's share of
the sentences BERTopic could assign (topic 0, the generic catch-all, and the -1 outliers are left
out of the denominator), and draws three files:

    fig1_composition_bars_col.png      the printed Figure 1: seven bars grouped by kind, one
                                       column of the two-column build (3.337 x 1.90 in)
    fig1_composition_stacked_col.png   the same shares as one stacked bar of three kinds, at the
                                       column width (3.337 x 0.40 in)
    fig1_composition_stacked.png       the stacked bar at the width of the single-column Word page

It prints the shares, which the caption quotes.  The topic table is a frozen intermediate: the
BERTopic fit read review text, which neither this repository nor the Hugging Face data holds
(README.md of this folder says more).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
COLUMN = 241.14749 / 72.27          # sigconf \columnwidth in inches
PT = 7.0                            # body is 9 pt; figure text is 2 pt under it
DPI = 300
GROUPS = ("content", "sentiment", "noncontent")
COLOR = {"content": "#1f4e79", "sentiment": "#e8971e", "noncontent": "#d9d9d9"}
EDGE = {"content": "#1f4e79", "sentiment": "#e8971e", "noncontent": "#404040"}
TEXTC = {"content": "white", "sentiment": "#1a1a1a", "noncontent": "#1a1a1a"}
NAME = {"content": "content-bearing", "sentiment": "sentiment / verdict",
        "noncontent": "other non-content"}


def load_shares(data):
    """(share per category, share per kind, labels, kinds), shares in percent"""
    counts = {}
    for line in (data / "fig1_bertopic_topics.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(-?\d+)\s*\|\s*(\d+)\s*\|", line)
        if m:
            counts[int(m.group(1))] = int(m.group(2))
    spec = json.loads((data / "fig1_topic_buckets.json").read_text(encoding="utf-8"))
    buckets, labels, kinds = spec["buckets"], spec["labels"], spec["kinds"]
    mapped = sorted(t for ts in buckets.values() for t in ts)
    assert len(mapped) == len(set(mapped)), "a topic appears in two buckets"
    assert mapped == sorted(t for t in counts if t not in (-1, 0)), "the mapping must cover topics 1..126"
    denom = sum(counts[t] for t in mapped)
    share = {k: 100.0 * sum(counts[t] for t in ts) / denom for k, ts in buckets.items()}
    gshare = {g: sum(s for k, s in share.items() if kinds[k] == g) for g in GROUPS}
    pshare, pgshare = printed_shares(share, gshare, kinds)
    print("denominator (assignable sentences): %d" % denom)
    for g in GROUPS:
        subs = ", ".join("%s %.1f%%" % (labels[k], pshare[k])
                         for k in sorted(share, key=share.get, reverse=True) if kinds[k] == g)
        print("  %-10s %5.1f%%   (%s)" % (g, pgshare[g], subs))
    return share, gshare, labels, kinds


def tenths(values, total):
    """Percentages rounded to tenths so that they add up to `total` tenths (largest remainder)."""
    floor = {k: int(v * 10) for k, v in values.items()}
    rest = sorted(values, key=lambda k: values[k] * 10 - floor[k], reverse=True)
    for k in rest[:total - sum(floor.values())]:
        floor[k] += 1
    return {k: n / 10 for k, n in floor.items()}


def printed_shares(share, gshare, kinds):
    """The shares as the figure prints them: the kinds add up to 100.0 and the categories of a
    kind to the kind's printed total (rounded one by one, 31.2 + 6.0 + 3.5 stood beside 40.6)."""
    pg = tenths(gshare, 1000)
    pk = {}
    for g in GROUPS:
        pk.update(tenths({k: s for k, s in share.items() if kinds[k] == g}, round(pg[g] * 10)))
    return pk, pg


def inch_axes(fig, width, height):
    """Axes covering the whole figure, with data units of one inch."""
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return ax


def stacked_col(gshare, out):
    """The three kinds as one stacked bar, one column wide."""
    width, height = COLUMN, 0.40
    bar_h, bar_top = 0.165, height - 0.045
    mid = bar_top - bar_h / 2
    pad = 0.012                      # half the outline, kept inside the figure
    fig = plt.figure(figsize=(width, height), dpi=DPI)
    ax = inch_axes(fig, width, height)
    left = pad
    scale = (width - 2 * pad) / 100.0
    printed = tenths(gshare, 1000)
    for g in GROUPS:
        w = gshare[g] * scale
        ax.add_patch(Rectangle((left, bar_top - bar_h), w, bar_h, facecolor=COLOR[g],
                               edgecolor=EDGE[g], linewidth=0.7))
        if g == "noncontent":
            # the dot texture of the Word figure, at the size it prints there:
            # three rows on a 0.04 in lattice, the middle row half a step off
            step = 0.0397
            xs = np.arange(left + 0.033, left + w - 0.026, step)
            for dy, phase in ((0.060, 0.0), (0.0, step / 2), (-0.060, 0.0)):
                x = xs + phase
                x = x[x < left + w - 0.026]
                ax.scatter(x, [mid + dy] * len(x), s=1.1, marker="o", color="#8c8c8c",
                           zorder=3, linewidths=0)
        label = ax.text(left + w / 2, mid, "%.1f%%" % printed[g], ha="center", va="center",
                        fontsize=PT, color=TEXTC[g], zorder=4)
        if g == "noncontent":
            label.set_path_effects([pe.withStroke(linewidth=2.5, foreground=COLOR[g])])
        ax.text(left + w / 2, bar_top - bar_h - 0.040, NAME[g], ha="center", va="top",
                fontsize=PT, color="#1a1a1a")
        left += w
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def bars_col(share, gshare, labels, kinds, out):
    """The printed Figure 1: every category its own bar, grouped by kind, one column wide.

    The three kinds keep the stacked bar's fills (the dotted texture for the non-content kind),
    and each kind of several categories is named once, with its total, in the room its own short
    bars leave. The printed shares of a kind's categories add up to its printed total.
    """
    pshare, pgshare = printed_shares(share, gshare, kinds)
    order = [k for g in GROUPS
             for k in sorted((k for k in share if kinds[k] == g), key=share.get, reverse=True)]
    ink = {"content": "#1f4e79", "sentiment": "#a5650a", "noncontent": "#404040"}
    hatch = {"content": None, "sentiment": None, "noncontent": "...."}
    width, height = COLUMN, 1.90
    left, bottom, top = 1.20, 0.36, 0.04          # inches around the plot
    with plt.rc_context({"hatch.color": "#6b6b6b", "hatch.linewidth": 0.5}):
        fig = plt.figure(figsize=(width, height), dpi=DPI)
        ax = fig.add_axes([left / width, bottom / height, (width - left - 0.06) / width,
                           (height - bottom - top) / height])
        ys = list(range(len(order) - 1, -1, -1))
        for y, k in zip(ys, order):
            g = kinds[k]
            ax.barh(y, share[k], height=0.72, color=COLOR[g], hatch=hatch[g], edgecolor=EDGE[g],
                    linewidth=0.6)
            ax.text(share[k] + 0.6, y, "%.1f%%" % pshare[k], va="center", fontsize=PT)
        ax.set_yticks(ys)
        ax.set_yticklabels([labels[k] for k in order], fontsize=PT)
        ax.set_xlabel("Share of categorized sentences (%)", fontsize=PT, labelpad=2)
        ax.set_xlim(0, 40)
        ax.set_xticks(range(0, 41, 10))
        ax.tick_params(axis="x", labelsize=PT, length=2, pad=2)
        ax.tick_params(axis="y", length=2, pad=2)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.spines[["top", "right"]].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.6)
        y_of = dict(zip(order, ys))
        for g in GROUPS:
            rows = [k for k in order if kinds[k] == g]
            if len(rows) < 2:
                continue
            y = (y_of[rows[-1]] + y_of[rows[-2]]) / 2
            ax.text(39.5, y, "%s\n%.1f%% in all" % (NAME[g], pgshare[g]), ha="right", va="center",
                    fontsize=PT, color=ink[g], linespacing=1.25)
        fig.savefig(out)
        plt.close(fig)
    print("wrote", out)


def stacked_word(gshare, out):
    """The stacked bar at the width of the single-column Word page (6.6 in, 200 dpi)."""
    with plt.rc_context({"hatch.color": "#404040"}):
        fig, ax = plt.subplots(figsize=(6.6, 0.62), dpi=200)
        left = 0.0
        printed = tenths(gshare, 1000)
        for g in GROUPS:
            w = gshare[g]
            ax.barh(0.5, w, left=left, height=0.72, color=COLOR[g], edgecolor=EDGE[g], linewidth=1.0)
            if g == "noncontent":
                # whole dots on a regular lattice, run across the segment; the label's glyph
                # outline clears its own room
                xs = np.arange(left + 0.7, left + w - 0.55, 0.85)
                for row, phase in ((0.5 + 0.26, 0.0), (0.5, 0.425), (0.5 - 0.26, 0.0)):
                    x = xs + phase
                    x = x[x < left + w - 0.55]
                    ax.scatter(x, [row] * len(x), s=2.2, marker="o", color="#8c8c8c", zorder=3,
                               clip_on=False)
            label = ax.text(left + w / 2, 0.5, "%.1f%%" % printed[g], ha="center", va="center",
                            fontsize=10.1, color=TEXTC[g], zorder=4)
            if g == "noncontent":
                label.set_path_effects([pe.withStroke(linewidth=3.6, foreground=COLOR[g])])
            ax.text(left + w / 2, -0.06, NAME[g], ha="center", va="top", fontsize=10.1,
                    color="#1a1a1a")
            left += w
        ax.set_xlim(0, left)
        ax.set_ylim(-0.5, 1.35)   # headroom above the bar renders as white margin
        ax.axis("off")
        fig.tight_layout(pad=0.1)
        fig.savefig(out, pad_inches=0.04, bbox_inches="tight")
        plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(HERE / "data"), help="folder with the two Figure 1 inputs")
    ap.add_argument("--out", default="paper_plots_out", help="folder the PNG files go to")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    share, gshare, labels, kinds = load_shares(Path(a.data))
    bars_col(share, gshare, labels, kinds, out / "fig1_composition_bars_col.png")
    stacked_col(gshare, out / "fig1_composition_stacked_col.png")
    stacked_word(gshare, out / "fig1_composition_stacked.png")


if __name__ == "__main__":
    main()
