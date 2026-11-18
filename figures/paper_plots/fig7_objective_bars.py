# -*- coding: utf-8 -*-
"""Figure 7: the six objectives of Table A1 as grouped bars, identity against semantics.

    python figures/paper_plots/fig7_objective_bars.py [--table data/rank_tables.json] [--out DIR]
    python figures/paper_plots/fig7_objective_bars.py --scores <k10_eval score output>

For each objective (frozen embedder, BYOL, VICReg, SimCLR-style, CE, I-CE) four bars: test-set tag
F1, name hit@1, stripped hit@5 and stripped hit@1, five-fold mean with the between-fold standard
deviation (ddof = 0) as error bars, every tower trained at the 2,048-sentence budget and read
against ten 4,096-sentence anchor draws.  The default input is the five-fold aggregate the paper's
tables print (data/rank_tables.json, block `body`; the same numbers as Table A1).  --scores takes
the JSON that `contrast_experiment/w9/eval/k10_eval.py score` writes for the cells I-CE|k, CE|k,
SimCLR|k, VICReg|k, BYOL|k and frozen|k and recomputes the aggregate from it; that output agrees
with the shipped aggregate to the third decimal, so the bars are the same but the file is not
byte for byte.

Two files:

    fig7_objective_bars_col.png   the printed Figure 7, one column of the two-column build (3.337 x 2.62 in)
    fig7_objective_bars.png       the same chart for the single-column Word page (3.9 x 2.34 in)
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
COLUMN = 241.14749 / 72.27          # sigconf \columnwidth in inches
PT = 7.0
ARMS = [("frozen", "Frozen"), ("BYOL", "BYOL"), ("VICReg", "VICReg"),
        ("SimCLR", "SimCLR"), ("CE", "CE"), ("I-CE", "I-CE (ours)")]
BRACKETS = [(0, 0, "no training"), (1, 3, "strong views only"), (4, 5, "strong + weak views")]
# columns of a `body` row: Name hit@1, Name hit@5, Stripped hit@1, Stripped hit@5, tag F1
COLS = ["pd_nm_neutral", "pd_h5_neutral", "pd_nm_noname", "pd_h5_noname", "pd_test_tag"]
NAME, STRIP, STRIP5, TAG = 0, 2, 3, 4
PITCH, SLOT = 1.42, 0.315
W = SLOT * 0.86
C_TAG, C_NAME, C_S5, C_S1 = "#bdb5a8", "#cfe0ea", "#7aa6c0", "#2b5468"
EB = {"lw": 0.6, "ecolor": "#6b6b6b"}


def body_from_scores(path):
    """the `body` rows (mean, population std over the five folds) from a k10_eval score output"""
    res = json.loads(Path(path).read_text(encoding="utf-8"))["results"]
    body = {}
    for arm, _ in ARMS:
        rows = [res["%s|%d" % (arm, k)] for k in range(5)]
        body[arm] = [[float(np.mean([r[c] for r in rows])) for c in COLS],
                     [float(np.std([r[c] for r in rows])) for c in COLS]]
    return body


def series(body):
    """{column: (means, stds)} in the order of ARMS"""
    return {k: ([body[a][0][k] for a, _ in ARMS], [body[a][1][k] for a, _ in ARMS])
            for k in (NAME, STRIP, STRIP5, TAG)}


def draw_bars(ax, s):
    """the four bars of every objective and the line through the stripped hit@1 bars"""
    xs = [i * PITCH for i in range(len(ARMS))]
    for offset, k, c in ((-1.5, TAG, C_TAG), (-0.5, NAME, C_NAME), (0.5, STRIP5, C_S5), (1.5, STRIP, C_S1)):
        ax.bar([x + offset * SLOT for x in xs], s[k][0], W, yerr=s[k][1], capsize=1.5, color=c,
               linewidth=0, error_kw=EB)
    ax.plot([x + 1.5 * SLOT for x in xs], s[STRIP][0], color="#c0392b", linewidth=1.0, marker="o",
            markersize=2.6, markerfacecolor="#c0392b", markeredgecolor="white", markeredgewidth=0.45,
            zorder=5, clip_on=False)
    return xs


def style_axes(ax, xs, labels):
    ax.set_xlim(xs[0] - PITCH / 2 - 0.1, xs[-1] + PITCH / 2 + 0.1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=PT)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=6.0)
    ax.set_ylabel("Score", fontsize=PT)
    ax.grid(axis="y", color="#9a9a9a", linewidth=0.45, alpha=0.32)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", length=0)


def draw_brackets(ax, xs, rule, tick, text):
    """the view-regime brackets under the tick labels, at axes heights rule / tick / text"""
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i0, i1, label in BRACKETS:
        x0, x1 = xs[i0] - 0.52, xs[i1] + 0.52
        ax.plot([x0, x1], [rule, rule], transform=trans, clip_on=False, color="#606060", linewidth=0.7)
        for xe in (x0, x1):
            ax.plot([xe, xe], [rule, tick], transform=trans, clip_on=False, color="#606060", linewidth=0.7)
        ax.annotate(label, xy=((x0 + x1) / 2, text), xycoords=trans, ha="center", va="top",
                    fontsize=PT, color="#404040")


def column_figure(body, out):
    """the printed Figure 7: legend above the bars, the label of the dashed rule on the rule"""
    s = series(body)
    width, height = COLUMN, 2.62
    ax_l, ax_b, ax_w, ax_h = 0.36, 0.47, width - 0.36 - 0.03, 1.82
    fig = plt.figure(figsize=(width, height), dpi=300)
    ax = fig.add_axes([ax_l / width, ax_b / height, ax_w / width, ax_h / height])
    xs = draw_bars(ax, s)
    best = max(s[STRIP][0])
    ax.axhline(best, color="#c0392b", linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=1)
    ax.annotate("best stripped\nhit@1  %.3f" % best, xy=(xs[0] - 1.5 * SLOT - W / 2, best), xytext=(0, 3),
                textcoords="offset points", ha="left", va="bottom", fontsize=PT, color="#c0392b",
                linespacing=1.25)
    style_axes(ax, xs, [label for _, label in ARMS])
    draw_brackets(ax, xs, -0.26 / ax_h, -0.21 / ax_h, -0.30 / ax_h)
    handles = [Patch(facecolor=C_TAG, label="Test-set TAG F1 (semantics)"),
               Patch(facecolor=C_NAME, label="Name hit@1 (name-intact)"),
               Patch(facecolor=C_S5, label="Stripped hit@5 (identity)"),
               Patch(facecolor=C_S1, label="Stripped hit@1 (identity)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 - 0.02 / height), ncol=2,
               fontsize=PT, frameon=False, borderpad=0, handlelength=1.1, handletextpad=0.35,
               labelspacing=0.45, columnspacing=1.4, borderaxespad=0)
    fig.savefig(out)
    plt.close(fig)
    print("wrote %s   best stripped hit@1 %.3f" % (out, best))


def word_figure(body, out, size=(3.9, 2.34)):
    """the single-column Word version: legend beside the panel, the rule run out into the margin"""
    s = series(body)
    fig, ax = plt.subplots(figsize=size, dpi=300)
    xs = draw_bars(ax, s)
    axes_x = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    best = max(s[STRIP][0])
    ax.plot([0, 1.10], [best, best], transform=axes_x, clip_on=False, color="#c0392b", linewidth=0.7,
            linestyle=(0, (4, 2.5)), zorder=1)
    ax.annotate("best stripped\nhit@1  %.3f" % best, xy=(1.13, best), xycoords=axes_x, ha="left",
                va="center", fontsize=7.0, color="#c0392b", linespacing=1.25)
    style_axes(ax, xs, [lbl for _, lbl in ARMS])
    draw_brackets(ax, xs, -0.145, -0.115, -0.168)
    handles = [Patch(facecolor=C_TAG, label="Test-set TAG F1\n(semantics)"),
               Patch(facecolor=C_NAME, label="Name hit@1\n(name-intact)"),
               Patch(facecolor=C_S5, label="Stripped hit@5\n(identity)"),
               Patch(facecolor=C_S1, label="Stripped hit@1\n(identity)")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.015, 0.30), fontsize=7.0, frameon=False,
              borderpad=0.2, handlelength=1.1, handletextpad=0.35, labelspacing=0.7, borderaxespad=0)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(bottom=0.185)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=str(HERE / "data" / "rank_tables.json"))
    ap.add_argument("--scores", default="", help="a k10_eval.py score output to aggregate instead")
    ap.add_argument("--out", default="paper_plots_out")
    a = ap.parse_args()
    body = body_from_scores(a.scores) if a.scores else \
        json.loads(Path(a.table).read_text(encoding="utf-8"))["body"]
    for arm, _ in ARMS:
        print("  %-8s stripped hit@1 %.3f  tag F1 %.3f" % (arm, body[arm][0][STRIP], body[arm][0][TAG]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    column_figure(body, out / "fig7_objective_bars_col.png")
    word_figure(body, out / "fig7_objective_bars.png")


if __name__ == "__main__":
    main()
