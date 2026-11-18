# -*- coding: utf-8 -*-
r"""Figure A2: UMAP of game anchors (top) and name-stripped queries (bottom) for four encoders.

    python contrast_experiment/w9/Pod/table1_recompute.py --plan rank --ckpt-dir <checkpoints> \
        --data-dir <data folder>                     # CUDA; writes fig5_encodings.npz
    python figures/paper_plots/figA2_umap.py --enc <dir>/fig5_encodings.npz --data-dir <data folder>
        -> figA2_umap.png

What is drawn (the caption of Figure A2):

* one joint UMAP per encoder over its 2,020 anchors and the 814 name-stripped wiki rewrites of the
  fold-0 universe (n_neighbors=15, min_dist=0.1, cosine, random_state=42), so anchors and queries
  share one plane; every map is centred, scaled to unit std and rigidly aligned (rotation or
  reflection) to the frozen embedder's map;
* colour = the most specific of six coarse tags a game carries (the rarest of the six it has);
  games with none of the six are not drawn; the top row shows the anchors of the universe games
  that carry one of the six, the bottom row the queries of the same games;
* in the bottom row a thin line joins a fixed random 180 of those queries to their own game's
  anchor (rng seed 0), omitted in the frozen column, where every line would cross the panel;
* under each bottom panel, the share of all 814 queries whose displacement is below their own
  game's nearest-neighbour margin (`inside_margin` of table1_results.json, the `inside` column
  that table1_recompute.py prints);
* the legend as a column on the right.

The figure printed in the paper is not this script's output: it was drawn by an earlier UMAP script
that did not survive, and its legend was then re-set beside the panels.  This script draws the same
design from the same encodings, so the panels, the colours, the lines and the clearance labels
follow the same rules, but the UMAP layout (where each cloud sits in its panel) differs.  The
percentages are computed, not drawn from the layout, and match the paper.

Options
    --enc PATH       fig5_encodings.npz (table1_recompute.py)
    --results PATH   table1_results.json (default contrast_experiment/w9/Pod/table1_results.json)
    --data-dir DIR   the data folder with tag_labels.npz and wiki_eval_split.json (or W9_DATA_DIR)
    --out PNG        default paper_plots_out/figA2_umap.png
    --seed 0         the 180-line sample
Needs umap-learn, scipy and matplotlib.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPACES = ["Frozen", "I-CE", "VICReg", "BYOL"]
TITLE = {"Frozen": ("Frozen embedder", "(mean pool)"),
         "I-CE": ("I-CE @4096, fold 0", "(ours)"),
         "VICReg": ("VICReg", "(epd 20/10/20)"),
         "BYOL": ("BYOL", None)}
SIX = ["Card Game", "Sports/Rhythm", "Racing/Driving", "Fighting/Melee", "Platformer", "Puzzle"]
COL = {"Card Game": "#d62728", "Sports/Rhythm": "#1f77b4", "Racing/Driving": "#2ca02c",
       "Fighting/Melee": "#9467bd", "Platformer": "#ff7f0e", "Puzzle": "#8c564b"}
N_LINES = 180
GREY = "#505050"


def l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def game_tags(data):
    """per game, the index into SIX of the rarest of the six tags it carries, or -1; and the names"""
    tl = np.load(data / "tag_labels.npz", allow_pickle=True)
    tag_names = [str(x) for x in tl["tag_names"]]
    Y = tl["y"]
    names = [str(x) for x in tl["names"]]
    freq = Y.sum(0)
    six_idx = [tag_names.index(t) for t in SIX]
    spec_order = sorted(range(6), key=lambda j: freq[six_idx[j]])
    game_tag = np.full(len(names), -1, np.int64)
    for g in range(len(names)):
        for j in spec_order:
            if Y[g, six_idx[j]] > 0:
                game_tag[g] = j
                break
    return game_tag, names


def joint_maps(enc, noname):
    """one joint UMAP per encoder, centred, scaled and aligned to the frozen map"""
    import umap
    maps = {}
    for s in SPACES:
        joint = np.vstack([l2(enc[f"Zg_{s}"]), l2(enc[f"Za_{s}"])[noname]]).astype(np.float32)
        e = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(joint)
        maps[s] = (e - e.mean(0)) / e.std()
        print(f"UMAP {s} done", flush=True)
    for s in SPACES:
        R, _ = orthogonal_procrustes(maps[s], maps["Frozen"])
        maps[s] = maps[s] @ R
    return maps


def draw_column(axes, s, e, NG, sel):
    """the anchor panel and the query panel of encoder s"""
    eg, eq = e[:NG], e[NG:]
    pts = np.vstack([eg[sel["games"]], eq[sel["q"]]])
    # square window on the 1st..99th percentile box of the drawn points,
    # so a handful of far outliers cannot shrink the cloud into a corner
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    half = 0.56 * max(hi[0] - lo[0], hi[1] - lo[1])
    gt, qg = sel["game_tag"], sel["q_game"]
    for r, ax in enumerate(axes):
        if r == 0:
            P, T = eg[sel["games"]], gt[sel["games"]]
        else:
            P, T = eq[sel["q"]], gt[qg[sel["q"]]]
            if s != "Frozen":
                for qi in sel["lines"]:
                    p0, p1 = eq[qi], eg[qg[qi]]
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=0.55, color=COL[SIX[gt[qg[qi]]]],
                            alpha=0.45, zorder=1)
        for j in range(6):
            m = T == j
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=8, c=COL[SIX[j]], linewidths=0, alpha=0.9, zorder=2)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_xticks([]); ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_linewidth(0.6)


def draw(maps, NG, sel, res, out):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(14.6, 7.0))
    gs = fig.add_gridspec(2, 4, left=0.065, right=0.835, top=0.885, bottom=0.075, wspace=0.06, hspace=0.10)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)])
    for c, s in enumerate(SPACES):
        draw_column(axes[:, c], s, maps[s], NG, sel)
        t1, t2 = TITLE[s]
        if t2:
            axes[0, c].set_title(t1, fontsize=15, pad=30)
            axes[0, c].text(0.5, 1.03, t2, transform=axes[0, c].transAxes, ha="center", va="bottom",
                            fontsize=12.5, color=GREY)
        else:
            axes[0, c].set_title(t1, fontsize=15, pad=8)
        axes[1, c].set_xlabel(f"{100 * res[s]['inside_margin']:.0f}% inside own clearance", fontsize=14.5,
                              labelpad=8)
    axes[0, 0].set_ylabel("Game anchors", fontsize=15, labelpad=10)
    axes[1, 0].set_ylabel("Name-stripped queries", fontsize=15, labelpad=26)
    axes[1, 0].text(-0.075, 0.5, "(line = distance to own anchor)", transform=axes[1, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=11.5, color=GREY)
    handles = [Line2D([0], [0], marker="o", ls="", mfc=COL[t], mec="none", ms=9, label=t) for t in SIX]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.845, 0.5), fontsize=13.5,
               frameon=False, labelspacing=1.1, handletextpad=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=sel["dpi"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enc", required=True, help="fig5_encodings.npz of table1_recompute.py")
    ap.add_argument("--results", default=str(ROOT / "contrast_experiment" / "w9" / "Pod" / "table1_results.json"))
    ap.add_argument("--data-dir", default=os.environ.get("W9_DATA_DIR", ""))
    ap.add_argument("--out", default=str(Path("paper_plots_out") / "figA2_umap.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()
    data = Path(a.data_dir)
    assert a.data_dir and (data / "tag_labels.npz").exists(), "--data-dir <data folder> (or W9_DATA_DIR) is required"
    game_tag, names = game_tags(data)
    enc = np.load(a.enc, allow_pickle=True)
    variants = [str(x) for x in enc["variants"]]
    noname = np.array([i for i in range(len(variants)) if variants[i] == "noname"])
    assert len(noname) == 814, len(noname)
    q_game = enc["gA"][noname]                                  # game index per query
    sp = json.loads((data / "wiki_eval_split.json").read_text())
    universe = set(sp["test"]) | set(sp["val"]) | set(sp["train"])
    uni = np.array([n.split("_")[0] in universe for n in names])
    sel = dict(game_tag=game_tag, q_game=q_game, dpi=a.dpi,
               games=np.where(uni & (game_tag >= 0))[0], q=np.where(game_tag[q_game] >= 0)[0])
    print(f"universe games with one of the six tags: {len(sel['games'])}; tagged queries: {len(sel['q'])}")
    rng = np.random.default_rng(a.seed)
    sel["lines"] = np.sort(rng.choice(sel["q"], size=min(N_LINES, len(sel["q"])), replace=False))
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))["t3"]
    maps = joint_maps(enc, noname)
    out = Path(a.out)
    draw(maps, enc["Zg_Frozen"].shape[0], sel, res, out)
    print(f"wrote {out}; lines drawn for {len(sel['lines'])} queries; clearance "
          + ", ".join(f"{s} {100 * res[s]['inside_margin']:.0f}%" for s in SPACES))


if __name__ == "__main__":
    main()
