# Neighborhood figures: Figures 8, 9 and A3, and the statistics of Appendix L

A neighborhood is a seed game and the seven held-out games nearest to it by
the cosine of CE's anchors. The paper reads retrieval inside such
neighborhoods three ways:

- **Figure 8**: name-stripped hit@1 by the genre of the seed game, over the
  814 neighborhoods of the five folds, for six encoders;
- **Figure A3**: the same over the 201 neighborhoods of the fixed split;
- **Figure 9**: one neighborhood of the genre that Figure 8 boxes, drawn for
  the six encoders, every query at its true distance from its own anchor;
- **Appendix L**: the statistics over the 201 fixed-split neighborhoods (how
  much shorter the lines of I-CE are than those of CE, which tower leads in
  how many neighborhoods, where misses go, and the rest).

This folder draws the figures and computes the statistics from the published
checkpoints. Nothing here trains.

## One command

```bash
python figures/neighborhoods/run_all.py --ckpt-dir checkpoints --data-dir /path/to/data
```

`checkpoints` is a download of
[`chironYu/latenarray-i2ce-checkpoints`](https://huggingface.co/chironYu/latenarray-i2ce-checkpoints)
and the data folder one of
[`chironYu/latenarray-i2ce-data`](https://huggingface.co/datasets/chironYu/latenarray-i2ce-data)
with the full pool joined (`join_full_pool.py`). The figures land in
`hood_work/figures/`, the numbers in `hood_work/fig_facts_4096k10.json` and
`hood_work/query_stats_4096k10.json`.

The first step is the heavy one: the ten 4,096-sentence anchor draws of
every game, read from the full pool, once for the fixed-split towers and
once for the five-fold towers of Figure 8. The fixed-split file took about
40 minutes on the machine the paper used; the five-fold one holds thirty
towers and takes longer. They are the files of `contrast_experiment/w9/eval/fixed_split_eval.py` and
`k10_eval.py`; if you have them already, pass `--fixed-draws`, `--cv-draws`
and `--cv-cells` and the rest takes a few minutes. A step whose output is
in the working folder is skipped on the next run.

## The steps

| Step | Script | Reads | Writes |
|---|---|---|---|
| fixed-split vectors | `prep.py` (GPU) | the six fixed-split towers and a+u @ views at their selected checkpoints (`eval/fixed_split_ckpt_plan.json`), `draws_fixed.npz` | `fixed/allgames.npz`, `fixed/frozen.npz`, `fixed/ckpt/ref_anchors_*.npy` |
| five-fold neighborhoods | `cv_hoods.py` | the five-fold ten-draw encodings | `cv/cv_hoods_4096.json` |
| Figure 8 | `fig_genre_bars.py --cv` | `cv/cv_hoods_4096.json` | `fig8_genres.png` |
| Figure 9 | `fig_triplet.py fig9`, `fig9col` | the fixed-split vectors, the five-fold neighborhoods (for the boxed genre) | `fig9_neighborhood.png` (full text width), `fig9_neighborhood_col.png` (one column, the printed figure), `fig_facts_4096k10.json` |
| Figure A3 | `fig_genre_bars.py` | the fixed-split vectors | `figA3_genres_fixed.png` |
| Appendix L | `fig_stats.py` | the fixed-split vectors | `query_stats_4096k10.json` |
| layout check | `layout_check.py` (`--layout-check`) | the fixed-split vectors | printed: how well a 2-D projection keeps the line lengths |

`fig_triplet.py` holds the definitions the others import: the
neighborhoods, the genre of a seed (the rarest of the 23 reduced tags it
carries), the rule that picks the neighborhood shown, and the layout of a
panel. Its docstring states the rule and the layout in full.
`all_titles.json` gives the Steam titles of the 2,020 games in the order of
`games.npz`; `hood_env.py` reads the three environment variables the
scripts take their paths from when run alone (`HOOD_WORK`, `HOOD_OUT`,
`W9_DATA_DIR`).

A query is scored by its hit rate over the ten anchor draws. Figure 9 draws
and marks the first draw, and each panel's title gives the mean number of
hits over the ten.

## What reproduces, and how exactly

The scripts are the ones that drew the paper's figures, with their paths
and figure numbers brought up to date. Fed the encodings the paper kept,
they give:

- the 814 five-fold neighborhoods with the same members and scores;
- the fixed-split queries and hit records bit for bit, from the published
  checkpoints;
- every number of `query_stats_4096k10.json` and of the neighborhood and
  population entries of `fig_facts_4096k10.json`;
- the four figure files byte for byte, under the matplotlib release each
  was drawn with (3.11.1 for Figure 8, 3.10.8 for Figures 9 and A3).
  Another release sets text a little differently, which moves an axis by
  a pixel or changes the height of a figure by a few.

A note on the directions of the lines in Figure 9. A query is drawn at its
true distance from its anchor, in the direction its displacement takes
under a linear map built from the pseudo-inverse of the eight centred
anchors. Eight centred points have rank seven, so the eighth direction,
whose singular value is round-off (about 1e-7), has to be left out: the
scripts compute the map in double precision without it, and the picture is
the same on every machine. The figure as first drawn kept that direction,
which made the line directions depend on the numpy build in their second
digit; `HOOD_MAP=drawn` (or `run_all.py --map drawn`) is that computation,
kept for comparison. The lengths of the lines, the positions of the
anchors and the hit marks are the same either way; of the recorded
numbers only the "as drawn" direction coherence differs.

## Requirements

`numpy`, `matplotlib`, `torch` with CUDA for `prep.py` and the draws,
`scipy` for `fig_stats.py`, `umap-learn` for `layout_check.py` only.
