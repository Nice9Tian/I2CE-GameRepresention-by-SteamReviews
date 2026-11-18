# Figures 1, 6, 7 and A2

The scripts that drew four of the paper's plots, with their inputs:

- **Figure 1**: what the sentences of Steam reviews are about, seven categories of a BERTopic topic model, as bars grouped into content-bearing, sentiment and other non-content;
- **Figure 6**: the gradient envelopes of cross-entropy and of the invariance term against the margin (Section 4.3);
- **Figure 7**: the six objectives of Table A1 as grouped bars, retrieval on both query registers against the test-set tag reading;
- **Figure A2**: a UMAP of game anchors and name-stripped queries for four encoders.

## One command

```bash
python figures/paper_plots/run_all.py --out paper_plots_out
```

draws Figures 1, 6 and 7 from the files in `data/` (matplotlib only). Figure A2 needs encodings
that a GPU step writes first:

```bash
python contrast_experiment/w9/Pod/table1_recompute.py --plan rank \
    --ckpt-dir checkpoints --data-dir /path/to/data            # writes fig5_encodings.npz
python figures/paper_plots/run_all.py --enc <its output folder>/fig5_encodings.npz --data-dir /path/to/data
```

`checkpoints` is a download of
[`chironYu/latenarray-i2ce-checkpoints`](https://huggingface.co/chironYu/latenarray-i2ce-checkpoints)
and the data folder one of
[`chironYu/latenarray-i2ce-data`](https://huggingface.co/datasets/chironYu/latenarray-i2ce-data).
Each script also runs alone; `--help` lists its options.

## The scripts

| Figure | Script | Reads | Writes |
|---|---|---|---|
| 1 | `fig1_composition.py` | `data/fig1_bertopic_topics.md`, `data/fig1_topic_buckets.json` | `fig1_composition_bars_col.png` (the printed figure, one column), `fig1_composition_stacked_col.png` and `fig1_composition_stacked.png` (the three kinds as one stacked bar, at one column and at the Word page width) |
| 6 | `fig6_envelopes.py` | nothing: the curves are the formulas of Section 4.3 at λ = 2, τ = 0.02, V = 4 | `fig6_envelopes.png` |
| 7 | `fig7_objective_bars.py` | `data/rank_tables.json` (block `body`), or with `--scores` a `k10_eval.py score` output | `fig7_objective_bars_col.png` (the printed figure, one column), `fig7_objective_bars.png` (the Word page version) |
| A2 | `figA2_umap.py` | `fig5_encodings.npz` of `table1_recompute.py`, `contrast_experiment/w9/Pod/table1_results.json` (the clearance labels), `tag_labels.npz` and `wiki_eval_split.json` of the data folder | `figA2_umap.png` |

`data/rank_tables.json` holds the five-fold aggregates the paper's tables print: `body` is the
single-source reading (Table A1 and Table S2), `merged` the merged-index reading (Table S3). Each
row is the mean and the population standard deviation over the five folds of Name hit@1, Name
hit@5, Stripped hit@1, Stripped hit@5 and test-set tag F1. `contrast_experiment/w9/eval/k10_eval.py`
recomputes them from the published checkpoints (its `score` and `table` steps); passing that
output to `fig7_objective_bars.py --scores` draws the figure from it.

## What reproduces, and how exactly

- **Figures 1, 6 and 7** come out byte for byte as the paper prints them under matplotlib 3.11.1
  (Python 3.12, numpy 2.x), the release they were drawn with. Under matplotlib 3.10.8 the
  files have the same size but text is set slightly differently, which changes up to about
  a tenth of the pixels, at the edges of glyphs and lines.
- **Figure 1's topic table cannot be rebuilt from public data.** The BERTopic fit
  (100,000 sentences sampled from the reviews of 2,000 games, UMAP with 15 neighbours, HDBSCAN
  with a minimum cluster size of 50, seed 42) read the review sentences themselves, and neither
  this repository nor the Hugging Face data holds review text, only sentence embeddings. The fit's
  output, the count and top words of each of its 127 topics, is shipped as
  `data/fig1_bertopic_topics.md`; the drawing step reads only the counts. The file is the fit's
  report with its last section, sample sentences from each topic, left out, and the path of its
  input file shortened. `data/fig1_topic_buckets.json` assigns the 126 topics other than the
  catch-all topic 0 to the seven categories (its `_provenance` entry says how; the paths there
  are those of the paper repository).
- **Figure A2 as printed was drawn by an earlier script that did not survive**, and its legend
  was re-set beside the panels afterwards. `figA2_umap.py` draws the same design from the same
  encodings, so the panel content follows the same rules and the clearance percentages are the
  printed ones (0%, 41%, 23%, 14%), but the UMAP layout differs: where each cloud lies in its panel
  is not the printed picture. Its output is the same on every run (UMAP with a fixed seed;
  checked with umap-learn 0.5.12, matplotlib 3.10.8). The encodings are reproducible:
  `table1_recompute.py` writes them from the published fold-0 checkpoints and the
  4,096-sentence anchor draw `wscan_gal_rev_g4096.npz`.

## Requirements

`numpy` and `matplotlib` for Figures 1, 6 and 7; `umap-learn` and `scipy` in addition for
Figure A2.
