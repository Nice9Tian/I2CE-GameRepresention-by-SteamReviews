# w9 — the full contrastive-experiment suite (release build)

Self-contained reproduction package for the paper's experiment campaign:
the I-CE recipe and every baseline/ablation arm (CE, MoCo-queue, BYOL,
VICReg-epd, slot/readout grids, twin-pack `pk` family, sliding-window `swin`
family, temperature sweep), on both the fixed split and 5-fold
cross-validation.

This build ships **code only** and performs **no repository
synchronisation** and **no cloud-provider API calls** (both were research
operations, removed on purpose). Every notebook assumes the code is already
present in this folder.

## Layout

```
w9/
├── Pod/                     # workers + one notebook per experiment
│   ├── w9_a100_worker.py    # fixed-split worker: every arm/loss family
│   ├── w9_cv_worker.py      # 5-fold worker (fold-inductive splits)
│   ├── w9_jobs.py           # claim files, labels, GPU detection, monitors
│   ├── h5_staging.py        # parallel copy of large assets to local disk
│   └── w9_*.ipynb           # experiments (see table below)
└── VICReg_review/           # minimal model/eval package the workers import
    ├── model.py             # tower, expander, VICReg loss
    └── text_variant_eval.py # ridge tag probe, split helpers, micro-F1
```

## Arm naming

Arm labels read `wcle_<tower recipe>_<loss><tau>`. The recipe token is
`i2ce` for the paper's tower (invariance weight 2 plus CE), `ce` for the
contrast-only baseline, `swin<w>step<S>loop<T>i2ce` for the windowed
teacher; the loss token is `ice` (I-CE), `ce`, `by` (BYOL) and so on; the
final `tf` means the temperature is frozen at 0.02 and `tl` that it is
learnable (`wcle_i2ce_icetl`). So `wcle_i2ce_icetf` is the published
recipe itself. Nothing in a suffix denotes a fine-tuning stage.

## Data prerequisites

The notebooks expect a data directory (default `/workspace/fusion_cache_w9`,
edit the constant in each notebook's first cell) holding the corpus assets:
`games.npz`, `wiki_eval.npz`, `wscan_gal_rev.npz`, `wscan_pool_rev.*`,
`ss_queries_rev.npz`, `ss_queries_rev_S.npy`, `wiki_clean_views.npz`,
`sp_raw_views.npz`, `tag_labels.npz`, `wiki_eval_split.json`,
`_tag_splitM.json`, and (for full-pool training) `full_pool_fp16.npy` +
`full_pool_meta.npz` with the `full_pool_READY` marker. The prebuilt folder
on Hugging Face,
[`chironYu/latenarray-i2ce-data`](https://huggingface.co/datasets/chironYu/latenarray-i2ce-data),
holds all of them, so nothing has to be rebuilt:

```bash
hf download chironYu/latenarray-i2ce-data --repo-type dataset --local-dir fusion_cache_w9
cd fusion_cache_w9 && python join_full_pool.py   # joins the full pool, checks SHA256SUMS, writes full_pool_READY
```

Point `DATA_SRC` (or `W9_DATA_DIR` below) at that folder. The data
pipeline in `dataset_builder/` produces the same files from the raw
reviews.

## Running

1. Put this `w9/` folder and the data directory on the machine (any CUDA
   box; 24 GB VRAM suffices for anchor budgets ≤ 2,048 sentences, 80 GB for
   4,096).
2. Open a notebook under `Pod/` and run its cells top to bottom. Each
   notebook is one experiment: it stages data, launches its workers, and
   prints the readout table at the end.
3. Multiple machines may run the same notebook against a shared volume:
   atomic claim files make them split the job queue safely; a tower's done
   marker is its final projection `.npz`.

| Notebook | Experiment |
|---|---|
| `w9_a100.ipynb` | wave-1 fixed-split arms |
| `w9_final_experiment.ipynb` | CE vs I-CE × {512…4096} × 5 folds (the paper's scaling table) |
| `w9_cv.ipynb` | 5-fold base pair |
| `w9_experiment_5fold_2.ipynb` | slot8 / MoCo-queue / BYOL / VICReg-epd × 5 folds @4096 |
| `w9_i2ce_t.ipynb` | temperature sweep × 5 folds × {512, 2048} |
| `w9_flash*.ipynb` | loss-ladder and structure grids |
| `w9_scale.ipynb`, `w9_mq.ipynb`, `w9_save_4096_tag.ipynb` | anchor-scale & anchor-supply arms |
| `w9_packageview.ipynb` | twin-pack `pk{N}` fractal family |
| `w9_scale_query.ipynb` | slot-count / readout capacity grid |
| `w9_swin.ipynb`, `w9_swin_5fold.ipynb` | sliding fresh-window CE field |
| `w9_i2ce_continue.ipynb` | budget-extension run |

The workers select checkpoints by the paper's `sum(exp(-rank))` over the
validation games' **neutral and noname** rewrites, against all 2,020 games.
Ranks start at 1; equal checkpoint scores prefer the earlier epoch. Tags,
review pseudo-queries and test-query metrics are report-only diagnostics.

### Reselecting existing checkpoints without training

`Pod/w9_reselect.py` reads saved `SPg`/`SPa` projections and query metadata,
scores every candidate, and writes a new checkpoint plan in a separate
directory. It does not train, import a training worker, run a tag probe, or
modify the source caches/checkpoints. Only NumPy is needed.

```bash
python contrast_experiment/w9/Pod/w9_reselect.py \
  --source-root /data/archive \
  --projection-dir /data/archive/w9_towers \
  --cache-dir /data/archive/fusion_cache_w9 \
  --checkpoint-dir /data/archive/w9_cv_out \
  --checkpoint-dir /data/archive/w9_out \
  --output-dir ./results/w9_rank_selection \
  --reference-plan /paper/data/w9/rank_ckpt_plan.json \
  --max-epoch 2000 --allow-short-runs
```

The optional reference plan scopes runs and compares the old epochs; it
does not supply the new choice. Without it, every run in the projection
directory is scored. Repeat `--run NAME` to select specific runs. `_foldN`
names use CV validation fold `(N+1)%5` (seed 20260711); other names use the
pinned split. Default coverage requires every 50 epochs through 2000.
`--allow-short-runs` permits contiguous shorter runs, including the archived
600-epoch warm-start trajectories, and reports the actual horizon. Missing
interior epochs are errors.

Read `selection_summary.json` and require `status: complete` before using
`rank_ckpt_plan.json`. The plan points at existing checkpoint files; the
per-run `rank_traj_*.json` records every score and source provenance.
Outputs are atomic and resume per run; `--overwrite` forces rescoring.
Use a separate output directory for each run subset. This operation uses
saved single-draw projections and does not recompute K=10 test tables.

CPU smoke tests (no training or corpus required):

```bash
python -B -m unittest discover -s contrast_experiment/w9/Pod -p test_w9_selection.py -v
```

### Reproducing Table A9 (the window-size sweep)

`Pod/table_a9_recompute.py` rebuilds every row of appendix Table A9 from
the projection caches the fixed-split worker writes beside each checkpoint
(`tower_<label>_ep<N>.npz`), with no GPU: for each of the four arms it
re-runs the worker's own `zs_from_arrays` on every checkpoint from epoch 50
to 2,000, selects the checkpoint by the paper's rank criterion (the sum of
`exp(-rank)` over the validation games' rewrites of both registers, ties
to the earlier epoch; `W9_SELECT=rvsel` switches to the worker's own
review-query score), reports the test-set retrieval columns of that
checkpoint, and scores the Section 5 test-set tag probe (ridge on the
1,613 training-gallery anchors, threshold on the 203 validation games,
micro-F1 on the 204 test games' name-intact rewrites). Point it at the
result directory and the cache with `LARICE_RESULTS` (or `W9_OUT`) and
`W9_CACHE`; it writes `Pod/table_a9_results.json` and prints the table.

```bash
python contrast_experiment/w9/Pod/table_a9_recompute.py
```

### Reproducing Table 1 and Figure 5 (displacement against margin)

`Pod/table1_recompute.py` re-encodes the fold-0 towers of the six-objective
comparison at the 4,096-sentence budget (`ckpt_w9cv_wcle_<arm>_fold0_g4096_fp_ep<N>.pt`,
epochs from the paper's rank plan: I-CE 2000, CE 650, SimCLR 1100, VICReg 150,
BYOL 250) on the single anchor draw `wscan_gal_rev_g4096.npz` and on the
wiki rewrites, then computes, per encoder, the mean query displacement
(1 - cos to the own anchor over the 814 name-stripped rewrites), the mean
nearest-neighbour margin over the 2,020 anchors, their ratio, and the share
of queries inside their own margin (Figure 5's clearance labels). It writes
`table1_results.json` and `fig5_encodings.npz`; `--plan cvsel --check`
scores the older worker-selected checkpoints instead and compares against
the shipped numbers. Needs a CUDA interpreter and the research caches
(`W9_FIG3`, `W9_CACHE`); the paper repository's `figures/fig5_umap.py` draws
Figure 5 from the encodings.

```bash
python contrast_experiment/w9/Pod/table1_recompute.py --plan rank
```

## From trained towers to the manuscript's numbers

A worker's own readout scores each checkpoint against the gallery it trained
with. The manuscript scores every five-fold tower differently: against ten
independent 4,096-sentence anchor packs per game, and reports the mean over
the ten. `eval/k10_eval.py` does that step.

```
eval/
├── k10_eval.py           # ten-draw encoding, scoring and the table
├── rank_ckpt_plan.json   # the checkpoint the rank rule picked for each run
└── i2bce_sel_pool.json   # the I + SimCLR checkpoints, picked by `select`
Pod/w9_selection.py       # the rank rule (sum of exp(-rank) on validation queries)
```

The draws read sentences from the full pool (`full_pool_fp16.npy` +
`full_pool_meta.npz`, built once by `Pod/w9_a100.ipynb` from
`embedding_h5.h5`); `--h5 <embedding_h5.h5>` reads the h5 instead and gives
the same draws. A draw depends only on the budget, the draw index and the
game, so any tower can be scored on the same ten packs.

The 110 checkpoints behind the manuscript's numbers (the 105 runs of
`rank_ckpt_plan.json` and the five folds of I + SimCLR) are on Hugging Face at
[`chironYu/latenarray-i2ce-checkpoints`](https://huggingface.co/chironYu/latenarray-i2ce-checkpoints),
about 181 MiB in all.

```bash
export W9_DATA_DIR=/path/to/fusion_cache_w9           # the prebuilt data folder above
hf download chironYu/latenarray-i2ce-checkpoints --local-dir checkpoints
python contrast_experiment/w9/eval/k10_eval.py cells  cells.json --ckpt-dir checkpoints contrast_experiment/w9/eval/i2bce_sel_pool.json
python contrast_experiment/w9/eval/k10_eval.py encode cells.json draws.npz      # all towers on 2,020 games x 10 draws
python contrast_experiment/w9/eval/k10_eval.py score  cells.json draws.npz scores.json
python contrast_experiment/w9/eval/k10_eval.py table  scores.json               # five-fold mean ± std per run family
```

`cells` takes every run of `rank_ckpt_plan.json` plus the runs of each
selection file given after it, and looks every checkpoint name up in
`--ckpt-dir`, so your own runs under the same names can stand in for the
downloaded ones. I + SimCLR's worker saved a 512-sentence gallery, so its
checkpoints were picked by `select`, which applies the rank rule to every
saved checkpoint of a run on the gallery the other five-fold runs select on;
`select` does the same for a run you trained yourself. `encode` resumes from
its `.part.npz` if interrupted.
