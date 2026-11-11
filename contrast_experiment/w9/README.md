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
hf download chironYu/latenarray-i2ce-data --repo-type dataset --local-dir fusion_cache_w9 --exclude "*_rawprefix.*"
cd fusion_cache_w9 && python join_full_pool.py   # joins the full pool, checks SHA256SUMS, writes full_pool_READY
```

Point `DATA_SRC` (or `W9_DATA_DIR` below) at that folder. The data
pipeline in `dataset_builder/` produces the same files from the raw
reviews.

The folder also holds `wscan_gal_rev_g1024.npz` and
`wscan_gal_rev_g4096.npz`, the fixed split's anchor packs drawn from the
whole corpus (a median of 4,096 sentences at 4,096). The fixed-split
worker uses them when they are present. The five-fold worker uses them
only for the multi-anchor `ma` arms; every other five-fold run builds its
packs from the 2,048-sentence review pool, the way the paper's five-fold
runs were trained, by one of two rules:

- `--anchor-cap pool`: the store-page prefix plus the game's whole review
  pool, with no review left out (a median of 2,068 sentences, the longest
  2,184). The paper's main five-fold towers are these runs, and their runs
  and checkpoints are named `_gpool`.
- `--anchor-cap 1024` or `2048`: the prefix plus whole reviews up to that
  many sentences in total (a median of 2,046 at 2,048); runs are named
  `_g1024`, `_g2048`.

The paper calls both the 2,048-sentence budget. The worker refuses a cap
above 2,048, which the pool cannot fill. Until 2026-11-11 the `pool` packs
had no name of their own: they were requested with a cap the pool never
reached and the runs were named after that cap, `_g4096`. The packs are
the same, sentence for sentence; the checkpoints and plan files were
renamed to `_gpool`.

At `--anchor-cap 512`, the default of both workers, the packs are the
shipped `wscan_gal_rev.npz`. Since 2026-11-07 its store-page prefix, and
that of `ss_queries_rev_S.npy`, is row-standardized, like every other input
row and like the packs the workers build on the GPU at 1,024 sentences and
above. The paper's 512-sentence rows come from towers retrained on these
packs (a run whose loss never reads the anchors kept its weights and was
read again on them). Both workers name a run on them `_d512`, for example
`w9cv_wcle_i2ce_icetf_fold0_d512_fp` and `w9_wcle_i2ce_icetf_d512_fp`, the
names the retrained checkpoints have on Hugging Face, and they stop when
the `wscan_gal_rev.npz` they are given still holds the prefix raw.

The runs made before that date trained on the raw prefix and have no tag in
their names; their checkpoints stay on Hugging Face. To repeat one, leave
out the `--exclude` and pass the worker `--raw-prefix`: it reads
`wscan_gal_rev_rawprefix.npz` and `ss_queries_rev_S_rawprefix.npy` and
names the run without the tag.

## Running

1. Put this `w9/` folder and the data directory on the machine (any CUDA
   box; 24 GB VRAM suffices for anchor budgets ≤ 1,024 sentences, 48 GB for
   2,048, the paper's budget, measured at a cap of 2,048 and estimated for
   `pool`; 4,096 needs 80 GB and no profile asks for it).
2. Open a notebook under `Pod/` and run its cells top to bottom. Each
   notebook is one experiment: it stages data, launches its workers, and
   prints the readout table at the end.
3. Multiple machines may run the same notebook against a shared volume:
   atomic claim files make them split the job queue safely; a tower's done
   marker is its final projection `.npz`.

| Notebook | Experiment |
|---|---|
| `w9_a100.ipynb` | wave-1 fixed-split arms |
| `w9_final_experiment.ipynb` | CE vs I-CE × {512, 1,024, 2,048, pool} × 5 folds (the paper's scaling table) |
| `w9_cv.ipynb` | 5-fold base pair |
| `w9_experiment_5fold_2.ipynb` | slot8 / MoCo-queue / BYOL / VICReg-epd (batch 192) / SimCLR-style × 5 folds @pool |
| `w9_i2ce_t.ipynb` | temperature sweep × 5 folds × {512, 2048} |
| `w9_sweeps_3fold.ipynb` | invariance weight, view count and latent queries × folds 0 to 2 @512 (Tables A4 to A6) |
| `w9_anchor_in_i_5fold.ipynb` | CE + anchor-in-I × 5 folds @512 (Table 2, Table A14) |
| `w9_docview_5fold.ipynb` | document view absent / LLM-rewritten × 5 folds @512 (Table A18) |
| `w9_vfa.ipynb` | views-first anchors × 5 folds @512 (Table S1) |
| `w9_dino_control.ipynb` | DINO at its published settings, fold 0 @pool (the control of Section 4.3, Table A1) |
| `w9_twostage_5fold.ipynb` | BYOL warm start, then the windowed teacher × 5 folds @pool (Table A12); the BYOL stage is fetched from Hugging Face, or trained, for folds that have no checkpoint |
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

### Reproducing the window-size sweep (not in the paper)

The window-size sweep of the windowed teacher (w = 84, 168 and 336 against
full coupling, on the fixed split) is an extra of this repository. The
appendix of September 2026 printed it as its Table A9; the paper no longer
prints it, and its Table A9 is now the baseline recipes. The file names
keep the old number.

`Pod/table_a9_recompute.py` rebuilds every row of the sweep's table from
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

### Reproducing Table A17 and Figure A2 (displacement against margin)

`Pod/table1_recompute.py` re-encodes the published fold-0 checkpoints of the
six-objective comparison (`ckpt_w9cv_wcle_<arm>_fold0_gpool_fp_ep<N>.pt`,
epochs from the paper's rank plan: I-CE 2000, CE 650, SimCLR 1100, VICReg 150,
BYOL 250) on the single 4,096-sentence anchor draw `wscan_gal_rev_g4096.npz` and on the
wiki rewrites, then computes, per encoder, the mean query displacement
(1 - cos to the own anchor over the 814 name-stripped rewrites), the mean
nearest-neighbour margin over the 2,020 anchors, their ratio, and the share
of queries inside their own margin (Figure A2's clearance labels). It writes
`table1_results.json` and `fig5_encodings.npz`; `--plan cvsel --check`
scores the older worker-selected checkpoints instead and compares against
the shipped numbers. Needs a CUDA interpreter and the research caches
(`W9_FIG3`, `W9_CACHE`); the paper repository's `figures/fig5_umap.py` draws
Figure A2 from the encodings.

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
├── k10_eval.py                  # ten-draw encoding, scoring and the table
├── rank_ckpt_plan.json          # every five-fold run the paper reads: fold, epoch, file, SHA-256
├── i2bce_sel_pool.json          # the I + SimCLR checkpoints, picked by `select`
├── fixed_split_ckpt_plan.json   # the fixed-split checkpoints (Table 3, Table A7, Figures 9, A1 and A3)
└── superseded_ckpt_plan.json    # published checkpoints that no number of the paper reads
Pod/w9_selection.py              # the rank rule (sum of exp(-rank) on validation queries)
```

The draws read sentences from the full pool (`full_pool_fp16.npy` +
`full_pool_meta.npz`, built once by `Pod/w9_a100.ipynb` from
`embedding_h5.h5`); `--h5 <embedding_h5.h5>` reads the h5 instead and gives
the same draws. A draw depends only on the budget, the draw index and the
game, so any tower can be scored on the same ten packs.

The checkpoints are on Hugging Face at
[`chironYu/latenarray-i2ce-checkpoints`](https://huggingface.co/chironYu/latenarray-i2ce-checkpoints),
265 files and about 580 MiB in all, every one named by a file of `eval/`:

- the 157 five-fold runs of `rank_ckpt_plan.json` and the five folds of
  I + SimCLR in `i2bce_sel_pool.json`. These are the five-fold towers the
  paper reads, the DINO-style control of Table A1 (fold 0 only) among
  them, and `cells` takes them all;
- the five BYOL checkpoints the two-stage runs started from, named by the
  `init_ckpt` field of those runs in `rank_ckpt_plan.json`;
- the 68 checkpoints of `fixed_split_ckpt_plan.json`, from 15 fixed-split
  runs: the seven that train with 512-sentence anchors (Table 3,
  Section 6.2), one of them with an earlier checkpoint kept; a+u @ views
  (Table 3); the three VICReg weightings of Table A7; and the towers of
  Figure 9, Figure A3 and Appendix L (CE and I-CE trained at 2,048
  sentences, the SimCLR-style tower, I + SimCLR). Two runs list more
  than their selected checkpoint under `trajectory`: all 40 checkpoints
  of the I-CE run, whose latent queries Figure A1 draws, and the
  SimCLR-style checkpoints from epoch 1,350 on, whose test curve
  Appendix L describes. `k10_eval.py` reads the five folds and does not
  score these;
- the 30 checkpoints of `superseded_ckpt_plan.json`, which no number of the
  paper reads: the 512-sentence runs made on the packs as built before
  2026-11-07, which the `_d512` runs replaced, and one fold of a VICReg arm
  that used the whole training set as its batch.

```bash
export W9_DATA_DIR=/path/to/fusion_cache_w9           # the prebuilt data folder above
hf download chironYu/latenarray-i2ce-checkpoints --local-dir checkpoints
python contrast_experiment/w9/eval/k10_eval.py cells  cells.json --ckpt-dir checkpoints
python contrast_experiment/w9/eval/k10_eval.py encode cells.json draws.npz      # all towers on 2,020 games x 10 draws
python contrast_experiment/w9/eval/k10_eval.py score  cells.json draws.npz scores.json
python contrast_experiment/w9/eval/k10_eval.py table  scores.json               # five-fold mean ± std per run family
```

`cells` takes every run of `rank_ckpt_plan.json` and of
`i2bce_sel_pool.json`, plus the runs of each selection file given after it,
and looks every checkpoint name up in `--ckpt-dir`, so your own runs under
the same names can stand in for the downloaded ones. The five families of
the main comparison are scored under the paper's names (`I-CE`, `CE`,
`SimCLR`, `VICReg`, `BYOL`); they are the `_gpool` runs, the names a run
with `--anchor-cap pool` gets. `--only <regex>` keeps the cells whose key
matches, to score the runs of one table: `--only "_d512$"` for the
512-sentence ones. A run under another name is scored when `cells` is given
a selection file for it (a JSON of `{key: {"ckpt", "fold", "ep"}}`, which
`select` writes for one run family); `superseded_ckpt_plan.json` is such a
file. I + SimCLR's worker saved a 512-sentence gallery, so its
checkpoints were picked by `select`, which applies the rank rule to every
saved checkpoint of a run on the gallery the other five-fold runs select on;
`select` does the same for a run you trained yourself. `encode` resumes from
its `.part.npz` if interrupted.
