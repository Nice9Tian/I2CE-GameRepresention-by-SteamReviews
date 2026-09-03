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
`full_pool_meta.npz` with the `full_pool_READY` marker. These are produced
by the data pipeline in `release/dataset_builder`.

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

Selection is deployment-faithful throughout: checkpoints are picked on
validation-fold review pseudo-queries; LLM rewrites are evaluation-only.

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
