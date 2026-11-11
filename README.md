# I2CE — Invariant Game Representation Learning from Steam Reviews

Code, data pipeline, appendices and supplement of the paper *Invariant Game
Representation Learning from Steam Reviews* (AICCC 2026), which introduces
the I-CE representation. A small attention-pooling tower
reads cached sentence embeddings of Steam reviews and produces one 128-d
vector per game. It is trained by a compound objective: a cross-entropy
term (CE) that classifies each fresh review view against a gallery of
every game's anchor pack, and an invariance term (I) that pulls the views
of one game together. The result is a single index that answers both
name-intact and name-stripped queries, and carries enough semantics for a
tag probe.

The tower is about 0.36 M trainable parameters over a frozen
Qwen3-Embedding-0.6B (1024-d, last-token pooling, no instruction prefix).
Every review is sentence-split and embedded once; training thereafter
reads only the cached vectors.

Published with the code, on Hugging Face:

- the cached tensors every experiment reads,
  [`chironYu/latenarray-i2ce-data`](https://huggingface.co/datasets/chironYu/latenarray-i2ce-data)
  (Section 4);
- the trained towers behind the paper's numbers, 204 checkpoints,
  [`chironYu/latenarray-i2ce-checkpoints`](https://huggingface.co/chironYu/latenarray-i2ce-checkpoints),
  with the script that scores them as the paper does
  ([`contrast_experiment/w9/README.md`](contrast_experiment/w9/README.md)).

---

## 1. The paper, its appendices and the supplement

Appendices A to N (Tables A1 to A18) hold what the paper body points to:
the five-fold cross-validation detail with the anchor-budget sweep (A),
the two query registers on one game (B), the temperature,
invariance-weight, view-count, latent-query and VICReg sweeps and the
windowed teacher (C), the gradient bounds behind the handover point of
Section 4.3 (D), the separation-gradient derivations behind Sections 4.3
and 6.3 (E), the query-rewrite prompts (F), the tag-vocabulary mapping
(G), the baseline recipes (H), the cost model at scale with the sharded
sliding window (I), the per-fold detail across anchor budgets (J), the
tag deficit read as a register crossing (K), the neighborhood figures
and the four-encoder map (L), the displacement and margin table (M), and
the document-view study (N). Tables A1 to A6, in Appendices A and C, are
the ones the paper also prints after its references, together with
Appendix B. Supplements A to C (Tables S1 to S4) carry the
teacher-exclusion result, the two anchor-reading regimes and the
training-set versus test-set query comparison.

[`document/APPENDIX.pdf`](document/APPENDIX.pdf) is the reading copy;
[`APPENDIX.md`](APPENDIX.md) renders in the browser and diffs as text;
[`document/APPENDIX.docx`](document/APPENDIX.docx) is the same document in Word.
[`SUPPLEMENT.md`](SUPPLEMENT.md), [`document/SUPPLEMENT.pdf`](document/SUPPLEMENT.pdf) and
[`document/SUPPLEMENT.docx`](document/SUPPLEMENT.docx) are the supplement in the same three
forms; the root keeps only the Markdown, the Word and PDF copies live in `document/`. Appendix letters and table numbers match the citations in the
paper.

Every retrieval number the manuscript prints is **zero-shot cosine retrieval**: no
retrieval head is trained anywhere, a query is the tower's encoding of
text it has never seen, and the candidates are all 2,020 games ranked by
cosine against the anchor gallery. In the result files that is the `zs_*`
/ `zsbest_*` family. The `ft4var_*` files come from the two-phase
name-recall head in `steam_reviews_framework/backhead_name.py`, a
separate line of work the manuscript does not report.

`contrast_experiment/w9/Pod/w9_profiles.py` states, cell by cell, which
training run backs which figure or table. Start there when you want to
reproduce one specific published number.

### Figure 10 (trend river) and its interactive page

[`figures/trend_river/`](figures/trend_river/) holds what the paper's
release-share river is made from: the page that clusters the game vectors
and paints the canvas, the scripts that render the figure, and the bundled
dataset (1,649 games: 128-d I-CE game vectors from the fold-0 run
`i2ce_icetf_fold0_g2048`, plus release times, coarse tags, titles).
`river_flat_fit.py` fits the levels and transfers the figure draws;
`river_ice_column.py` and `river_ice_render.py` regenerate the paper's PNG
headlessly, at one column and at full width. Open
`cluster_share_stack.html` in any browser, no build step, to explore the
clusters: there the page paints its own pressure-driven event model, which
the figure drew until its bands were fitted to the release shares. The
directory's README has the data schema, the fit, both drawings and the
commands.

---

## 2. Using the I2CE model

Dependencies run strictly one way: experiment → framework → main_model.

```
main_model/                the tower alone (LariceTower + LariceConfig),
                           task-agnostic, standalone-publishable.
                           Tensor protocol [data, view]:
                           x[B,V,S,D] + mask  ->  z[B,V,out_dim]
steam_reviews_framework/   the Steam binding: protocol (splits, the
                           inductive exclusion rule, the vsel selection
                           score) / sampler / anchors / trainer /
                           backhead_name / backhead_tag / run.py
contrast_experiment/       the baseline and ablation suite: contrast_models
                           (CE, BYOL, ArcFace, gate and I-dose variants),
                           contrast_heads, run.py, report.py,
                           pod/ (multi-machine route),
                           w9/ (the manuscript's campaign)
dataset_builder/           corpus reconstruction: reviews (Kaggle -> clean
                           -> split -> embed -> h5), corpora (wiki scrape
                           -> clean -> LLM rewrites), build_assets,
                           API templates
figures/                   the interactive trend-river source
data/                      every heavy artefact (gitignored except READMEs)
```

### The tower on its own

`main_model/` is task-agnostic and depends on nothing in this repository.
Any task whose samples lay out as `[data, view]` can use it unchanged:

```python
from main_model import LariceConfig, LariceTower, ice_loss

cfg = LariceConfig(num_queries=4, dim_model=128, input_dim=1024,
                   num_views=4, tau=0.02, inv_weight=2.0, readout="pool")
tower = LariceTower(cfg)

z = tower(x, mask)                          # [B, V, out_dim], L2-normalised
loss = ice_loss(z, gallery, targets, cfg)   # CE (data axis) + I (view axis)
```

`gallery` is the tower's own encoding of every item's anchor set,
recomputed with gradient each step. See [`main_model/README.md`](main_model/README.md).

### The deployable variant (pfc)

The objective the paper reports re-encodes the full anchor gallery with
gradient at every step. The variant proposed for deployment keeps the
tower and the loss and changes only the anchor supply: a sharded sliding
window over a memory-mapped pack store, whose device memory is constant
in the catalog size (Section 8 of the paper; Appendix I, equation (I.3)).
[`main_model/pfc_sample.py`](main_model/pfc_sample.py) is the
self-contained CPU walk-through and
[`main_model/README.md`](main_model/README.md) documents it; the research
implementation is `--pfc-shards K --pfc-window w --full-pool-path <mmap>`
on a swin arm of
[`contrast_experiment/w9/Pod/w9_cv_worker.py`](contrast_experiment/w9/Pod/w9_cv_worker.py).
[`main_model/gallery_sample.py`](main_model/gallery_sample.py) shows how the
anchor packs of the published recipe are built.

### Training on the Steam corpus

```bash
pip install -r requirements.txt

# One I-CE tower (the manuscript's objective) on the fixed split
python steam_reviews_framework/run.py

# ... on a cross-validation fold
python steam_reviews_framework/train_champion.py --cv-fold 0

# ... with the CE-gated ablation instead of the manuscript's objective
python steam_reviews_framework/train_champion.py --arm cegate2

# The whole contrast roster, then the comparison table
python contrast_experiment/run.py [--cv]
```

Every step is resume-safe: rerunning skips whatever already exists, so an
interrupted run continues where it stopped.

Fixed throughout, and identical to the manuscript:

- **Corpus.** 2,020 games, reviews crawled 2020-2024, 6.6 M reviews, 73 M
  sentences. A game enters only if at least 500 of its reviews survive a
  300-character floor.
- **Fully inductive.** No text of a held-out game reaches any training
  stage: not its reviews, its documents, its pseudo-queries, nor a
  gallery-negative gradient. The training-time gallery covers train games
  only; the full 2,020-game gallery is used with a frozen tower at
  evaluation time.
- **Strong views.** Whole reviews, never truncated, drawn by rejection
  sampling with acceptance `a(r) = 0.2 + 0.7*(r-r_min)/(r_max-r_min)`
  recomputed per game, until the view holds at least 16 sentences. Three
  review views plus one document view (raw wiki text where one exists,
  else the store page, else a fourth review view; the LLM-rewritten
  document is the Appendix N ablation).
- **Anchor pack (the weak view).** Store-page prefix, then whole reviews
  to a sentence budget, re-encoded with gradient at every step.
- **Tower.** 4 latent queries, 128-d, 4 heads, one cross-attention layer
  over the raw 1024-d sentence embeddings, mean-pooled over slots
  (`readout="pool"`), then a two-layer MLP to an L2-normalized vector.
- **Optimizer.** AdamW, lr 5e-4, weight decay 1e-4, batch 192 games,
  16 steps per epoch, gradient clipping at 5.0, AMP. Frozen `tau = 0.02`,
  invariance weight 2.0.
- **CE scope.** Every game in the step is classified, which is Equation
  (1) of the manuscript. The CE *gate* — CE firing only on the games that
  carry a document view — is an ablation, reachable with `--arm cegate2`,
  never the published objective.
- **Selection.** Checkpoints every 50 epochs, no online early stopping.
  The deployed checkpoint is picked post-hoc on validation-fold queries,
  never on test.
- **Splits.** `dataset_builder/wiki_eval_split.json` (seed 20260711) is
  the authoritative file and ships with the code. The fixed split is
  204 test / 203 val / 407 train of the 814-game wiki universe; the 204 + 203
  held-out games leave a 1,613-game training gallery. `--cv-fold k` and `--cv` permute
  the same 814 games into five folds (fold k = test, fold k+1 = val, the
  other three = train), which leaves 1,694 or 1,695 per fold.

---

## 3. The three reproduction profiles

The campaign behind the manuscript is
[`contrast_experiment/w9/`](contrast_experiment/w9/README.md): workers,
notebooks, and the minimal model package they import, self-contained so
the folder can be copied to a GPU host on its own.

Which of its runs backs which published number is declared in
[`contrast_experiment/w9/Pod/w9_profiles.py`](contrast_experiment/w9/Pod/w9_profiles.py).
Each cell there carries the figures and tables it feeds, and three
profiles select from it:

| Profile | Scope | Cells | Towers | Anchor budgets |
|---|---|---|---|---|
| `fullTest` | every experiment the paper, the appendix **or** the supplement reports | 19 | 193 | 512 – 2,048 |
| `paperTest` | every experiment the paper body reports, at its own budget | 7 | 67 | 512, 2,048 |
| `litePaperTest` | the same as `paperTest`, every budget clamped to 1,024 | 7 | 67 | 512, 1,024 |

```bash
python contrast_experiment/w9/Pod/w9_profiles.py     # print the three plans
```

Four things worth knowing before you pick one.

**The clamp is a ceiling, not a setting.** `litePaperTest` lowers only the
cells the paper ran above 1,024. A cell the paper ran at 512 stays at 512,
because rerunning it at 1,024 would not reproduce the published row. What
moves is the 2,048 five-fold block, to a budget the paper itself measures
as within 0.02 of the full configuration on every reading (Section 4.2,
Table A2) and which fits a 24 GB desktop GPU.

**`paperTest` and `litePaperTest` train the same number of towers.** They
differ only in what those towers cost. `fullTest` is nearly three times the
count; its largest cell is the temperature sweep (30 towers). The body's
cells include the five-fold towers trained at 512 sentences that Table 2
tests (CE + anchor-in-I against I-CE and CE), beside the main `pool` block.

**Twenty-two tables are not twenty-two campaigns.** Tables A1, A13, A15 and
A16 of the appendix are different readouts of one shared five-fold
@2,048 tower set, and Tables S2 to S4 of the supplement read those same
towers plus the swin and two-stage teachers of `apx.swin.5fold` and
`apx.twostage`. The grid is much smaller than the table count suggests.

**The 2,048-sentence budget comes as two pack rules.** The main five-fold
cells train on `pool` packs: the store-page prefix plus the game's whole
2,048-sentence review pool, with no review left out (a median of 2,068
sentences, the longest 2,184). Their runs and checkpoints are named
`_gpool`; these are the towers of Figure 7 and of the tables that read
them. A cap of 2,048 stops at 2,048 sentences in total, prefix included (a
median of 2,046), and names its runs `_g2048`; the 2,048 rung of the anchor
ladder and the temperature sweep use it. No profile trains above this
budget. The towers are scored against 4,096-sentence anchors, which costs
no training memory, and the window-size sweep, the one cell on
4,096-sentence packs, is not in the paper and in no profile.
[`contrast_experiment/w9/README.md`](contrast_experiment/w9/README.md) has
the details.

Not listed in any profile, because neither the paper, the appendix nor
the supplement mentions them: the twin-pack `pk*` family, the slot/readout capacity
grid, multi-anchor `ma2*`, the async simulation, and hard-negative
`nemesis`. They remain reachable through `w9_jobs.FS_JOBS` and the
notebooks.

### Hardware

The anchor budget sets the requirement, because the gallery is re-encoded
with gradient at every step. A 24 GB desktop GPU covers budgets up to
1,024 sentences, where retrieval has already saturated (a 22.3 GiB
peak, within 0.02 of the full configuration on every reading); the
2,048-sentence budget, the paper's and the default here, peaks at
35.0 GiB at a cap of 2,048 and needs a 48 GB card; a 2,000-epoch run took
4.6 to 6.3 wall-clock hours on A100s shared between folds. `pool` packs
are up to 2,184 sentences wide. Their peak has not been measured at that
width; interpolating between the measured peaks at widths of 2,048 and
4,096 gives about 36.7 GiB.

At inference the cost is ordinary: one forward pass through the frozen
0.6B embedder (under 3 GiB) plus a millisecond-level inner-product search
against the pre-computed anchors.

---

## 4. Building the data

Both entry points share one data-preparation pipeline, and every step is
resume-safe.

```bash
python steam_reviews_framework/run.py --data-only    # prepare, do not train
python dataset_builder/rebuild_data.py --check       # what is missing, and why
```

**Prebuilt data: nothing to rebuild.** Every tensor this pipeline builds is
on Hugging Face at
[`chironYu/latenarray-i2ce-data`](https://huggingface.co/datasets/chironYu/latenarray-i2ce-data).
With that folder in place neither review h5 is needed and no sentence is
embedded again: `run.py` finds every asset, reads the tag labels from
`tag_labels.npz` instead of `text_h5.h5`, and rebuilds the anchor packs at
the profile's budget from the shipped review pool (into
`wscan_gal_rev_pool<budget>.npz`; the shipped `wscan_gal_rev.npz` holds
512-sentence packs and is left as it is). The packaged entry points need
neither the full pool, nor the two w9 galleries, nor the two `_rawprefix`
files; what they do need is about 21 GiB in all:

```bash
hf download chironYu/latenarray-i2ce-data --repo-type dataset --local-dir data/assets \
  --exclude "full_pool_fp16.npy.part*" "wscan_gal_rev_g*.npz" "*_rawprefix.*"
python steam_reviews_framework/run.py --data-only    # reports "the review h5 is not needed"
```

The w9 campaign and `contrast_experiment/w9/eval/k10_eval.py` also read the
140 GiB full pool. Download the whole folder (180 GiB with
`--exclude "*_rawprefix.*"`) and run
`python join_full_pool.py` inside it: it joins `full_pool_fp16.npy` from its
four parts, checks it against `SHA256SUMS` and writes the `full_pool_READY`
marker the notebooks wait for (keep about 150 GB free for the join). That
folder is then the w9 data directory as it is.

The store-page prefix of the anchor packs and the pseudo-queries is
row-standardized like every other input row. Up to 2026-11-07 the
512-sentence `wscan_gal_rev.npz` and `ss_queries_rev_S.npy` carried it raw
(row std about 2.9). The paper's 512-sentence rows come from towers
retrained on the current packs, and the w9 workers name a run on them
`_d512`, as the retrained checkpoints on Hugging Face are named. The files
as they were are kept on Hugging Face as `wscan_gal_rev_rawprefix.npz` and
`ss_queries_rev_S_rawprefix.npy` (10 GiB), for repeating a run of that
time: the w9 workers read them with `--raw-prefix` and name the run without
the tag. `data/assets/README.md` has the details.

Building from scratch instead:

1. **Corpora (bundled, reproducibility first).** `wikipage.zip`
   (wiki_clean / variants / llm, 814 games) and `storepage.zip` (six
   store-page corpora, 1,811 games) ship in
   `steam_reviews_framework/corpora_bundles/` and unpack into
   `data/corpora/`. The bundled texts always win: Wikipedia is never
   re-scraped and the LLM is never re-run, so results cannot drift with
   live wiki edits or non-deterministic rewrites.
2. **Review files.** The 73-million-sentence embedding h5 and the
   text/tag h5 are downloaded when `LARICE_EMBED_H5_URL` /
   `LARICE_TEXT_H5_URL` are set, or rebuilt once from the Kaggle dump via
   `dataset_builder/reviews/`. They are read only to build the tensor
   assets, so this step is skipped when the assets are already there.
3. **Tensor assets.** `dataset_builder/build_assets.py` fills in whatever
   is missing.

### The anchor budget follows the profile

The anchor budget — sentences per pack, `GCAP` — is the number that
decides whether a run fits the machine, because the gallery is re-encoded
with gradient at every step:

| budget | build array (2,020 × cap × 1,024 fp16) | training peak | fits |
|---|---|---|---|
| 512 | 2.1 GB | 8.4 GiB | anything |
| 1,024 | 4.2 GB | 22.3 GiB | 24 GB desktop GPU |
| 2,048 | 8.5 GB | 35.0 GiB | 48 GB card |
| 4,096 (no profile) | 16.9 GB | ~61 GiB | 80 GB A100 only |

**You do not set it by hand.** `--profile` reaches the data build, so the
packs are built at the budget the profile trains at:

```bash
python steam_reviews_framework/run.py --profile litePaperTest   # 1,024
python steam_reviews_framework/run.py --profile paperTest       # 2,048
python contrast_experiment/run.py --profile litePaperTest --cv
```

`dataset_builder/profiles.py` holds that mapping and is the single source
of truth; `w9_profiles.py` reads `LITE_CAP` from it. To build at a budget
no profile names, pass `--anchor-cap N` to `build_assets.py` directly, or
set `LARICE_ANCHOR_CAP`.

Assets already on disk are not silently reused at the wrong size. The
gallery's own second axis *is* the budget it was built at, so the build
compares and stops rather than letting training run at a budget you did
not ask for:

```
anchor budget mismatch: the assets in <dir> were built at 2048 sentences,
this run wants 1024. Training reads the built packs, so it would silently
run at 2048, not 1024.
```

The w9 workers take `--anchor-cap` per job, which is how one campaign
spans several budgets; `GCAP` governs only the packaged path.

The other build constants, at the top of `dataset_builder/build_assets.py`:

| Constant | Default | Meaning |
|---|---|---|
| `CAP`, `TOPK` | 2048, 3 | review pool budget per game / gold-guarantee count |
| `QCAP`, `QPG` | 512, 4 | pseudo-queries: anchor-shaped, 4 per game |
| `SEED` | 20260711 | the split and sampling seed; also `wiki_eval_split.json` |

Credentials live in `dataset_builder/llmAPI.txt` (corpus rewriting) and
`dataset_builder/embeddingAPI.txt` (cloud embedding endpoint), both
gitignored, both with a `*.template.txt` next to them. They can also be
typed into the settings block at the top of
`steam_reviews_framework/run.py`.

### Where the artefacts live

Every heavy artefact lives under `data/`, outside the code tree, and each
location is overridable so an existing layout can be linked in without
copying:

```
LARICE_DATA_ROOT   data root (default: ./data)
LARICE_ASSETS      training/eval tensors     LARICE_RESULTS  checkpoints + result jsons
LARICE_CORPORA     text corpora              LARICE_EMBED_H5 / LARICE_TEXT_H5  review h5
```

See [`data/README.md`](data/README.md) for what each sub-directory holds
and which stage writes it.

### Requirements

Python 3.11+, `numpy`, `torch`, `h5py`, `scikit-learn`, `scipy`,
`requests`. The data pipeline additionally needs `wtpsplit` (the SaT
sentence splitter), `transformers` and `sentence-transformers` (the local
Qwen3-Embedding backend), and `pandas` (Kaggle review preparation). A run
that only trains from prepared assets needs none of that second group.

---

## Citing

*Invariant Game Representation Learning from Steam Reviews.* AICCC 2026.
The appendices and the supplement the paper cites are the documents of
Section 1. Code and documents are under the MIT license ([`LICENSE`](LICENSE)).
