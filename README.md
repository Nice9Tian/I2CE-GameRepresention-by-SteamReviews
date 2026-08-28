# I2CE — Invariant Game Representation Learning from Steam Reviews

Code, data pipeline and supplementary material for the I-CE representation
described in the submitted manuscript. A small attention-pooling tower
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

---

## 1. The paper and its appendices

Appendices A to L and Tables A1 to A15: the separation-gradient
derivation behind Section 4.3, the per-fold cross-validation detail, the
baseline recipes, the cost model at scale, the document-view study, the
windowed teacher and its three sweeps, the teacher-exclusion result, the
two anchor-reading regimes, the trainset/testset comparison, the
query-rewrite prompts, and the tag-vocabulary mapping.

[`APPENDIX.pdf`](APPENDIX.pdf) is the reading copy;
[`APPENDIX.md`](APPENDIX.md) renders in the browser and diffs as text;
[`APPENDIX.docx`](APPENDIX.docx) is the same document in Word. Appendix
letters and table numbers match the citations in the paper.

Every number the manuscript prints is **zero-shot cosine retrieval**: no
retrieval head is trained anywhere, a query is the tower's encoding of
text it has never seen, and the candidates are all 2,020 games ranked by
cosine against the anchor gallery. In the result files that is the `zs_*`
/ `zsbest_*` family. The `ft4var_*` files come from the two-phase
name-recall head in `steam_reviews_framework/backhead_name.py`, a
separate line of work the manuscript does not report.

`contrast_experiment/w9/Pod/w9_profiles.py` states, cell by cell, which
training run backs which figure or table. Start there when you want to
reproduce one specific published number.

### Interactive Figure 6 (trend river)

[`figures/trend_river/`](figures/trend_river/) holds the interactive
generator of the paper's content-trend river plus its bundled dataset
(1,919 games: tower centroids, release times, coarse tags, titles). Open
`cluster_share_stack.html` in any browser, no build step. Its README has
the data schema and the pressure-driven event model behind the stream
junctions.

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

- **Corpus.** 2,020 games released 2020-2024, 6.6 M reviews, 73 M
  sentences. A game enters only if at least 500 of its reviews survive a
  300-character floor.
- **Fully inductive.** No text of a held-out game reaches any training
  stage: not its reviews, its documents, its pseudo-queries, nor a
  gallery-negative gradient. The training-time gallery covers train games
  only; the full 2,020-game gallery is used with a frozen tower at
  evaluation time.
- **Strong views.** Whole reviews, never truncated, drawn by rejection
  sampling with acceptance `a(L) = 0.2 + 0.7*(L-Lmin)/(Lmax-Lmin)`
  recomputed per game, until the view holds at least 16 sentences. Three
  review views plus one document view (LLM-rewritten wiki where one
  exists, else the store page, else a fourth review view).
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
  204 test / 203 val / 407 excluded of the 814-game wiki universe, which
  leaves a 1,613-game training gallery. `--cv-fold k` and `--cv` permute
  the same 814 games into five folds (fold k = test, fold k+1 = val, the
  other three = train), which leaves 1,694 per fold.

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
| `fullTest` | every experiment the paper **or** the appendix reports | 14 | 150 | 512 – 4,096 |
| `paperTest` | every experiment the paper body reports, at its own budget | 5 | 52 | 512, 4,096 |
| `litePaperTest` | the same as `paperTest`, every budget clamped to 1,024 | 5 | 52 | 512, 1,024 |

```bash
python contrast_experiment/w9/Pod/w9_profiles.py     # print the three plans
```

Three things worth knowing before you pick one.

**The clamp is a ceiling, not a setting.** `litePaperTest` lowers only the
cells the paper ran above 1,024. A cell the paper ran at 512 stays at 512,
because rerunning it at 1,024 would not reproduce the published row. What
moves is the 4,096 five-fold block, to a budget the paper itself measures
as within 0.02 of the full configuration on every reading (Section 4.2,
Table A2) and which fits a 24 GB desktop GPU.

**`paperTest` and `litePaperTest` train the same number of towers.** They
differ only in what those towers cost. `fullTest` roughly triples the
count, mostly through two cells: the anchor ladder (30 towers) and the
temperature sweep (30 towers).

**Fifteen appendix tables are not fifteen campaigns.** Tables A6, A13,
A14 and A15 are four different readouts of one shared five-fold @4,096
tower set. The grid is much smaller than the table count suggests.

Not listed in any profile, because neither the paper nor the appendix
mentions them: the twin-pack `pk*` family, the slot/readout capacity
grid, multi-anchor `ma2*`, the async simulation, and hard-negative
`nemesis`. They remain reachable through `w9_jobs.FS_JOBS` and the
notebooks.

### Hardware

The anchor budget sets the requirement, because the gallery is re-encoded
with gradient at every step. A 24 GB desktop GPU covers budgets up to
2,048 sentences, and retrieval has already saturated by 1,024 (a 22.3 GiB
peak, within 0.02 of the full configuration on every reading). The
4,096-sentence budget occupies a single 80 GB A100, about 61 GiB peak and
roughly 6.7 wall-clock hours for a 2,000-epoch run.

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
   `dataset_builder/reviews/`.
3. **Tensor assets.** `dataset_builder/build_assets.py` fills in whatever
   is missing.

### The parameters that matter

Set at the top of `dataset_builder/build_assets.py`:

| Constant | Default | Meaning |
|---|---|---|
| `GCAP` | 4096 | anchor budget: sentences per pack (doc prefix, then whole reviews) |
| `CAP`, `TOPK` | 2048, 3 | review pool budget per game / gold-guarantee count |
| `QCAP`, `QPG` | 512, 4 | pseudo-queries: anchor-shaped, 4 per game |
| `SEED` | 20260711 | the split and sampling seed; also `wiki_eval_split.json` |

**`GCAP` is the one to think about.** It sizes the anchor gallery the
whole campaign trains against, and it costs: the build allocates
`2,020 x GCAP x 1,024` fp16 in host RAM and writes it out, so 4,096 needs
about 17 GB against 2 GB at 512. Match it to the profile you intend to
run — `GCAP = 1024` for `litePaperTest` on a desktop GPU, 4,096 for
`paperTest` and `fullTest`. The w9 workers override it per job with
`--anchor-cap`, so this constant governs only the packaged path.

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
