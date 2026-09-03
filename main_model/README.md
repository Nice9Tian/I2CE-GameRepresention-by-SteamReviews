# larice — Latent Represent I-CE

A small (≈ 0.36 M params), task-agnostic set-representation tower
(architecture lineage: SetPoolN) that turns a *set* of upstream
embeddings (sentences, patches, events, …) into one invariant vector.
`Q` learnable latent queries (`num_queries`) cross-attend over the set; the training
objective factorises into two orthogonal forces:

- **CE (data axis)** — an InfoNCE pull toward the item's *anchor* in a
  gallery of all items, buying **identity**;
- **I (view axis)** — cosine alignment across independently sampled views
  of the same item, buying **semantic stability**.

Both terms fire on every item in the step: negatives buy identity,
alignment buys meaning. That is the objective the manuscript reports.

`ce_loss` also takes an optional `gate` mask, and it is **off by
default**. It exists for one family of ablations (`cegate*`, `igate*`,
`rgate*` in `contrast_experiment/`) that restrict which items contribute
a CE *positive* term; masked-out items stay gallery negatives throughout.
Passing a gate is a deliberate departure from the published recipe.

## Tensor protocol

The leading two axes of every input are **`[data, view]`**:

```
x    : [B, V, S, D_in]   float   S = set elements, D_in = upstream dim
mask : [B, V, S]         bool    True = padding
out  : [B, V, out_dim]           L2-normalised
```

Single-view tasks use `V = 1` (a rank-3 `[B, S, D_in]` input is accepted
and treated as `V = 1`). Loss semantics follow the axes: `invariance_loss`
reduces along **view**, `ce_loss` reduces along **data**. Any task whose
samples can be laid out this way can use the tower unchanged.

## Readout

Default output is the **concat** of the `N` query slots
(`out_dim = Q × dim_model`) — the general-purpose representation; each
slot is a deployable sub-space of its own.

> **Note — name/identity recall:** for retrieval-style tasks where the
> query must hit one specific item (name recall), the **pooled** readout
> (`readout="pool"`: mean over slots → MLP → L2) consistently outperforms
> concat in our experiments. Use `LariceConfig(readout="pool")` there.

## Usage

```python
from main_model import LariceConfig, LariceTower, ice_loss

cfg = LariceConfig(num_queries=4, dim_model=128, input_dim=1024,
                  num_views=4, tau=0.02, inv_weight=2.0, readout="pool")
tower = LariceTower(cfg)

z = tower(x, mask)                      # [B, V, out_dim]
loss = ice_loss(z, gallery, targets, cfg)          # published objective
# loss = ice_loss(z, gallery, targets, cfg, gate=has_doc_view)  # ablation
```

`ice_loss` / `ce_loss` were previously named `champion_loss` /
`gated_ce_loss`; both old names still import as aliases.

`gallery` is the tower's own encoding of every item's anchor set
(recomputed each step, gradients on — "batch = all" negatives). A fixed,
medium-cardinality anchor per item removes the need for EMA targets,
negative mining, or memory queues.

Config surface: `num_queries (Q)`, `num_views (NV)`, `tau`
(frozen / learnable), `inv_weight`, `dim_model`, `num_heads`,
`input_dim`, `readout`. The CE gate is not part of it: it is an
ablation-only argument of the loss call, defaulting to off.

## Anchor gallery: how the packs are set up (`gallery_sample.py`)

`gallery` is not a free choice: the published recipe draws one fixed anchor
pack per item (store-page or document sentences first,
then whole reviews in a fixed random permutation until the sentence budget
`m_a` is full, a review that does not fit being skipped rather than cut),
and re-encodes every training item's pack at every step with gradients on.
`gallery_sample.py` is a self-contained, CPU-runnable walk-through of that
supply on synthetic embeddings: pack construction (`build_anchor_pack`), the
standing gallery (`encode_gallery`, the `gallery_train` pattern), the fresh
strong views (`sample_review_view`, the length-based acceptance rule) and
the `ice_loss` call. The production code is `dataset_builder/build_assets.py`
and `steam_reviews_framework/{anchors,sampler,train}.py`. The windowed
teacher (swin) of the appendix is a different anchor supply and lives in
`contrast_experiment/w9/Pod/w9_a100_worker.py`
(arm `wcle_swin168step84loop2i2ce_icetf`; five-fold in `w9_cv_worker.py`).

## Deployable variant: sharded sliding window over a memory-mapped store (`pfc_sample.py`)

The full standing gallery is the objective the manuscript reports; the
variant we propose for deployment keeps the tower and the loss and changes
only the anchor supply. The catalog sits on a ring split into `K` shards (the `κ` of Appendix H);
per micro-pass each shard fresh-encodes the next `w` packs after its own
pointer (gradients on, gradient-checkpointed so only the `[w, dim]` outputs
stay live) and advances by `S`; the CE softmax of a view runs over
`[own anchors of the batch | window of shard 1 | ... | window of shard K]`
with batch items masked inside the windows; each micro-pass is
back-propagated immediately and the invariance term once per step; the pack
store is a memory-mapped fp16 array from which only the current windows are
paged in. Device memory is constant in the catalog size and a `K`-shard
single process computes exactly what `K` workers would with an all-reduce
of their window logits. Per-step compute is Θ(T·(G + K·w)·m_a·Q·d), equation
(H.3) of Appendix H (which writes the shard count κ), and `K = 1` is the
windowed teacher of the appendix (`w = 168`, `S = 84`, `T = 2`). `pfc_sample.py` is the self-contained CPU
walk-through (`ShardedRing`, `encode_window`, `pfc_step`); the research
implementation is `--pfc-shards K --pfc-window w --full-pool-path <mmap>`
on a swin arm (`wcle_swin168step84loop2i2ce_icetf`, `w = 168`, `S = 84`,
`T = 2`) of `contrast_experiment/w9/Pod/w9_cv_worker.py`.
