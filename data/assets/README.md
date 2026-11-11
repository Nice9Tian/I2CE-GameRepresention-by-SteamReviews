# data/assets/ — training / evaluation tensor assets

All produced by `dataset_builder/build_assets.py` (fixed SEED=20260711;
resume-safe: existing files are skipped). `gidx` always indexes rows of
`games.npz`.

| File | Structure | Meaning |
|---|---|---|
| `games.npz` | `names` (2020,) object | **the gidx basis** — order = `game_names` of embedding_h5 |
| `wiki_clean_views.npz` `wiki_llm_views.npz` `sp_raw_views.npz` | `S` (N, MAXLEN, 1024) fp16 zero-padded; `S_len` (N,) i32; `gidx` (N,) i64; `names` (N,) object | full-document views: SaT split (no cap) + per-sentence embedding. Row-standardized (rown) at load time before entering the tower |
| `wscan_pool_rev.npy` | (2020, 2048, 1024) fp16 | the review pool: 2,048-sentence budget per game; **gold guarantee** = the 3 longest reviews enter unconditionally, the rest fill in whole at random; vectors already per-row mean0/std1 |
| `wscan_pool_rev_rid.npy` | (2020, 2048) i32 | per-row review id in the pool (−1 = padding) — the basis of review-level rejection sampling |
| `wscan_pool_rev_len.npy` | (2020,) i32 | used rows per game |
| `wscan_gal_rev.npz` | `gal` (2020, 512, 1024) fp16; `gal_len` i32; `gal_doc_len` i32 | **the anchors**: sp_raw document sentences FIRST, row-standardized like every other row, whole reviews fill the 512 budget; `gal_doc_len` = doc-prefix length (0 = no doc) — the head's doc-gating masks the prefix through it |
| `wscan_gal_rev_rawprefix.npz` | as above | the 512 packs as built before 2026-11-07, sp_raw prefix copied raw (row std ~2.9): what the @512 runs made before that date trained on; only for repeating them (see the note below) |
| `ss_queries_rev.npz` | `off` (8081,) i64; `gidx` (8080,) i64 | pseudo-query index: 4 per game, anchor-shaped (doc prefix + whole reviews @512), flat-stored |
| `ss_queries_rev_S.npy` | (Σ sentences, 1024) fp16, ~8.5 GB | pseudo-query vectors (sp_raw prefix row-standardized); query j = `S[off[j]:off[j+1]]`. **Loaded as mmap**. Report-only diagnostics, never a selection or a published number |
| `ss_queries_rev_S_rawprefix.npy` | as above | the pseudo-query vectors as built before 2026-11-07 (raw sp_raw prefix) |
| `wiki_eval.npz` | `S` (3256, MAXLEN, 1024) fp16; `S_len`; `gidx`; `names`; `variants` object | evaluation queries = 814 games × 4 variants, **complete text, no truncation** |
| `_tag_split.json` | train/val/test game-name lists | the tag-probe split (seed 42, 0.7/0.15) |

The protocol split is NOT in this directory: the authoritative
`wiki_eval_split.json` (seed 20260711, 204/203/407) ships with the code
under `dataset_builder/`.

**The store-page prefix (2026-11-07).** `build_assets.py` used to copy the sp_raw
sentences raw into the anchor packs and the pseudo-queries, in front of review
sentences that are row-standardized; the tower has no input normalization, so in
the 512-sentence packs the prefix entered at about three times the scale of every
other input. The w9 workers' own GPU-built packs (1,024 sentences and above) always
took the prefix from `load_views("sp_raw_views.npz")`, i.e. standardized, and the
prebuilt `wscan_gal_rev_g1024.npz` / `_g4096.npz` carry it standardized. `build_assets.py` now
standardizes the prefix the same way (`rown_prefix`), and the prebuilt data on
Hugging Face carries the fixed `wscan_gal_rev.npz` and `ss_queries_rev_S.npy`. The
paper's @512 rows come from towers retrained on the fixed packs. The w9 workers
(`contrast_experiment/w9/Pod/`) name a run on them `_d512` and stop when the
`wscan_gal_rev.npz` they are given still holds the prefix raw. To repeat a run made
before the fix, pass the worker `--raw-prefix`: it reads `wscan_gal_rev_rawprefix.npz`
(and `ss_queries_rev_S_rawprefix.npy` for the report-only pseudo-query columns) and
names the run as it was named then, without the tag.
