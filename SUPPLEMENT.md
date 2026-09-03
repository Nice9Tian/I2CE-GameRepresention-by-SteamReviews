# Invariant Game Representation Learning from Steam Reviews: Supplementary Material

Supplements A–C hold three studies that the appendices (A–L) no longer carry: the teacher-exclusion result that Appendix L cites (Supplement A, Table S1), the two anchor-reading regimes behind the body’s headline protocol (Supplement B, Tables S2 and S3) and the training-set versus test-set query comparison (Supplement C, Table S4). Section, table and equation numbers without an S or A prefix refer to the body.

## SUPPLEMENT A: THE ANCHOR MUST EXCLUDE THE STRONG VIEWS

The tag gap of Section 8 invites a softer anchor: if the fixed anchor pack, which shares reviews with a strong view only by chance (about 29% of a median game’s sentences sit in the pack at $m_{a}=4,096$), is what forces the strong view to discard view-specific content, then letting the anchor see those sentences might relax the pressure. We tested exactly this (arm vfa): at every step, each batch game’s anchor pack is opened with the game’s four strong views and the fixed pack fills the remainder, so the anchor pack becomes a superset of the strong view’s evidence; evaluation anchors stay the fixed packs. The result refutes the hypothesis and then some (Table S1). Both retrieval axes collapse (Name hit@1 $0.901\to 0.638$, Stripped $0.657\to 0.398$, zero of five paired folds) and the tag reading falls rather than rises ($0.705\to 0.670$). The diagnosis is unambiguous: every fold selects the epoch-50 checkpoint, the earliest written, so the tower is already past its best at the first write and degrades monotonically thereafter. The mechanism is positive leakage. Once the anchor vector $a_{g}$ contains the strong view’s own sentences, the contrastive positive is trivially matched and the softmax stops pushing the strong view to recognize the game from anything else; at evaluation the anchor reverts to the fixed pack the strong view never learned to key on, and identity, with the tag geometry that rides on it, is gone. The anchor must not be built from the strong view’s own sentences: this independence is what lets contrast manufacture identity, and the tag gap is its intrinsic price, not a free lunch a softer anchor can recover.

**Table S1:** Views-first anchors (vfa), a softer anchor that fails. Each batch game’s anchor pack is opened with the strong view’s own four views before the fixed pack; five-fold @512 (test-set queries (ts), a single anchor draw per fold) against the identical-split I-CE reference. Retrieval collapses on both axes (zero of five paired folds) and the tag reading does not rise; every fold selects the epoch-50 checkpoint. The tag F1 column is the review-probe reading (the @512 references predate the held-out test-set probe), consistent within this table. Both rows also predate the selection rule of Section 5 and share the earlier criterion, so the comparison is internal to the table; the collapse it reports is not a selection artifact, since every fold of the softer anchor picks its earliest checkpoint and degrades from there.

| Objective | Name hit@1 | Stripped hit@1 | tag F1 |
| --- | --- | --- | --- |
| Views-first anchors (vfa) @512 | 0.638 ± 0.055 | 0.398 ± 0.041 | 0.670 ± 0.021 |
| I-CE (reference, same split) @512 | 0.901 ± 0.021 | 0.657 ± 0.028 | 0.705 ± 0.015 |

## SUPPLEMENT B: THE TWO ANCHOR-READING REGIMES

Anchor packs are drawn, not given: a game’s reviews far exceed the 4,096-sentence cap, so each pack is one sample of the game’s material (Section 5). This supplement reports every headline tower under both readings of ten such draws. Table S2 is the single-source reading the body uses; Table S3 pools the ten encodings into one index vector first. On Stripped hit@1 the ordering and the sign of every paired comparison are identical in the two tables; the only rank changes outside the Stripped hit@1 column are between towers separated by less than the fold spread; what pooling changes is at most 0.008 Stripped hit@1 on the tower that gains most from averaging its draws, computed before rounding.

**Table S2:** The single-source reading: each of the ten anchor packs is scored on its own and the scores are averaged (five-fold mean ± std, test-set queries (ts), 4,096-sentence anchors). This is the protocol behind the body’s headline numbers, reproduced here for comparison with Table S3.

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE (ours) | 0.954 ± 0.013 | 0.997 ± 0.004 | 0.741 ± 0.018 | 0.928 ± 0.020 | 0.685 ± 0.019 |
| swin-I-CE (~26% gradient window) | 0.934 ± 0.014 | 0.994 ± 0.004 | 0.709 ± 0.027 | 0.922 ± 0.010 | 0.691 ± 0.014 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.930 ± 0.011 | 0.991 ± 0.008 | 0.709 ± 0.019 | 0.917 ± 0.005 | 0.705 ± 0.022 |
| CE (contrast only) | 0.938 ± 0.013 | 0.988 ± 0.010 | 0.673 ± 0.014 | 0.889 ± 0.011 | 0.664 ± 0.016 |
| SimCLR-style (in-batch views) | 0.918 ± 0.024 | 0.990 ± 0.007 | 0.632 ± 0.029 | 0.882 ± 0.022 | 0.707 ± 0.019 |
| I-CE (EMA + memory bank) | 0.936 ± 0.010 | 0.991 ± 0.007 | 0.636 ± 0.016 | 0.881 ± 0.016 | 0.711 ± 0.016 |
| VICReg (expander wiring, v/i/c = 20/10/20) | 0.823 ± 0.010 | 0.960 ± 0.016 | 0.451 ± 0.017 | 0.729 ± 0.033 | 0.708 ± 0.022 |
| BYOL | 0.441 ± 0.033 | 0.737 ± 0.026 | 0.284 ± 0.017 | 0.553 ± 0.043 | 0.712 ± 0.017 |
| Frozen embedder (mean pool) | 0.425 ± 0.033 | 0.626 ± 0.014 | 0.209 ± 0.020 | 0.380 ± 0.031 | 0.588 ± 0.009 |

**Table S3:** The merged-index reading: the ten anchor encodings are averaged into one index vector per game before scoring (same towers, folds and test-set queries (ts) as Table S2). Pooling adds at most 0.008 Stripped hit@1, computed before rounding; the only rank swaps are between towers separated by less than the fold spread.

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE (ours) | 0.956 ± 0.019 | 0.996 ± 0.005 | 0.747 ± 0.021 | 0.931 ± 0.020 | 0.684 ± 0.026 |
| swin-I-CE (~26% gradient window) | 0.941 ± 0.018 | 0.994 ± 0.004 | 0.716 ± 0.026 | 0.926 ± 0.011 | 0.681 ± 0.019 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.932 ± 0.014 | 0.991 ± 0.008 | 0.716 ± 0.019 | 0.920 ± 0.005 | 0.706 ± 0.020 |
| CE (contrast only) | 0.950 ± 0.012 | 0.989 ± 0.008 | 0.678 ± 0.019 | 0.893 ± 0.013 | 0.658 ± 0.020 |
| SimCLR-style (in-batch views) | 0.919 ± 0.027 | 0.990 ± 0.006 | 0.634 ± 0.027 | 0.886 ± 0.021 | 0.704 ± 0.021 |
| I-CE (EMA + memory bank) | 0.944 ± 0.013 | 0.993 ± 0.006 | 0.641 ± 0.012 | 0.887 ± 0.013 | 0.717 ± 0.014 |
| VICReg (expander wiring, v/i/c = 20/10/20) | 0.830 ± 0.012 | 0.962 ± 0.020 | 0.453 ± 0.014 | 0.735 ± 0.037 | 0.705 ± 0.025 |
| BYOL | 0.447 ± 0.034 | 0.743 ± 0.022 | 0.284 ± 0.014 | 0.554 ± 0.046 | 0.715 ± 0.020 |
| Frozen embedder (mean pool) | 0.450 ± 0.034 | 0.642 ± 0.015 | 0.215 ± 0.031 | 0.393 ± 0.035 | 0.578 ± 0.015 |

## SUPPLEMENT C: TRAINING-SET VERSUS TEST-SET QUERIES

Every headline number in this paper is measured on test-set queries (ts), games held out of the fold the tower trained on. Because the 814-game evaluation universe also contains games inside the training set, the same measurement can be repeated on training-set queries (tr): about 488 per fold against 162 or 163 test-set queries, ranked against the same 2,020 candidates by the same finished tower. The difference is a generalization gap, and the frozen embedder anchors its interpretation: having trained on nothing, it scores the same on both populations at this resolution (0.425 / 0.425), so any gap a trained tower shows is what training memorized rather than what the queries happen to ask. Table S4 gives both populations for every headline tower.

**Table S4:** Retrieval on training-set queries (tr) versus test-set queries (ts), five-fold means at the 4,096-sentence anchor budget, single-source reading. $\mathrm{Gap}=(\mathrm{tr})- (\mathrm{ts})$, computed before rounding. The frozen row is the control: no training, and no gap at this resolution.

| Objective | Name hit@1 (tr) | Name hit@1 (ts) | Gap | Stripped hit@1 (tr) | Stripped hit@1 (ts) | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| I-CE (ours) | 0.991 | 0.954 | +0.037 | 0.740 | 0.741 | −0.002 |
| swin-I-CE (~26% gradient window) | 0.983 | 0.934 | +0.049 | 0.716 | 0.709 | +0.007 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.979 | 0.930 | +0.049 | 0.708 | 0.709 | −0.000 |
| CE (contrast only) | 0.986 | 0.938 | +0.048 | 0.735 | 0.673 | +0.062 |
| SimCLR-style (in-batch views) | 0.987 | 0.918 | +0.069 | 0.655 | 0.632 | +0.023 |
| I-CE (EMA + memory bank) | 0.978 | 0.936 | +0.042 | 0.613 | 0.636 | −0.024 |
| VICReg (expander wiring, v/i/c = 20/10/20) | 0.902 | 0.823 | +0.079 | 0.480 | 0.451 | +0.029 |
| BYOL | 0.732 | 0.441 | +0.291 | 0.355 | 0.284 | +0.072 |
| Frozen embedder (mean pool) | 0.425 | 0.425 | +0.000 | 0.209 | 0.209 | −0.000 |

Two readings stand out. On name-intact queries every trained tower gives something back when the game is new, from 0.037 for I-CE to 0.291 for BYOL, and the ordering of that gap tracks how much of a tower’s identity resolution is anchored rather than remembered. On name-stripped queries the contrast is sharper still: plain CE scores 0.735 on games it trained on but only 0.673 on held-out ones, a 0.062 gap, while I-CE reads 0.740 and 0.741, a gap of −0.002 that runs the other way. The invariance term raises the name-stripped score and simultaneously removes the memorization component from it.
