# Invariant Game Representation Learning from Steam Reviews

Appendices A-H

*Supplementary material to the AICCC 2026 submission (PA0049)*

## APPENDIX A: FIVE-FOLD CROSS-VALIDATION DETAIL

Table A1 gives the per-fold results behind Table A2's largest-budget row: I-CE at the 4,096-sentence anchor budget fold by fold, with its seed-paired CE partner (identical fold splits) summarized as mean ± std, evaluated zero-shot under the protocol of Section 5 (validation-selected checkpoints, Section 5.3; retrieval among all 2,020 games).

*Table A1: Per-fold detail at the 4,096-sentence anchor budget, testset queries (ts). I-CE beats CE on Stripped hit@1 in five of five paired folds at this budget, and in 20 of 20 across the four budgets of Table A2.*

| Fold | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE fold 0 | 0.970 | 1.000 | 0.728 | 0.920 | 0.689 |
| I-CE fold 1 | 0.955 | 1.000 | 0.745 | 0.931 | 0.665 |
| I-CE fold 2 | 0.931 | 0.989 | 0.715 | 0.904 | 0.662 |
| I-CE fold 3 | 0.952 | 0.994 | 0.749 | 0.922 | 0.697 |
| I-CE fold 4 | 0.962 | 1.000 | 0.769 | 0.963 | 0.709 |
| **I-CE mean ± std** | **0.954 ± 0.013** | **0.997 ± 0.004** | **0.741 ± 0.018** | **0.928 ± 0.020** | **0.685 ± 0.019** |
| **CE mean ± std** | **0.938 ± 0.013** | **0.988 ± 0.010** | **0.673 ± 0.014** | **0.889 ± 0.011** | **0.664 ± 0.016** |

*Table A2: Ablation 1 in full — CE vs I-CE at anchor budgets of 512–4,096 sentences (testset queries (ts), five-fold mean ± std, the two objectives seed-paired per fold on identical splits). Increasing the budget yields one significant step — 512 to 1,024 sentences (p = 0.003) — followed by a plateau in which the three larger budgets are indistinguishable (p ≥ 0.20); I-CE leads CE at every budget (20 of 20 paired folds on Stripped hit@1), so the invariance constraint remains beneficial at all four scales. The windowed teacher (Appendix D) appears at its trained budget.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| CE @512 | 0.891 ± 0.018 | 0.983 ± 0.005 | 0.614 ± 0.011 | 0.853 ± 0.019 | 0.687 ± 0.011 |
| **I-CE @512** | **0.916 ± 0.009** | **0.991 ± 0.003** | **0.644 ± 0.024** | **0.872 ± 0.010** | **0.692 ± 0.013** |
| CE @1024 | 0.922 ± 0.016 | 0.988 ± 0.011 | 0.655 ± 0.007 | 0.873 ± 0.022 | 0.668 ± 0.013 |
| **I-CE @1024** | **0.942 ± 0.015** | **0.997 ± 0.002** | **0.727 ± 0.006** | **0.918 ± 0.006** | **0.693 ± 0.022** |
| CE @2048 | 0.929 ± 0.011 | 0.989 ± 0.007 | 0.671 ± 0.015 | 0.882 ± 0.014 | 0.657 ± 0.014 |
| **I-CE @2048** | **0.947 ± 0.016** | **0.996 ± 0.003** | **0.728 ± 0.009** | **0.918 ± 0.010** | **0.684 ± 0.013** |
| CE @4096 | 0.938 ± 0.013 | 0.988 ± 0.010 | 0.673 ± 0.014 | 0.889 ± 0.011 | 0.664 ± 0.016 |
| **I-CE @4096** | **0.954 ± 0.013** | **0.997 ± 0.004** | **0.741 ± 0.018** | **0.928 ± 0.020** | **0.685 ± 0.019** |
| swin-I-CE @4096 (windowed) | 0.934 ± 0.014 | 0.994 ± 0.004 | 0.709 ± 0.027 | 0.922 ± 0.010 | 0.691 ± 0.014 |

Two observations hold at fold level. First, the invariance gain is not a split artifact: I-CE wins Stripped hit@1 in every fold at this budget, while the tag reading rises alongside. Second, the fold-to-fold spread is large — up to ±0.03 on Stripped hit@1 — which is why this paper quotes no single-split difference smaller than that anywhere. The EMA + memory-bank run has since landed and is tabulated in Table A13, the BYOL and VICReg runs in the body's Figure 3.

## APPENDIX B: THE BASELINE RECIPES OF FIGURE 3

Every trained baseline shares the full scaffolding of Section 4 — the same 4-query cross-attention tower, the same four views per game per step (three review views and one document view), the same corpus, batch (192 games), optimizer (AdamW, lr 5e-4, weight decay 1e-4), 2,000-epoch budget, temperature τ = 0.02 where a softmax exists, and the validation-only selection of Section 5.3. They differ only in the loss, exactly as follows.

Frozen embedder (mean pool). No tower and no training: a game is the L2-normalized mean of its anchor-pack sentence embeddings, and a query is the L2-normalized mean of its own sentences. This row measures the raw geometry every other row starts from.

CE (contrast only). Equation (2) alone: each of the four student views is classified against the full teacher gallery, re-encoded with gradient at every step; the invariance term of Equation (1) is removed. This isolates what the softmax supplies without multi-view agreement.

SimCLR-style (in-batch views). No anchors anywhere in training. The step's 4 × 192 view encodings form the key set of an NT-Xent loss [4]: view v of a game takes its ring sibling — view v+1 (mod 4) of the same game — as the positive, the game's other own views are masked as false negatives, and every other game's views serve as negatives at τ = 0.02. The tower is unchanged; only the contrast lives in view space instead of anchor space. The anchor gallery is encoded once at evaluation time, by the finished tower.

VICReg (epd 20/10/20). No negatives and no gallery. Each view encoding passes through an expander MLP, and the VICReg loss [1] is applied to expander outputs over the six view pairs with variance/invariance/covariance weights 20/10/20; views are drawn for 192 games per step, with the variance term estimating per-dimension spread within the step's view batch; drawing the whole pool per step (batch = all) was tested and yielded no improvement (Appendix Table A6). Anchors are encoded only at evaluation time. Appendix Table A6 tabulates the weight sweep behind this recipe.

BYOL. An online tower plus a two-layer predictor head, and a target tower that is an exponential moving average of the online weights (momentum 0.996) behind a stop-gradient. The loss is 1 − cos between the predictor's output for one view and the target tower's encoding of another, averaged over ordered view pairs; the document view participates like any other. Collapse is prevented by the predictor/EMA asymmetry rather than by negatives.

CE + anchor-in-I (Table 2). Equation (2) unchanged, but the invariance term of Equation (1) ranges over the ten pairs among the four student views and the game’s own teacher anchor, instead of the six pairs among views alone. It is the control of Section 6.2: the same constant-weight attraction on the anchor that the decomposed objectives use, with the softmax left in place. Two-stage: BYOL warm start → windowed (Table A13). Stage one is the BYOL recipe above, run to its own selection point; stage two re-initializes the tower from those weights and trains the windowed teacher of Appendix D for 600 epochs, 30% of the from-scratch discriminative budget. Selection happens inside stage two. I-CE (EMA + memory bank; Table A13). The fully coupled teacher is replaced by an EMA shadow tower (momentum 0.99) that encodes the current batch's anchors into a 3,072-key memory bank; each view runs a cross-entropy against the ring (its freshly written key is the positive, stale same-game keys are masked), and the invariance term is unchanged. Gradient reaches the student side only — the property whose price Table A13 measures.

## APPENDIX C: THE DOCUMENT VIEW — PRESENT, ABSENT, REWRITTEN

The document view is one of the four student views, and this ablation isolates it at the 512-sentence anchor budget — five folds per cell, seed-paired with the raw-document tower (Table A3). Two verdicts. First, the document view helps, modestly: removing it costs 0.008 Stripped hit@1 and 0.032 Name hit@1, with tag readings flat — the document supplies a cross-register bridge for name-free retrieval, not semantic breadth. Second, and less obviously, rewriting the document is better than keeping it: the LLM-rewritten tower leads the raw-document tower on the stripped columns (+0.008 Stripped hit@1, +0.011 Stripped hit@5) and ties it on the name columns, while reading tags -0.002 higher. This is the contamination of Section 2.2 showing up as a measurable effect rather than a worry. Wikipedia-derived text is exactly the material a public sentence embedder is likely to have memorized, and a memorized document pulls the game’s representation toward a recalled surface instead of the review consensus we want it to summarize; a faithful sentence-wise rewrite preserves the content while breaking the verbatim match, so it acts as a decontaminating filter. The firewall we adopted for safety turns out to pay for itself, and single-split readings of this ablation — which had ranged from apparent parity to an apparent 0.05 collapse depending on the selected checkpoint — are superseded by the fold-level account above.

*Table A3: The document view — present, absent, rewritten (testset queries (ts), five-fold mean ± std at the 512-sentence anchor budget, validation-selected checkpoints, zero-shot, seed-paired folds). The document view is worth 0.008 Stripped hit@1 (and 0.032 Name hit@1), and rewriting it helps: the LLM-rewritten row leads raw documents on the stripped columns (+0.008 Stripped hit@1) and ties on the name columns. The firewall is not a cost but a gain.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| **I-CE, document view present** | **0.916 ± 0.009** | **0.991 ± 0.003** | **0.644 ± 0.024** | **0.872 ± 0.010** | **0.692 ± 0.013** |
| I-CE, no document view | 0.883 ± 0.016 | 0.981 ± 0.007 | 0.636 ± 0.016 | 0.871 ± 0.013 | 0.680 ± 0.024 |
| I-CE, LLM-rewritten documents | 0.914 ± 0.010 | 0.990 ± 0.002 | 0.652 ± 0.023 | 0.882 ± 0.009 | 0.689 ± 0.016 |

## APPENDIX D: THE WINDOWED TEACHER AND THE WINDOW-SIZE, VICREG, AND TEMPERATURE SWEEPS

The windowed teacher deferred from Section 4 is specified below, followed by the three sweeps behind design choices. The window-size scan and the VICReg grid are single-split and should be read at the ±0.03 resolution of Section 5.3; the temperature sweep is five-fold.

swin-I-CE bounds the fully coupled teacher’s memory ceiling. The catalog is laid out on a ring with a persistent pointer p. Each step runs two micro-passes: micro-pass l ∈ {0, 1} freshly encodes the W = 168 anchors starting at ring position p + l·S (stride S = 84, so consecutive passes overlap by half a window) together with the current batch’s own anchors, forms its own CE partition — every view’s positive sits in the batch block, the window anchors serve as negatives — and backpropagates immediately, freeing the window’s activations before the next pass; the pointer then advances by 2S, sweeping the full catalog every ten steps so that every anchor takes a regular, gradient-coupled turn. Per step this touches about 27% of the gallery with gradient, bounding the gradient-coupled encoding to 22% of the full backward graph (Table A5); the tower, the views, and the invariance term are unchanged. Table A13 quantifies the cost of this variant five-fold; the window-size sweep (W = 84–336) is Table A4.

*Table A4: The window-size sweep behind the windowed teacher (fixed split, testset queries (ts) of that split, 4,096-sentence anchors, validation-selected checkpoints; a single anchor draw rather than the ten-draw expectation of the main tables). Coverage counts the training-fold anchor packs re-encoded with gradient per step (batch plus the union of the two micro-pass windows). Halving or doubling the window moves nothing beyond single-split noise — the smallest window already holds the plateau, so memory, not coverage, should pick W. Tag readings are the test-set probe of Section 5.3, recomputed on this split so that they sit on the same scale as the rest of the paper; their spread here is single-split noise, and at fold level swin and full coupling read tags at parity (Table A13, which gives the five-fold account of W = 168).*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| swin, W = 84 (step 42, 19.7% coverage) | 0.946 | 1.000 | 0.691 | 0.902 | 0.714 |
| **swin, W = 168 (step 84, 27.5% coverage)** | **0.941** | **0.995** | **0.691** | **0.892** | **0.704** |
| swin, W = 336 (step 168, 43.1% coverage) | 0.941 | 1.000 | 0.686 | 0.897 | 0.668 |
| **I-CE, fully coupled (100% coverage)** | **0.951** | **1.000** | **0.706** | **0.917** | **0.679** |

*Table A5: What the sliding window bounds. The teacher gallery is the activation ceiling of the fully coupled objective — it re-encodes every training anchor pack with gradient at once — whereas swin holds only the batch's own anchors plus one W = 168 window per micro-pass and frees it before the next, so the gradient-coupled encoding at its peak is 360 anchors, 22% of the full graph. Both variants keep the whole catalog resident: swin bounds the activation, not the store, which is why the memory saving is bounded by the persistent ring rather than reaching the full 4.5×. Anchor packs are capped at 4,096 sentences and realized packs are shorter, so the sentence-slot counts are ceilings. Counts are quoted on the fixed split of Appendix D (1,613 training games); a five-fold gallery is 1,694 and scales the same way.*

|  | Full-gallery I-CE | swin-I-CE (W = 168) |
| --- | --- | --- |
| **Anchors re-encoded with gradient per backward (peak)** | 1,613 | 360 (192 batch + 168 window) |
| **as a fraction of the training gallery** | 100% | 22% |
| **Sentence-slots in the backward graph (4,096 budget)** | ≈ 6.6 M | ≈ 1.5 M |
| **Anchor packs held resident (the ring)** | 1,613 (all) | 1,613 (all) |

The VICReg grid below is older than the rest of the protocol: it predates validation selection and its towers were logged only at trajectory peaks, so Table A6 reports each arm's peak stripped hit@1 and omits tag readings, which cannot be re-scored under the test-set protocol of Section 5.3.

*Table A6: The VICReg weight sweep (fixed split, testset queries (ts) of that split, 512-sentence anchors at evaluation, a single anchor draw; trajectory peaks). All rows use the canonical wiring — variance, invariance, and covariance all on the expander-output pair; the earlier centroid wiring (invariance as an MSE between unit-norm view centroids) collapsed retrieval outright at every view width (hit@1 ≤ 0.03) and is omitted. V/I/C = 20/10/20 is the best cell and the image-domain 25/25/1 trails it. Drawing views for the whole training pool per step (batch = all) provides true population moments in the variance term at roughly eight times the step cost; its one completed fold scored within fold noise of the batch-192 recipe (stripped hit@1 0.387 under the single-draw protocol, against that recipe’s 0.440 ± 0.021 five-fold mean under the averaged-anchor protocol), so the body's five-fold VICReg run (Figure 3) trains at batch 192. I-CE and CE peaks under the same readout anchor the scale.*

| Objective | Name hit@1 | Stripped hit@1 |
| --- | --- | --- |
| V/I/C = 25/25/1 | 0.373 | 0.216 |
| **V/I/C = 20/10/20 (recipe weights)** | **0.578** | **0.284** |
| V/I/C = 20/10/15 | 0.574 | 0.265 |
| 25/25/1, batch = all | 0.328 | 0.201 |
| 20/10/20, batch = all | 0.495 | 0.250 |
| **I-CE (reference, same readout)** | **0.926** | **0.672** |
| CE (reference, same readout) | 0.922 | 0.618 |

The temperature sweep, unlike the two scans above, is five-fold and carries the Section 7 verdict directly.

*Table A7: The temperature sweep behind the Section 7 verdict and the frozen setting of Section 4.2 (testset queries (ts), five-fold mean ± std at the 512- and 2,048-sentence anchor budgets; the τ = 0.10 cell at 512 has four folds). Among fixed settings τ = 0.02 is retrieval-optimal at both budgets: softening to 0.05/0.10 costs 0.060–0.226 stripped hit@1, and what it buys is the tag reading — τ = 0.05 lifts it into the negative-light band of Section 6.1 — so on the soft side temperature trades the tag reading against identity. The learnable variant (initialized at 0.02, τ clamped to [0.005, 0.2]) never settles at an interior optimum: in all five folds it sharpens monotonically to the clamp floor within 250 epochs and trains there, an effective fixed τ = 0.005, beating the frozen recipe by 0.010 stripped hit@1 in five folds of five and by 0.008 on tags — the sharp-side headroom that the frozen-for-comparability setting of Section 4.2 leaves to deployment.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| **τ = 0.02 (recipe), 512** | **0.916 ± 0.009** | **0.991 ± 0.003** | **0.644 ± 0.024** | **0.872 ± 0.010** | **0.692 ± 0.013** |
| τ = 0.05, 512 | 0.896 ± 0.017 | 0.987 ± 0.007 | 0.584 ± 0.021 | 0.859 ± 0.014 | 0.704 ± 0.017 |
| τ = 0.10, 512 (four folds) | 0.828 ± 0.027 | 0.956 ± 0.020 | 0.490 ± 0.019 | 0.770 ± 0.016 | 0.701 ± 0.021 |
| **τ = 0.02 (recipe), 2,048** | **0.947 ± 0.016** | **0.996 ± 0.003** | **0.728 ± 0.009** | **0.918 ± 0.010** | **0.684 ± 0.013** |
| τ = 0.05, 2,048 | 0.938 ± 0.018 | 0.992 ± 0.004 | 0.655 ± 0.027 | 0.890 ± 0.015 | 0.710 ± 0.015 |
| τ = 0.10, 2,048 | 0.883 ± 0.032 | 0.978 ± 0.014 | 0.502 ± 0.012 | 0.785 ± 0.012 | 0.707 ± 0.015 |
| learnable τ, 2,048 | 0.951 ± 0.008 | 0.995 ± 0.005 | 0.738 ± 0.006 | 0.920 ± 0.015 | 0.693 ± 0.015 |

## APPENDIX E: THE TEACHER MUST EXCLUDE THE STUDENT’S VIEWS

The tag gap of Section 7 invites a softer teacher: if the fixed anchor pack — which never contains the sentences a student view just read — is what forces the student to discard view-specific content, then letting the teacher see those sentences might relax the pressure. We tested exactly this (arm vfa): at every step, each batch game’s teacher pack is opened with the game’s four fresh student views and the fixed pack fills the remainder, so the teacher becomes a superset of the student’s evidence; evaluation anchors stay the fixed packs. The result refutes the hypothesis and then some (Table A8). Both retrieval axes collapse — name hit@1 0.901 → 0.638, stripped 0.657 → 0.398, zero of five paired folds — and the tag reading falls rather than rises (0.705 → 0.670). The diagnosis is unambiguous: every fold selects the epoch-50 checkpoint, the earliest written, so the tower is already past its best at the first write and degrades monotonically thereafter. The mechanism is positive leakage. Once the teacher vector a_g contains the student’s own view sentences, the contrastive positive is trivially matched and the softmax stops pushing the student to recognize the game from anything else; at evaluation the teacher reverts to the fixed pack the student never learned to key on, and identity — with the tag geometry that rides on it — is gone. The teacher must be content the student cannot see: this isolation is what lets contrast manufacture identity, and the tag gap is its intrinsic price, not a free lunch a softer teacher can recover.

*Table A8: Views-first anchors (vfa), a softer teacher that fails. Each batch game’s teacher pack is opened with the student’s own four views before the fixed pack; five-fold @512 (testset queries (ts), a single anchor draw per fold) against the identical-split I-CE reference. Retrieval collapses on both axes (zero of five paired folds) and the tag reading does not rise; every fold selects the epoch-50 checkpoint. The TAG column is the review-probe reading (the @512 references predate the held-out test-set probe), consistent within this table. Both rows also predate the selection rule of Section 5.3 and share the earlier criterion, so the comparison is internal to the table; the collapse it reports is not a selection artifact, since every fold of the softer teacher picks its earliest checkpoint and degrades from there.*

| Objective | Name hit@1 | Stripped hit@1 | TAG F1 |
| --- | --- | --- | --- |
| Views-first anchors (vfa) @512 | 0.638 ± 0.055 | 0.398 ± 0.041 | 0.670 ± 0.021 |
| I-CE (reference, same split) @512 | 0.901 ± 0.021 | 0.657 ± 0.028 | 0.705 ± 0.015 |

## APPENDIX F: THE TWO ANCHOR-READING REGIMES

Anchor packs are drawn, not given: a game's reviews far exceed the 4,096-sentence cap, so each pack is one sample of the game's material (Section 5.1). This appendix reports every headline tower under both readings of ten such draws. Table A9 is the single-source reading the body uses; Table A10 pools the ten encodings into one index vector first. The ordering of methods, the sign of every paired comparison, and the size of the identity/semantics trade are identical in the two tables; what pooling changes is at most 0.008 Stripped hit@1, on the tower that gains most from averaging its draws.

*Table A9: The single-source reading — each of the ten anchor packs is scored on its own and the scores are averaged (five-fold mean ± std, testset queries (ts), 4,096-sentence anchors). This is the protocol of every table in the body, reproduced here for comparison with Table A10.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE (ours) | 0.954 ± 0.013 | 0.997 ± 0.004 | 0.741 ± 0.018 | 0.928 ± 0.020 | 0.685 ± 0.019 |
| swin-I-CE (~27% gradient window) | 0.934 ± 0.014 | 0.994 ± 0.004 | 0.709 ± 0.027 | 0.922 ± 0.010 | 0.691 ± 0.014 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.930 ± 0.011 | 0.991 ± 0.008 | 0.709 ± 0.019 | 0.917 ± 0.005 | 0.705 ± 0.022 |
| CE (contrast only) | 0.938 ± 0.013 | 0.988 ± 0.010 | 0.673 ± 0.014 | 0.889 ± 0.011 | 0.664 ± 0.016 |
| SimCLR-style (in-batch views) | 0.918 ± 0.024 | 0.990 ± 0.007 | 0.632 ± 0.029 | 0.882 ± 0.022 | 0.707 ± 0.019 |
| I-CE (EMA + memory bank) | 0.936 ± 0.010 | 0.991 ± 0.007 | 0.636 ± 0.016 | 0.881 ± 0.016 | 0.711 ± 0.016 |
| VICReg (expander wiring, v/i/c = 20/10/20) | 0.823 ± 0.010 | 0.960 ± 0.016 | 0.451 ± 0.017 | 0.729 ± 0.033 | 0.708 ± 0.022 |
| BYOL | 0.441 ± 0.033 | 0.737 ± 0.026 | 0.284 ± 0.017 | 0.553 ± 0.043 | 0.712 ± 0.017 |
| Frozen embedder (mean pool) | 0.425 ± 0.033 | 0.626 ± 0.014 | 0.209 ± 0.020 | 0.380 ± 0.031 | 0.588 ± 0.009 |

*Table A10: The merged-index reading — the ten anchor encodings are averaged into one index vector per game before scoring (same towers, folds and testset queries (ts) as Table A9). Pooling adds at most 0.008 Stripped hit@1 and no method changes rank.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE (ours) | 0.956 ± 0.019 | 0.996 ± 0.005 | 0.747 ± 0.021 | 0.931 ± 0.020 | 0.684 ± 0.026 |
| swin-I-CE (~27% gradient window) | 0.941 ± 0.018 | 0.994 ± 0.004 | 0.716 ± 0.026 | 0.926 ± 0.011 | 0.681 ± 0.019 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.932 ± 0.014 | 0.991 ± 0.008 | 0.716 ± 0.019 | 0.920 ± 0.005 | 0.706 ± 0.020 |
| CE (contrast only) | 0.950 ± 0.012 | 0.989 ± 0.008 | 0.678 ± 0.019 | 0.893 ± 0.013 | 0.658 ± 0.020 |
| SimCLR-style (in-batch views) | 0.919 ± 0.027 | 0.990 ± 0.006 | 0.634 ± 0.027 | 0.886 ± 0.021 | 0.704 ± 0.021 |
| I-CE (EMA + memory bank) | 0.944 ± 0.013 | 0.993 ± 0.006 | 0.641 ± 0.012 | 0.887 ± 0.013 | 0.717 ± 0.014 |
| VICReg (expander wiring, v/i/c = 20/10/20) | 0.830 ± 0.012 | 0.962 ± 0.020 | 0.453 ± 0.014 | 0.735 ± 0.037 | 0.705 ± 0.025 |
| BYOL | 0.447 ± 0.034 | 0.743 ± 0.022 | 0.284 ± 0.014 | 0.554 ± 0.046 | 0.715 ± 0.020 |
| Frozen embedder (mean pool) | 0.450 ± 0.034 | 0.642 ± 0.015 | 0.215 ± 0.031 | 0.393 ± 0.035 | 0.578 ± 0.015 |

## APPENDIX G: TRAINSET VERSUS TESTSET QUERIES

Every headline number in this paper is measured on testset queries (ts) — games held out of the fold the tower trained on. Because the 814-game evaluation universe also contains games inside the training pool, the same measurement can be repeated on trainset queries (tr): about 488 per fold against 163 testset queries, ranked against the same 2,020 candidates by the same finished tower. The difference is a generalization gap, and the frozen embedder anchors its interpretation: having trained on nothing, it scores identically on both populations (0.425 / 0.425), so any gap a trained tower shows is what training memorized rather than what the queries happen to ask. Table A11 gives both populations for every headline tower.

*Table A11: Retrieval on trainset queries (tr) versus testset queries (ts), five-fold means at the 4,096-sentence anchor budget, single-source reading. Gap = (tr) − (ts). The frozen row is the control: no training, no gap.*

| Objective | Name hit@1 (tr) | Name hit@1 (ts) | Gap | Stripped hit@1 (tr) | Stripped hit@1 (ts) | Gap |
| --- | --- | --- | --- | --- | --- | --- |
| I-CE (ours) | 0.991 | 0.954 | +0.037 | 0.740 | 0.741 | -0.002 |
| swin-I-CE (~27% gradient window) | 0.983 | 0.934 | +0.049 | 0.716 | 0.709 | +0.007 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.979 | 0.930 | +0.049 | 0.708 | 0.709 | -0.000 |
| CE (contrast only) | 0.986 | 0.938 | +0.048 | 0.735 | 0.673 | +0.062 |
| SimCLR-style (in-batch views) | 0.987 | 0.918 | +0.069 | 0.655 | 0.632 | +0.023 |
| I-CE (EMA + memory bank) | 0.978 | 0.936 | +0.042 | 0.613 | 0.636 | -0.024 |
| VICReg (expander wiring, v/i/c = 20/10/20) | 0.902 | 0.823 | +0.079 | 0.480 | 0.451 | +0.029 |
| BYOL | 0.732 | 0.441 | +0.291 | 0.355 | 0.284 | +0.072 |
| Frozen embedder (mean pool) | 0.425 | 0.425 | +0.000 | 0.209 | 0.209 | -0.000 |

Two readings stand out. On name-intact queries every trained tower gives something back when the game is new — from 0.037 for I-CE to 0.291 for BYOL — and the ordering of that gap tracks how much of a tower's identity resolution is anchored rather than remembered. On name-stripped queries the contrast is sharper still: plain CE scores 0.735 on games it trained on but only 0.673 on held-out ones, a 0.062 gap, while I-CE reads 0.740 and 0.741 — no gap at all. The invariance term does not merely raise the name-stripped score; it removes the part of that score which was memorization. This is the same conclusion Section 6.1 draws from the factorization, measured from the other side.

## APPENDIX H: COST AT SCALE

Write N for the catalog size, P for the anchor-pack cap in sentences, d = 1,024 for the embedding width, B = 192 for the games in a step, and W = 168, S = 84, L = 2 for the window, stride and micro-passes of the teacher in Appendix D. The tower reads one pack with Q = 4 latent queries, so encoding a pack costs Θ(P·Q·d) work and holds Θ(P·d) activation. Everything below follows from those two facts.

The fully coupled teacher re-encodes every pack with gradient at every step. Its cost is Θ(N·P·Q·d) in time and Θ(N·P·d) in memory — both linear in the catalog. At our scale the backward graph holds 6.9 M sentence slots — 13 GiB of fp16 input before any activation is stored (cf. Table A5, whose 6.6 M counts the fixed split’s 1,613 packs) (Table A5); the same expression at N = 10⁶ asks for 7.6 TiB, which no accelerator holds. This is the wall the third limitation of Section 7 names.

The windowed teacher removes N from both expressions. A micro-pass encodes the batch's own packs plus one window and frees them before the next, so a step costs Θ(L·(B + W)·P·Q·d) in time and Θ((B + W)·P·d) in memory. At B = 192 and W = 168 that is 1.47 M sentence slots, 2.8 GiB of input, whether the catalog holds two thousand entries or ten million. Neither expression contains N. What the window does not bound, however, is the ring itself: the windowed teacher still keeps every anchor pack resident (Table A5), so Θ(N·P·d) of store sits on the device even though only Θ((B + W)·P·d) of it is ever differentiated. Measured at our scale the fully coupled teacher peaks at about 61 GiB at P = 4,096, and about 13 GiB of any variant’s footprint is that resident ring; the windowed teacher’s peak was not separately measured. The window has bounded the backward graph but not the catalog, and Table A12 shows where each variant stops.

*Table A12: The cost model of the three teachers, with the fully coupled I-CE teacher normalized to 1 at the same catalog size. N is the catalog, P the anchor-pack cap, d the embedding width, Q = 4 latent queries, B = 192 games per step, and W = 168, S = 84, L = 2 the window, stride and micro-passes. Ratios are quoted at our N = 1,694; the windowed teacher's step contains no N, so both of its first two rows fall as 1/N, but its ring keeps every pack resident and therefore stays linear in the catalog on the device. The streamed store — the full-corpus pipeline of this paper — is what removes that last term, paying I/O for it. The measured peak is at P = 4,096 (fold 0, torch.cuda.max_memory_allocated); the last row extrapolates each variant's device budget for N on one 80 GiB accelerator, assuming the linear dependence on N.*

| Quantity | I-CE, fully coupled | swin-I-CE (W = 168) | swin + streamed store |
| --- | --- | --- | --- |
| **Step time** | **Θ(N·P·Q·d) — 1** | **Θ(L(B+W)·P·Q·d) — 0.43** | **Θ(L(B+W)·P·Q·d) — 0.43** |
| **Peak activation** | **Θ(N·P·d) — 1** | **Θ((B+W)·P·d) — 0.21** | **Θ((B+W)·P·d) — 0.21** |
| **Resident anchor store** | **Θ(N·P·d) — 1** | **Θ(N·P·d) — 1** | **Θ((B+W)·P·d) — 0.21** |
| **Device memory in N** | **linear** | **linear** | **constant** |
| **Per-step device I/O** | **none** | **none** | **Θ(L·S·P·d) ≈ 1.3 GiB** |
| **Measured peak at N = 1,694** | **≈ 61 GiB** | **not measured** | **not measured** |
| **Extrapolated largest N, one 80 GiB device** | **≈ 2,200** | **—** | **not memory-bound** |

The third variant closes that last term by giving up residency, and it is the pipeline that built the corpus of this paper. The merged pool is a single memory-mapped fp16 array on disk: the build writes and reads it a million rows at a time, so that assembling 23,373 games never holds more than a chunk, and the training loop gathers each anchor pack from that map instead of from a resident ring. The device then holds only what it differentiates, Θ((B + W)·P·d), and the catalog leaves device memory altogether. What replaces it is I/O: the ring advances L·S = 168 packs per step and only the new ones must be paged, Θ(L·S·P·d) — 1.3 GiB per step at P = 4,096, 336 MiB at P = 1,024 — which a prefetch thread hides behind the forward pass. This is the trade-off the design makes, I/O for space, and the last row of Table A12 shows what it gains: on one 80 GiB accelerator the fully coupled teacher, from its measured 61 GiB at N = 1,694, extrapolates to a ceiling near N ≈ 2,200; the windowed teacher’s ring still grows linearly in the catalog; the streamed variant alone has no such bound, its device footprint constant in the catalog. The distinction is not hypothetical at our own next step: the 23,373-game corpus needs 183 GiB of packs at P = 4,096, which no single device holds, but 46 GiB at P = 1,024, which one does. Two quantities do still grow with the catalog, and neither is memory. The ring advances 168 packs per step, so an anchor takes a gradient turn once every N/(L·S) steps — ten steps at our N = 1,694, 139 at 23,373, and 5,952, about 370 epochs of sixteen steps, at a million; equivalently, per-step coverage is (B + W)/N, falling from 21% here to 1.5% at full-corpus scale and 0.04% at a million. A million-entity store is 7.6 TiB at P = 4,096 or 1.9 TiB at P = 1,024 — an ordinary disk array, read at about a gigabyte per step.

The practical reading is that memory, the usual constraint, ceases to be one: a million-entity catalog trains inside a single 80 GiB accelerator at constant step cost, with the anchor store on disk and paged. What replaces it is a coverage question — how rarely an anchor may take its turn before the objective stops learning. Our evidence spans 21% to 100% per-step coverage, the range in which the windowed teacher gives up 0.032 Stripped hit@1 against full coupling (Table A13); it does not reach the 0.04% regime and we do not claim it. The architecture makes million-scale training affordable; whether it stays effective there is the open question the third limitation should be read as posing.

*Table A13: The cost of the anchor-supply economies, moved here from the body in the phase-3 revision (testset queries (ts), five-fold mean ± std at the 4,096-sentence anchor budget). The sliding fresh-window variant (swin, Appendix D) re-encodes ~27% of the gallery with gradient per step; the two-stage row hands a BYOL warm start to the windowed teacher for the last 600 epochs; the EMA + memory-bank variant replaces the coupled teacher with an EMA shadow encoder feeding a 3,072-key bank (Appendix B). Full coupling is the ceiling: the windowed teacher and the two-stage recipe each give up 0.032 Stripped hit@1, the memory-bank economy 0.105.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set TAG F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE, fully coupled teacher | 0.954 ± 0.013 | 0.997 ± 0.004 | 0.741 ± 0.018 | 0.928 ± 0.020 | 0.685 ± 0.019 |
| swin-I-CE (~27% gradient window) | 0.934 ± 0.014 | 0.994 ± 0.004 | 0.709 ± 0.027 | 0.922 ± 0.010 | 0.691 ± 0.014 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.930 ± 0.011 | 0.991 ± 0.008 | 0.709 ± 0.019 | 0.917 ± 0.005 | 0.705 ± 0.022 |
| I-CE (EMA + memory bank) | 0.936 ± 0.010 | 0.991 ± 0.007 | 0.636 ± 0.016 | 0.881 ± 0.016 | 0.711 ± 0.016 |

## APPENDIX I: THE QUERY-REWRITE PROMPTS

All four query registers come from one chat-completions model (temperature 0.7) that reads a held-out game’s full wiki article and writes an English description of it. The instructions below were issued in Chinese and are translated faithfully here; the verbatim strings, the retry policy and the model identifier are in the released code. An output shorter than 300 characters is re-requested with the elaboration instruction, up to three attempts, and the article is truncated to 12,000 characters before it is sent.

ALGORITHM 1: The rewrite instructions behind the four query registers

user message  —  "Game: {title}" then the article text
neutral (Name hit@k)  —  Read the full game page text the user provides and summarize it into a descriptive article about the game. Preserve the real information about the game’s content, gameplay and mechanics in full; do not invent anything that is not on the page. Register: neutral. Write in English and output the article body directly.
name-stripped (Stripped hit@k)  —  Read the full game page text the user provides and summarize it into a descriptive article about the game, in a neutral register. Preserve the real information about the game’s content, gameplay and mechanics in full; do not invent anything that is not on the page. But do not reveal the game’s name, the characters’ names, the items’ names or anything like them — replace them all with imagined, invented words. Write in English and output the article body directly.
praising (Appendix K)  —  the neutral instruction with the register clause replaced by: Register: praising and affirming.
critical (Appendix K)  —  the neutral instruction with the register clause replaced by: Register: negative and critical.
elaboration retry (any register)  —  Your previous answer was too short. Please write a considerably MORE DETAILED article (at least 300 words) in the requested style, covering the game’s content, story, world and mechanics from the page text above.

## APPENDIX J: REDUCING THE STEAM TAG VOCABULARY

A single mapping file is the source of truth for every tag number in this paper: each of the 202 fine tags is assigned either to one coarse class or to the sentinel "del".

The listing below is the whole definition: a class is present on a game if any of the Steam tags on its right-hand side is, and the vote weights Steam attaches are used only to resolve which source is strongest, never as a target — the probe predicts presence, so a game either carries a class or does not. On our 2,020 games this yields a 2,020 × 23 binary matrix of density 0.202 — 4.6 classes per game at the mean, 5 at the median, and between 35 and 1,395 games per class.

ALGORITHM 2: The 23 TAG classes, each written as the original Steam tags it absorbs

Action/Adventure = [Action, Action-Adventure, Adventure, Combat, Hack and Slash, Souls-like]
Bullet Hell = [Bullet Hell]
Card Game = [Card Game, Card Battler, Deckbuilding, Roguelike Deckbuilder]
Co-op/Multiplayer = [Class-Based, Co-op, Co-op Campaign, Competitive, Local Co-Op, Local Multiplayer, Massively Multiplayer, Multiplayer, Online Co-Op, PvE, PvP, Split Screen, Team-Based]
Crime/Mystery = [Assassin, Crime, Detective, Mystery, Political]
Fantasy = [Fantasy, Dark Fantasy, Dragons, Gothic, Magic, Medieval, Mythology]
Fighting/Melee = [3D Fighter, Beat 'em up, Character Action Game, Fighting, Martial Arts, Spectacle fighter, Swordplay]
Historical/War = [Alternate History, Historical, Military, War, World War II]
Horror = [Horror, Demons, Psychological Horror, Supernatural, Survival Horror, Vampire, Zombies]
Narrative/Choices = [Choices Matter, Choose Your Own Adventure, Conversation, Interactive Fiction, Lore-Rich, Multiple Endings, Narration, Narrative, Point & Click, Story Rich, Text-Based, Visual Novel, Walking Simulator]
Open World/Survival = [Base-Building, Building, Crafting, Exploration, Fishing, Hunting, Mining, Open World, Open World Survival Craft, Sandbox, Survival]
Platformer = [Platformer, 2D Platformer, 3D Platformer, Metroidvania, Parkour]
Puzzle = [Puzzle]
RPG = [RPG, Action RPG, CRPG, JRPG, MMORPG, Party-Based RPG, Strategy RPG, Tactical RPG]
Racing/Driving = [Driving, Racing, Tanks, Vehicular Combat]
Roguelike = [Action Roguelike, Dungeon Crawler, Procedural Generation, Rogue-like, Rogue-lite]
Sci-fi/Cyberpunk = [Aliens, Cyberpunk, Dystopian, Futuristic, Post-apocalyptic, Robots, Sci-fi, Space]
Shooter/FPS = [FPS, Looter Shooter, Shooter, Third-Person Shooter]
Simulation/Management = [Agriculture, Automation, Automobile Sim, City Builder, Colony Sim, Farming Sim, Life Sim, Management, Resource Management, Simulation, Space Sim]
Sports/Rhythm = [Rhythm, Sports]
Stealth/Immersive = [Immersive Sim, Stealth]
Strategy/Tactics = [4X, Grand Strategy, RTS, Real Time Tactics, Strategy, Tactical, Tower Defense, Wargame]
Turn-Based = [Turn-Based, Turn-Based Combat, Turn-Based Strategy, Turn-Based Tactics]
discarded = [2D, 3D, Anime, Arcade, Atmospheric, Beautiful, Cartoony, Casual, Character Customization, Cinematic, Classic, Colorful, Comedy, Controller, Cute, Dark, Dark Humor, Dating Sim, Destruction, Difficult, Drama, Early Access, Economy, Emotional, Family Friendly, Fast-Paced, Female Protagonist, First-Person, Free to Play, Funny, Gore, Great Soundtrack, Gun Customization, Hand-drawn, Indie, Inventory Management, Isometric, LGBTQ+, Loot, Mature, Memes, Music, Nudity, Old School, Physics, Pixel Graphics, Psychedelic, Psychological, Realistic, Relaxing, Replay Value, Retro, Romance, Sexual Content, Singleplayer, Soundtrack, Stylized, Surreal, Third Person, Thriller, Violent]

## APPENDIX K: TONE-REWRITTEN QUERIES

Each held-out game’s article is rewritten by the same model in a praising and in a critical voice (Appendix I); names stay intact, so the Neutral column is Table 2’s Name hit@1, and the last column is the paired per-fold difference.

*Table A14: Tone-rewritten queries (testset queries (ts), hit@1 among all 2,020 games, five-fold mean ± std; same towers, checkpoints and protocol as Table 2). Tone moves retrieval by at most 0.05 anywhere — against the 0.2+ that removing names costs (Table 2). The two anchored-contrast towers are the only ones a critical voice never helps — I-CE loses in five folds of five, CE in four — while the four other spaces gain on average, the frozen embedder most (+0.051).*

| Objective | Neutral | Praising | Critical | Crit − Neut |
| --- | --- | --- | --- | --- |
| Frozen embedder (mean pool) | 0.209 ± 0.016 | 0.217 ± 0.019 | 0.260 ± 0.019 | +0.051 |
| CE (contrast only) | 0.673 ± 0.011 | 0.686 ± 0.007 | 0.666 ± 0.010 | −0.007 |
| SimCLR-style (in-batch views) | 0.632 ± 0.010 | 0.655 ± 0.011 | 0.654 ± 0.013 | +0.022 |
| VICReg (epd 20/10/20) | 0.451 ± 0.012 | 0.485 ± 0.023 | 0.468 ± 0.022 | +0.017 |
| BYOL | 0.284 ± 0.016 | 0.315 ± 0.014 | 0.317 ± 0.019 | +0.033 |
| I-CE (ours, @4096) | 0.954 ± 0.013 | 0.963 ± 0.013 | 0.941 ± 0.010 | −0.013 |

*Table A15: TAG micro-F1 on the same tone-rewritten queries (five-fold mean ± std; the probes are Table 2’s TAG column exactly — one ridge per gallery draw, K = 10, held-out protocol of Section 5 — so the Neutral column is Table 2’s TAG column and only the query register changes). The direction inverts against Table A14: every trained tower reads tags from the critical rewrite as well as or better than from the neutral one — I-CE +0.012 in five folds of five — while the frozen embedder loses −0.018 in five of five. The identity cost a critical voice carries in Table A14 does not extend to the semantic reading.*

| Objective | Neutral | Praising | Critical | Crit − Neut |
| --- | --- | --- | --- | --- |
| Frozen embedder (mean pool) | 0.605 ± 0.017 | 0.584 ± 0.021 | 0.587 ± 0.020 | −0.018 |
| CE (contrast only) | 0.682 ± 0.015 | 0.672 ± 0.014 | 0.686 ± 0.021 | +0.003 |
| SimCLR-style (in-batch views) | 0.697 ± 0.015 | 0.683 ± 0.018 | 0.702 ± 0.019 | +0.005 |
| VICReg (epd 20/10/20) | 0.698 ± 0.012 | 0.687 ± 0.011 | 0.701 ± 0.014 | +0.003 |
| BYOL | 0.672 ± 0.017 | 0.665 ± 0.013 | 0.673 ± 0.021 | +0.002 |
| I-CE (ours, @4096) | 0.685 ± 0.019 | 0.673 ± 0.021 | 0.697 ± 0.021 | +0.012 |

## APPENDIX L: GRADIENT BOUNDS AND THE HANDOVER POINT

The handover point between the contrastive push-pull and the invariance term, where the two forces intersect, follows from their maximum gradient magnitudes.

The contrastive (CE) loss exerts a pull on the positive anchor and a push on the negative anchors. Under symmetric confusion, the gradient on a view's embedding z with respect to CE is proportional to (1 − p_g)a_g − Σ p_h a_h, where p_g is the classification confidence on the positive anchor a_g. By factoring out (1 − p_g), the term becomes a difference between a_g and the weighted centroid of the negative anchors. Since both lie within the unit ball, their maximum Euclidean distance is 2 (attained when perfectly opposed). Thus, the maximum magnitude (envelope) of the contrastive force is strictly bounded by 2(1 − p_g)/τ. The force adaptively brakes, vanishing linearly as p_g approaches 1.

The invariance (I) term across V views computes the average cosine disagreement. For any single view, its gradient collects the pull from the other V − 1 unit vectors. The maximum possible length of their sum is V − 1, which happens when the other views are aligned. The gradient envelope of the weighted invariance term λI is therefore bounded by (2λ / (V(V − 1))) × (V − 1) = 2λ/V. This force is a constant bounding envelope that does not decay with training confidence.

Equating the two envelopes gives the precise handover point where the contrastive force drops below the invariance force: 2(1 − p_g)/τ = 2λ/V, which simplifies to 1 − p_g = λτ/V. In our architecture, V = 4. Setting λ = 2 naturally normalizes the invariance force envelope to a unit force (1.0). Coupled with our empirically optimal temperature τ = 0.02, this formula yields 1 − p_g = 0.01. Therefore, the parameters calibrate the gradients to gracefully hand over dominance exactly when the model reaches 99% classification confidence.
