# Invariant Game Representation Learning from Steam Reviews

Appendices A–L

*Appendices to the AICCC 2026 submission (PA0049)*

## APPENDIX A: THE TWO QUERY REGISTERS ON ONE GAME

Both registers of Section 5 are written from the same wiki article by the same model and differ only in the instruction (Appendix F): name-intact keeps the names and is scored as Name hit@k, name-stripped is scored as Stripped hit@k, and the openings below are verbatim, for one held-out game.

*LISTING 1: The two query registers on one held-out game*

```
Name-intact (Name hit@k)
Miasma Chronicles is a tactical role-playing video game developed by The Bearded Ladies and published by 505 Games in 2023. Set in a post-apocalyptic version of the United States, it combines turn-based tactics with stealth game mechanics. The story follows a world devastated by a force known as the miasma, which can transform or kill living things. More than a century after this catastrophe, a woman leaves her son, Elvis, in the care of a robot named Diggs. Elvis also receives a glove left behind by his mother that can control the miasma. Players take control of Elvis as he searches for his mother and tries to uncover more about the miasma.

Name-stripped (Stripped hit@k)
This tactical role-playing game takes place in a ruined future version of the United States, where a mysterious force has devastated the land by transforming or killing living things. The story follows a young man who is left in the care of a robot guardian after his mother disappears, and who later sets out to find her while uncovering the truth behind the strange force at the center of the world. He carries a special glove left by his mother, which can control this force and serves as a key part of both the narrative and the gameplay.
```

## APPENDIX B: FIVE-FOLD CROSS-VALIDATION DETAIL

Table A1 gives the per-fold results behind Table A2’s largest-budget row, seed-paired CE and I-CE at the 4,096-sentence budget under the protocol of Section 5 (retrieval among all 2,020 games).

*Table A1: Per-fold detail at the 4,096-sentence anchor budget, test-set queries (ts). I-CE beats CE in five of five paired folds on both query registers at this budget. Means and standard deviations are computed before rounding.*

| Fold | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE fold 0 | 0.970 | 1.000 | 0.728 | 0.920 | 0.689 |
| I-CE fold 1 | 0.955 | 1.000 | 0.745 | 0.931 | 0.665 |
| I-CE fold 2 | 0.931 | 0.989 | 0.715 | 0.904 | 0.662 |
| I-CE fold 3 | 0.952 | 0.994 | 0.749 | 0.922 | 0.697 |
| I-CE fold 4 | 0.962 | 1.000 | 0.769 | 0.963 | 0.709 |
| **I-CE mean ± std** | **0.954 ± 0.013** | **0.997 ± 0.004** | **0.741 ± 0.018** | **0.928 ± 0.020** | **0.685 ± 0.019** |
| CE fold 0 | 0.934 | 0.988 | 0.687 | 0.889 | 0.681 |
| CE fold 1 | 0.945 | 1.000 | 0.693 | 0.899 | 0.667 |
| CE fold 2 | 0.919 | 0.971 | 0.667 | 0.869 | 0.649 |
| CE fold 3 | 0.933 | 0.988 | 0.656 | 0.901 | 0.681 |
| CE fold 4 | 0.959 | 0.993 | 0.662 | 0.888 | 0.642 |
| **CE mean ± std** | **0.938 ± 0.013** | **0.988 ± 0.010** | **0.673 ± 0.014** | **0.889 ± 0.011** | **0.664 ± 0.016** |

*Table A2: Ablation 1 in full: CE vs I-CE at anchor budgets of 512–4,096 sentences (test-set queries (ts), five-fold mean ± std, the two objectives seed-paired per fold on identical splits). Increasing the budget yields one significant step, 512 to 1,024 sentences (p-value 0.003), followed by a plateau in which the three larger budgets are indistinguishable (p-values $\ge 0.19$); I-CE leads CE at every budget (20 of 20 paired folds on Stripped hit@1). The windowed teacher (swin, Appendix J) appears at its trained budget.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
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

Table A13 in Appendix K repeats the fold-level detail at every budget, Table A7 tabulates the EMA + memory-bank variant, and Table A3 gives the full five-fold account of every objective family, the numbers the body’s Figure 4 plots.

*Table A3: The self-supervision families in full: the five-fold account behind the body’s Figure 4 (test-set queries (ts), five-fold mean ± std at the 4,096-sentence anchor budget; “epd” marks VICReg wired on the expander outputs, $v/i/c=20/10/20$; zero-shot cosine retrieval among all 2,020 games; test-set tag micro-F1 under the held-out protocol of Section 5). Rows are sorted by Stripped hit@1, worst first; the first row is the no-tower baseline, the frozen embedder’s mean-pooled sentence vectors under the identical protocol. The two capabilities factorize: every negative-light objective lands in the same narrow tag band while conceding name-stripped retrieval, and I-CE is the only row that keeps name-stripped retrieval while reading tags above CE.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| Frozen embedder (mean pool) | 0.425 ± 0.033 | 0.626 ± 0.014 | 0.209 ± 0.020 | 0.380 ± 0.031 | 0.588 ± 0.009 |
| BYOL | 0.441 ± 0.033 | 0.737 ± 0.026 | 0.284 ± 0.017 | 0.553 ± 0.043 | 0.712 ± 0.017 |
| VICReg (epd 20/10/20) | 0.823 ± 0.010 | 0.960 ± 0.016 | 0.451 ± 0.017 | 0.729 ± 0.033 | 0.708 ± 0.022 |
| SimCLR-style (in-batch views) | 0.918 ± 0.024 | 0.990 ± 0.007 | 0.632 ± 0.029 | 0.882 ± 0.022 | 0.707 ± 0.019 |
| CE (contrast only) | 0.938 ± 0.013 | 0.988 ± 0.010 | 0.673 ± 0.014 | 0.889 ± 0.011 | 0.664 ± 0.016 |
| **I-CE (ours)** | **0.954 ± 0.013** | **0.997 ± 0.004** | **0.741 ± 0.018** | **0.928 ± 0.020** | **0.685 ± 0.019** |

## APPENDIX C: THE BASELINE AND VARIANT RECIPES

Every trained baseline shares the full scaffolding of Section 4: the same 4-query cross-attention tower, the same four views per game per step (three review views and one document view where a document exists), the same corpus, batch (192 games), optimizer (AdamW, lr 5e-4, weight decay 1e-4), 2,000-epoch budget, temperature $\tau =0.02$ where a softmax exists, and the validation-only selection of Section 5. Table A4 gives each arm’s departure from that scaffolding.

*Table A4: The baseline and variant recipes. The first five rows are the Figure 4 baselines; apart from the untrained frozen-embedder row, each keeps the tower and the training recipe and changes the objective together with the anchor supply and teacher that objective implies. The last four change the objective and, where noted, the schedule or the anchor supply.*

| Arm | Objective | Anchor supply | Schedule or teacher | Notes |
| --- | --- | --- | --- | --- |
| Frozen embedder (mean pool) | None. A game is the L2-normalized mean of its anchor-pack sentence embeddings, a query the L2-normalized mean of its own sentences | Anchor packs read at evaluation only | No tower and no training | Measures the raw geometry every other row starts from |
| CE (contrast only) | Equation (2) alone; the invariance term of Equation (1) is removed | Full anchor gallery, re-encoded with gradient at every step | Fully coupled teacher | Isolates what the softmax supplies without multi-view agreement |
| SimCLR-style (in-batch views) | NT-Xent (normalized temperature-scaled cross-entropy) [4] at $\tau =0.02$ over the step’s $4\times 192$ view encodings: view $v$ of a game takes view $v+1(\mathrm{mod}4)$ of the same game as its positive, the game’s other own views are masked as false negatives, every other game’s views are negatives | No anchors anywhere in training; the gallery is encoded once at evaluation time by the finished tower | Tower unchanged; no anchor teacher | The contrast lives in view space instead of anchor space |
| VICReg (epd 20/10/20) | VICReg [1] on expander-MLP outputs over the six view pairs, variance/invariance/covariance weights 20/10/20; no negatives | Anchors encoded at evaluation time only | Views drawn for 192 games per step, the variance term estimating per-dimension spread within the step’s view batch | Drawing the whole pool per step (batch = all) yielded no improvement; Table A11 is the weight sweep |
| BYOL | $1- \cos$ between the predictor’s output for one view and the target tower’s encoding of another, averaged over ordered view pairs; the document view participates like any other | No gallery in training | Online tower plus a two-layer predictor head, and a target tower that is an exponential moving average of the online weights (momentum 0.996) behind a stop-gradient | Collapse is prevented by the predictor/EMA asymmetry rather than by negatives |
| CE + anchor-in-I (Table 2) | Equation (2) unchanged, but the invariance term of Equation (1) ranges over the ten pairs among the four strong views and the game’s own anchor vector instead of the six pairs among views alone | Full anchor gallery, re-encoded with gradient at every step | Fully coupled teacher | The control of Section 6.2: constant-weight attraction on the anchor with the softmax left in place |
| Two-stage: BYOL warm start → windowed (Table A7) | Stage one is the BYOL row above; stage two is the windowed teacher of Appendix J | Windowed, $w=168$ | Stage one runs to its own selection point; stage two re-initializes the tower from those weights and trains 600 epochs, 30% of the from-scratch discriminative budget | Selection happens inside stage two |
| I-CE (EMA + memory bank; Table A7) | Cross-entropy against the ring, the view’s freshly written key the positive and stale same-game keys masked; the invariance term is unchanged | An EMA shadow tower encodes the current batch’s anchors into a 3,072-key memory bank | EMA shadow tower, momentum 0.99, replacing the fully coupled teacher | Gradient reaches the strong-view side only; the 3,072 keys are negatives in number but inert in the gradient, the sense in which this tower counts as negative-light in Section 6.1 |
| Align + uniformity (Table 2) | The anchored softmax replaced by an alignment term plus a kernel-uniformity term, over a grid of align-to-uniformity ratios and kernel temperatures of 2 and 25 | Repulsion aimed at the anchor gallery or at the batch views | Fully coupled teacher | The decomposition sweep of Section 6.2; Table 2 prints two representative arms and the geometry files in this repository carry the sweep |

## APPENDIX D: THE SEPARATION GRADIENTS OF SECTION 4.3

This appendix works out Equations (2), (3), (6) and (7) over the noise model of Equation (4), with a remark on Equation (8): why sample repulsion aims along draw noise, why anchor repulsion does not, why the invariance term leaves only the semantic difference, and why the coupled softmax brakes its own forces. Each derivation names its convention: Equation (4) is stated for the pooled input while the repulsion steps read $c$ and $\epsilon$ as free components of the encoding [5], and the invariance steps linearize the tower, whose noise response is what that term suppresses.

**The noise model.** Pool an $m$-sentence pack drawn from game $g$:

$$
x={c}_{g}+{\epsilon }_{m},E[{\epsilon }_{m}]=0,\mathrm{Cov}({\epsilon }_{m})=\frac{{\Sigma }_{g}}{m} \tag{D.1}
$$

with ${c}_{g}$ the draw-invariant component, content plus the static bias its reviewers share, and ${\epsilon }_{m}$ that draw’s fluctuation, ${\Sigma }_{g}$ being the covariance of one sentence embedding of $g$. Packs enter at two scales, ${m}_{v}$ sentences for a view and ${m}_{a}$ for an anchor, the document view of Section 4.1 entering not as a third scale, carrying no draw noise of its own, but as the fixed offset ${\delta }_{g}$ of (D.18),

$$
{m}_{v}\ge 16,\; 512\le {m}_{a}\le 4,096,{m}_{a}\gg {m}_{v},\mathrm{tr}\mathrm{Cov}({\epsilon }_{v})\gg \mathrm{tr}\mathrm{Cov}({\epsilon }_{a}) \tag{D.2}
$$

a strong view at its 16-sentence floor (23 at the median) against an anchor cap realized packs fall short of. The $1/m$ scaling assumes independent pooled draws, which whole-review sampling and attention pooling violate, so only ratios of the two scales are used. Encoding both, $f({x}_{v})=z$ and $f({x}_{a})=a$, views scatter and anchors hold still.

Two approximations sit here. Different draw laws and the anchor’s store-page prefix leave ${c}_{g}$ common only up to a static offset of the kind ${\delta }_{g}$ below, which averaging does not remove. And a view meets its own anchor pack only by chance, about 29% of a median game’s sentences at ${m}_{a}=4,096$, so ${\epsilon }_{v}$ and ${\epsilon }_{a}$ count as independent.

**Why sample-to-sample repulsion aims along noise.** In the free-variable convention, with one negative view per competing game in the batch, Equation (6) reads

$$
{\nabla }_{{z}_{v}}{\ell }_{\mathrm{SimCLR}}=\frac{1}{\tau }\sum_{h\neq g} {q}_{h}({z}_{h}- {z}_{v' }) \tag{D.3}
$$

with the in-batch weight

$$
{q}_{h}=\frac{\exp (\langle {z}_{v},{z}_{h}\rangle /\tau )}{\exp (\langle {z}_{v},{z}_{v' }\rangle /\tau )+\sum_{k\neq g} \exp (\langle {z}_{v},{z}_{k}\rangle /\tau )} \tag{D.4}
$$

putting the rest on the positive partner ${z}_{v' }$, so the full set of weights, positive included, sums to one and the $h=g$ term drops from the gradient sum. The partner is another fresh draw of $g$, so its noise is what enters

$$
{z}_{h}- {z}_{v' }=({c}_{h}- {c}_{g})+({\epsilon }_{h}- {\epsilon }_{v' }) \tag{D.5}
$$

whose second half is unbiased term by term, the two draws being zero-mean by (D.1) and, coming from different games, independent, and has mean square

$$
E\| {\epsilon }_{h}- {\epsilon }_{v' }{\| }^{2}=\frac{\mathrm{tr}{\Sigma }_{h}+\mathrm{tr}{\Sigma }_{g}}{{m}_{v}} \tag{D.6}
$$

Rank turns on hard pairs, and at a semantic hard pair, defined by $\| {c}_{h}- {c}_{g}\| \ll \sqrt{(\mathrm{tr}{\Sigma }_{h}+\mathrm{tr}{\Sigma }_{g})/{m}_{v}}$ with the noise norm concentrating near its root mean square (the scale carried to the encodings by the linearization of (D.13) below),

$$
\| {\epsilon }_{h}- {\epsilon }_{v' }\| \gg \| {c}_{h}- {c}_{g}\| \tag{D.7}
$$

so the push aims predominantly along noise. Hard pairs are selected by encoded similarity, and one hard only because two draws collided is corrected by a noise-aligned push. In the large the semantic term dominates, which is why sample-repelled towers spread widely yet carve narrow nearest-neighbor margins.

**Why anchor-mediated repulsion does not.** Still free-variable, Equation (7) pushes a view against the anchor gallery,

$$
{\nabla }_{{z}_{v}}{\ell }_{\mathrm{CE}}=\frac{1}{\tau }\sum_{h\neq g} {p}_{h}({a}_{h}- {a}_{g}),\sum_{h\neq g} {p}_{h}=1- {p}_{g} \tag{D.8}
$$

so the descent step is one pull and $n- 1$ pushes,

$$
- {\nabla }_{{z}_{v}}{\ell }_{\mathrm{CE}}=\frac{1}{\tau }[(1- {p}_{g}){a}_{g}- \sum_{h\neq g} {p}_{h}{a}_{h}] \tag{D.9}
$$

Write ${p}_{h|g}$ for the softmax weight of anchor $h$ under a view ${z}_{g}$ of game $g$. Call $(g,\; h)$ symmetrically confused when ${p}_{h|g}={p}_{g|h}$ and ${p}_{k|g}={p}_{k|h}$ for every other anchor $k$, which by normalization forces ${p}_{g|g}={p}_{h|h}$; subtracting the descent steps of the views ${z}_{g}$ and ${z}_{h}$, ${z}_{g}- {z}_{h}$ moves along

$$
(\frac{1- {p}_{g|g}}{\tau }+\frac{{p}_{h|g}}{\tau })({a}_{g}- {a}_{h}) \tag{D.10}
$$

both coefficients read off (D.9), the matched third-party weights cancelling in the subtraction so that only the $(g,\; h)$ terms survive and the step is parallel to ${a}_{g}- {a}_{h}$, which decomposes as

$$
{a}_{g}- {a}_{h}=({c}_{g}- {c}_{h})+({\epsilon }_{\mathrm{ag}}- {\epsilon }_{\mathrm{ah}}) \tag{D.11}
$$

whose second half is, by the variance asymmetry, a static residual and not a fresh draw,

$$
\frac{E\| {\epsilon }_{\mathrm{ag}}- {\epsilon }_{\mathrm{ah}}{\| }^{2}}{E\| {\epsilon }_{h}- {\epsilon }_{v' }{\| }^{2}}=\frac{{m}_{v}}{{m}_{a}} \tag{D.12}
$$

an amplitude ratio $\sqrt{{m}_{v}/{m}_{a}}$, about $1/5.7$ at the 16-sentence floor against the 512-sentence anchor budget, and larger for longer views. Read in the encodings through the linearization of (D.13) below, one Jacobian and one sentence-level covariance ${\Sigma }_{g}$ serving both packs of a game, the ratio is unchanged; it covers the draw-noise half of the anchor residual, the store-page offset of the kind ${\delta }_{g}$ being static and not shrinking with ${m}_{a}$. The anchored path has a lower noise floor, tracking the semantic difference far below the scale $\sqrt{(\mathrm{tr}{\Sigma }_{h}+\mathrm{tr}{\Sigma }_{g})/{m}_{v}}$ where sample repulsion loses it. The residual is static within a run, replaced by the ten fresh anchor draws of the evaluation protocol (Section 5), whose effect Tables S2 and S3 of the supplementary document bound at 0.008 Stripped hit@1. Through ${p}_{h}$ the softmax also concentrates the whole repulsion budget on whichever anchors currently compete. Equation (8) sits at the opposite extreme: its mean gives every competitor weight $1/B$, $B$ the view encodings in a step (Section 4.3), against a ${p}_{h}$ that is of order one at a hard competitor early in training, so matching one anchored step takes order $B$ mean steps, each carrying an independent draw, and the accumulated noise variance grows by the same factor.

**Why the invariance term suppresses the noise response.** Switching to the linearization, expand about the invariant component, with $J$ the Jacobian at $c$,

$$
f(c+\epsilon )\approx f(c)+J\epsilon ,\; J=\frac{\partial f}{\partial x} \tag{D.13}
$$

and since the tower outputs unit vectors, differentiating $\| f{\| }^{2}=1$ gives

$$
f(c{)}^{T}J=0 \tag{D.14}
$$

which kills the first-order term of the cosine, and the same constraint fixes each output’s radial second-order term at $- \| J\epsilon {\| }^{2}/2$, so that $\| z\| =1$ holds to that order, leaving

$$
\langle {z}_{i},{z}_{j}\rangle =1+{\epsilon }_{i}^{T}{J}^{T}J{\epsilon }_{j}- \frac{\| J{\epsilon }_{i}{\| }^{2}+\| J{\epsilon }_{j}{\| }^{2}}{2} \tag{D.15}
$$

and therefore

$$
1- \cos ({z}_{i},{z}_{j})=\frac{\| J({\epsilon }_{i}- {\epsilon }_{j}){\| }^{2}}{2} \tag{D.16}
$$

For independent draws of equal covariance every pair has expectation $\mathrm{tr}(J\mathrm{Cov}({\epsilon }_{v}){J}^{T})$, and the $2/(V(V- 1))$ prefactor of Equation (3) averages the $V(V- 1)/2$ pairs, so

$$
E[I]=\mathrm{tr}(J\mathrm{Cov}({\epsilon }_{v}){J}^{T}) \tag{D.17}
$$

with coefficient one, the case of a game without a document view (198 in a fold). The document view rides on ${c}_{g}$ with its static register offset ${\delta }_{g}$, its own fixed deviation folded in,

$$
{z}_{\mathrm{doc}}\approx f({c}_{g})+J{\delta }_{g} \tag{D.18}
$$

and, for a game with a document view (1,496 of the 1,694 in a fold), each of the $V- 1$ pairs involving it swaps one view’s noise for that offset, contributing $\| J{\delta }_{g}{\| }^{2}/2$ and half the noise term,

$$
E[I]=\frac{V- 1}{V}\mathrm{tr}(J\mathrm{Cov}({\epsilon }_{v}){J}^{T})+\frac{\| J{\delta }_{g}{\| }^{2}}{V} \tag{D.19}
$$

a floor of $\| J{\delta }_{g}{\| }^{2}/4$ at $V=4$ that no drawing removes, which is why closing ${\delta }_{g}$ falls here. Being symmetric, the term closes ${\delta }_{g}$ at a compromise, not by moving the document alone: the game-independent part ${b}_{\mathrm{doc}}- {b}_{\mathrm{rev}}$ of ${\delta }_{g}=({b}_{\mathrm{doc}}- {b}_{\mathrm{rev}})+{t}_{g}$ (Section 4.1) leaves inter-game differences untouched, while the drift ${t}_{g}$ does not and is absorbed into $f({c}_{g})$. Minimizing $I$ presses the noise term of (D.19) to zero in expectation,

$$
E\| J{\epsilon }_{v}{\| }^{2}\to 0,{z}_{v}- {z}_{v' }\approx J({\epsilon }_{v}- {\epsilon }_{v' })\to 0 \tag{D.20}
$$

suppressing the tower’s noise response, not the noise. Two assumptions carry this: the linearization holds in the high-noise regime of the strong views, and $\mathrm{Cov}({\epsilon }_{v})$ must span a subspace distinct enough from ${c}_{g}- {c}_{h}$ for one to be suppressed without the other. Cross-entropy forbids total collapse, the constant tower costing $\ln n\approx 7.4$ nats per view, but not the loss of semantic directions inside the noise subspace, which we read as the tag deficit of Section 8.

**What the two terms leave.** The conventions meet here, assuming the tower keeps Jacobian rank along the semantic directions while its noise response is suppressed. With ${J}_{g}$ and ${J}_{h}$ the Jacobians at ${c}_{g}$ and ${c}_{h}$,

$$
{z}_{g}- {z}_{h}=(f({c}_{g})- f({c}_{h}))+({J}_{g}{\epsilon }_{g}- {J}_{h}{\epsilon }_{h}) \tag{D.21}
$$

whose second half Equation (3) sends to zero while Equation (7) holds the first half apart through the anchors, each view pinned to its own anchor at the CE optimum and the anchors pushed apart by (D.23), so at the optimum

$$
{z}_{g}- {z}_{h}=f({c}_{g})- f({c}_{h}) \tag{D.22}
$$

up to the drift ${t}_{g}$ absorbed into $f({c}_{g})$ and the static anchor residual of (D.12). Displacement contracts while the margin, an anchor-side quantity made by repulsion, roughly holds, so the displacement-to-margin ratio retrieval depends on falls. The contraction is derived for in-distribution draws; extending it to the zero-shot cross-register queries of Section 5 assumes a query’s displacement lies in the same subspace, the step the document view makes plausible.

**The coupled softmax brake.** Differentiating Equation (2) against the anchors, again free-variable,

$$
{\nabla }_{{a}_{g}}{\ell }_{\mathrm{CE}}=- \frac{1- {p}_{g}}{\tau }{z}_{v},{\nabla }_{{a}_{h}}{\ell }_{\mathrm{CE}}=\frac{{p}_{h}}{\tau }{z}_{v} \tag{D.23}
$$

so pull and pushes carry the weights of one distribution and vanish together as classification succeeds. Decomposing into alignment and uniformity [5] loses the coupling: with ${\ell }_{\mathrm{align}}=- \langle {z}_{v},{a}_{g}\rangle$, the inner-product form (on the sphere the squared-distance alignment of [5] differs by a factor of two, which the direction claim does not use), at weight $\alpha$, the align-to-uniformity ratio of Appendix C with the uniformity weight normalized to one,

$$
{\nabla }_{{a}_{g}}(\alpha {\ell }_{\mathrm{align}})=- \alpha {z}_{v} \tag{D.24}
$$

leaving a constant-weight attraction, $\alpha$ against the $(1- {p}_{g})/\tau$ of (D.23), that pulls the low-variance anchor toward the high-variance view after the games separate, the weak-view prediction of Section 1 tested in Section 6.2. That test varies more than the coupling, so it fixes the direction, not the size. Anchors are treated as constants when differentiating with respect to the views, and the views as constants when differentiating with respect to the anchors, and every claim here is a direction or a dominance relation, not a rate.

## APPENDIX E: GRADIENT BOUNDS AND THE HANDOVER POINT

This appendix locates the handover between the contrastive push-pull and the invariance term, the crossing of their gradient envelopes that the empirically chosen $\tau =0.02$ and $\lambda =2$ place at 99% confidence. Both envelopes are gradients with respect to the encodings, read as free variables as in Appendix D’s repulsion steps and taken with the tangential projection of the sphere so that they are commensurable; AdamW’s per-coordinate rescaling is not modeled.

**Unit force.** One view enters $V- 1$ of Equation (3)’s $V(V- 1)/2$ pairs, each pulling by at most one,

$$
\| {\nabla }_{{z}_{v}}(\lambda I)\| \le \lambda \frac{2}{V(V- 1)}(V- 1)=\frac{2\lambda }{V}=1 \tag{E.1}
$$

with equality when the other views are aligned with each other and orthogonal to ${z}_{v}$; $\lambda =V/2$ makes the bound exactly one, the recipe’s unit of force, flat in confidence though the realized force is not, falling with the residual disagreement.

**The ceiling.** The descent step of Equation (D.9) is $(1- {p}_{g})/\tau$ times the difference between ${a}_{g}$ and the weighted centroid of the competitors, both in the unit ball and so at most 2 apart,

$$
\| {\nabla }_{{z}_{v}}{\ell }_{\mathrm{CE}}\| \le \frac{2(1- {p}_{g})}{\tau }\le \frac{2}{\tau }=100 \tag{E.2}
$$

so $\tau$ sets the ceiling, a bound the tangential projection can only tighten, which the brake of Appendix D spends linearly as ${p}_{g}\to 1$.

**The handover.** Writing ${p}^{*}$ for the value of ${p}_{g}$ at which the braked envelope falls to the invariance envelope, the recipe values $V=4$, $\lambda =2$, $\tau =0.02$ give

$$
\frac{2(1- {p}^{*})}{\tau }=\frac{2\lambda }{V},\; 1- {p}^{*}=\frac{\lambda \tau }{V}=0.01 \tag{E.3}
$$

so a view’s envelope hands over once classified at 99% confidence: ceiling times handover margin is unit force, $(2/\tau )(\lambda \tau /V)=2\lambda /V$, and the empirically chosen temperature (Table A12) restates as $\tau =(V/\lambda )(1- {p}^{*})=2\times 0.01$, the factor 2 being $V/\lambda$, which equals $\lambda$ here only because $\lambda =\sqrt{V}$. Only the product $\lambda \tau /V$ enters the crossing; $\tau$ was fixed by the sweep of Table A12 and $\lambda$ carries no sweep, so unit force reads the fixed values rather than justifying them. In margin units, $s=\langle {z}_{v},{a}_{g}\rangle - \langle {z}_{v},{a}_{h}\rangle =\tau \ln ({p}_{g}/{p}_{h})$ at any gallery size, and the residual mass $\lambda \tau /V$ placed on one competitor, the worst case, gives the gap Section 4.3 quotes, ${s}^{*}=\tau \ln (V/(\lambda \tau )- 1)=0.02\ln 99\approx 0.092$. The crossing compares worst-case envelopes attained at incompatible configurations, so it is a parameter-scale statement rather than a measured handover.

## APPENDIX F: THE QUERY-REWRITE PROMPTS

Each query is an English description that GPT-5.4-mini at temperature 0.7 writes from the held-out game’s full wiki article, truncated to 12,000 characters, called through an OpenAI-compatible gateway; the instructions below were issued in Chinese and are translated faithfully here, an output shorter than 300 characters re-requested with the elaboration instruction, which asks for at least 300 words and so for well more than the trigger requires, up to three attempts, and the verbatim strings and the retry policy in the released code. The name-stripped instruction asks for invented replacement names, but the model complies literally in only 113 of the 814 queries and generalizes the name away in the other 701, so the register is better read as name-free than as name-substituted.

*ALGORITHM 1: The rewrite instructions behind the two query registers*

```
user message  —  "Game: {title}" then the article text
neutral (Name hit@k)  —  Read the full game page text the user provides and summarize it into a descriptive article about the game. Preserve the real information about the game’s content, gameplay and mechanics in full; do not invent anything that is not on the page. Register: neutral. Write in English and output the article body directly.
name-stripped (Stripped hit@k)  —  Read the full game page text the user provides and summarize it into a descriptive article about the game, in a neutral register. Preserve the real information about the game’s content, gameplay and mechanics in full; do not invent anything that is not on the page. But do not reveal the game’s name, the characters’ names, the items’ names or anything like them — replace them all with imagined, invented words. Write in English and output the article body directly.
elaboration retry (any register)  —  Your previous answer was too short. Please write a considerably MORE DETAILED article (at least 300 words) in the requested style, covering the game’s content, story, world and mechanics from the page text above.
```

## APPENDIX G: REDUCING THE STEAM TAG VOCABULARY

A single mapping file is the source of truth for every tag number in this paper: each of the 202 fine tags is assigned either to one coarse class or to the sentinel “discarded”. A class is present on a game if any of the Steam tags on its right-hand side is, and the vote weights Steam attaches are used only to resolve which source is strongest, never as a target, since the probe predicts presence. On our 2,020 games this yields a $2,020\times 23$ binary matrix of density 0.202, 4.6 classes per game at the mean, 5 at the median, and between 35 and 1,395 games per class.

*ALGORITHM 2: The 23 TAG classes, each written as the original Steam tags it absorbs*

```
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
```

## APPENDIX H: COST AT SCALE

Write $R$ for the number of resident anchor packs (the fold gallery of Section 4.1, so $R=n=1,694$ here; the whole catalog, and so $R=N$, at deployment scale), ${m}_{a}$ for the anchor-pack cap in sentences (the budget of Section 4.1), $d=1,024$ for the embedding width, $G=192$ for the games in a step, and $w=168$, $S=84$, $T=2$ for the window, stride and micro-passes of the teacher in Appendix J. The tower reads one pack with $Q=4$ latent queries, so encoding a pack costs $\Theta ({m}_{a}\cdot Q\cdot d)$ work and holds $\Theta ({m}_{a}\cdot d)$ activation.

The fully coupled teacher re-encodes every pack with gradient at every step, so its step time and its peak activation are

$$
\Theta (R\cdot {m}_{a}\cdot Q\cdot d),\; \Theta (R\cdot {m}_{a}\cdot d) \tag{H.1}
$$

both linear in the catalog. At our scale the backward graph holds 6.9 M sentence slots, 13 GiB of fp16 input before any activation is stored (cf. Table A10, whose 6.6 M counts the fixed split’s 1,613 packs); the same expression at $R={10}^{6}$ asks for 7.6 TiB, which no accelerator holds. This is the wall the scale limitation of Section 8 names.

The measured cost matches (Table A5). Peak device allocation rises by 7.2× across an 8× rise in ${m}_{a}$ and is dominated throughout by the with-gradient re-encoding of the gallery rather than by the anchor store, which accounts for only 1.7 to 13.2 GiB of it across the four budgets. Allocation is a property of the configuration rather than of the accelerator, so it transfers: the three smaller budgets fit a 48 GiB card, while the 4,096-sentence budget needs an 80 GiB one. Inference is unaffected by the budget: a query costs one forward pass through the frozen 0.6B embedder (< 3 GiB) and a dot product against the finished anchors, an inner-product search that runs in milliseconds.

*Table A5: Measured cost of the fully coupled teacher at each anchor budget (peak is the fold-0 maximum device allocation over a training step, per process and unaffected by packing; the 4,096-sentence wall clock is the five-fold range). Only the 4,096-sentence budget had an A100 80 GB to itself, so its elapsed time is a clean per-run figure. The smaller budgets ran several folds packed onto one card, so their wall clock bounds a dedicated run from above: every timed run at 512 and at 1,024 sentences finished inside four hours while sharing its card. The one timed trace at 2,048 is too thin to bound and is left blank.*

| Anchor budget ${m}_{a}$ | Peak device allocation | Wall clock, 2,000 epochs |
| --- | --- | --- |
| 512 sentences | 8.4 GiB | < 4 h |
| 1,024 sentences | 22.3 GiB | < 4 h |
| 2,048 sentences | 35.0 GiB | — |
| **4,096 sentences (recipe)** | **60.6 GiB** | **≈ 6.7 h (6.4–7.2, five folds)** |

At 1,024 sentences, the plateau of Table A2, every retrieval and tag reading stays within 0.02 of the 4,096-sentence recipe at a peak 2.7× smaller, which fits the 24 GB of an RTX 3090 or 4090; the 22.3 GiB leaves little headroom, so such a card should be running headless.

The windowed teacher removes $R$ from both expressions. A micro-pass encodes the batch’s own packs plus one window and frees them before the next, so a step costs

$$
\Theta (T\cdot (G+w)\cdot {m}_{a}\cdot Q\cdot d),\; \Theta ((G+w)\cdot {m}_{a}\cdot d) \tag{H.2}
$$

in time and in memory (Table A6). At $G=192$ and $w=168$ that is 1.47 M sentence slots, 2.8 GiB of input, whether the catalog holds two thousand entries or ten million. What the window does not bound is the ring itself: the windowed teacher still keeps every anchor pack resident (Table A10), so $\Theta (R\cdot {m}_{a}\cdot d)$ of store sits on the device even though only $\Theta ((G+w)\cdot {m}_{a}\cdot d)$ of it is ever differentiated. Measured at our scale the fully coupled teacher peaks at about 61 GiB at ${m}_{a}=4,096$, and about 13 GiB of either teacher’s footprint is that resident ring; the windowed teacher’s peak was not separately measured.

*Table A6: The cost model of the three teachers, with the fully coupled I-CE teacher normalized to 1 at the same catalog size. $R$ is the resident-pack count, ${m}_{a}$ the anchor-pack cap, $d$ the embedding width, $Q=4$ latent queries, $G=192$ games per step, and $w=168$, $S=84$, $T=2$ the window, stride and micro-passes. Ratios are quoted at our $R=1,694$; the windowed teacher’s step contains no $R$, so its first two quantities fall as $1/R$, but its ring keeps every pack resident and therefore stays linear in the catalog on the device. The streamed store, the full-corpus pipeline of this paper, is what removes that last term, paying I/O for it. The measured peak is at ${m}_{a}=4,096$ (fold 0, torch.cuda.max_memory_allocated); the last row extrapolates each variant’s device budget for $R$ on one 80 GiB accelerator, assuming the linear dependence on $R$.*

| Quantity | I-CE, fully coupled | swin-I-CE (w = 168) | swin + streamed store |
| --- | --- | --- | --- |
| Step time | $\Theta (R\cdot {m}_{a}\cdot Q\cdot d)$ | $\Theta (T\cdot (G+w)\cdot {m}_{a}\cdot Q\cdot d)$ | $\Theta (T\cdot (G+w)\cdot {m}_{a}\cdot Q\cdot d)$ |
| ratio to I-CE | 1 | 0.43 | 0.43 |
| Peak activation | $\Theta (R\cdot {m}_{a}\cdot d)$ | $\Theta ((G+w)\cdot {m}_{a}\cdot d)$ | $\Theta ((G+w)\cdot {m}_{a}\cdot d)$ |
| ratio to I-CE | 1 | 0.21 | 0.21 |
| Resident anchor store | $\Theta (R\cdot {m}_{a}\cdot d)$ | $\Theta (R\cdot {m}_{a}\cdot d)$ | $\Theta ((G+w)\cdot {m}_{a}\cdot d)$ |
| ratio to I-CE | 1 | 1 | 0.21 |
| Device memory in R | linear | linear | constant |
| Per-step device I/O | none | none | $\Theta (T\cdot S\cdot {m}_{a}\cdot d)$ ≈ 1.3 GiB |
| Measured peak at R = 1,694 | ≈ 61 GiB | not measured | not measured |
| Extrapolated largest R, one 80 GiB device | ≈ 2,200 | not measured | not memory-bound |

The streamed store closes that last term by giving up residency, and it is the pipeline that built the corpus of this paper. The merged pool is a single memory-mapped fp16 array on disk: the build writes and reads it a million rows at a time, so that assembling the 23,373-game merged pool (from which the 500-review floor of Section 2.2 keeps the 2,020 games of this paper) never holds more than a chunk, and the training loop gathers each anchor pack from that map instead of from a resident ring. The device then holds only what it differentiates, $\Theta ((G+w)\cdot {m}_{a}\cdot d)$, and pays I/O in place of space: the ring advances $T\cdot S=168$ packs per step and only the new ones must be paged, $\Theta (T\cdot S\cdot {m}_{a}\cdot d)$, 1.3 GiB per step at ${m}_{a}=4,096$ and 336 MiB at ${m}_{a}=1,024$, which a prefetch thread hides behind the forward pass. On one 80 GiB accelerator the fully coupled teacher extrapolates from its measured 61 GiB at $R=1,694$ to a ceiling near $R\approx 2,200$, the windowed teacher’s ring still grows linearly in the catalog, and the streamed variant alone has no such bound; the 23,373-game corpus needs 183 GiB of packs at ${m}_{a}=4,096$, which no single device holds, but 46 GiB at ${m}_{a}=1,024$, which one does.

**The sharded window.** The deployable form of the streamed variant splits the ring into $\kappa$ shards, each sweeping its own window of $w$ packs per micro-pass; the CE partition of a view is the union of the batch’s own anchors and the $\kappa$ windows, and each window encode is gradient-checkpointed so that only its $w$ output vectors stay live. A step then costs

$$
\Theta (T\cdot (G+\kappa \cdot w)\cdot {m}_{a}\cdot Q\cdot d) \tag{H.3}
$$

in time while device memory stays at one window, $\Theta ((G+w)\cdot {m}_{a}\cdot d)$, constant in the catalog; a single process with $\kappa$ shards computes exactly what $\kappa$ workers would after an all-reduce of their window logits, and $\kappa =1$ is the windowed teacher above. The released code carries it as the sharded option of the five-fold worker on a swin arm, with the memory-mapped pool as the store, and main_model/pfc_sample.py walks through the supply on synthetic data.

Two quantities do still grow with the catalog, and neither is memory. The ring advances 168 packs per step, so an anchor takes a gradient turn once every $R/(T\cdot S)$ steps: ten steps at our $R=1,694$, 139 at 23,373, and 5,952, about 370 epochs of sixteen steps, at a million; equivalently, peak concurrent coverage is $(G+w)/R$, falling from 21% here to 1.5% at full-corpus scale and 0.04% at a million. A million-entity store is 7.6 TiB at ${m}_{a}=4,096$ or 1.9 TiB at ${m}_{a}=1,024$, an ordinary disk array read at about a gigabyte per step. Our evidence spans 21% to 100% peak concurrent coverage, the range in which the windowed teacher gives up 0.032 Stripped hit@1 against full coupling (Table A7); it does not reach the 0.04% regime and we do not claim it.

*Table A7: The cost of the anchor-supply economies (test-set queries (ts), five-fold mean ± std at the 4,096-sentence anchor budget). The sliding fresh-window variant (swin, Appendix J) re-encodes ~26% of the five-fold gallery with gradient per step (27% on the fixed split of Appendix J); the two-stage row hands a BYOL warm start to the windowed teacher for the last 600 epochs; the EMA + memory-bank variant replaces the coupled teacher with an EMA shadow encoder feeding a 3,072-key bank (Appendix C).*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE, fully coupled teacher | 0.954 ± 0.013 | 0.997 ± 0.004 | 0.741 ± 0.018 | 0.928 ± 0.020 | 0.685 ± 0.019 |
| swin-I-CE (~26% gradient window) | 0.934 ± 0.014 | 0.994 ± 0.004 | 0.709 ± 0.027 | 0.922 ± 0.010 | 0.691 ± 0.014 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.930 ± 0.011 | 0.991 ± 0.008 | 0.709 ± 0.019 | 0.917 ± 0.005 | 0.705 ± 0.022 |
| I-CE (EMA + memory bank) | 0.936 ± 0.010 | 0.991 ± 0.007 | 0.636 ± 0.016 | 0.881 ± 0.016 | 0.711 ± 0.016 |

## APPENDIX I: THE DOCUMENT VIEW: PRESENT, ABSENT, REWRITTEN

The document view is one of the four strong views, and this ablation isolates it at the 512-sentence anchor budget, five folds per cell, seed-paired with the raw-document tower of the recipe, which keeps the original documents (Table A8).

*Table A8: The document view: present, absent, rewritten (test-set queries (ts), five-fold mean ± std at the 512-sentence anchor budget, checkpoints selected by the rank criterion of Section 5, zero-shot, seed-paired folds; deltas are computed before rounding). The document view is worth 0.008 Stripped hit@1 (and 0.032 Name hit@1), and rewriting it helps: the LLM-rewritten row leads raw documents on the stripped columns (+0.008 Stripped hit@1) and ties on the name columns. The firewall would be a gain, not a cost.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| **I-CE, raw documents (recipe)** | **0.916 ± 0.009** | **0.991 ± 0.003** | **0.644 ± 0.024** | **0.872 ± 0.010** | **0.692 ± 0.013** |
| I-CE, no document view | 0.883 ± 0.016 | 0.981 ± 0.007 | 0.636 ± 0.016 | 0.871 ± 0.013 | 0.680 ± 0.024 |
| I-CE, LLM-rewritten documents | 0.914 ± 0.010 | 0.990 ± 0.002 | 0.652 ± 0.023 | 0.882 ± 0.009 | 0.689 ± 0.016 |

## APPENDIX J: THE WINDOWED TEACHER, THE WINDOW-SIZE, VICREG AND TEMPERATURE SWEEPS, AND THE TAG READING AGAINST THE MARGIN

The windowed teacher deferred from Section 4 is specified below, followed by the three sweeps behind design choices. The window-size scan and the VICReg grid are single-split and should be read at the $\pm 0.03$ resolution of Section 5; the temperature sweep is five-fold.

swin-I-CE lays the catalog on a ring with a persistent pointer $u$ and runs two micro-passes per step: micro-pass $l\in \{0,\; 1\}$ freshly encodes the $w=168$ anchors starting at ring position $u+l\cdot S$ (stride $S=84$, so consecutive passes overlap by half a window) together with the current batch’s own anchors, forms its own CE partition (every view’s positive sits in the batch block, the window anchors serve as negatives) and backpropagates immediately, freeing the window’s activations before the next pass. The pointer then advances by $2S$, sweeping the full catalog every ten steps so that every anchor takes a regular, gradient-coupled turn, which per step touches about 27% of the gallery with gradient and bounds the gradient-coupled encoding to 22% of the full backward graph (Table A10); the tower, the views, and the invariance term are unchanged. Table A7 quantifies the cost of this variant five-fold; the window-size sweep ($w=84–336$) is Table A9.

*Table A9: The window-size sweep behind the windowed teacher (fixed split, test-set queries (ts) of that split, 4,096-sentence anchors; a single anchor draw rather than the ten-draw expectation of the main tables). Every row’s checkpoint is selected by the rank criterion of Section 5, the sum of exp(−rank) over the validation games’ rewrites of both registers, over its first 2,000 epochs, the same rule as the five-fold tables; the cap matters for the reference arm alone, whose run continued to 4,000 epochs. Coverage counts the training-set anchor packs re-encoded with gradient per step (batch plus the union of the two micro-pass windows). Halving or doubling the window moves Stripped hit@1 by at most 0.044 with no monotone trend, at the single-split resolution: the smallest window already holds the plateau, so memory, not coverage, should pick $w$. Tag readings are the test-set probe of Section 5, recomputed on this split so that they sit on the same scale as the rest of the paper; their spread here is single-split noise, and at fold level swin and full coupling read tags at parity (Table A7, which gives the five-fold account of $w=168$). The fully coupled row is the anchor-budget ladder’s tower re-selected under the same criterion; it leads the windowed arms by 0.04 to 0.08 Stripped hit@1 on this split, in line with the five-fold 0.032 of Table A7. All four rows are regenerated by one script in the repository.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| swin, w = 84 (step 42, 19.7% coverage) | 0.956 | 0.990 | 0.725 | 0.912 | 0.707 |
| **swin, w = 168 (step 84, 27.5% coverage)** | **0.936** | **0.990** | **0.681** | **0.907** | **0.689** |
| swin, w = 336 (step 168, 43.1% coverage) | 0.946 | 0.990 | 0.711 | 0.897 | 0.676 |
| **I-CE, fully coupled (100% coverage)** | **0.975** | **1.000** | **0.765** | **0.897** | **0.678** |

*Table A10: What the sliding window bounds. The fully coupled objective re-encodes every training-set anchor pack with gradient at once; swin holds only the batch’s own anchors plus one $w=168$ window per micro-pass and frees it before the next, so its gradient-coupled encoding peaks at 360 anchors, 22% of the full graph. Both variants keep the whole catalog resident, so the saving is bounded by the persistent ring rather than reaching the full 4.5×. Sentence-slot counts are ceilings (packs are capped at 4,096 sentences and realized packs are shorter), quoted on the fixed split of Section 4.1 (1,613 training-set games); a five-fold gallery of 1,694 or 1,695 scales the same way.*

|  | Full-gallery I-CE | swin-I-CE (w = 168) |
| --- | --- | --- |
| **Anchors re-encoded with gradient per backward (peak)** | 1,613 | 360 (192 batch + 168 window) |
| **as a fraction of the anchor gallery** | 100% | 22% |
| **Sentence slots in the backward graph (4,096 budget)** | ≈ 6.6 M | ≈ 1.5 M |
| **Anchor packs held resident (the ring)** | 1,613 (all) | 1,613 (all) |

The VICReg grid below predates validation selection and its towers were logged only at trajectory peaks, so Table A11 reports each arm’s peak Stripped hit@1 and omits tag readings, which cannot be re-scored under the test-set protocol of Section 5.

*Table A11: The VICReg weight sweep (fixed split, test-set queries (ts) of that split, 512-sentence anchors at evaluation, a single anchor draw; trajectory peaks). All rows use the canonical wiring (variance, invariance, and covariance all on the expander-output pair); the earlier centroid wiring (invariance as a mean-squared error between unit-norm view centroids) collapsed retrieval outright at every view width (hit@1 $\le 0.03$) and is omitted. $v/i/c=20/10/20$ is the best cell and the image-domain 25/25/1 trails it. Drawing views for the whole training set per step (batch = all) provides true population moments in the variance term at roughly eight times the step cost; its one completed run reads 0.250 Stripped hit@1 against the batch-192 recipe’s 0.284 under this table’s own protocol, no improvement at the $\pm 0.03$ resolution, so the body’s five-fold VICReg run (Figure 4) trains at batch 192. I-CE and CE peaks under the same readout anchor the scale.*

| Objective | Name hit@1 | Stripped hit@1 |
| --- | --- | --- |
| v/i/c = 25/25/1 | 0.373 | 0.216 |
| **v/i/c = 20/10/20 (recipe weights)** | **0.578** | **0.284** |
| v/i/c = 20/10/15 | 0.574 | 0.265 |
| 25/25/1, batch = all | 0.328 | 0.201 |
| 20/10/20, batch = all | 0.495 | 0.250 |
| **I-CE (reference, same readout)** | **0.926** | **0.672** |
| CE (reference, same readout) | 0.922 | 0.618 |

The temperature sweep of Table A12 is five-fold and carries the Section 8 verdict directly. The learnable variant (initialized at 0.02, $\tau$ clamped to $[0.005,\; 0.2]$) never settles at an interior optimum: in all five folds it sharpens monotonically to the clamp floor within 250 epochs and trains there, an effective fixed $\tau =0.005$, beating the frozen recipe by 0.010 Stripped hit@1 in five folds of five and by 0.008 on tags (computed before rounding), the sharp-side headroom that the frozen setting of Section 4.2 leaves to deployment.

*Table A12: The temperature sweep behind the Section 8 verdict and the frozen setting of Section 4.2 (test-set queries (ts), five-fold mean ± std at the 512- and 2,048-sentence anchor budgets; the $\tau =0.10$ cell at 512 has four folds; checkpoints selected by the rank criterion of Section 5 on the validation fold, the table reporting their test-set readings). Among fixed settings $\tau =0.02$ is retrieval-optimal at both budgets: softening to 0.05/0.10 costs 0.060–0.226 Stripped hit@1, and what it buys is the tag reading: $\tau =0.05$ lifts it to the edge of the negative-light band of Section 6.1, so on the soft side temperature trades the tag reading against identity.*

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| **τ = 0.02 (recipe), 512** | **0.916 ± 0.009** | **0.991 ± 0.003** | **0.644 ± 0.024** | **0.872 ± 0.010** | **0.692 ± 0.013** |
| τ = 0.05, 512 | 0.896 ± 0.017 | 0.987 ± 0.007 | 0.584 ± 0.021 | 0.859 ± 0.014 | 0.704 ± 0.017 |
| τ = 0.10, 512 (four folds) | 0.828 ± 0.027 | 0.956 ± 0.020 | 0.490 ± 0.019 | 0.770 ± 0.016 | 0.701 ± 0.021 |
| **τ = 0.02 (recipe), 2,048** | **0.947 ± 0.016** | **0.996 ± 0.003** | **0.728 ± 0.009** | **0.918 ± 0.010** | **0.684 ± 0.013** |
| τ = 0.05, 2,048 | 0.938 ± 0.018 | 0.992 ± 0.004 | 0.655 ± 0.027 | 0.890 ± 0.015 | 0.710 ± 0.015 |
| τ = 0.10, 2,048 | 0.883 ± 0.032 | 0.978 ± 0.014 | 0.502 ± 0.012 | 0.785 ± 0.012 | 0.707 ± 0.015 |
| learnable τ, 2,048 | 0.951 ± 0.008 | 0.995 ± 0.005 | 0.738 ± 0.006 | 0.920 ± 0.015 | 0.693 ± 0.015 |

**The tag reading against the margin.** Pairing the fold-0 margins of Table 1 with the five-fold tag means of Table A3 for the five trained towers (BYOL 0.047 / 0.712, SimCLR 0.106 / 0.707, VICReg 0.108 / 0.708, I-CE 0.176 / 0.685, CE 0.191 / 0.664) gives

$$
\mathrm{tag}\approx 0.735- 0.32\times \mathrm{margin} \tag{J.1}
$$

with Pearson −0.91 and Spearman −0.90, the one rank inversion being SimCLR against VICReg, 0.002 apart in margin and 0.001 in tag. Pairing the same margins with those towers’ fold-0 tag readings instead gives

$$
\mathrm{tag}\approx 0.762- 0.39\times \mathrm{margin} \tag{J.2}
$$

with Pearson −0.87 and Spearman −0.70, single-fold tag readings being the noisier side. I-CE sits 0.006 above the five-fold line and CE 0.010 below it; the regression is not printed in the body because five points under two protocols support a direction, not a coefficient.

## APPENDIX K: PER-FOLD DETAIL ACROSS THE ANCHOR BUDGETS

Table A13 gives, fold by fold, the two numbers the body quotes across budgets. Values at 512 to 2,048 sentences are the ten-draw means behind Table A2, the 4,096-sentence rows those behind Tables A1 and A3; the two objectives are seed-paired on identical splits, so each column is a paired comparison.

*Table A13: CE versus I-CE per fold at every anchor budget (test-set queries, ten anchor draws per fold, mean ± std over the five folds). Upper block: Stripped hit@1, which I-CE wins in 20 of 20 paired folds. Lower block: test-set tag F1, which I-CE wins in 18 of 20, the exceptions being fold 3 at 512 and fold 1 at 4,096 sentences.*

| Stripped hit@1 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
| --- | --- | --- | --- | --- | --- | --- |
| CE @512 | 0.626 | 0.620 | 0.594 | 0.609 | 0.619 | 0.614 ± 0.011 |
| I-CE @512 | 0.653 | 0.639 | 0.684 | 0.612 | 0.631 | 0.644 ± 0.024 |
| CE @1,024 | 0.663 | 0.647 | 0.645 | 0.661 | 0.659 | 0.655 ± 0.007 |
| I-CE @1,024 | 0.737 | 0.723 | 0.719 | 0.728 | 0.727 | 0.727 ± 0.006 |
| CE @2,048 | 0.684 | 0.684 | 0.682 | 0.647 | 0.660 | 0.671 ± 0.015 |
| I-CE @2,048 | 0.732 | 0.740 | 0.721 | 0.733 | 0.714 | 0.728 ± 0.009 |
| CE @4,096 | 0.687 | 0.693 | 0.667 | 0.656 | 0.662 | 0.673 ± 0.014 |
| I-CE @4,096 | 0.728 | 0.745 | 0.715 | 0.749 | 0.769 | 0.741 ± 0.018 |

| Test-set tag F1 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
| --- | --- | --- | --- | --- | --- | --- |
| CE @512 | 0.700 | 0.673 | 0.677 | 0.699 | 0.684 | 0.687 ± 0.011 |
| I-CE @512 | 0.712 | 0.678 | 0.679 | 0.691 | 0.698 | 0.692 ± 0.013 |
| CE @1,024 | 0.678 | 0.649 | 0.667 | 0.658 | 0.686 | 0.668 ± 0.013 |
| I-CE @1,024 | 0.714 | 0.657 | 0.678 | 0.702 | 0.714 | 0.693 ± 0.022 |
| CE @2,048 | 0.670 | 0.630 | 0.653 | 0.664 | 0.665 | 0.657 ± 0.014 |
| I-CE @2,048 | 0.686 | 0.666 | 0.673 | 0.701 | 0.695 | 0.684 ± 0.013 |
| CE @4,096 | 0.681 | 0.667 | 0.649 | 0.681 | 0.642 | 0.664 ± 0.016 |
| I-CE @4,096 | 0.689 | 0.665 | 0.662 | 0.697 | 0.709 | 0.685 ± 0.019 |

## APPENDIX L: THE TAG DEFICIT AS A REGISTER CROSSING

The tag probe of Section 5 is fit in one register and read in another, so its ranking of the towers mixes tag content with register transfer. Table A14 separates the two on the shipped rank-selected checkpoints, all five folds at the 4,096-sentence anchor budget, the only budget at which every tower of Table A3 exists (BYOL, the SimCLR-style arm and VICReg have no selected checkpoints at 512). The cross-register column is the shipped protocol unchanged: for each of the ten anchor draws behind Table A3, a ridge readout is fit on the 1,694 or 1,695 training-pool anchors (review register), its threshold is picked on validation-game anchors, and it is scored on the name-intact wiki-rewrite query vectors of the test games (document register), the ten micro-F1 values then averaged; recomputed from the shipped checkpoints, this column returns every tag entry of Table A3 exactly. The in-domain column changes one thing: the same ridge with the same threshold is scored on the test-game anchors of the same draw, so the two readings differ only in the register of the vectors read. Transfer loss is in-domain minus cross-register. The register offset is the norm of the mean anchor-to-query displacement over the training games of the fold that carry a wiki query (about 488 per fold), measured in the tower’s 128-d output space.

*Table A14: Where the tag deficit lives (five folds at the 4,096-sentence anchor budget, shipped rank-selected checkpoints, ten anchor draws, mean ± std over folds). The cross-register column is the shipped readout of Section 5 and reproduces the tag column of Table A3; the in-domain column reuses that ridge and its threshold on test-game anchors. Rows are ordered by transfer loss. The margin column repeats Table 1 (fold-0, single draw). The frozen embedder is omitted: its readout runs in the 1,024-d input space and is not comparable to the 128-d towers.*

| Objective | NN margin (Table 1) | Cross-register tag F1 | In-domain tag F1 | Transfer loss | Register offset |
| --- | --- | --- | --- | --- | --- |
| CE (contrast only) | 0.191 | 0.664 ± 0.016 | 0.758 ± 0.024 | 0.094 | 0.297 |
| I-CE (ours) | 0.176 | 0.685 ± 0.019 | 0.763 ± 0.019 | 0.078 | 0.217 |
| SimCLR-style (in-batch views) | 0.106 | 0.707 ± 0.019 | 0.753 ± 0.015 | 0.046 | 0.075 |
| VICReg (v/i/c = 20/10/20) | 0.108 | 0.708 ± 0.022 | 0.743 ± 0.016 | 0.034 | 0.141 |
| BYOL | 0.047 | 0.712 ± 0.017 | 0.734 ± 0.016 | 0.022 | 0.045 |

Table A15 gives the paired differences behind the body’s claims: the cross-register deficit holds against all four negative-light towers in every fold, and the in-domain reversal holds against the three towers of Figure 4 and not against CE, which the body states accordingly.

*Table A15: Paired differences across the five folds, I-CE minus the named tower, at the 4,096-sentence budget: the cross-register readout of Section 5 and the in-domain readout of Table A14, with the paired $t$ (four degrees of freedom) and the folds I-CE wins. The EMA + memory-bank tower of Table A7 has the cross-register reading only; its in-domain readout was not measured.*

| I-CE minus | Cross-register Δ | t | Folds won | In-domain Δ | t | Folds won |
| --- | --- | --- | --- | --- | --- | --- |
| BYOL | −0.028 | −3.74 | 0 of 5 | +0.029 | 9.41 | 5 of 5 |
| VICReg | −0.024 | −3.56 | 0 of 5 | +0.020 | 6.93 | 5 of 5 |
| SimCLR-style | −0.022 | −4.05 | 0 of 5 | +0.010 | 2.81 | 4 of 5 |
| EMA + memory bank | −0.027 | −13.44 | 0 of 5 | not measured | n/a | n/a |
| CE | +0.021 | 1.68 | 4 of 5 | +0.004 | 0.90 | 4 of 5 |

**Why identity pressure costs transfer.** Every anchor pack is a store-page prefix followed by reviews (Section 4.1), so the gallery an anchored objective repels against is review register in the main, and a game’s document view has to leave its own anchor’s neighborhood far enough to be told apart from the other 1,693 or 1,694 review anchors, most cheaply in a direction no review anchor occupies, orthogonal to the review manifold. The invariance term is the only force pulling that view back onto its anchor and it closes the gap only part of the way: I-CE reads tags in domain level with CE and above the three negative-light towers yet pays the second-largest transfer loss, while CE, with no invariance term, pays the largest loss and carries the largest offset. BYOL and VICReg never push a view away from a review anchor, so their two registers stay close and they lose least in transfer, which is how they win the shipped tag column while reading tags worst in domain; the SimCLR-style arm sits between the families, its in-batch negatives drawn from both registers, so its repulsion does not single out the document direction.

**What this does not license.** Subtracting the training-estimated offset from every test query lifts I-CE’s cross-register reading from 0.685 to 0.700 but costs it retrieval (Stripped hit@1 0.741 to 0.728), and for an affine readout a constant shift only moves each tag’s threshold: a recalibration, not a repair of the geometry. Letting the anchor pack see the strong views’ own sentences does not help either (the vfa arm of Supplement A). If a correction exists it belongs on the repulsive side, admitting document-register negatives into the gallery, which we have not tested.
