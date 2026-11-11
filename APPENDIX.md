# Invariant Game Representation Learning from Steam Reviews

Appendices A–N

*Appendices to the AICCC 2026 submission (PA0049)*

## APPENDIX A: FIVE-FOLD CROSS-VALIDATION DETAIL

Tables A13 and A14 in online Appendix J give the fold-level detail behind Tables A1 and A2, and Table A12 in online Appendix I tabulates the EMA + memory-bank variant.

**Table A1:** The self-supervision families in full, the five-fold account behind the body’s Figure 7 (test-set queries (ts), five-fold mean ± std, protocol as in Figure 7). Rows are sorted by Stripped hit@1, worst first; “epd” marks VICReg wired on the expander outputs, $v/i/c=20/10/20$. The DINO-style row is the control of Section 4.3, DINO at its published settings (online Appendix H, Table A9), trained on fold 0 only; its ± is over the ten anchor draws, and I-CE on the same fold reads 0.970, 1.000, 0.728, 0.920 and 0.689.

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| Frozen embedder (mean pool) | 0.425 ± 0.033 | 0.626 ± 0.014 | 0.209 ± 0.020 | 0.380 ± 0.031 | 0.588 ± 0.009 |
| BYOL | 0.441 ± 0.033 | 0.737 ± 0.026 | 0.284 ± 0.017 | 0.553 ± 0.043 | 0.712 ± 0.017 |
| VICReg (epd 20/10/20) | 0.823 ± 0.010 | 0.960 ± 0.016 | 0.451 ± 0.017 | 0.729 ± 0.033 | 0.708 ± 0.022 |
| DINO-style (fold 0 only) | 0.829 ± 0.008 | 0.960 ± 0.003 | 0.472 ± 0.012 | 0.780 ± 0.010 | 0.721 ± 0.007 |
| SimCLR-style (in-batch views) | 0.918 ± 0.024 | 0.990 ± 0.007 | 0.632 ± 0.029 | 0.882 ± 0.022 | 0.707 ± 0.019 |
| CE (contrast only) | 0.938 ± 0.013 | 0.988 ± 0.010 | 0.673 ± 0.014 | 0.889 ± 0.011 | 0.664 ± 0.016 |
| **I-CE (ours)** | **0.954 ± 0.013** | **0.997 ± 0.004** | **0.741 ± 0.018** | **0.928 ± 0.020** | **0.685 ± 0.019** |

**Table A2:** I-CE trained at anchor budgets of 512, 1,024 and 2,048 sentences (test-set queries (ts), five-fold mean ± std, read as in Table A1; the 2,048-sentence row is its I-CE). On Stripped hit@1, increasing the budget yields one significant step, 512 to 1,024 sentences (p-value 0.019), followed by a plateau: 1,024 and 2,048 sentences are indistinguishable (p-value 0.76).

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE @512 | 0.942 ± 0.017 | 0.997 ± 0.004 | 0.715 ± 0.010 | 0.918 ± 0.016 | 0.685 ± 0.014 |
| I-CE @1,024 | 0.947 ± 0.018 | 0.998 ± 0.003 | 0.738 ± 0.009 | 0.922 ± 0.007 | 0.690 ± 0.021 |
| **I-CE @2,048** | **0.954 ± 0.013** | **0.997 ± 0.004** | **0.741 ± 0.018** | **0.928 ± 0.020** | **0.685 ± 0.019** |

## APPENDIX B: THE TWO QUERY REGISTERS ON ONE GAME

Both registers of Section 5 are written from the same wiki article by the same model and differ only in the instruction (Appendix F); the openings below are verbatim.

**LISTING 1:** The two query registers on one held-out game

```
Name-intact (Name hit@k)
Miasma Chronicles is a tactical role-playing video game developed by The Bearded Ladies and published by 505 Games in 2023. Set in a post-apocalyptic version of the United States, it combines turn-based tactics with stealth game mechanics. The story follows a world devastated by a force known as the miasma, which can transform or kill living things. More than a century after this catastrophe, a woman leaves her son, Elvis, in the care of a robot named Diggs. Elvis also receives a glove left behind by his mother that can control the miasma. Players take control of Elvis as he searches for his mother and tries to uncover more about the miasma.

Name-stripped (Stripped hit@k)
This tactical role-playing game takes place in a ruined future version of the United States, where a mysterious force has devastated the land by transforming or killing living things. The story follows a young man who is left in the care of a robot guardian after his mother disappears, and who later sets out to find her while uncovering the truth behind the strange force at the center of the world. He carries a special glove left by his mother, which can control this force and serves as a key part of both the narrative and the gameplay.
```

## APPENDIX C: THE TEMPERATURE, INVARIANCE-WEIGHT, VIEW-COUNT, LATENT-QUERY AND VICREG SWEEPS, THE WINDOWED TEACHER, AND THE TAG READING AGAINST THE MARGIN

The five sweeps behind design choices come first, followed by the windowed teacher behind the sliding-window design of Section 8. The temperature sweep is five-fold; the invariance weight and the view count are swept over three folds; the latent queries are swept over three folds at the 512-sentence budget and over five folds at the full budget; the VICReg grid is single-split and should be read at the $\pm 0.03$ resolution of Section 5.

In the temperature sweep of Table A3, the learnable variant (initialized at 0.02, $\tau$ clamped to $[0.005,\ 0.2]$ ) never settles at an interior optimum: in all five folds it sharpens monotonically to the clamp floor within 250 epochs and trains there, an effective fixed $\tau =0.005$, beating the frozen recipe by 0.010 Stripped hit@1 in four folds of five and by 0.009 on tags (computed before rounding), the sharp-side headroom that the frozen setting of Section 4.2 leaves to deployment.

**Table A3:** The temperature sweep behind the Section 8 verdict and the frozen setting of Section 4.2 (test-set queries (ts), five-fold mean ± std, read as in Table A1). The 2,048-sentence rows are seed-paired on the same anchor packs; their recipe row is a second run of the recipe at that budget (the towers of Table A2 read 0.741 ± 0.018 Stripped hit@1).

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| **τ = 0.02 (recipe), 512** | **0.942 ± 0.017** | **0.997 ± 0.004** | **0.715 ± 0.010** | **0.918 ± 0.016** | **0.685 ± 0.014** |
| τ = 0.05, 512 | 0.928 ± 0.026 | 0.993 ± 0.008 | 0.642 ± 0.026 | 0.895 ± 0.014 | 0.715 ± 0.017 |
| τ = 0.10, 512 | 0.876 ± 0.028 | 0.973 ± 0.012 | 0.520 ± 0.015 | 0.793 ± 0.009 | 0.705 ± 0.020 |
| **τ = 0.02 (recipe), 2,048** | **0.949 ± 0.016** | **0.997 ± 0.003** | **0.732 ± 0.010** | **0.923 ± 0.013** | **0.683 ± 0.013** |
| τ = 0.05, 2,048 | 0.941 ± 0.018 | 0.992 ± 0.004 | 0.657 ± 0.028 | 0.891 ± 0.017 | 0.711 ± 0.014 |
| τ = 0.10, 2,048 | 0.887 ± 0.033 | 0.978 ± 0.013 | 0.506 ± 0.015 | 0.786 ± 0.015 | 0.709 ± 0.016 |
| learnable τ, 2,048 | 0.950 ± 0.008 | 0.996 ± 0.004 | 0.742 ± 0.007 | 0.921 ± 0.014 | 0.692 ± 0.014 |

**The invariance weight.** No dose of Table A4 separates from the recipe on any column (p ≥ 0.12), and every dose stays 0.03 to 0.05 Stripped hit@1 above CE, so the invariance term is what lifts I-CE over CE and its weight sits on a plateau from 1 to 3; the highest dose gives a little back (−0.015 Stripped hit@1, behind in all three folds), the first sign of the plateau’s edge. Appendix D reads the same setting from the gradients: only the product λτ/V enters the handover point, so the weight and the temperature trade against each other and the five-fold sweep of Table A3 fixes the pair.

**Table A4:** The invariance weight λ over folds 0 to 2 (test-set queries (ts), three-fold mean ± std; towers trained at the 512-sentence anchor budget and scored against 4,096-sentence anchors, ten draws per fold, checkpoints selected by the rank criterion of Section 5, seed-paired folds).

| Invariance weight | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| λ = 0 (CE) | 0.924 ± 0.024 | 0.993 ± 0.005 | 0.677 ± 0.015 | 0.892 ± 0.013 | 0.661 ± 0.028 |
| λ = 1 | 0.940 ± 0.033 | 0.997 ± 0.003 | 0.725 ± 0.018 | 0.911 ± 0.015 | 0.687 ± 0.019 |
| **λ = 2 (recipe)** | **0.942 ± 0.022** | **0.999 ± 0.001** | **0.718 ± 0.004** | **0.910 ± 0.017** | **0.682 ± 0.015** |
| λ = 3 | 0.946 ± 0.014 | 0.993 ± 0.009 | 0.716 ± 0.002 | 0.911 ± 0.005 | 0.687 ± 0.019 |
| λ = 4 | 0.928 ± 0.018 | 0.992 ± 0.003 | 0.703 ± 0.012 | 0.917 ± 0.009 | 0.681 ± 0.009 |

**The view count.** Every count of Table A5 trails I-CE in two or three of the three folds, and none separates on any column except V = 6, which trails by 0.006 Stripped hit@1 and 0.005 Name hit@5, each in all three folds (p = 0.028 and 0.029), gaps below the resolution of any single fold. Further review views buy nothing over these folds; that does not show the method is insensitive to V, and under the per-view-sum convention a further view also moves the handover point λτ/V (Appendix D), which the V = 6, λ = 3 row restores without changing the reading. The flat axis matches the multi-crop ablation of DINO [24], where six local crops trail ten by 0.8 points of ImageNet linear accuracy at 100 epochs but by 0.2 at 300. Read against the 512-sentence anchor packs at every saved checkpoint, the counts here also differ most early: the fold means of the five counts span 0.033 validation and 0.018 test Stripped hit@1 over epochs 50 to 300, V = 3 lowest on both, and 0.013 and 0.011 over epochs 1,000 to 2,000.

**Table A5:** The number of strong views V over folds 0 to 2: one document view and V − 1 review views (protocol, budget and selection as in Table A4); the last row raises λ with V so that λ/V stays at the recipe’s value.

| Strong views | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| V = 3 | 0.946 ± 0.014 | 0.997 ± 0.003 | 0.710 ± 0.011 | 0.908 ± 0.019 | 0.684 ± 0.018 |
| **V = 4 (recipe)** | **0.942 ± 0.022** | **0.999 ± 0.001** | **0.718 ± 0.004** | **0.910 ± 0.017** | **0.682 ± 0.015** |
| V = 5 | 0.939 ± 0.021 | 0.993 ± 0.007 | 0.708 ± 0.012 | 0.908 ± 0.013 | 0.677 ± 0.019 |
| V = 6 | 0.931 ± 0.023 | 0.994 ± 0.001 | 0.713 ± 0.005 | 0.895 ± 0.011 | 0.685 ± 0.016 |
| V = 7 | 0.928 ± 0.025 | 0.995 ± 0.005 | 0.714 ± 0.017 | 0.909 ± 0.007 | 0.674 ± 0.025 |
| V = 6, λ = 3 | 0.933 ± 0.015 | 0.992 ± 0.006 | 0.707 ± 0.015 | 0.908 ± 0.012 | 0.691 ± 0.010 |

**The number of latent queries.** From four queries up the axis is flat under both protocols; below four it is not (Table A6). Over folds 0 to 2 the one- and two-query towers trail I-CE on Stripped hit@1 in all three folds, by 0.017 and 0.031 (p = 0.18 and 0.031), the one-query tower on Name hit@1 as well, by 0.028 (p = 0.047); the eight- and sixteen-query towers lead it in three and two of the three folds, neither separating (p = 0.20 and 0.64), and over the five folds the eight-query tower trails the recipe on tags in four folds of five without separating. Averaged right after the attention, the slots amount to one attention readout whose weights, head by head, are the mean of the slots’ weights: Q sets how many attention patterns the reader mixes, not how many vectors it keeps, and for the stripped queries one or two patterns are too few where four are enough. The four queries do not stay alike (Figure A1): close together at the first checkpoint, they part between epochs 200 and 450, rearrange until about epoch 800 and keep distinct positions from then on.

**Table A6:** The number of latent queries under both protocols (test-set queries (ts)): the three-fold rows as in Table A4, the five-fold rows seed-paired per fold and read as Table A1 is, Q = 4 being its I-CE. The paired differences over the five folds (Q = 8 minus Q = 4) are −0.003 ± 0.009 Name hit@1, +0.003 ± 0.042 Stripped hit@1 and −0.021 ± 0.022 test-set tag F1 (p = 0.52, 0.90 and 0.11). The readout is the mean over the slots throughout.

| Latent queries | Protocol | Anchor budget | Name hit@1 | Stripped hit@1 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| Q = 1 | folds 0–2 | 512 | 0.915 ± 0.023 | 0.701 ± 0.008 | 0.679 ± 0.021 |
| Q = 2 | folds 0–2 | 512 | 0.936 ± 0.016 | 0.687 ± 0.012 | 0.661 ± 0.015 |
| **Q = 4 (recipe)** | **folds 0–2** | **512** | **0.942 ± 0.022** | **0.718 ± 0.004** | **0.682 ± 0.015** |
| Q = 8 | folds 0–2 | 512 | 0.945 ± 0.014 | 0.740 ± 0.021 | 0.691 ± 0.013 |
| Q = 16 | folds 0–2 | 512 | 0.930 ± 0.018 | 0.727 ± 0.026 | 0.683 ± 0.020 |
| **Q = 4 (recipe)** | **five folds** | **2,048** | **0.954 ± 0.013** | **0.741 ± 0.018** | **0.685 ± 0.019** |
| Q = 8 | five folds | 2,048 | 0.951 ± 0.011 | 0.744 ± 0.026 | 0.664 ± 0.025 |

> **Image — `assets/paper/figA1_slot_pca.png`** (3.34 in wide, 1.75 in tall, docx name "SlotTrajectory")
>
> **Alt text.** A line chart of four lines, one per latent-query slot, each in its own stroke (slot 1 solid, slot 2 short dashes, slot 3 dash-dot, slot 4 long dashes) and numbered at its right end, against training epoch from 0 to 2,000, the vertical axis each query's coordinate on the first principal component. The four lines start close together near zero, separate between epochs 200 and 450, cross and rearrange until about epoch 800, and then run nearly flat at four distinct levels to the end, slot 1 highest, then slots 4, 2 and 3; a dotted vertical line labeled selected marks the selected checkpoint at epoch 1,750.

**Figure A1:** The four latent queries of the fixed-split I-CE tower (Figure 9) over its 2,000 epochs: each slot’s query at every saved checkpoint, projected on the first principal component of all 160 (36% of their variance); dotted, the checkpoint the rank criterion selects.

**Table A7:** The VICReg weight sweep (fixed split, test-set queries (ts) of that split, read as Table 3 is: 4,096-sentence anchors, a single anchor draw, every tower at the checkpoint the rank criterion of Section 5 picks on the split’s validation queries over its whole run). All rows use the canonical wiring (variance, invariance, and covariance all on the expander-output pair); the earlier centroid wiring (invariance as a mean-squared error between unit-norm view centroids) collapsed retrieval outright at every view width (hit@1 $\le 0.03$ ) and is omitted. VICReg does not read the anchors in training, so these are the towers of the original sweep, each of whose 40 checkpoints (epochs 50 to 2,000) encodes the 4,096-sentence anchors for the rank criterion, as the anchor-free towers of Table 3 do; the two whole-training-set (batch = all) runs of that sweep are omitted, and the body’s five-fold VICReg run (Figure 7) trains at batch 192. I-CE and CE are the rows of Table 3, which anchor the scale. The image-domain 25/25/1 trails the other two cells on retrieval and on tags; every cell trails CE and I-CE by at least 0.21 Stripped hit@1 and reads tags above both; the rank criterion takes two of the cells at epoch 100.

| Objective | Epoch | Name hit@1 | Stripped hit@1 | Test-set tag F1 |
| --- | --- | --- | --- | --- |
| v/i/c = 25/25/1 | 550 | 0.618 | 0.363 | 0.708 |
| **v/i/c = 20/10/20 (recipe weights)** | **100** | **0.814** | **0.451** | **0.725** |
| v/i/c = 20/10/15 | 100 | 0.799 | 0.456 | 0.718 |
| **I-CE (Table 3)** | **2000** | **0.946** | **0.725** | **0.677** |
| CE (Table 3) | 1900 | 0.907 | 0.667 | 0.640 |

**The windowed teacher.** swin-I-CE lays the catalog on a ring with a persistent pointer $u$ and runs two micro-passes per step: micro-pass $l\in \lbrace 0,\ 1\rbrace$ freshly encodes the $w=168$ anchors starting at ring position $u+l\cdot S$ (stride $S=84$, so consecutive passes overlap by half a window) together with the current batch’s own anchors, forms its own CE partition (every view’s positive sits in the batch block, the window anchors serve as negatives) and backpropagates immediately, freeing the window’s activations before the next pass. The pointer then advances by $2S$, sweeping the full catalog every ten steps so that every anchor takes a regular, gradient-coupled turn, which per step touches about 27% of the gallery with gradient and bounds the gradient-coupled encoding to 22% of the full backward graph (Table A8); the tower, the views, and the invariance term are unchanged. Table A12 quantifies the cost of this variant five-fold.

**Table A8:** What the sliding window bounds. The fully coupled objective re-encodes every training-set anchor pack with gradient at once; swin holds only the batch’s own anchors plus one $w=168$ window per micro-pass and frees it before the next, so its gradient-coupled encoding peaks at 360 anchors, 22% of the full graph. Both variants keep the whole catalog resident, so the saving is bounded by the persistent ring rather than reaching the full 4.5×. Sentence-slot counts are ceilings at the 2,048-sentence budget (realized packs are shorter), quoted on the fixed split of Section 5 (1,613 training-set games); a five-fold gallery of 1,694 or 1,695 scales the same way.

|  | Full-gallery I-CE | swin-I-CE (w = 168) |
| --- | --- | --- |
| **Anchors re-encoded with gradient per backward (peak)** | 1,613 | 360 (192 batch + 168 window) |
| **as a fraction of the anchor gallery** | 100% | 22% |
| **Sentence slots in the backward graph (2,048 budget)** | ≈ 3.3 M | ≈ 0.74 M |
| **Anchor packs held resident (the ring)** | 1,613 (all) | 1,613 (all) |

**The tag reading against the margin.** Pairing the fold-0 margins of Table A17 with the five-fold tag means of Table A1 for the five trained towers (BYOL 0.047 / 0.712, SimCLR 0.106 / 0.707, VICReg 0.108 / 0.708, I-CE 0.176 / 0.685, CE 0.191 / 0.664) gives

$$
\mathrm{tag}\approx 0.735- 0.32\times \mathrm{margin} \qquad (\text{C.1})
$$

with Pearson −0.91 and Spearman −0.90, the one rank inversion being SimCLR against VICReg, 0.002 apart in margin and 0.001 in tag. Pairing the same margins with those towers’ fold-0 tag readings instead gives

$$
\mathrm{tag}\approx 0.762- 0.39\times \mathrm{margin} \qquad (\text{C.2})
$$

with Pearson −0.87 and Spearman −0.70, single-fold tag readings being the noisier side. I-CE sits 0.006 above the five-fold line and CE 0.010 below it; the regression is not printed in the body because five points under two protocols support a direction, not a coefficient.

## APPENDIX D: GRADIENT BOUNDS AND THE HANDOVER POINT

This appendix locates the handover between the contrastive push-pull and the invariance term, the crossing of their gradient envelopes that the empirically chosen $\tau =0.02$ and $\lambda =2$ place at 99% confidence. Both envelopes are gradients with respect to the encodings, read as free variables as in Appendix E’s repulsion steps and taken with the tangential projection of the sphere so that they are commensurable; AdamW’s per-coordinate rescaling is not modeled.

**Unit force.** One view enters $V- 1$ of Equation (3)’s $V(V- 1)/2$ pairs, each pulling by at most one,

$$
\Vert \nabla_{z_{v}}(\lambda I)\Vert \le \lambda \frac{2}{V(V- 1)}(V- 1)=\frac{2\lambda }{V}=1 \qquad (\text{D.1})
$$

with equality when the other views are aligned with each other and orthogonal to $z_{v}$ ; $\lambda =V/2$ makes the bound exactly one, the recipe’s unit of force, flat in confidence though the realized force is not, falling with the residual disagreement.

**The ceiling.** The descent step of Equation (E.9) is $(1- p_{g})/\tau$ times the difference between $a_{g}$ and the weighted centroid of the competitors, both in the unit ball and so at most 2 apart,

$$
\Vert \nabla_{z_{v}}\ell_{\mathrm{CE}}\Vert \le \frac{2(1- p_{g})}{\tau }\le \frac{2}{\tau }=100 \qquad (\text{D.2})
$$

so $\tau$ sets the ceiling, a bound the tangential projection can only tighten, which the brake of Appendix E spends linearly as $p_{g}\to 1$.

**The handover.** Writing ${p}^{\ast }$ for the value of $p_{g}$ at which the braked envelope falls to the invariance envelope, the recipe values $V=4$, $\lambda =2$, $\tau =0.02$ give

$$
\frac{2(1- {p}^{\ast })}{\tau }=\frac{2\lambda }{V},\ 1- {p}^{\ast }=\frac{\lambda \tau }{V}=0.01 \qquad (\text{D.3})
$$

so a view’s envelope hands over once classified at 99% confidence: ceiling times handover margin is unit force, $(2/\tau )(\lambda \tau /V)=2\lambda /V$, and the empirically chosen temperature (Table A3) restates as $\tau =(V/\lambda )(1- {p}^{\ast })=2\times 0.01$, the factor 2 being $V/\lambda$, which equals $\lambda$ here only because $\lambda =\sqrt{V}$. Only the product $\lambda \tau /V$ enters the crossing; $\tau$ was fixed by the sweep of Table A3 and $\lambda$ sits on the plateau of Table A4, so unit force reads the fixed values rather than justifying them. In margin units (Equation (8)), $s_{h}=\langle z_{v},a_{g}\rangle - \langle z_{v},a_{h}\rangle =\tau \ln (p_{g}/p_{h})$ at any gallery size, and the residual mass $\lambda \tau /V$ placed on one competitor, the worst case, gives the gap $s_{h}^{\ast }=\tau \ln (V/(\lambda \tau )- 1)=0.02\ln 99\approx 0.092$. The crossing compares worst-case envelopes attained at incompatible configurations, so it is a parameter-scale statement rather than a measured handover.

## APPENDIX E: THE SEPARATION GRADIENTS OF SECTIONS 4.3 AND 6.3

This appendix works out Equations (2), (3), (6), (10) and (11) over the noise model of Equation (4), with a remark on Equation (12): why sample repulsion aims along draw noise, why anchor repulsion does not, why the invariance term leaves only the semantic difference, and why the coupled softmax brakes its own forces. Each derivation names its convention: Equation (4) is stated for the pooled input while the repulsion steps read $c$ and $\epsilon$ as free components of the encoding [5], and the invariance steps linearize the tower, whose noise response is what that term suppresses.

**The noise model.** Pool an $m$ -sentence pack drawn from game $g$:

$$
x=c_{g}+\epsilon_{m},\ E[\epsilon_{m}]=0,\ \mathrm{Cov}(\epsilon_{m})=\frac{\Sigma_{g}}{m} \qquad (\text{E.1})
$$

with $c_{g}$ the draw-invariant component, content plus the static bias its reviewers share, and $\epsilon_{m}$ that draw’s fluctuation, $\Sigma_{g}$ being the covariance of one sentence embedding of $g$. Packs enter at two scales, $m_{v}$ sentences for a view and $m_{a}$ for an anchor, the document view of Section 4.1 entering not as a third scale, carrying no draw noise of its own, but as the fixed offset $\delta_{g}$ of (E.18),

$$
m_{v}\ge 16,\ 512\le m_{a}\le 2,048,\ m_{a}\gg m_{v},\ \mathrm{tr}\mathrm{Cov}(\epsilon_{v})\gg \mathrm{tr}\mathrm{Cov}(\epsilon_{a}) \qquad (\text{E.2})
$$

a strong view at its 16-sentence floor (23 at the median) against an anchor cap realized packs fall short of. The $1/m$ scaling assumes independent pooled draws, which whole-review sampling and attention pooling violate, so only ratios of the two scales are used. Encoding both, $f(x_{v})=z$ and $f(x_{a})=a$, views scatter and anchors hold still.

Two approximations sit here. Different draw laws and the anchor’s store-page prefix leave $c_{g}$ common only up to a static offset of the kind $\delta_{g}$ below, which averaging does not remove. And a view meets its own anchor pack only by chance, about 14% of a median game’s sentences at $m_{a}=2,048$, so $\epsilon_{v}$ and $\epsilon_{a}$ count as independent.

**Why sample-to-sample repulsion aims along noise.** In the free-variable convention, with one negative view per competing game in the batch, Equation (11) reads

$$
\nabla_{z_{v}}\ell_{\mathrm{SimCLR}}=\frac{1}{\tau }\sum_{h\neq g} q_{h}(z_{h}- z_{v' }) \qquad (\text{E.3})
$$

with the in-batch weight

$$
q_{h}=\frac{\exp (\langle z_{v},z_{h}\rangle /\tau )}{\exp (\langle z_{v},z_{v' }\rangle /\tau )+\sum_{k\neq g} \exp (\langle z_{v},z_{k}\rangle /\tau )} \qquad (\text{E.4})
$$

putting the rest on the positive partner $z_{v' }$, so the full set of weights, positive included, sums to one and the $h=g$ term drops from the gradient sum. The partner is another fresh draw of $g$, so its noise is what enters

$$
z_{h}- z_{v' }=(c_{h}- c_{g})+(\epsilon_{h}- \epsilon_{v' }) \qquad (\text{E.5})
$$

whose second half is unbiased term by term, the two draws being zero-mean by (E.1) and, coming from different games, independent, and has mean square

$$
E\Vert \epsilon_{h}- \epsilon_{v' }{\Vert }^{2}=\frac{\mathrm{tr}\Sigma_{h}+\mathrm{tr}\Sigma_{g}}{m_{v}} \qquad (\text{E.6})
$$

Rank turns on hard pairs, and at a semantic hard pair, defined by $\Vert c_{h}- c_{g}\Vert \ll \sqrt{(\mathrm{tr}\Sigma_{h}+\mathrm{tr}\Sigma_{g})/m_{v}}$ with the noise norm concentrating near its root mean square (the scale carried to the encodings by the linearization of (E.13) below),

$$
\Vert \epsilon_{h}- \epsilon_{v' }\Vert \gg \Vert c_{h}- c_{g}\Vert \qquad (\text{E.7})
$$

so the push aims predominantly along noise. Hard pairs are selected by encoded similarity, and one hard only because two draws collided is corrected by a noise-aligned push. In the large the semantic term dominates, which is why sample-repelled towers spread widely yet carve narrow nearest-neighbor margins.

**Why anchor-mediated repulsion does not.** Still free-variable, Equation (6) pushes a view against the anchor gallery,

$$
\nabla_{z_{v}}\ell_{\mathrm{CE}}=\frac{1}{\tau }\sum_{h\neq g} p_{h}(a_{h}- a_{g}),\ \sum_{h\neq g} p_{h}=1- p_{g} \qquad (\text{E.8})
$$

so the descent step is one pull and $n- 1$ pushes,

$$
{}- \nabla_{z_{v}}\ell_{\mathrm{CE}}=\frac{1}{\tau }[(1- p_{g})a_{g}- \sum_{h\neq g} p_{h}a_{h}] \qquad (\text{E.9})
$$

Write $p_{h|g}$ for the softmax weight of anchor $h$ under a view $z_{g}$ of game $g$. Call $(g,\ h)$ symmetrically confused when $p_{h|g}=p_{g|h}$ and $p_{k|g}=p_{k|h}$ for every other anchor $k$, which by normalization forces $p_{g|g}=p_{h|h}$ ; subtracting the descent steps of the views $z_{g}$ and $z_{h}$, $z_{g}- z_{h}$ moves along

$$
(\frac{1- p_{g|g}}{\tau }+\frac{p_{h|g}}{\tau })(a_{g}- a_{h}) \qquad (\text{E.10})
$$

both coefficients read off (E.9), the matched third-party weights cancelling in the subtraction so that only the $(g,\ h)$ terms survive and the step is parallel to $a_{g}- a_{h}$, which decomposes as

$$
a_{g}- a_{h}=(c_{g}- c_{h})+(\epsilon_{\mathrm{ag}}- \epsilon_{\mathrm{ah}}) \qquad (\text{E.11})
$$

whose second half is, by the variance asymmetry, a static residual and not a fresh draw,

$$
\frac{E\Vert \epsilon_{\mathrm{ag}}- \epsilon_{\mathrm{ah}}{\Vert }^{2}}{E\Vert \epsilon_{h}- \epsilon_{v' }{\Vert }^{2}}=\frac{m_{v}}{m_{a}} \qquad (\text{E.12})
$$

an amplitude ratio $\sqrt{m_{v}/m_{a}}$, about $1/5.7$ at the 16-sentence floor against the 512-sentence anchor budget, and larger for longer views. Read in the encodings through the linearization of (E.13) below, one Jacobian and one sentence-level covariance $\Sigma_{g}$ serving both packs of a game, the ratio is unchanged; it covers the draw-noise half of the anchor residual, the store-page offset of the kind $\delta_{g}$ being static and not shrinking with $m_{a}$. The anchored path has a lower noise floor, tracking the semantic difference far below the scale $\sqrt{(\mathrm{tr}\Sigma_{h}+\mathrm{tr}\Sigma_{g})/m_{v}}$ where sample repulsion loses it. The residual is static within a run, replaced by the ten fresh anchor draws of the evaluation protocol (Section 5), whose effect Tables S2 and S3 of the supplementary document bound at 0.008 Stripped hit@1. Through $p_{h}$ the softmax also concentrates the whole repulsion budget on whichever anchors currently compete. Equation (12) sits at the opposite extreme: its mean gives every competitor weight $1/B$, $B$ the view encodings in a step (Section 6.3), against a $p_{h}$ that is of order one at a hard competitor early in training, so matching one anchored step takes order $B$ mean steps, each carrying an independent draw, and the accumulated noise variance grows by the same factor.

**Why the invariance term suppresses the noise response.** Switching to the linearization, expand about the invariant component, with $J$ the Jacobian at $c$,

$$
f(c+\epsilon )\approx f(c)+J\epsilon ,\ J=\frac{\partial f}{\partial x} \qquad (\text{E.13})
$$

and since the tower outputs unit vectors, differentiating $\Vert f{\Vert }^{2}=1$ gives

$$
f(c{)}^{T}J=0 \qquad (\text{E.14})
$$

which kills the first-order term of the cosine, and the same constraint fixes each output’s radial second-order term at $- \Vert J\epsilon {\Vert }^{2}/2$, so that $\Vert z\Vert =1$ holds to that order, leaving

$$
\langle z_{i},z_{j}\rangle =1+\epsilon_{i}^{T}{J}^{T}J\epsilon_{j}- \frac{\Vert J\epsilon_{i}{\Vert }^{2}+\Vert J\epsilon_{j}{\Vert }^{2}}{2} \qquad (\text{E.15})
$$

and therefore

$$
1- \cos (z_{i},z_{j})=\frac{\Vert J(\epsilon_{i}- \epsilon_{j}){\Vert }^{2}}{2} \qquad (\text{E.16})
$$

For independent draws of equal covariance every pair has expectation $\mathrm{tr}(J\mathrm{Cov}(\epsilon_{v}){J}^{T})$, and the $2/(V(V- 1))$ prefactor of Equation (3) averages the $V(V- 1)/2$ pairs, so

$$
E[I]=\mathrm{tr}(J\mathrm{Cov}(\epsilon_{v}){J}^{T}) \qquad (\text{E.17})
$$

with coefficient one, the case of a game without a document view (198 in a fold). The document view rides on $c_{g}$ with its static register offset $\delta_{g}$, its own fixed deviation folded in,

$$
z_{\mathrm{doc}}\approx f(c_{g})+J\delta_{g} \qquad (\text{E.18})
$$

and, for a game with a document view (1,496 of the 1,694 in a fold), each of the $V- 1$ pairs involving it swaps one view’s noise for that offset, contributing $\Vert J\delta_{g}{\Vert }^{2}/2$ and half the noise term,

$$
E[I]=\frac{V- 1}{V}\mathrm{tr}(J\mathrm{Cov}(\epsilon_{v}){J}^{T})+\frac{\Vert J\delta_{g}{\Vert }^{2}}{V} \qquad (\text{E.19})
$$

a floor of $\Vert J\delta_{g}{\Vert }^{2}/4$ at $V=4$ that no drawing removes, which is why closing $\delta_{g}$ falls here. Being symmetric, the term closes $\delta_{g}$ at a compromise, not by moving the document alone: the game-independent part $b_{\mathrm{doc}}- b_{\mathrm{rev}}$ of $\delta_{g}=(b_{\mathrm{doc}}- b_{\mathrm{rev}})+t_{g}$ leaves inter-game differences untouched, while the drift $t_{g}$ does not and is absorbed into $f(c_{g})$. Minimizing $I$ presses the noise term of (E.19) to zero in expectation,

$$
E\Vert J\epsilon_{v}{\Vert }^{2}\to 0,\ z_{v}- z_{v' }\approx J(\epsilon_{v}- \epsilon_{v' })\to 0 \qquad (\text{E.20})
$$

suppressing the tower’s noise response, not the noise. Two assumptions carry this: the linearization holds in the high-noise regime of the strong views, and $\mathrm{Cov}(\epsilon_{v})$ must span a subspace distinct enough from $c_{g}- c_{h}$ for one to be suppressed without the other. Cross-entropy forbids total collapse, the constant tower costing $\ln n\approx 7.4$ nats per view, but not the loss of semantic directions inside the noise subspace, which we read as the tag deficit of Section 8.

**What δ<sub>g</sub> measures.** Pooling each text in the frozen 1,024-dimensional space the way the mean-pool baseline of Section 6.1 does, a game’s document sits at cosine 0.869 from its own reviews for a store-page description (median over 1,811 games) and 0.861 for a wiki article (814 games), while two different games’ review vectors sit at 0.946: the register crossing is wider than the gap between games, and the frozen row of Table A1 is what that geometry costs. Split as above, the game-independent part carries 0.352 and 0.365 of a median offset of 0.512 and 0.527, so neither half of δ<sub>g</sub> dominates. The two document sources are not one register either: the document view is the wiki article for 814 games and the store-page description for the rest (Section 4.1), and their mean offsets point 0.65 apart in cosine, so the b<sub>doc</sub> above is an average over two registers rather than a constant of the corpus. The LLM rewrite of online Appendix N barely moves it, 0.977 from the raw article’s offset, which is why that arm changes no retrieval column beyond the fold spread.

**What the two terms leave.** The conventions meet here, assuming the tower keeps Jacobian rank along the semantic directions while its noise response is suppressed. With $J_{g}$ and $J_{h}$ the Jacobians at $c_{g}$ and $c_{h}$,

$$
z_{g}- z_{h}=(f(c_{g})- f(c_{h}))+(J_{g}\epsilon_{g}- J_{h}\epsilon_{h}) \qquad (\text{E.21})
$$

whose second half Equation (3) sends to zero while Equation (6) holds the first half apart through the anchors, each view pinned to its own anchor at the CE optimum and the anchors pushed apart by (E.23), so at the optimum

$$
z_{g}- z_{h}=f(c_{g})- f(c_{h}) \qquad (\text{E.22})
$$

up to the drift $t_{g}$ absorbed into $f(c_{g})$ and the static anchor residual of (E.12). Displacement contracts while the margin, an anchor-side quantity made by repulsion, roughly holds, so the displacement-to-margin ratio retrieval depends on falls. The contraction is derived for in-distribution draws; extending it to the zero-shot cross-register queries of Section 5 assumes a query’s displacement lies in the same subspace, the step the document view makes plausible.

**The coupled softmax brake.** Differentiating Equation (2) against the anchors, again free-variable,

$$
\nabla_{a_{g}}\ell_{\mathrm{CE}}=- \frac{1- p_{g}}{\tau }z_{v},\ \nabla_{a_{h}}\ell_{\mathrm{CE}}=\frac{p_{h}}{\tau }z_{v} \qquad (\text{E.23})
$$

so pull and pushes carry the weights of one distribution and vanish together as classification succeeds. Decomposing into alignment and uniformity [5] loses the coupling: with $\ell_{\mathrm{align}}=- \langle z_{v},a_{g}\rangle$, the inner-product form (on the sphere the squared-distance alignment of [5] differs by a factor of two, which the direction claim does not use), at weight $\alpha$, the align-to-uniformity ratio of Appendix H with the uniformity weight normalized to one,

$$
\nabla_{a_{g}}(\alpha \ell_{\mathrm{align}})=- \alpha z_{v} \qquad (\text{E.24})
$$

leaving a constant-weight attraction, $\alpha$ against the $(1- p_{g})/\tau$ of (E.23), that pulls the low-variance anchor toward the high-variance view after the games separate. Extending the invariance term of Equation (3) to the anchor, I′ over the ten pairs among the four strong views and the anchor in place of the six among views, leaves the softmax whole and adds a pull of the same kind,

$$
\nabla_{a_{g}}(\lambda I' )=- \frac{2\lambda }{V(V+1)}\sum_{v=1}^{V} z_{v} \qquad (\text{E.25})
$$

the second half of Equation (10), whose first half is (E.23), at weight $2\lambda /(V(V+1))=0.2$ per view for $\lambda =2,\ V=4$ ; the braked weight of (E.23) falls below it once $1- p_{g}<0.004$. This is the weak-view prediction of Section 1 tested in Section 6.2. That test varies more than the coupling, so it fixes the direction, not the size. Anchors are treated as constants when differentiating with respect to the views, and the views as constants when differentiating with respect to the anchors, and every claim here is a direction or a dominance relation, not a rate.

## APPENDIX F: THE QUERY-REWRITE PROMPTS

Each query is an English description that GPT-5.4-mini at temperature 0.7 writes from the held-out game’s full wiki article, truncated to 12,000 characters, called through an OpenAI-compatible gateway; the instructions below were issued in Chinese and are translated faithfully here, an output shorter than 300 characters re-requested with the elaboration instruction, which asks for at least 300 words and so for well more than the trigger requires, up to three attempts, and the verbatim strings and the retry policy in the released code. The name-stripped instruction asks for invented replacement names, but the model complies literally in only 113 of the 814 queries and generalizes the name away in the other 701, so the register is better read as name-free than as name-substituted.

**ALGORITHM 1:** The rewrite instructions behind the two query registers

```
user message  —  "Game: {title}" then the article text
neutral (Name hit@k)  —  Read the full game page text the user provides and summarize it into a descriptive article about the game. Preserve the real information about the game’s content, gameplay and mechanics in full; do not invent anything that is not on the page. Register: neutral. Write in English and output the article body directly.
name-stripped (Stripped hit@k)  —  Read the full game page text the user provides and summarize it into a descriptive article about the game, in a neutral register. Preserve the real information about the game’s content, gameplay and mechanics in full; do not invent anything that is not on the page. But do not reveal the game’s name, the characters’ names, the items’ names or anything like them — replace them all with imagined, invented words. Write in English and output the article body directly.
elaboration retry (any register)  —  Your previous answer was too short. Please write a considerably MORE DETAILED article (at least 300 words) in the requested style, covering the game’s content, story, world and mechanics from the page text above.
```

## APPENDIX G: REDUCING THE STEAM TAG VOCABULARY

A single mapping file is the source of truth for every tag number in this paper: each of the 202 fine tags is assigned either to one coarse class or to the sentinel “discarded”. A class is present on a game if any of the Steam tags on its right-hand side is, and the vote weights Steam attaches are used only to resolve which source is strongest, never as a target, since the probe predicts presence. On our 2,020 games this yields a $2,020\times 23$ binary matrix of density 0.202, 4.6 classes per game at the mean, 5 at the median, and between 35 and 1,395 games per class.

**ALGORITHM 2:** The 23 TAG classes, each written as the original Steam tags it absorbs

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

## APPENDIX H: THE BASELINE AND VARIANT RECIPES

Every trained baseline shares the full scaffolding of Section 4: the same 4-query cross-attention tower, the same four views per game per step (three review views and one document view where a document exists), the same corpus, batch (192 games), optimizer (AdamW, lr 5e-4, weight decay 1e-4), 2,000-epoch budget, temperature $\tau =0.02$ where a softmax exists, and the validation-only selection of Section 5. Table A9 gives each arm’s departure from that scaffolding.

**Table A9:** The baseline and variant recipes. The first five rows are the Figure 7 baselines and the sixth the DINO-style control of Table A1; apart from the untrained frozen-embedder row, each keeps the tower and the training recipe and changes the objective together with the anchor supply and teacher that objective implies. The last four change the objective and, where noted, the schedule or the anchor supply.

| Arm | Objective | Anchor supply | Schedule or teacher | Notes |
| --- | --- | --- | --- | --- |
| Frozen embedder (mean pool) | None. A game is the L2-normalized mean of its anchor-pack sentence embeddings, a query the L2-normalized mean of its own sentences | Anchor packs read at evaluation only | No tower and no training | Measures the raw geometry every other row starts from |
| CE (contrast only) | Equation (2) alone; the invariance term of Equation (1) is removed | Full anchor gallery, re-encoded with gradient at every step | Fully coupled teacher | Isolates what the softmax supplies without multi-view agreement |
| SimCLR-style (in-batch views) | NT-Xent (normalized temperature-scaled cross-entropy) [4] at $\tau =0.02$ over the step’s $4\times 192$ view encodings: view $v$ of a game takes view $v+1(\mathrm{mod}4)$ of the same game as its positive, the game’s other own views are masked as false negatives, every other game’s views are negatives | No anchors anywhere in training; the gallery is encoded once at evaluation time by the finished tower | Tower unchanged; no anchor teacher | The contrast lives in view space instead of anchor space |
| VICReg (epd 20/10/20) | VICReg [1] on expander-MLP outputs over the six view pairs, variance/invariance/covariance weights 20/10/20; no negatives | Anchors encoded at evaluation time only | Views drawn for 192 games per step, the variance term estimating per-dimension spread within the step’s view batch | Table A7 is the weight sweep |
| BYOL | $1- \cos$ between the predictor’s output for one view and the target tower’s encoding of another, averaged over ordered view pairs; the document view participates like any other | No gallery in training | Online tower plus a two-layer predictor head, and a target tower that is an exponential moving average of the online weights (momentum 0.996) behind a stop-gradient | Collapse is prevented by the predictor/EMA asymmetry rather than by negatives |
| DINO-style (Table A1, fold 0) | DINO [24] as published: the cross-entropy from a teacher’s centered and sharpened softmax over K = 65,536 prototypes to the student’s softmax, over every pair of a teacher view and another view; student temperature 0.1, teacher 0.04 rising to 0.07 over the first tenth of training; no negatives, no anchor gallery and no invariance term | No anchors in training; the teacher’s two views hold up to 512 sentences each: both open with the game’s store page, and each review of its 2,048-sentence review pool, taken whole and in random order, enters the first view if it still fits, otherwise the second if it fits there; re-dealt every step; anchors encoded at evaluation time by the teacher | Teacher an exponential moving average of the student tower and a three-layer projection head (2,048 hidden units, 256-d bottleneck, weight-normalized last layer), momentum 0.996 rising to 1, centering momentum 0.9, DINO’s learning-rate and weight-decay (0.04 to 0.4) schedules; the student sees the teacher’s two views and four more, three review views drawn from the full review library and one document view | DINO’s published settings, none swept; retrieval reads the teacher’s 128-d tower output; trained on fold 0 only |
| CE + anchor-in-I (Table 3) | Equation (2) unchanged, but the invariance term of Equation (1) ranges over the ten pairs among the four strong views and the game’s own anchor vector instead of the six pairs among views alone | Full anchor gallery, re-encoded with gradient at every step | Fully coupled teacher | The control of Section 6.2: constant-weight attraction on the anchor with the softmax left in place |
| Two-stage: BYOL warm start → windowed (Table A12) | Stage one is the BYOL row above; stage two is the windowed teacher of Appendix C | Windowed, $w=168$ | Stage one runs to its own selection point; stage two re-initializes the tower from those weights and trains 600 epochs, 30% of the from-scratch discriminative budget | Selection happens inside stage two |
| I-CE (EMA + memory bank; Table A12) | Cross-entropy against the ring, the view’s freshly written key the positive and stale same-game keys masked; the invariance term is unchanged | An EMA shadow tower encodes the current batch’s anchors into a 3,072-key memory bank | EMA shadow tower, momentum 0.99, replacing the fully coupled teacher | Gradient reaches the strong-view side only; the 3,072 keys are negatives in number but inert in the gradient, the sense in which this tower counts as negative-light in Section 6.1 |
| Align + uniformity (Table 3) | The softmax of Equation (2) replaced by an alignment term plus a kernel-uniformity term, over a grid of align-to-uniformity ratios and kernel temperatures of 2 and 25 | Repulsion aimed at the anchor gallery or at the batch views | Fully coupled teacher | The decomposition sweep of Section 6.2; Table 3 prints two representative arms, aimed at the gallery at a ratio of 12.5 : 1 with the invariance term over the strong views and the anchor, and aimed at the views at 2 : 1 with it over the strong views alone, and the geometry files in this repository carry the sweep |

## APPENDIX I: COST AT SCALE

Write $R$ for the number of resident anchor packs (the fold gallery of Section 4.1, so $R=n=1,694$ here; the whole catalog, and so $R=N$, at deployment scale), $m_{a}$ for the anchor-pack cap in sentences (the budget of Section 4.1), $d=1,024$ for the embedding width, $G=192$ for the games in a step, and $w=168$, $S=84$, $T=2$ for the window, stride and micro-passes of the teacher in Appendix C. The tower reads one pack with $Q=4$ latent queries, so encoding a pack costs $\Theta (m_{a}\cdot Q\cdot d)$ work and holds $\Theta (m_{a}\cdot d)$ activation.

The fully coupled teacher re-encodes every pack with gradient at every step, so its step time and its peak activation are

$$
\Theta (R\cdot m_{a}\cdot Q\cdot d),\ \Theta (R\cdot m_{a}\cdot d) \qquad (\text{I.1})
$$

both linear in the catalog. At our scale, 1,694 packs of 2,048 sentences, the backward graph holds 3.5 M sentence slots, 6.6 GiB of fp16 input before any activation is stored (Table A8 counts 3.3 M for the fixed split’s 1,613 packs); the same expression at $R={10}^{6}$ asks for 3.8 TiB, which no accelerator holds. This is the wall the scale limitation of Section 8 names.

The measured cost matches (Table A10). Peak device allocation rises by 4.2× across a 4× rise in $m_{a}$ and is dominated throughout by the with-gradient re-encoding of the gallery rather than by the anchor store, which accounts for only 1.7 to 6.6 GiB of it across the three budgets. Allocation is a property of the configuration rather than of the accelerator, so it transfers: all three budgets, the recipe’s 2,048 sentences included, fit a 48 GiB card. Inference is unaffected by the budget: a query costs one forward pass through the frozen 0.6B embedder (< 3 GiB) and a dot product against the finished anchors, an inner-product search that runs in milliseconds.

**Table A10:** Measured cost of the fully coupled teacher at each anchor budget (peak is the fold-0 maximum device allocation over a training step, per process and unaffected by packing). The recipe’s budget is 2,048 sentences. The timed runs shared their cards, several folds packed onto one, so their wall clock bounds a dedicated run from above: every timed run at 512 and at 1,024 sentences finished inside four hours, and the sixteen at 2,048 (the recipe once, the temperature variants of Table A3 fifteen times) in 4.6 to 6.3 hours, the recipe’s own run the slowest.

| Anchor budget $m_{a}$ | Peak device allocation | Wall clock, 2,000 epochs |
| --- | --- | --- |
| 512 sentences | 8.4 GiB | < 4 h |
| 1,024 sentences | 22.3 GiB | < 4 h |
| **2,048 sentences** | **35.0 GiB** | **≤ 6.3 h (4.6–6.3, 16 runs)** |

At 1,024 sentences, the plateau of Table A2, every retrieval and tag reading stays within 0.02 of the recipe at a peak 1.6× smaller, which fits the 24 GB of an RTX 3090 or 4090; the 22.3 GiB leaves little headroom, so such a card should be running headless.

The windowed teacher removes $R$ from both expressions. A micro-pass encodes the batch’s own packs plus one window and frees them before the next, so a step costs

$$
\Theta (T\cdot (G+w)\cdot m_{a}\cdot Q\cdot d),\ \Theta ((G+w)\cdot m_{a}\cdot d) \qquad (\text{I.2})
$$

in time and in memory (Table A11). At $G=192$ and $w=168$ that is 0.74 M sentence slots, 1.4 GiB of input at the 2,048-sentence budget, whether the catalog holds two thousand entries or ten million. What the window does not bound is the ring itself: the windowed teacher still keeps every anchor pack resident (Table A8), so $\Theta (R\cdot m_{a}\cdot d)$ of store sits on the device even though only $\Theta ((G+w)\cdot m_{a}\cdot d)$ of it is ever differentiated. Measured at our scale the fully coupled teacher peaks at 35 GiB at $m_{a}=2,048$, and about 6.6 GiB of either teacher’s footprint is that resident ring; the windowed teacher’s peak was not separately measured.

**Table A11:** The cost model of the three teachers, with the fully coupled I-CE teacher normalized to 1 at the same catalog size. $R$ is the resident-pack count, $m_{a}$ the anchor-pack cap, $d$ the embedding width, $Q=4$ latent queries, $G=192$ games per step, and $w=168$, $S=84$, $T=2$ the window, stride and micro-passes. Ratios are quoted at our $R=1,694$ ; the windowed teacher’s step contains no $R$, so its first two quantities fall as $1/R$, but its ring keeps every pack resident and therefore stays linear in the catalog on the device. The streamed store, the full-corpus pipeline of this paper, is what removes that last term, paying I/O for it. The measured peak is at $m_{a}=2,048$ (fold 0, torch.cuda.max_memory_allocated); the last row extrapolates each variant’s device budget for $R$ on one 80 GiB accelerator, assuming the linear dependence on $R$.

| Quantity | I-CE, fully coupled | swin-I-CE (w = 168) | swin + streamed store |
| --- | --- | --- | --- |
| Step time | $\Theta (R\cdot m_{a}\cdot Q\cdot d)$ | $\Theta (T\cdot (G+w)\cdot m_{a}\cdot Q\cdot d)$ | $\Theta (T\cdot (G+w)\cdot m_{a}\cdot Q\cdot d)$ |
| ratio to I-CE | 1 | 0.43 | 0.43 |
| Peak activation | $\Theta (R\cdot m_{a}\cdot d)$ | $\Theta ((G+w)\cdot m_{a}\cdot d)$ | $\Theta ((G+w)\cdot m_{a}\cdot d)$ |
| ratio to I-CE | 1 | 0.21 | 0.21 |
| Resident anchor store | $\Theta (R\cdot m_{a}\cdot d)$ | $\Theta (R\cdot m_{a}\cdot d)$ | $\Theta ((G+w)\cdot m_{a}\cdot d)$ |
| ratio to I-CE | 1 | 1 | 0.21 |
| Device memory in R | linear | linear | constant |
| Per-step device I/O | none | none | $\Theta (T\cdot S\cdot m_{a}\cdot d)$ ≈ 0.66 GiB |
| Measured peak at R = 1,694 | ≈ 35 GiB | not measured | not measured |
| Extrapolated largest R, one 80 GiB device | ≈ 3,900 | not measured | not memory-bound |

The streamed store closes that last term by giving up residency, and it is the pipeline that built the corpus of this paper. The merged pool is a single memory-mapped fp16 array on disk: the build writes and reads it a million rows at a time, so that assembling the 23,373-game merged pool (from which the 500-review floor of Section 2.2 keeps the 2,020 games of this paper) never holds more than a chunk, and the training loop gathers each anchor pack from that map instead of from a resident ring. The device then holds only what it differentiates, $\Theta ((G+w)\cdot m_{a}\cdot d)$, and pays I/O in place of space: the ring advances $T\cdot S=168$ packs per step and only the new ones must be paged, $\Theta (T\cdot S\cdot m_{a}\cdot d)$, 672 MiB per step at $m_{a}=2,048$ and 336 MiB at $m_{a}=1,024$, which a prefetch thread hides behind the forward pass. On one 80 GiB accelerator the fully coupled teacher extrapolates from its measured 35 GiB at $R=1,694$ to a ceiling near $R\approx 3,900$, the windowed teacher’s ring still grows linearly in the catalog, and the streamed variant alone has no such bound; the 23,373-game corpus needs 91 GiB of packs at $m_{a}=2,048$, which no single device holds, but 46 GiB at $m_{a}=1,024$, which one does.

**The sharded window.** The deployable form of the streamed variant splits the ring into $\kappa$ shards, each sweeping its own window of $w$ packs per micro-pass; the CE partition of a view is the union of the batch’s own anchors and the $\kappa$ windows, and each window encode is gradient-checkpointed so that only its $w$ output vectors stay live. A step then costs

$$
\Theta (T\cdot (G+\kappa \cdot w)\cdot m_{a}\cdot Q\cdot d) \qquad (\text{I.3})
$$

in time while device memory stays at one window, $\Theta ((G+w)\cdot m_{a}\cdot d)$, constant in the catalog; a single process with $\kappa$ shards computes exactly what $\kappa$ workers would after an all-reduce of their window logits, and $\kappa =1$ is the windowed teacher above. The released code carries it as the sharded option of the five-fold worker on a swin arm, with the memory-mapped pool as the store, and main_model/pfc_sample.py walks through the supply on synthetic data.

Two quantities do still grow with the catalog, and neither is memory. The ring advances 168 packs per step, so an anchor takes a gradient turn once every $R/(T\cdot S)$ steps: ten steps at our $R=1,694$, 139 at 23,373, and 5,952, about 370 epochs of sixteen steps, at a million; equivalently, peak concurrent coverage is $(G+w)/R$, falling from 21% here to 1.5% at full-corpus scale and 0.04% at a million. A million-entity store is 3.8 TiB at $m_{a}=2,048$ or 1.9 TiB at $m_{a}=1,024$, an ordinary disk array read at about a gigabyte per step. Our evidence spans 21% to 100% peak concurrent coverage, the range in which the windowed teacher gives up 0.032 Stripped hit@1 against full coupling (Table A12); it does not reach the 0.04% regime and we do not claim it.

**Table A12:** The cost of the anchor-supply economies (test-set queries (ts), five-fold mean ± std; trained and scored as in Table A1). The sliding fresh-window variant (swin, Appendix C) re-encodes ~26% of the five-fold gallery with gradient per step (27% on the fixed split of Appendix C); the two-stage row hands a BYOL warm start to the windowed teacher for the last 600 epochs; the EMA + memory-bank variant replaces the coupled teacher with an EMA shadow encoder feeding a 3,072-key bank (Appendix H).

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| I-CE, fully coupled teacher | 0.954 ± 0.013 | 0.997 ± 0.004 | 0.741 ± 0.018 | 0.928 ± 0.020 | 0.685 ± 0.019 |
| swin-I-CE (~26% gradient window) | 0.934 ± 0.014 | 0.994 ± 0.004 | 0.709 ± 0.027 | 0.922 ± 0.010 | 0.691 ± 0.014 |
| Two-stage: BYOL warm start → windowed, 600 ep | 0.930 ± 0.011 | 0.991 ± 0.008 | 0.709 ± 0.019 | 0.917 ± 0.005 | 0.705 ± 0.022 |
| I-CE (EMA + memory bank) | 0.936 ± 0.010 | 0.991 ± 0.007 | 0.636 ± 0.016 | 0.881 ± 0.016 | 0.711 ± 0.016 |

## APPENDIX J: PER-FOLD DETAIL ACROSS THE ANCHOR BUDGETS

Table A13 gives the per-fold results behind the CE and I-CE rows of Table A1, the seed-paired towers of Figure 7, under the protocol of Section 5 (retrieval among all 2,020 games).

**Table A13:** Per-fold detail of the Figure 7 towers (trained at the 2,048-sentence anchor budget, scored against 4,096-sentence anchors), test-set queries (ts). I-CE beats CE in five of five paired folds on both query registers at this budget. Means and standard deviations are computed before rounding.

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

The 512-sentence rows were retrained on anchor packs whose store-page prefix is standardized per row, as every other input is; the packs first used for them kept that prefix unstandardized, which depressed every 512-sentence row. Table A14 gives, fold by fold, the two numbers the body quotes across budgets. Every value is a ten-draw mean, the I-CE rows those behind Table A2 and the last pair of rows also those behind Tables A1 and A13; the two objectives are seed-paired on identical splits, so each column is a paired comparison. The CE + anchor-in-I rows extend the invariance term to the game’s anchor (Section 6.2) under the folds, seeds, sampler and schedule of the 512-sentence rows; at the selected checkpoints, over the ten 4,096-sentence anchor draws, their nearest-neighbor margin is 0.160 against I-CE’s 0.164, narrower in every fold.

**Table A14:** CE versus I-CE per fold at every anchor budget (test-set queries, ten anchor draws per fold, mean ± std over the five folds). Upper block: Stripped hit@1, which I-CE wins in all 15 paired folds. Lower block: test-set tag F1, which I-CE wins in 12 of 15, the exceptions being folds 2 and 3 at 512 sentences and fold 1 at 2,048 sentences.

| Stripped hit@1 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
| --- | --- | --- | --- | --- | --- | --- |
| CE @512 | 0.664 | 0.698 | 0.669 | 0.664 | 0.686 | 0.676 ± 0.013 |
| I-CE @512 | 0.725 | 0.715 | 0.715 | 0.724 | 0.698 | 0.715 ± 0.010 |
| CE + anchor-in-I @512 | 0.715 | 0.726 | 0.750 | 0.706 | 0.727 | 0.725 ± 0.015 |
| CE @1,024 | 0.676 | 0.656 | 0.667 | 0.669 | 0.675 | 0.668 ± 0.007 |
| I-CE @1,024 | 0.756 | 0.733 | 0.729 | 0.733 | 0.738 | 0.738 ± 0.009 |
| CE @2,048 | 0.687 | 0.693 | 0.667 | 0.656 | 0.662 | 0.673 ± 0.014 |
| I-CE @2,048 | 0.728 | 0.745 | 0.715 | 0.749 | 0.769 | 0.741 ± 0.018 |

| Test-set tag F1 | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean ± std |
| --- | --- | --- | --- | --- | --- | --- |
| CE @512 | 0.692 | 0.624 | 0.668 | 0.690 | 0.661 | 0.667 ± 0.025 |
| I-CE @512 | 0.700 | 0.679 | 0.665 | 0.681 | 0.701 | 0.685 ± 0.014 |
| CE + anchor-in-I @512 | 0.710 | 0.676 | 0.680 | 0.704 | 0.715 | 0.697 ± 0.016 |
| CE @1,024 | 0.677 | 0.644 | 0.659 | 0.639 | 0.691 | 0.662 ± 0.020 |
| I-CE @1,024 | 0.712 | 0.662 | 0.668 | 0.701 | 0.708 | 0.690 ± 0.021 |
| CE @2,048 | 0.681 | 0.667 | 0.649 | 0.681 | 0.642 | 0.664 ± 0.016 |
| I-CE @2,048 | 0.689 | 0.665 | 0.662 | 0.697 | 0.709 | 0.685 ± 0.019 |

## APPENDIX K: THE TAG DEFICIT AS A REGISTER CROSSING

The tag probe of Section 5 is fit in one register and read in another, so its ranking of the towers mixes tag content with register transfer. Table A15 separates the two on the shipped rank-selected checkpoints, all five folds of the Table A1 towers, the only setting in which every tower exists (BYOL, the SimCLR-style arm and VICReg have no selected checkpoints at 512). The cross-register column is the shipped protocol unchanged: for each of the ten anchor draws behind Table A1, a ridge readout is fit on the 1,694 or 1,695 training-pool anchors (review register), its threshold is picked on validation-game anchors, and it is scored on the name-intact wiki-rewrite query vectors of the test games (document register), the ten micro-F1 values then averaged; recomputed from the shipped checkpoints, this column returns every tag entry of Table A1 exactly. The in-domain column changes one thing: the same ridge with the same threshold is scored on the test-game anchors of the same draw, so the two readings differ only in the register of the vectors read. Transfer loss is in-domain minus cross-register. The register offset is the norm of the mean anchor-to-query displacement over the training games of the fold that carry a wiki query (about 488 per fold), measured in the tower’s 128-d output space.

**Table A15:** Where the tag deficit lives (the five folds of Table A1, shipped rank-selected checkpoints, ten anchor draws, mean ± std over folds). The cross-register column is the shipped readout of Section 5 and reproduces the tag column of Table A1; the in-domain column reuses that ridge and its threshold on test-game anchors. Rows are ordered by transfer loss. The margin column repeats Table A17 (fold-0, single draw). The frozen embedder is omitted: its readout runs in the 1,024-d input space and is not comparable to the 128-d towers.

| Objective | NN margin (Table A17) | Cross-register tag F1 | In-domain tag F1 | Transfer loss | Register offset |
| --- | --- | --- | --- | --- | --- |
| CE (contrast only) | 0.191 | 0.664 ± 0.016 | 0.758 ± 0.024 | 0.094 | 0.297 |
| I-CE (ours) | 0.176 | 0.685 ± 0.019 | 0.763 ± 0.019 | 0.078 | 0.217 |
| SimCLR-style (in-batch views) | 0.106 | 0.707 ± 0.019 | 0.753 ± 0.015 | 0.046 | 0.075 |
| VICReg (v/i/c = 20/10/20) | 0.108 | 0.708 ± 0.022 | 0.743 ± 0.016 | 0.034 | 0.141 |
| BYOL | 0.047 | 0.712 ± 0.017 | 0.734 ± 0.016 | 0.022 | 0.045 |

Table A16 gives the paired differences behind the body’s claims: the cross-register deficit holds against all four negative-light towers in every fold, and the in-domain reversal holds against the three towers of Figure 7 and not against CE, which the body states accordingly.

**Table A16:** Paired differences across the five folds, I-CE minus the named tower, in the towers of Table A1: the cross-register readout of Section 5 and the in-domain readout of Table A15, with the paired $t$ (four degrees of freedom) and the folds I-CE wins. The EMA + memory-bank tower of Table A12 has the cross-register reading only; its in-domain readout was not measured.

| I-CE minus | Cross-register Δ | t | Folds won | In-domain Δ | t | Folds won |
| --- | --- | --- | --- | --- | --- | --- |
| BYOL | −0.028 | −3.74 | 0 of 5 | +0.029 | 9.41 | 5 of 5 |
| VICReg | −0.024 | −3.56 | 0 of 5 | +0.020 | 6.93 | 5 of 5 |
| SimCLR-style | −0.022 | −4.05 | 0 of 5 | +0.010 | 2.81 | 4 of 5 |
| EMA + memory bank | −0.027 | −13.44 | 0 of 5 | not measured | n/a | n/a |
| CE | +0.021 | 1.68 | 4 of 5 | +0.004 | 0.90 | 4 of 5 |

**Why identity pressure costs transfer.** Every anchor pack is a store-page prefix followed by reviews (Section 4.1), so the gallery an anchored objective repels against is review register in the main, and a game’s document view has to leave its own anchor’s neighborhood far enough to be told apart from the other 1,693 or 1,694 review anchors, most cheaply in a direction no review anchor occupies, orthogonal to the review manifold. The invariance term is the only force pulling that view back onto its anchor and it closes the gap only part of the way: I-CE reads tags in domain level with CE and above the three negative-light towers yet pays the second-largest transfer loss, while CE, with no invariance term, pays the largest loss and carries the largest offset. BYOL and VICReg never push a view away from a review anchor, so their two registers stay close and they lose least in transfer, which is how they win the shipped tag column while reading tags worst in domain; the SimCLR-style arm sits between the families, its in-batch negatives drawn from both registers, so its repulsion does not single out the document direction.

**What this does not license.** Subtracting the training-estimated offset from every test query lifts I-CE’s cross-register reading from 0.685 to 0.700 but costs it retrieval (Stripped hit@1 0.741 to 0.728), and for an affine readout a constant shift only moves each tag’s threshold: a recalibration, not a repair of the geometry. Letting the anchor pack see the strong views’ own sentences does not help either (the vfa arm of Supplement A). If a correction exists it belongs on the repulsive side, admitting document-register negatives into the gallery, which we have not tested.

## APPENDIX L: THE NEIGHBORHOOD FIGURES AND THE FOUR-ENCODER MAP

**The four-encoder map.** Figure A2 maps four of the six encoders of Table A17 on fold 0 of the five-fold split: one joint Uniform Manifold Approximation and Projection (UMAP) per encoder over its 2,020 anchors and the 814 name-stripped queries, the frozen embedder, I-CE, VICReg and BYOL side by side. Section 6.1 cites it for the frozen embedder’s clearance.

> **Image — `assets/paper/fig5_umap.png`** (6.00 in wide, 2.95 in tall, docx name "UMAPComparison")
>
> **Alt text.** A 4-by-2 grid of UMAP scatter plots sharing one legend of six tags, listed top to bottom: Card Game, Sports/Rhythm, Racing/Driving, Fighting/Melee, Platformer, Puzzle. Columns are encoders: Frozen embedder (mean pool), I-CE @4096 fold 0 (ours), VICReg (epd 20/10/20), BYOL. The top row shows the game anchors, the bottom row the name-stripped queries, with a line drawn from each of 180 sampled queries to its own anchor (no lines in the frozen panel), so longer lines mean worse placement. Under the frozen embedder the anchors form one diffuse cloud with the tag classes interleaved, and its queries collapse into a single narrow vertical strip far from the anchors: 0% inside own clearance. I-CE spreads the anchors into visibly separated tag-class groups and lands its queries on top of them, with mostly short lines: 41% inside own clearance. VICReg reaches 23% and BYOL 14%, both with many long lines crossing the map.

**Figure A2:** Uniform Manifold Approximation and Projection (UMAP) visualizations of game anchors (top) and name-stripped queries (bottom) across four encoders: the frozen embedder, our I-CE tower of Figure 7, VICReg and BYOL. Color marks the six tags in the legend (319 games); lines join a fixed random 180 of those queries to their own anchors, omitted in the frozen column for legibility. Percentages count all 814 fold-0 queries, train, validation and test alike, landing inside their own game’s clearance, against one draw of 4,096-sentence anchors (that of Table A17).

**The encoders of Figures 8 and 9 and of the neighborhood statistics.** Figure 8 uses the five-fold towers and the ten anchor draws of Figure 7 and Table A1 (frozen embedder, VICReg, the SimCLR-style tower, CE and I-CE) and adds I + SimCLR, the SimCLR-style tower with I-CE’s invariance term, trained for this figure under the SimCLR-style five-fold protocol (the same view sampler over the full review pool, 2,000 epochs per fold, a checkpoint every 50). Its checkpoints are chosen by the validation rule of Section 5 on the gallery the other five-fold runs select on; its test-set Stripped hit@1 over the five folds is 0.623 ± 0.019, against 0.632 ± 0.029 for the SimCLR-style tower. Shrinking the anchors costs every tower little: read at their Table A1 checkpoints against one draw of 512-sentence anchors, the SimCLR-style tower returns 0.622 ± 0.024 and VICReg 0.420 ± 0.020, a loss of 0.01 and 0.03 from their 4,096-sentence readings in Table A1, and CE and I-CE trained at that budget return 0.638 ± 0.027 and 0.703 ± 0.034, a loss of 0.04 and 0.01 from theirs in Table A14; so CE’s lead over the SimCLR-style tower is 0.016 at 512 sentences and 0.041 at 4,096. Figure 9, Figure A3 and the 201-neighborhood statistics below use the fixed-split towers, each an instance of the Section 4.2 tower on the fixed 204/203 split, read like the five-fold towers against the ten draws of 4,096-sentence anchors, a query’s score being the share of the ten in which it retrieves its own game. CE and I-CE are the fixed-split runs trained at the 2,048-sentence budget, their checkpoints chosen by the validation rule of Section 5 over their 2,000 epochs (epochs 1,150 and 1,750). The SimCLR-style tower (in-batch view-against-view softmax, the recipe of Appendix H), the same tower with I-CE’s invariance term added (I + SimCLR), VICReg (the arm of Table A7) and a+u @ views (align + uniformity at the views, the arm of Table 3) never read an anchor in training; each was trained on this split for 2,000 epochs, a checkpoint every 50, and its checkpoint is chosen by the same rule on a 4,096-sentence anchor pack of this split (epochs 1,350, 1,350, 100 and 50). On this split the six drawn encoders reach 0.229 (frozen embedder), 0.446 (VICReg), 0.640 (SimCLR-style), 0.634 (I + SimCLR), 0.675 (CE) and 0.743 (I-CE) test-set Stripped hit@1 (five-fold: 0.209, 0.451, 0.632, 0.623, 0.673 and 0.741), with a standard deviation over the ten draws of at most 0.017, and the SimCLR-style tower’s test curve, read against that pack, shows no sustained rise after epoch 1,350 (0.59 to 0.65). The frozen embedder follows the definition of Section 6.1 (the first row of Table A17): a game is the L2-normalized mean of its anchor-pack sentence embeddings, a query the L2-normalized mean of its own sentences, every sentence row-standardized as everywhere else.

**The neighborhoods and how one was chosen.** Every test-set game with a wiki article is a seed; its neighborhood is the seed plus its seven nearest neighbors, by the cosine of CE’s anchors averaged over the ten draws, among the held-out (validation and test) wiki games, chosen without regard to genre, 201 distinct neighborhoods in all. Each game contributes three queries, the wiki article as original text, the LLM rewrite with names kept and the rewrite with names stripped; a query’s score under a tower is the share of the ten anchor draws in which its nearest of the 2,020 anchors is its own game, a neighborhood’s accuracy is the sum of its 24 scores, and its name-stripped accuracy that of the eight stripped queries alone, the measure of Figures 8 and A3. A neighborhood’s genre is its seed’s, the rarest of the 23 reduced tags of Appendix G the seed carries, no tag for the 10 seeds carrying none, 21 rows in all. Over all 201 neighborhoods CE and the SimCLR-style tower are nearly level (the SimCLR-style tower ahead by 0.2 queries on average; CE ahead in 43% of neighborhoods, behind in 54%); CE leads VICReg by 2.9 (ahead in 90%; 89% on the name-stripped eight), the SimCLR-style tower leads a+u @ views by 8.2 (ahead in 200 of the 201 neighborhoods; 96%) and I + SimCLR leads it by 8.0 (200 of 201; 96%); the invariance term adds 1.2 queries to CE (ahead in 77%, tied in 4%, behind in 19%, the median gap 0.9 queries) and takes 0.2 from the SimCLR-style tower (ahead in 36%, behind in 60%); I-CE leads the SimCLR-style tower by 1.0 (ahead in 68%, tied in 4%, behind in 27%). CE’s lead over VICReg is 2.6 queries in the least confusable quarter and 3.0 in the most confusable (Spearman 0.07 over overlapping neighborhoods, a descriptive figure); I-CE’s lead over CE is 0.8 in the least confusable quarter, 1.1 in the next and 1.4 in each of the two most confusable. CE trails the SimCLR-style tower in 109 of the 201 neighborhoods, 32 of them in the most confusable quarter; in 100 of the 109 I-CE is ahead of CE and in 54 ahead of the SimCLR-style tower, whereas I + SimCLR is ahead of the SimCLR-style tower in 32; that count conditions on CE’s outcome and the neighborhoods overlap, so it describes the mechanism rather than testing it. The neighborhood of Figure 9 is chosen by a rule that reads all six towers alike and no tower’s gain. The box in Figures 8 and A3 marks the genre with at least five neighborhoods whose profile, its six encoder means of name-stripped accuracy, lies nearest the profile of all neighborhoods: Stealth/Immersive over the five-fold neighborhoods (Euclidean distance 2.9 points; Racing/Driving next at 5.4, Platformer at 6.7) and on the fixed split of Figure A3 as well (4.5; Crime/Mystery next at 8.0). The genre on which the six encoders disagree most, Card Game (standard deviation of the six means 23.5 points), lies 28.6 points from that profile, farther than every genre but Puzzle, so the box takes the typical genre rather than the extreme one. Figure 9 opens the fixed-split Stealth/Immersive neighborhood that is most like its genre: of the genre’s 17 neighborhoods, the one whose six-encoder profile over all 24 queries, the points a panel draws, lies nearest the genre’s mean profile (Euclidean distance 11.49 points, Tainted Grail: The Fall of Avalon second at 15.47, Still Wakes the Deep third at 19.0), Abiotic Factor and its seven nearest held-out games (Wizard with a Gun, Nightingale, The Forever Winter, Satisfactory, The Riftbreaker, DYSMANTLE and Grounded); the rule was fixed before the ten draws were read, and it was applied unchanged when the anchor-free towers were extended to 2,000 epochs and re-selected, which moved the pick from the Tainted Grail neighborhood to this one. Each of its six mean hit counts lies within 0.8 standard deviations of the genre’s mean, but the difference between two counts is a coarser thing: the invariance term’s gain over CE is 1.2 queries there against a mean of 1.7 over the genre’s 17 neighborhoods (ahead in 16, behind in one; larger than there in eight) and 1.2 over the 201, and CE trails the SimCLR-style tower by 0.6 queries against 0.2 on average over the 201 (it trails by more in seven of the 17 and in 85 of the 201); the panel illustrates, the tallies above are the evidence. The five-fold neighborhoods of Figure 8 are built the same way inside each fold: every game the fold holds out for test is a seed, its seven nearest held-out games of that fold (test or validation) by that fold’s CE anchor cosine, the ten draws averaged, are its neighbors, and each game contributes the two LLM rewrites, 16 queries per neighborhood, the wiki original not being among the five-fold queries; a query’s score is its hit rate over the ten draws; 814 neighborhoods in all, each wiki game a seed exactly once.

**What Figure 8 shows.** The rows run from a mean pairwise CE anchor cosine of 0.63 (Sports/Rhythm) down to 0.51 (Simulation/Management); only Narrative/Choices has fewer than five neighborhoods (two). The a+u @ views arm has no five-fold run and BYOL (0.284 ± 0.017) would sit between the frozen embedder and VICReg in every row, so neither is drawn. No two bars coincide and none is at zero. In every row I-CE exceeds the other five encoders. CE exceeds the frozen embedder and VICReg in every row, the SimCLR-style tower in 18 (the exceptions are Bullet Hell, Card Game and Narrative/Choices) and I + SimCLR in 16; I + SimCLR leads the SimCLR-style tower in 11 rows and trails it in 10. Per neighborhood, I-CE is ahead of CE on the name-stripped eight in 64%, tied in 7% and behind in 29%, 0.53 queries of 8 on average, and ahead in every fold (0.34 to 0.79); I + SimCLR and the SimCLR-style tower are level (ahead in 47%, behind in 44%, a mean difference of 0.00), I + SimCLR ahead in two folds of five. CE leads the SimCLR-style tower in 56% of neighborhoods and VICReg in 90% over the 16 queries (58% and 85% on the name-stripped eight).

> **Image — `assets/paper/figA2_genres_fixed.png`** (6.10 in wide, 2.88 in tall, docx name "GenreOverviewFixed")
>
> **Alt text.** The same chart as Figure 8 on the fixed split's 201 neighborhoods, scored over the same ten anchor draws, 21 rows from Horror at the top to Simulation/Management at the bottom; the teal CE bar reaches between 50.4 and 87.3 percent and leads the gray, sand and lavender bars and the purple cap in 9 rows, dark teal caps appear in 16 rows, hatched teal spans in 4, dark purple caps in 8 and hatched purple spans in 12, and a black box outlines the Stealth/Immersive row (n = 17), whose bars reach 18.3, 47.9, 67.1 and 66.0 percent with a hatched purple span from 66.7 to 67.1 and a dark teal cap to 77.7.

**Figure A3:** Figure 8 on the fixed split: name-stripped hit@1 by genre over its 201 held-out neighborhoods under the fixed-split towers, drawn and boxed by the same rules.

**What Figure A3 shows.** The rows run from a mean pairwise CE anchor cosine of 0.68 (Horror) down to 0.53 (Simulation/Management); the six rows with fewer than five neighborhoods (Narrative/Choices, Bullet Hell, Horror, Fantasy, RPG, Simulation/Management) rest on at most 24 queries each and are to be read with caution. The a+u @ views arm is left out of Figure A3 for clarity; its numbers are in Table 3. No bar is at zero; two pairs coincide, the SimCLR-style tower and I + SimCLR in Horror, CE and I-CE in Simulation/Management. CE exceeds the frozen embedder and the three view-side towers in 9 of the 21 rows; I-CE exceeds CE in 16 rows, ties it in one and trails it in four (Strategy/Tactics, Sci-fi/Cyberpunk, Bullet Hell and Narrative/Choices, of five, seven, two and one neighborhoods), and exceeds the SimCLR-style tower in every row but RPG, Narrative/Choices and Bullet Hell, whereas I + SimCLR trails the SimCLR-style tower in 12 of the 21 rows and ties it in one.

**How the neighborhood panels are drawn.** Each panel is a map on which one kind of distance is exact. The eight anchors are placed by a two-dimensional metric MDS (SMACOF, the best of five starts) of their true pairwise chord distances between unit vectors, oriented like their principal plane and rotated or reflected onto the I-CE panel by orthogonal Procrustes; the 28 spacings come out with a median relative error of 19% to 25% and a maximum of 45% to 49% per panel (stress-1 0.20 to 0.25), the price of eight points that span seven dimensions (their principal plane would hold 48% to 61% of their variance). Each query is then drawn at its true chord distance from its own anchor, in the direction its displacement takes under the linear map, built from the pseudo-inverse of the centered anchors, that sends the anchors to their layout; so the length of every line in Figure 9 is exact and every panel is on one absolute scale, whereas a query’s position relative to the other anchors carries no information, and hits and misses are the true outcome among all 2,020 anchors. A two-dimensional projection does not keep those lengths comparable across panels: the plane of the eight anchors holds 4% to 15% of the queries’ displacement, and pooled over the six panels the query-to-anchor lengths of a per-panel PCA of the 32 points correlate with the true displacements at 0.29, those of an aligned UMAP at −0.25 (0.30 to 0.93 within a single panel’s PCA; `figures/layout_check.py`), which is why the lines are placed by their measured length instead. The frozen embedder’s displacements share one direction more nearly than any tower’s (mean resultant length 0.79 over the 24 unit displacement vectors, 0.96 as drawn, against 0.42 to 0.60 for the towers in the full space and 0.19 to 0.41 as drawn; for 24 unrelated directions the expected value is about 0.2), the register gap of Sections 2.3 and 4.1. The effective dimensionality (participation ratio of the anchor covariance) of the 2,020 anchors, on the draw shown, is 18.9 for the frozen embedder, 18.7 for VICReg, 51.1 for the SimCLR-style tower, 31.9 for CE and 38.9 for I-CE. Scripts: `figures/extract_fixed4096.py` (CUDA) writes the fixed-split vectors and re-reads Table 3; `data/w9/k10_eval.py` (CUDA) draws and scores the ten anchor packs, `figures/fixed4096_k10.py` reads the fixed-split towers against them and `figures/cv_hoods.py 4096` builds the five-fold neighborhoods; `figures/fig_genre_bars.py --cv --4096` draws Figure 8 and without the flags Figure A3; `figures/fig_triplet.py` selects the neighborhood, draws Figure 9 and records in `data/fig_distribution/fig_facts_4096k10.json` … (`figures/fig_stats.py` writes `query_stats_4096k10.json`) the layout errors, the miss destinations, the near-tie counts and the effective dimensionalities quoted here; `figures/fig_stats.py` computes the held-out query statistics and the neighborhood pair statistics; `figures/layout_check.py` the correlations of projected line lengths with the true displacements.

**What the panels of Figure 9 show.** The neighborhood’s mean pairwise CE anchor cosine is 0.650, in the second most confusable quarter of the 201 (the most confusable begins at 0.671); the genre rule, not confusability, selects it. Over the 407 held-out games and their three query versions, the ten draws averaged, a query’s cosine to its own anchor and to the nearest other anchor are 0.893 and 0.849 for the SimCLR-style tower, 0.905 and 0.865 with the invariance term, 0.647 and 0.567 for CE and 0.826 and 0.762 for I-CE; the standard deviation, over a game’s three query versions, of the cosine to its own anchor is 0.029, 0.027, 0.051 and 0.030; the share of queries retrieving their own game is 0.830, 0.825, 0.845 and 0.879. In the shown neighborhood, on the draw the panels show, the mean chord displacement of the 24 queries is 0.50 (frozen embedder), 0.64 (VICReg), 0.52 (SimCLR-style), 0.86 (CE), 0.49 (I + SimCLR) and 0.58 (I-CE), and the version-to-version scatter 0.30, 0.50, 0.44, 0.66, 0.41 and 0.40 in the same order. These lengths hardly depend on the neighborhood: over the 201 the six mean displacements are 0.55, 0.55, 0.46, 0.84, 0.43 and 0.59, CE’s lines are the longest of the six in every neighborhood and the two in-batch towers’ the two shortest in 192, and from CE to I-CE the lines shorten in every neighborhood, by 23% to 38% (30% on average, 33% in the shown one), and the scatter falls in every neighborhood, by 31% to 44% (39% on average, 40% in the shown one). The hit counts of one neighborhood are coarser. Over the ten draws CE retrieves 18 to 20 of the 24 queries there and I-CE 20 to 21, 19.2 and 20.4 on average, so the invariance term adds 1.2 queries to CE, its mean over the 201; it takes 1.5 from the SimCLR-style tower (19.8 to 18.3; 0.2 over the 201), whose queries sit within 0.02 of a tie between their own and another anchor on 8 of these 24 on the draw shown (9 with the term; 3 for CE and 3 for I-CE). On that draw I-CE misses four queries, the name-stripped rewrite of Nightingale and all three queries of Satisfactory, and CE six, all three queries of Nightingale, the wiki original of Abiotic Factor and the wiki original and the name-stripped rewrite of Satisfactory; each is taken by a game outside the eight. Over the 201 neighborhoods the six encoders return 8.5, 17.0, 20.1, 19.9, 19.9 and 21.1 of 24 queries on average, in the same order, and their nearest-neighbor margins over the held-out games are 0.012, 0.112, 0.098, 0.181, 0.089 and 0.166. On the fixed split VICReg’s name-stripped queries land nearer another game’s anchor than their own on average (cosine 0.81 against 0.79), as the frozen embedder’s do (0.86 against 0.84); the other towers’ land nearer their own.

## APPENDIX M: THE DISPLACEMENT AND MARGIN TABLE

Table A17 gives in full the fold-0 reading of query displacement against nearest-neighbor margin whose key numbers Section 6.1 quotes in its text beside Figures 8 and 9.

**Table A17:** Query displacement against nearest-neighbor margin, both measured as mean 1 − cos distance. Every row is a fold-0 encoder of the five-fold split (Table A1), read against 4,096-sentence anchors from a single anchor draw. A ratio below one means a typical query lands inside its own game’s clearance, the margin ball around its anchor, a stricter test than rank one. Displacement averages the fold’s full 814-query name-stripped register, the population of Figure A2; ratios are computed before rounding; epd = expander wiring, v/i/c = 20/10/20.

| Method | Query displacement (1 − cos) | NN margin (1 − cos) | Displacement / margin |
| --- | --- | --- | --- |
| **Frozen embedder (mean pool)** | **0.157** | **0.012** | **13.07** |
| BYOL | 0.148 | 0.047 | 3.12 |
| CE (contrast only) | 0.389 | 0.191 | 2.04 |
| VICReg (epd 20/10/20) | 0.217 | 0.108 | 2.02 |
| SimCLR-style (in-batch views) | 0.138 | 0.106 | 1.30 |
| **I-CE (ours)** | **0.201** | **0.176** | **1.14** |

## APPENDIX N: THE DOCUMENT VIEW: PRESENT, ABSENT, REWRITTEN

The document view is one of the four strong views, and this ablation isolates it in towers trained at the 512-sentence anchor budget, five folds per cell, seed-paired with the raw-document tower of the recipe, which keeps the original documents (Table A18).

**Table A18:** The document view: present, absent, rewritten (test-set queries (ts), five-fold mean ± std; towers trained at the 512-sentence anchor budget and scored against 4,096-sentence anchors, checkpoints selected by the rank criterion of Section 5, zero-shot, seed-paired folds; deltas are computed before rounding). The document view is worth 0.026 Name hit@1 and 0.032 Stripped hit@1, each in four folds of five (paired t(4) = 3.04 and 3.03, p = 0.039 for both). Rewriting it moves neither hit@1 measurably (−0.004 Stripped hit@1, p = 0.56; +0.006 Name hit@1, p = 0.42), though it costs 0.013 Stripped hit@5 in all five folds (p = 0.011), and in the frozen space the rewrite’s register offset points only 0.977 in cosine from the raw article’s (online Appendix E). A rewrite firewall would therefore cost little. The last row is CE at this budget, the CE @512 towers of Table A14: without the document view I-CE stays above it on the stripped columns and on tag F1 and falls just below it on the name columns.

| Objective | Name hit@1 | Name hit@5 | Stripped hit@1 | Stripped hit@5 | Test-set tag F1 |
| --- | --- | --- | --- | --- | --- |
| **I-CE, raw documents (recipe)** | **0.942 ± 0.017** | **0.997 ± 0.004** | **0.715 ± 0.010** | **0.918 ± 0.016** | **0.685 ± 0.014** |
| I-CE, no document view | 0.916 ± 0.029 | 0.985 ± 0.007 | 0.683 ± 0.025 | 0.901 ± 0.014 | 0.678 ± 0.023 |
| I-CE, LLM-rewritten documents | 0.948 ± 0.021 | 0.995 ± 0.002 | 0.711 ± 0.009 | 0.905 ± 0.021 | 0.685 ± 0.016 |
| CE, raw documents (Table A14) | 0.920 ± 0.021 | 0.988 ± 0.009 | 0.676 ± 0.013 | 0.890 ± 0.016 | 0.667 ± 0.025 |
