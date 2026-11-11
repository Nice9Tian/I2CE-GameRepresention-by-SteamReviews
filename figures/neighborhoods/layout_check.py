# -*- coding: utf-8 -*-
r"""Are query-to-anchor line lengths faithful in a 2-D projection?  The check Appendix L quotes: per-panel
PCA of the 32 points (the neighborhood of fig_triplet), UMAP per encoder on all 2,020 anchors and the 24
queries, and AlignedUMAP across the six encoders; within-panel and pooled correlations of the drawn
lengths with the true chord displacements.

    python layout_check.py      (needs umap-learn and scipy)
"""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.argv = [sys.argv[0]]; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import fig_triplet as ft
import umap
from scipy.stats import pearsonr, spearmanr
T = list(ft.T); K = len(T); own = ft.gidx[T]
MODELS = ["Frozen", "VICReg", "SimCLR-style", "I + SimCLR", "CE", "I-CE"]
NA = ft.gal["CE"].shape[0]                                   # 2,020 anchors
qi = NA + np.arange(3 * K); ai = np.repeat(own, 3)          # row of each query and of its own anchor in the stacked matrix


def faith(Y, true, label):
    d2 = np.linalg.norm(Y[qi] - Y[ai], axis=1)
    r = pearsonr(d2, true)[0]; rho = spearmanr(d2, true)[0]; ratio = d2 / true
    # does the 2-D nearest anchor agree with the true nearest anchor?
    D = np.linalg.norm(Y[qi][:, None, :] - Y[:NA][None, :, :], axis=2); nn2 = D.argmin(1)
    return d2, f"{label:34s} r {r:+.2f}  rho {rho:+.2f}  ratio cv {ratio.std() / ratio.mean():.2f}  2-D nearest anchor = true nearest {int((nn2 == TRUE_NN).sum())}/24"


X = {}; true = {}; TRUE_NN_ALL = {}
for m in MODELS:
    G = ft.gal[m]; Q = ft.nrm(ft.QSm[m][T]).reshape(3 * K, -1); X[m] = np.vstack([G, Q])
    true[m] = np.linalg.norm(Q - G[ai], axis=1)                                            # Euclidean = sqrt(2 - 2 cos)
    TRUE_NN_ALL[m] = (Q @ G.T).argmax(1)
pooled = {"PCA-32 (B1, absolute)": ([], []), "AlignedUMAP": ([], [])}
for m in MODELS:
    TRUE_NN = TRUE_NN_ALL[m]
    # PCA / classical MDS of the 32 points, absolute scale
    P = np.vstack([X[m][own], X[m][NA:]]); P0 = P - P.mean(0); U, s, _ = np.linalg.svd(P0, full_matrices=False); Y32 = U[:, :2] * s[:2]
    d2 = np.linalg.norm(Y32[K:] - Y32[np.repeat(np.arange(K), 3)], axis=1)
    print(f"{m:13s} PCA-32 (B1)                        r {pearsonr(d2, true[m])[0]:+.2f}  rho {spearmanr(d2, true[m])[0]:+.2f}  ratio cv {(d2 / true[m]).std() / (d2 / true[m]).mean():.2f}")
    pooled["PCA-32 (B1, absolute)"][0].extend(d2); pooled["PCA-32 (B1, absolute)"][1].extend(true[m])
    Y = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X[m])
    _, line = faith(Y, true[m], f"{m:13s} UMAP per encoder (2,044 pts)"); print(line)
slices = [X[m] for m in MODELS]; rel = [{i: i for i in range(NA + 3 * K)} for _ in range(len(MODELS) - 1)]
al = umap.AlignedUMAP(n_neighbors=15, min_dist=0.1, metric="cosine", alignment_regularisation=1e-2, random_state=0).fit(slices, relations=rel)
for m, Y in zip(MODELS, al.embeddings_):
    TRUE_NN = TRUE_NN_ALL[m]; d2, line = faith(np.asarray(Y), true[m], f"{m:13s} AlignedUMAP (6 slices)"); print(line)
    pooled["AlignedUMAP"][0].extend(d2); pooled["AlignedUMAP"][1].extend(true[m])
for k, (a, b) in pooled.items():
    a = np.array(a); b = np.array(b); print(f"POOLED over 6 encoders x 24 lines, {k}: r {pearsonr(a, b)[0]:+.2f}  rho {spearmanr(a, b)[0]:+.2f}")
print("done")
