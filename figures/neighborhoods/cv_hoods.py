# -*- coding: utf-8 -*-
r"""The 814 five-fold neighborhoods behind Figure 8.

Folds are the worker's: `np.random.default_rng(20260711).permutation(814)` over the sorted wiki
app ids, `np.array_split(5)`; fold k tests on part k and validates on part k+1.  In each fold every
test game seeds one neighborhood: itself and the seven held-out games (test or validation of that
fold) nearest to it by the cosine of that fold's CE anchors, averaged over the draws.  A
neighborhood's score for an encoder is the share of its eight name-stripped queries whose nearest
of the 2,020 anchors is their own game, a query's score being its hit rate over the ten
4,096-sentence anchor draws (the protocol of Figure 7).  Every wiki game is a seed exactly once, so
the five folds give 814 neighborhoods.

The vectors are the ten-draw encodings of contrast_experiment/w9/eval/k10_eval.py for the towers of
Figure 8 (SimCLR-style, VICReg, CE, I-CE and I + SimCLR on the five folds, and the frozen embedder):

    E=contrast_experiment/w9/eval
    python $E/k10_eval.py cells  cells_fig8.json --ckpt-dir checkpoints --only "^(I-CE|CE|SimCLR|VICReg|BYOL)\||i2bce"
    python $E/k10_eval.py encode cells_fig8.json draws_fig8.npz
    python figures/neighborhoods/cv_hoods.py --draws draws_fig8.npz --cells cells_fig8.json

    -> <HOOD_WORK>/cv/cv_hoods_4096.json, and the genre table printed
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hood_env  # noqa: E402

# the cells of k10_eval.py: the main families under the paper's names, I + SimCLR under its key in i2bce_sel_pool.json
KEY = {"SimCLR-style": "SimCLR|%d", "VICReg": "VICReg|%d", "BYOL": "BYOL|%d", "I + SimCLR": "w9cv_wcle_i2bce_icetf_fold%d_fp@pool",
       "CE": "CE|%d", "I-CE": "I-CE|%d"}
K = 8


def nrm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", required=True, help="the file `k10_eval.py encode` wrote for the five-fold towers")
    ap.add_argument("--cells", required=True, help="the cells file it was given (for the selected epochs)")
    a = ap.parse_args()
    C = hood_env.need_data_dir(); OUTJ = hood_env.FACTS / "cv" / "cv_hoods_4096.json"
    names = [str(x) for x in np.load(C / "games.npz", allow_pickle=True)["names"]]
    appid = [n.split("_")[0] for n in names]; a2g = {x: i for i, x in enumerate(appid)}
    A = np.load(C / "wiki_eval.npz", allow_pickle=True)
    variants = np.array([str(x) for x in A["variants"]]); gidx = np.asarray(A["gidx"])
    row_non = {int(g): i for i, (g, v) in enumerate(zip(gidx, variants)) if v == "noname"}
    row_neu = {int(g): i for i, (g, v) in enumerate(zip(gidx, variants)) if v == "neutral"}
    sp = json.load(open(C / "wiki_eval_split.json"))
    universe = sorted(set(sp["test"]) | set(sp["val"]) | set(sp["train"]))
    perm = np.random.default_rng(20260711).permutation(len(universe)); folds = np.array_split(perm, 5)
    tl = np.load(C / "tag_labels.npz", allow_pickle=True); TAG = [str(x) for x in tl["tag_names"]]; Y = tl["y"]; fr = Y.sum(0)

    def genre23(g):
        c = np.where(Y[g] > 0)[0]
        return TAG[min(c, key=lambda j: fr[j])] if len(c) else "no tag"

    D = np.load(a.draws); cells = json.load(open(a.cells, encoding="utf-8"))
    assert int(D["cap"]) == 4096 and int(D["games"]) == len(names), "the draws file must hold every game at 4,096 sentences (no --games)"
    towers = [m for m in KEY if all(KEY[m] % k in D.files for k in range(5))]; frozen = "frozen" in D.files
    assert "CE" in towers, "the CE towers of the five folds are needed: neighbors are taken by their anchors"

    def vectors(m, k):
        """galleries [draws, 2020, d], queries [3256, d] and the epoch of encoder m in fold k"""
        key = "frozen" if m == "Frozen" else KEY[m] % k
        return nrm(D[key]).transpose(1, 0, 2), nrm(D["Q|" + key]), (None if m == "Frozen" else int(cells[key]["ep"]))

    print("towers with all five folds:", towers, "| frozen:", frozen)
    hoods = []; fold_h1 = {m: [] for m in towers + (["Frozen"] if frozen else [])}; best_ep = {}
    for k in range(5):
        test = [a2g[universe[i]] for i in folds[k]]; val = [a2g[universe[i]] for i in folds[(k + 1) % 5]]
        held = np.array(test + val)
        V = {}
        for m in towers + (["Frozen"] if frozen else []):
            G, q, ep = vectors(m, k); V[m] = (G, q)
            if ep is not None:
                best_ep.setdefault(m, []).append(ep)
        hit = {}                                                  # per encoder: game -> name-stripped hit, name-intact hit (rates over the draws)
        for m, (G, q) in V.items():
            hs = np.mean([np.argmax(q[[row_non[x] for x in held]] @ g.T, axis=1) == held for g in G], axis=0)
            hn = np.mean([np.argmax(q[[row_neu[x] for x in held]] @ g.T, axis=1) == held for g in G], axis=0)
            hs = {int(x): float(v) for x, v in zip(held, hs)}; hn = {int(x): float(v) for x, v in zip(held, hn)}
            hit[m] = (hs, hn); fold_h1[m].append(float(np.mean([hs[x] for x in test])))
        gce = V["CE"][0][0] if len(V["CE"][0]) == 1 else nrm(V["CE"][0].mean(0)); Cm = gce[held] @ gce[held].T; pos = {int(x): i for i, x in enumerate(held)}
        for s in test:
            order = np.argsort(-Cm[pos[s]]); nb = [int(held[j]) for j in order if held[j] != s][:K - 1]
            t = [int(s)] + nb; ii = [pos[x] for x in t]
            sub = Cm[np.ix_(ii, ii)][np.triu_indices(K, 1)]
            hoods.append(dict(fold=k, seed=int(s), games=t, genre=genre23(int(s)), mean_cos=float(sub.mean()),
                              acc_s={m: float(np.mean([hit[m][0][x] for x in t])) for m in V},
                              acc_n={m: float(np.mean([hit[m][1][x] for x in t])) for m in V}))
    enc = list(hoods[0]["acc_s"])
    OUTJ.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUTJ, "w", encoding="utf-8").write(json.dumps(dict(encoders=enc, best_ep=best_ep, fold_stripped_h1=fold_h1, hoods=hoods)))
    print(f"{len(hoods)} neighborhoods; selected epochs:", best_ep)
    print("test-set name-stripped hit@1 per fold (mean ± std):")
    for m in enc:
        v = np.array(fold_h1[m]); print(f"  {m:14s} {v.mean():.3f} ± {v.std(ddof=1):.3f}   {np.round(v, 3).tolist()}")
    genres = {}
    for h in hoods:
        genres.setdefault(h["genre"], []).append(h)
    order = sorted(genres, key=lambda g: -np.mean([h["mean_cos"] for h in genres[g]]))
    print(f"\n{'genre':22s} {'n':>3s} {'cos':>5s} | " + " ".join(f"{m:>12s}" for m in enc))
    for g in order:
        hs = genres[g]
        print(f"{g:22s} {len(hs):3d} {np.mean([h['mean_cos'] for h in hs]):5.3f} | " + " ".join(f"{100 * np.mean([h['acc_s'][m] for h in hs]):12.1f}" for m in enc))
    if "I-CE" in enc and "CE" in enc:
        d = np.array([h["acc_s"]["I-CE"] - h["acc_s"]["CE"] for h in hoods]) * K
        print(f"\nI-CE minus CE, name-stripped queries per neighborhood of 8: mean {d.mean():+.2f}; ahead {np.mean(d > 0):.0%}, tied {np.mean(d == 0):.0%}, behind {np.mean(d < 0):.0%}")
        for k in range(5):
            dk = np.array([h["acc_s"]["I-CE"] - h["acc_s"]["CE"] for h in hoods if h["fold"] == k]) * K
            print(f"   fold {k}: mean {dk.mean():+.2f} over {len(dk)} neighborhoods")
        pg = {g: 100 * (np.mean([h["acc_s"]["I-CE"] for h in genres[g]]) - np.mean([h["acc_s"]["CE"] for h in genres[g]])) for g in order}
        print("   genres where I-CE leads / ties / trails CE:", sum(v > 0 for v in pg.values()), sum(v == 0 for v in pg.values()), sum(v < 0 for v in pg.values()))
    print("wrote", OUTJ)


if __name__ == "__main__":
    main()
