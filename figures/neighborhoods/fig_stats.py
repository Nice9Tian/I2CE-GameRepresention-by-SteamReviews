# -*- coding: utf-8 -*-
"""The query-level numbers behind Figures 8, 9 and A3 that Appendix L quotes, on the fixed split against
the ten 4,096-sentence anchor draws of prep.py, written to <HOOD_WORK>/query_stats_4096k10.json.

Held-out games (validation + test, 407), three query versions each (wiki original, LLM
rewrite with names, rewrite without names), all 2,020 anchors:
    own       cosine of the query to its own game's anchor
    other     cosine to the nearest other anchor
    margin    own - other (a hit is a positive margin)
    scatter   std of own over a game's three versions (how consistently a game's queries land)
    disp      1 - own for the name-stripped version (the displacement of Table A17)
    nn_margin 1 - cosine between a game's anchor and its nearest other anchor (the margin of Table A17)
The 201 held-out neighborhoods (fig_triplet.py): per-tower accuracy over 24 queries, pairwise
comparisons, the CE minus SimCLR-style lead by confusability quartile, the neighborhoods where CE
falls behind the SimCLR-style tower, and the lengths of the lines Figure 9 draws.

    python fig_stats.py     (needs scipy)
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent; sys.argv = [sys.argv[0]]; sys.path.insert(0, str(HERE))
import fig_triplet as ft   # noqa: E402

M = ft.models; nrm = ft.nrm; out = {}
held = np.where(ft.status != "train")[0]; own_idx = ft.gidx[held]
q = {}
for m in M:
    Q = nrm(ft.QSm[m][held]); n, V, D = Q.shape; oi = np.repeat(own_idx, V); hit = ft.SCm[m][held]; per = []
    for G in np.moveaxis(ft.gal_draws(m), 1, 0):             # one pass a draw (the one pack with HOOD_DRAWS unset); every statistic is the mean over the draws
        S = Q.reshape(-1, D) @ G.T
        own = S[np.arange(len(oi)), oi]; S[np.arange(len(oi)), oi] = -9; other = S.max(1)
        own = own.reshape(n, V); other = other.reshape(n, V)
        C = G @ G.T; C[np.arange(len(G)), np.arange(len(G))] = -9; nnm = 1 - C[own_idx].max(1)
        per.append([own.mean(), other.mean(), (own - other).mean(), own.std(1).mean(), own[:, 2].mean(), other[:, 2].mean(), (1 - own[:, 2]).mean(), nnm.mean()])
    o, ot, mg, sc, os_, ots, dsp, nn = np.mean(per, axis=0)
    q[m] = dict(own=float(o), other=float(ot), margin=float(mg), scatter=float(sc),
                own_stripped=float(os_), other_stripped=float(ots), disp_stripped=float(dsp),
                nn_margin=float(nn), ratio=float(dsp / nn),
                hit_stripped=float(hit[:, 2].mean()), hit_all=float(hit.mean()), n_games=int(n))
out["held_out_queries"] = q
print(f"{'tower':13s} {'own':>6s} {'other':>6s} {'margin':>7s} {'scatter':>8s} {'disp':>6s} {'nn_m':>6s} {'ratio':>6s} {'hit_s':>6s}")
for m in M:
    r = q[m]; print(f"{m:13s} {r['own']:6.3f} {r['other']:6.3f} {r['margin']:+7.3f} {r['scatter']:8.4f} {r['disp_stripped']:6.3f} {r['nn_margin']:6.3f} {r['ratio']:6.2f} {r['hit_stripped']:6.3f}")

H = ft.hoods; allh = list(H); cos = np.array([H[t]["mean_cos"] for t in allh]); NP = ft.NP
acc = {m: np.rint(np.array([H[t]["acc"][m] for t in allh]) * NP) for m in M}     # integer hit counts (a float product once counted a tie as a loss)
if ft.DRAWS > 1:                                                                   # mean hits over the anchor draws, to a thousandth of a query: equal means still tie
    acc = {m: np.round(np.array([H[t]["acc"][m] for t in allh]) * NP, 3) for m in M}
qs = np.quantile(cos, [0.25, 0.5, 0.75]); quart = np.digitize(cos, qs)
nb = dict(n=len(allh), n_queries=NP, mean_hits={m: float(acc[m].mean()) for m in M},
          quartile_cos_ranges=[[float(cos[quart == k].min()), float(cos[quart == k].max())] for k in range(4)],
          quartile_hits={m: [float(acc[m][quart == k].mean()) for k in range(4)] for m in M})
pairs = {}
for a, b in (("A+U @ views", "SimCLR-style"), ("A+U @ views", "I + SimCLR"), ("SimCLR-style", "CE"), ("VICReg", "CE"), ("SimCLR-style", "I + SimCLR"), ("CE", "I-CE"), ("Frozen", "SimCLR-style")):
    if a not in acc or b not in acc:                                               # a tower this budget's file does not hold
        continue
    d = acc[b] - acc[a]; rho, p = spearmanr(cos, d)
    pairs[f"{b} minus {a}"] = dict(mean=float(d.mean()), ahead=float(np.mean(d > 0)), tied=float(np.mean(d == 0)), behind=float(np.mean(d < 0)),
                                  by_quartile=[float(d[quart == k].mean()) for k in range(4)], spearman_vs_cos=float(rho), p=float(p))
nb["pairs"] = pairs
lose = acc["CE"] < acc["SimCLR-style"]
nb["ce_behind_simclr"] = dict(n=int(lose.sum()), mean_cos=float(cos[lose].mean()), in_top_quartile=int((lose & (quart == 3)).sum()),
                              hits={m: float(acc[m][lose].mean()) for m in M},
                              ice_over_ce=float(np.mean(acc["I-CE"][lose] > acc["CE"][lose])), ice_over_simclr=float(np.mean(acc["I-CE"][lose] > acc["SimCLR-style"][lose])),
                              isimclr_over_simclr=float(np.mean(acc["I + SimCLR"][lose] > acc["SimCLR-style"][lose])))
g = acc["I-CE"] - acc["CE"]; nb["gap_ice_ce"] = dict(median=float(np.median(g)), ahead=float(np.mean(g > 0)), tied=float(np.mean(g == 0)), behind=float(np.mean(g < 0)))


def chords(m):
    """per wiki game: the mean chord distance of its three queries to its own anchor (mean over the anchor draws), the line a
    panel of Figure 9 draws, and the mean chord distance between its three query versions, which no anchor enters"""
    Qn = nrm(ft.QSm[m]); G = ft.gal_draws(m)[ft.gidx]
    d = np.sqrt(np.maximum(2 - 2 * np.einsum("nvd,nkd->nvk", Qn, G), 0)).mean((1, 2))
    iu = np.triu_indices(Qn.shape[1], 1); sc = np.array([np.linalg.norm(x[:, None] - x[None], axis=2)[iu].mean() for x in Qn])
    return d, sc


def spread(v):
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()), q25=float(np.quantile(v, 0.25)), q75=float(np.quantile(v, 0.75)), positive=int((v > 0).sum()))


six = [m for m in ft.SCALE_TOWERS if m in M]; LN = {m: chords(m) for m in six}
disp = {m: np.array([LN[m][0][list(t)].mean() for t in allh]) for m in six}; scat = {m: np.array([LN[m][1][list(t)].mean() for t in allh]) for m in six}
nb["lines"] = dict(mean_displacement={m: float(disp[m].mean()) for m in six}, mean_scatter={m: float(scat[m].mean()) for m in six},
                   ce_to_ice_shorter=spread(1 - disp["I-CE"] / disp["CE"]), ce_to_ice_scatter_smaller=spread(1 - scat["I-CE"] / scat["CE"]),
                   simclr_to_isimclr_shorter=spread(1 - disp["I + SimCLR"] / disp["SimCLR-style"]),
                   ce_longest_of_six=int(sum(disp["CE"][i] >= max(disp[m][i] for m in six) - 1e-12 for i in range(len(allh)))),
                   inbatch_two_shortest=int(sum(max(disp["SimCLR-style"][i], disp["I + SimCLR"][i]) <= min(disp[m][i] for m in six if m not in ("SimCLR-style", "I + SimCLR")) + 1e-12
                                                for i in range(len(allh)))))
out["neighbourhoods"] = nb
ln = nb["lines"]; print("lines: mean displacement " + " ".join(f"{m} {ln['mean_displacement'][m]:.3f}" for m in six) + f"; CE to I-CE shorter by {100 * ln['ce_to_ice_shorter']['mean']:.1f}% "
                        f"({100 * ln['ce_to_ice_shorter']['min']:.1f} to {100 * ln['ce_to_ice_shorter']['max']:.1f}, in {ln['ce_to_ice_shorter']['positive']} of {len(allh)}); scatter smaller by "
                        f"{100 * ln['ce_to_ice_scatter_smaller']['mean']:.1f}% ({100 * ln['ce_to_ice_scatter_smaller']['min']:.1f} to {100 * ln['ce_to_ice_scatter_smaller']['max']:.1f}); "
                        f"CE longest in {ln['ce_longest_of_six']}, the in-batch towers the two shortest in {ln['inbatch_two_shortest']}")
for k, v in pairs.items():
    print(f"{k:32s} mean {v['mean']:+5.1f}  ahead {100*v['ahead']:3.0f}%  tied {100*v['tied']:3.0f}%  behind {100*v['behind']:3.0f}%  by quartile " + " ".join(f"{x:+.1f}" for x in v["by_quartile"]) + f"  rho {v['spearman_vs_cos']:+.2f} p {v['p']:.3f}")
b = nb["ce_behind_simclr"]; print(f"CE behind SimCLR in {b['n']} ({b['in_top_quartile']} in the top quartile): " + " ".join(f"{m} {b['hits'][m]:.1f}" for m in M) + f"; I-CE > CE {100*b['ice_over_ce']:.0f}%, I-CE > SimCLR {100*b['ice_over_simclr']:.0f}%, I+SimCLR > SimCLR {100*b['isimclr_over_simclr']:.0f}%")
(ft.DATA / f"query_stats{ft._SFX}.json").write_text(json.dumps(out, indent=1), encoding="utf-8"); print("wrote", ft.DATA / f"query_stats{ft._SFX}.json")
