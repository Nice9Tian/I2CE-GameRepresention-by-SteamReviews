# -*- coding: utf-8 -*-
"""Table 2: the paired tests over the five folds, from per-fold readings.

    python contrast_experiment/w9/eval/paired_tests.py                    # the rows Table 2 prints
    python contrast_experiment/w9/eval/paired_tests.py --all              # every pair, Table A19's tests too
    python contrast_experiment/w9/eval/paired_tests.py --main <k10_eval score output> --d512 <k10_eval score output>

Two groups of rows, each read from a file of per-fold readings:

  --main   the towers of Figure 7 (trained at the 2,048-sentence budget, read against ten
           4,096-sentence anchor draws): cells I-CE|k, CE|k, SimCLR|k, ... as `k10_eval.py cells`
           names them.  Default paper_readings/multianchor_rank_K10.json, the per-fold detail the
           paper's five-fold tables print (Table A13).
  --d512   the towers trained at 512 sentences on the current packs, read the same way: I-CE,
           CE, CE + anchor-in-I, and for --all the document-view arms of Table A19.  Default
           paper_readings/scores_c2cv_cap4096_k10.json, the reading the paper prints.

A file written by `k10_eval.py score` serves for either; the 512-sentence cells are then named
after their runs in rank_ckpt_plan.json (i2ce_icetf_fold<k>_d512, ...), and both namings are
looked up.  Recomputed draws agree with the shipped readings to the third decimal, so a p value
may move in its last printed digit.

Statistics: the paired difference first named minus second, its sd (ddof = 1), the folds in
which the first leads, Student's t with four degrees of freedom and its two-sided p, Cohen's d_z,
and the exact Wilcoxon signed-rank p (which cannot fall below 0.0625 at five folds).  Numpy only:
the t tail is the regularized incomplete beta, the Wilcoxon p an enumeration of the 32 sign
patterns; both agree with scipy's ttest_rel and wilcoxon(mode="exact").
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
READ = HERE / "paper_readings"
FOLDS = range(5)
# tag F1 is `tag_perdraw` in the paper's per-fold file and `pd_test_tag` in a k10_eval output
TAG = ("tag_perdraw", "pd_test_tag")
# the cells of the 512-sentence runs: (name in the paper's reading, name `k10_eval.py cells` gives)
D512 = {"I-CE": ("base.wcle_i2ce_icetf|{k}", "i2ce_icetf_fold{k}_d512"),
        "CE": ("base.wcle_ce_cetf|{k}", "ce_cetf_fold{k}_d512"),
        "CE + anchor-in-I": ("wcle_ai2ce_icetf|{k}", "ai2ce_icetf_fold{k}_d512"),
        "no document view": ("wcle_nodoc_i2ce_icetf|{k}", "nodoc_i2ce_icetf_fold{k}_d512"),
        "LLM-rewritten": ("wcle_i2ce_icetf_wllm|{k}", "i2ce_icetf_fold{k}_d512_wllm")}


# --------------------------------------------------------------------------- distributions

def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lead = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                    + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return lead * _betacf(a, b, x) / a
    return 1.0 - lead * _betacf(b, a, 1.0 - x) / b


def t_sf_two_sided(t, df):
    """Two-sided p of Student's t."""
    if df <= 0:
        return float("nan")
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def rankdata(values):
    """Average ranks, ties shared (the Wilcoxon convention)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def wilcoxon_exact(diff):
    """Two-sided exact Wilcoxon signed-rank p by enumerating the sign patterns.

    At n = 5 the smallest value this can return is 2/32 = 0.0625, reached
    whenever one arm leads in all five folds.
    """
    d = [float(x) for x in diff if x != 0.0]
    n = len(d)
    ranks = rankdata([abs(x) for x in d])
    total = sum(ranks)
    w_obs = min(sum(r for r, x in zip(ranks, d) if x > 0),
                total - sum(r for r, x in zip(ranks, d) if x > 0))
    hits = 0
    for signs in itertools.product((0, 1), repeat=n):
        w_plus = sum(r for r, s in zip(ranks, signs) if s)
        if min(w_plus, total - w_plus) <= w_obs + 1e-12:
            hits += 1
    return hits / 2 ** n


# --------------------------------------------------------------------------- #
# the comparison itself


def paired(a, b):
    """Paired statistics of a - b, both indexed by fold."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    t = d.mean() / (sd / math.sqrt(n))
    return dict(mean_a=a.mean(), mean_b=b.mean(), diff=d.mean(), sd=sd, n=n,
                ahead=int((d > 0).sum()), t=t, p_t=t_sf_two_sided(t, n - 1),
                p_w=wilcoxon_exact(d), dz=d.mean() / sd,
                per_fold=[round(float(x), 4) for x in d])


# --------------------------------------------------------------------------- reading the files


def load(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw.get("results", raw)          # a k10_eval output keeps its cells under `results`


def field(cell, name):
    names = TAG if name == "tag" else (name,)
    for n in names:
        if n in cell:
            return cell[n]
    raise KeyError(name)


def series(R, patterns, name):
    """the five per-fold values of the first naming of a run that is in R"""
    for pat in patterns:
        keys = [pat.format(k=k) for k in FOLDS]
        if all(k in R for k in keys):
            return np.array([field(R[k], name) for k in keys], float)
    raise KeyError("no cells %s in the file" % " or ".join(patterns))


def show(label, r):
    print(f"{label:<58s}{r['diff']:+.3f} ± {r['sd']:.3f}   {r['ahead']}/{r['n']}   t(4) = {r['t']:5.2f}"
          f"   p = {r['p_t']:.4f}   d_z = {r['dz']:.2f}   (Wilcoxon p = {r['p_w']:.4f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main", default=str(READ / "multianchor_rank_K10.json"))
    ap.add_argument("--d512", default=str(READ / "scores_c2cv_cap4096_k10.json"))
    ap.add_argument("--all", action="store_true", help="every pair, not only the rows Table 2 prints")
    a = ap.parse_args()
    M, D = load(a.main), load(a.d512)
    metrics = [("Stripped hit@1", "pd_nm_noname"), ("Name hit@1", "pd_nm_neutral"), ("Test-set tag F1", "tag")]

    def main_s(arm, f):
        return series(M, ["%s|{k}" % arm], f)

    def d512_s(arm, f):
        return series(D, D512[arm], f)

    print("Table 2: first named minus second, Δ ± sd over the five folds, folds ahead, t(4), p, d_z\n")
    print("I-CE vs. CE  (towers of Figure 7)")
    for m, f in metrics:
        show("  " + m, paired(main_s("I-CE", f), main_s("CE", f)))
    for arm in (["SimCLR"] + (["VICReg", "BYOL", "frozen"] if a.all else [])):
        print("I-CE vs. %s" % ("SimCLR-style" if arm == "SimCLR" else arm))
        for m, f in metrics if a.all else metrics[:1]:
            show("  " + m, paired(main_s("I-CE", f), main_s(arm, f)))
    for other, rows in (("I-CE", (0, 2)), ("CE", (0,))):
        print("CE + anchor-in-I vs. %s  (towers trained at 512 sentences)" % other)
        for i in (range(3) if a.all else rows):
            m, f = metrics[i]
            show("  " + m, paired(d512_s("CE + anchor-in-I", f), d512_s(other, f)))
    if a.all:
        print("Table A19: the document view (towers trained at 512 sentences)")
        for m, f in metrics:
            show("  %s: raw document - no document view" % m, paired(d512_s("I-CE", f), d512_s("no document view", f)))
            show("  %s: LLM-rewritten - raw document" % m, paired(d512_s("LLM-rewritten", f), d512_s("I-CE", f)))


if __name__ == "__main__":
    main()
