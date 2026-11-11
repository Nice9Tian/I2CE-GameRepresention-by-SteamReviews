# -*- coding: utf-8 -*-
r"""Level bands joined by transfers: the numbers the release-share river drew from 2026-10-13 to 2026-11-10.

In this repository (figures/trend_river/) the data file is `games_raw.js` and
the fit is written beside this script; the paths below are those of the
paper's own repository.

Since 2026-11-11 the figure (Figure 10) draws river_shift_fit.py's periods and ribbons instead; that script's
docstring says why.  This one is kept so that the earlier fit can be drawn again (--fit river_assets/flat_fit_ice.json
on either renderer), and river_shift_fit.py reads the page's clusters through export_lanes() below.

Position in the chain (as it was)
---------------------------------
    river_ice_prep.py   -> river_assets/games_raw_ice.js     (data, once)
    river_flat_fit.py   -> river_assets/flat_fit_ice.json    (this script)
    river_ice_render.py -> the full-width figure the Word build embeds   (--fit river_assets/flat_fit_ice.json)
    river_ice_column.py -> the one-column figure of the two-column build (same option)

What is fitted
--------------
The six clusters are the river page's own (seeded spherical k-means at K = 6 on the I-CE game vectors, run in the
browser), so they are read back from the page: headless Chrome loads it once and hands over the lane of every game.

Only the games released from --start on are used (2020.0: 1,504 of the 1,649; the three years before hold 145 games,
30 of them in 2019).  Their release share by cluster is smoothed over the release date with a Gaussian kernel
(--sigma, 0.25 years) on a grid of 24 steps a year, and then described by

    share_k(t) ~ level_k + sum_m w_m * ([k == to_m] - [k == from_m]) * ramp_m(t),
    ramp_m(t) = clip((t - start_m) / (end_m - start_m), 0, 1)

that is, every cluster keeps its level except while a transfer is under way, and a transfer moves the share w_m from
one cluster to another between two dates.  Transfers are added one at a time, each time the one that brings the
description closest to the smoothed shares (least squares), with all levels, amounts and dates refitted after every
addition (dates by coordinate descent on the grid), until no cluster is further than --tol (4 points of share) from
its smoothed share at any date.  A transfer lasts at least --mindur (0.25 years).

The figure draws the levels as bands of constant thickness and each transfer as a ribbon that leaves the source band
at start_m and joins the target band at end_m.

Reading the result
------------------
* The levels are close to the measured shares by construction (the report prints the largest and the rms distance).
* A ribbon's length is mostly the smoothing window: with 60 to 100 games a quarter the data do not resolve how long a
  shift took.
* Which cluster a share is paired with is not fixed by the data: a fall in one cluster is matched by rises in others,
  and the pairing is the least-squares one.  --resample N refits N bootstrap resamples of the games and reports how
  often each transfer comes back (same pair, overlapping dates) and how its amount varies with pair and dates held.

Run with:
    py figures/river_flat_fit.py                    # -> river_assets/flat_fit_ice.json
    py figures/river_flat_fit.py --resample 200     # also -> river_assets/flat_fit_ice.resample.json (minutes)
Needs numpy and Chrome (the same Chrome river_ice_render.py uses).
"""
import argparse
import html
import json
import re
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import river_ice_render as rir  # noqa: E402
from build_river_render import bake  # noqa: E402

DENS = 24                      # grid steps a year (the river page's own grid)
EXPORT_TAIL = """
<script>
// hand the page's clusters to river_flat_fit.py
window.addEventListener('load', function(){
  var d=document.createElement('div'); d.id='lanesout'; d.style.display='none';
  d.textContent=JSON.stringify({year0:YEAR0, k:K, n:RAW_N,
    labels:CLUSTER_LABELS.map(function(c){return {label:c.label, n:c.n};}),
    lane_of:Array.from(LANE_OF), tau:Array.from(RAW_TAU),
    dist:CLUSTER_DIST});   // 1 - cos between cluster centroids over the largest one (river_shift_fit.py)
  document.body.appendChild(d);
});
</script>
"""


def export_lanes(assets, data="games_raw_ice.js"):
    """Load the river page headless and return its clusters: year0, labels (lane 0 = bottom of the figure), the lane
    and the release time (years since year0) of every game."""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "lanes_export.html"
        bake(page, (Path(assets) / data).resolve().as_uri(), EXPORT_TAIL, src=Path(assets) / "cluster_share_stack.html")
        dom = rir.run_chrome(page, 1200, 800, 1)
    m = re.search(r'<div id="lanesout"[^>]*>(.*?)</div>', dom, re.S)
    if not m:
        raise SystemExit("the river page reached READY without handing over its clusters")
    out = json.loads(html.unescape(m.group(1)))
    if not (len(out["lane_of"]) == len(out["tau"]) == out["n"]):
        raise SystemExit("clusters and release times do not line up: %d, %d, %d" % (len(out["lane_of"]), len(out["tau"]), out["n"]))
    return out


def smoothed_shares(tau, lane, grid, sigma, K):
    """[steps, K] kernel share of each lane at each grid date, and the effective number of games behind it"""
    Wk = np.exp(-0.5 * ((grid[:, None] - tau[None, :]) / sigma) ** 2)
    S = np.stack([Wk[:, lane == k].sum(1) for k in range(K)], 1)
    return S / S.sum(1, keepdims=True), Wk.sum(1) ** 2 / (Wk ** 2).sum(1)


class FlatFit:
    """levels joined by transfers on the grid T; a transfer is (from lane, to lane, start step, end step)"""

    def __init__(self, T, K, mindur):
        self.T, self.K, self.n = T, K, len(T)
        self.mind = max(1, int(round(mindur * DENS)))
        self.AB = [(a, b) for a in range(self.n) for b in range(a + self.mind, self.n)]
        ramp = np.stack([self.ramp(a, b) for a, b in self.AB])
        self.RC = ramp - ramp.mean(1, keepdims=True)
        self.DEN = 2 * (self.RC ** 2).sum(1) + 1e-12
        self.PAIRS = list(combinations(range(K), 2))

    def ramp(self, a, b):
        return np.clip((self.T - self.T[a]) / (self.T[b] - self.T[a]), 0, 1)

    def design(self, transfers):
        X = np.zeros((self.n, self.K, self.K + len(transfers)))
        for k in range(self.K):
            X[:, k, k] = 1
        for m, (i, j, a, b) in enumerate(transfers):
            r = self.ramp(a, b)
            X[:, j, self.K + m] = r
            X[:, i, self.K + m] = -r
        return X

    def fit(self, transfers, S):
        """least squares for the levels and the amounts; returns (levels, amounts, fitted [steps, K], squared error)"""
        X = self.design(transfers)
        beta = np.linalg.lstsq(X.reshape(self.n * self.K, -1), S.reshape(-1), rcond=None)[0]
        F = X @ beta
        return beta[:self.K], beta[self.K:], F, float(((S - F) ** 2).sum())

    def best_addition(self, transfers, S, exact=60):
        """the transfer whose addition lowers the squared error most: every pair and every pair of dates is screened
        by its correlation with the residual, and the best `exact` candidates are refitted exactly"""
        _, _, F, sse0 = self.fit(transfers, S)
        R = S - F
        cand = []
        for i, j in self.PAIRS:
            num = self.RC @ (R[:, j] - R[:, i])
            gain = num ** 2 / self.DEN
            for q in np.argsort(-gain)[:exact // 4 + 1]:
                cand.append((gain[q], (i, j) if num[q] > 0 else (j, i), self.AB[q]))
        cand.sort(key=lambda c: -c[0])
        best = (sse0, None)
        for _, (i, j), (a, b) in cand[:exact]:
            s = self.fit(transfers + [(i, j, a, b)], S)[3]
            if s < best[0] - 1e-14:
                best = (s, (i, j, a, b))
        return best[1]

    def refine(self, transfers, S, sweeps=8, reach=10):
        """coordinate descent on the two dates of every transfer, levels and amounts refitted each time"""
        transfers = list(transfers)
        cur = self.fit(transfers, S)[3]
        for _ in range(sweeps):
            moved = False
            for m, (i, j, a, b) in enumerate(transfers):
                for a2 in range(max(0, a - reach), min(self.n - 1, a + reach) + 1):
                    for b2 in range(max(a2 + self.mind, b - reach), min(self.n - 1, b + reach) + 1):
                        trial = transfers[:m] + [(i, j, a2, b2)] + transfers[m + 1:]
                        s = self.fit(trial, S)[3]
                        if s < cur - 1e-12:
                            cur, transfers, moved = s, trial, True
                            a, b = a2, b2
            if not moved:
                break
        return transfers

    def path(self, S, count):
        """the fits with 0, 1, ..., count transfers (fewer if no transfer improves the fit any more)"""
        transfers, out = [], [self.summary([], S)]
        for _ in range(count):
            new = self.best_addition(transfers, S)
            if new is None:
                break
            transfers = self.refine(transfers + [new], S)
            out.append(self.summary(transfers, S))
        return out

    def summary(self, transfers, S):
        levels, amounts, F, _ = self.fit(transfers, S)
        gap = np.abs(S - F)
        s, k = np.unravel_index(gap.argmax(), gap.shape)
        return dict(transfers=[tuple(int(x) for x in t) for t in transfers], amounts=[float(w) for w in amounts],
                    levels=[float(x) for x in levels], fitted=F, rms=float(np.sqrt((gap ** 2).mean())),
                    largest=float(gap.max()), largest_lane=int(k), largest_step=int(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(rir.DEFAULT_ASSETS))
    ap.add_argument("--data", default=rir.DEFAULT_DATA)
    ap.add_argument("--out", default=None, help="the fit (default <assets>/flat_fit_ice.json)")
    ap.add_argument("--start", type=float, default=2020.0, help="first release date used and left end of the axis")
    ap.add_argument("--end", type=float, default=2024.75, help="right end of the axis")
    ap.add_argument("--sigma", type=float, default=0.25, help="standard deviation of the smoothing kernel, years")
    ap.add_argument("--tol", type=float, default=0.04, help="largest distance allowed between a level and the smoothed share")
    ap.add_argument("--mindur", type=float, default=0.25, help="shortest transfer, years")
    ap.add_argument("--max", type=int, default=10, help="give up after this many transfers")
    ap.add_argument("--resample", type=int, default=0, help="bootstrap resamples of the games for the stability report")
    a = ap.parse_args()
    assets = Path(a.assets).resolve()
    out = Path(a.out).resolve() if a.out else assets / "flat_fit_ice.json"

    page = export_lanes(assets, a.data)
    y0, K = page["year0"], page["k"]
    labels = [c["label"] for c in page["labels"]]
    tau_all, lane_all = np.array(page["tau"], float), np.array(page["lane_of"], int)
    keep = tau_all >= a.start - y0 - 1e-9
    tau, lane = tau_all[keep], lane_all[keep]
    T = (a.start - y0) + np.arange(int(round((a.end - a.start) * DENS)) + 1) / DENS
    S, neff = smoothed_shares(tau, lane, T, a.sigma, K)
    ff = FlatFit(T, K, a.mindur)
    year = lambda step: round(y0 + float(T[step]), 4)

    steps = ff.path(S, a.max)
    chosen = next((s for s in steps if s["largest"] <= a.tol), None)
    if chosen is None:
        raise SystemExit("no fit within %.1f points after %d transfers; raise --max or --tol" % (100 * a.tol, len(steps) - 1))
    F = chosen["fitted"]
    order = sorted(range(len(chosen["transfers"])), key=lambda m: chosen["transfers"][m][2])
    transfers = []
    for m in order:
        i, j, s0, s1 = chosen["transfers"][m]
        transfers.append({"from": i, "to": j, "from_label": labels[i], "to_label": labels[j],
                          "share": round(chosen["amounts"][m], 6), "start": year(s0), "end": year(s1)})
    fit = {
        "made_by": "figures/river_flat_fit.py", "data": a.data, "year0": y0,
        "games_total": int(len(tau_all)), "games_used": int(keep.sum()),
        "start": a.start, "end": a.end, "steps_per_year": DENS, "sigma_years": a.sigma,
        "tolerance": a.tol, "min_transfer_years": a.mindur,
        "labels": labels, "lane_games_used": [int((lane == k).sum()) for k in range(K)],
        "levels_start": [round(x, 6) for x in chosen["levels"]],
        "levels_end": [round(float(x), 6) for x in F[-1]],
        "transfers": transfers,
        "fit": {"rms": round(chosen["rms"], 5), "largest": round(chosen["largest"], 5),
                "largest_lane": labels[chosen["largest_lane"]], "largest_year": year(chosen["largest_step"])},
        "path": [{"transfers": len(s["transfers"]), "rms": round(s["rms"], 5), "largest": round(s["largest"], 5)} for s in steps],
        "effective_games": [round(float(x), 1) for x in (neff.min(), neff.max())],
        "grid": [year(s) for s in range(len(T))],
        "smoothed": [[round(float(v), 5) for v in row] for row in S],
        "fitted": [[round(float(v), 5) for v in row] for row in F],
    }
    out.write_text(json.dumps(fit, ensure_ascii=False, indent=1), encoding="utf-8")

    print("games %d of %d released from %.2f on; Gaussian kernel, sigma %.2f y; grid %.2f to %.2f, %d steps a year; "
          "effective games %.0f to %.0f" % (keep.sum(), len(tau_all), a.start, a.sigma, a.start, a.end, DENS, neff.min(), neff.max()))
    print("transfers  rms    largest distance to the smoothed share (points)")
    for s in steps:
        print("   %2d     %5.2f   %5.2f%s" % (len(s["transfers"]), 100 * s["rms"], 100 * s["largest"],
                                           "   <- drawn (first within %.1f)" % (100 * a.tol) if s is chosen else ""))
    print("levels (lane 0 = bottom of the figure), start -> end; smoothed share start -> end:")
    for k in range(K):
        print("  %d  %-44s %5.1f%% -> %5.1f%%    %5.1f%% -> %5.1f%%" % (k, labels[k], 100 * chosen["levels"][k], 100 * F[-1, k], 100 * S[0, k], 100 * S[-1, k]))
    print("transfers (source -> target, share moved, start -> end):")
    for t in transfers:
        print("  %d -> %d  %4.1f%%  %.2f -> %.2f   %s -> %s" % (t["from"], t["to"], 100 * t["share"], t["start"], t["end"],
                                                              t["from_label"].split(" / ")[0], t["to_label"].split(" / ")[0]))
    print("fit ->", out)

    if a.resample:
        rng = np.random.default_rng(20261013)
        M = len(chosen["transfers"])
        back, free, held = np.zeros(M), [[] for _ in range(M)], [[] for _ in range(M)]
        for _ in range(a.resample):
            idx = rng.integers(0, len(tau), len(tau))
            Sb, _ = smoothed_shares(tau[idx], lane[idx], T, a.sigma, K)
            again = ff.path(Sb, M)[-1]
            amounts = ff.fit(chosen["transfers"], Sb)[1]
            for m, (i, j, s0, s1) in enumerate(chosen["transfers"]):
                held[m].append(float(amounts[m]))
                for (i2, j2, a2, b2), w2 in zip(again["transfers"], again["amounts"]):
                    if (i2, j2) == (i, j) and min(s1, b2) - max(s0, a2) > 0:      # same pair, overlapping dates
                        back[m] += 1
                        free[m].append(w2)
                        break
        rows = []
        print("resampling the games %d times (same number of transfers refitted freely):" % a.resample)
        for m in order:
            i, j, s0, s1 = chosen["transfers"][m]
            h = np.array(held[m])
            rows.append({"from": i, "to": j, "share": round(chosen["amounts"][m], 6), "start": year(s0), "end": year(s1),
                         "found_again": round(float(back[m] / a.resample), 4), "amount_mean": round(float(h.mean()), 5),
                         "amount_sd": round(float(h.std()), 5), "amount_above_zero": round(float((h > 0).mean()), 4)})
            print("  %d -> %d  %4.1f%%  %.2f -> %.2f   found again in %3.0f%%; with its pair and dates held, %.1f +- %.1f points" % (
                i, j, 100 * chosen["amounts"][m], year(s0), year(s1), 100 * back[m] / a.resample, 100 * h.mean(), 100 * h.std()))
        rs = out.with_suffix(".resample.json")
        rs.write_text(json.dumps({"made_by": "figures/river_flat_fit.py --resample %d" % a.resample, "seed": 20261013,
                                  "resamples": a.resample, "transfers": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("stability ->", rs)


if __name__ == "__main__":
    main()
