# -*- coding: utf-8 -*-
r"""Periods, levels and ribbons of Figure 10 (release-share river), with no date set by hand.

In this repository (figures/trend_river/) the data file is `games_raw.js` and
the fit is written beside this script; the paths below are those of the
paper's own repository.

Position in the chain
---------------------
    river_ice_prep.py   -> river_assets/games_raw_ice.js      (data, once)
    river_shift_fit.py  -> river_assets/shift_fit_ice.json    (this script)
    river_ice_render.py -> the full-width figure the Word build embeds   (reads the fit)
    river_ice_column.py -> the one-column figure of the two-column build (reads the fit)
river_flat_fit.py, the least-squares fit the figure drew from 2026-10-13 to 2026-11-10, is kept: its
flat_fit_ice.json still draws that fit's transfers (--fit river_assets/flat_fit_ice.json on either renderer).

Why the least-squares fit was replaced (2026-11-11)
---------------------------------------------------
Release shares sum to one, so the data say how much every cluster gained or lost between two stretches of time, not
which cluster's loss became which cluster's gain; the least-squares fit picked one pairing and its dates, and they
came back in 66%, 25% and 18% of resamples of the games.  Here every step is a stated rule, a chance mechanism is
tested where there is one, and what is a drawing convention is named as such.

The rules
---------
1. Clusters.  The river page's own (seeded spherical k-means at K = 6 on the I-CE game vectors), read back from the
   page as in river_flat_fit.py.  Only the games released from --start on are used (2020.0: 1,504 of the 1,649).
2. Periods.  Between two boundaries the six release shares are taken as constant (a multinomial).  Boundaries may
   fall on month starts, with at least --minlen months (3) in a period; for a given number of boundaries the best
   placement maximises the likelihood (dynamic programming).  A boundary is added only while its gain in
   log-likelihood is larger than in 95% (--level) of --shuffles shuffles in which the release months are permuted
   among the games of each current period, the best placement being searched again in every shuffle.  The test
   says that the composition changed, not where: the fit also records where one boundary alone fits next best
   (`single_boundary`: the best month at least six months from the best one, and its gain).
3. Levels.  A band's thickness in a period is the cluster's share of that period's releases.
4. Ribbons.  Between two consecutive periods the losses are carried to the gains by the cheapest transport, the cost
   of moving share from one cluster to another being 1 - cos between the two cluster centroids (the page hands
   the costs over divided by the largest of them, which leaves the plan as it is; `cost` in the fit).  The pairing is
   therefore the least movement in the embedding that accounts for the change, a convention and not an observation.
   A movement under --min-share (one point) is not drawn; the levels drawn after a boundary are the levels before
   it plus the drawn ribbons, and the report prints how far that leaves them from the measured shares.
5. Ribbon length.  The data fix where the composition changes, not how long each movement took, so the length of a
   ribbon is a convention too: a ribbon is as long as a run of releases has to be for a shift of its size to stand
   two standard errors clear between two such runs, n = 8 p (1 - p) / w^2 games with w the share moved and p = 1/K,
   turned into years at the mean release rate, at least --mindur and at most the axis.  It is centred on its
   boundary and pushed inside the axis where it would stick out.  A large movement is thus drawn short and a small
   one long.
6. Rows.  The page orders the clusters by their mean coordinate on a one-dimensional UMAP; the row order carries no
   quantity.  --bottom moves one cluster to the bottom row (the author's placement of 2026-11-11: with the
   simulation cluster under the puzzle cluster, the three ribbons that leave the puzzle band no longer share one
   edge of it).

--test N adds the evidence behind what is drawn (-> shift_fit_ice.test.json):
  * boundaries: the games are resampled N times and the best placement searched again, for as many boundaries
    as the data have and for one boundary alone: how often a boundary falls within three months of each chosen
    one, and where one boundary alone falls (near the best month, near the runner-up, by calendar year).  The
    resamples are not put through the thresholds above: resampling with replacement inflates the likelihood
    gains (17 on average here, against 10.9 in the data), so a threshold made for shuffles would pass them;
  * ribbons: N shuffles of the release months among all the games, each given its own best boundaries (as many as
    the data have) and its own transport.  A ribbon's p value is the share of shuffles in which the same pair carries
    at least as much; `any` is the share in which some ribbon, whichever, is at least as large.  With the periods
    held, the games are resampled N times: `present` is the share of resamples in which the pair carries a point or
    more.

Run with:
    py figures/river_shift_fit.py                 # -> river_assets/shift_fit_ice.json (about 20 seconds)
    py figures/river_shift_fit.py --test 2000     # also -> river_assets/shift_fit_ice.test.json (about a minute)
Needs numpy and Chrome (the same Chrome river_ice_render.py uses).
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import river_ice_render as rir  # noqa: E402
from river_flat_fit import export_lanes  # noqa: E402

STEP = 12                      # boundaries may fall on month starts
APART = 6                      # months: how far from the best single boundary its runner-up is looked for
SEED = 20261112


# ------------------------------------------------------------------ periods
def counts(mo, ln, M, K):
    C = np.zeros((M, K))
    np.add.at(C, (mo, ln), 1)
    return C


def seg_cost_table(C):
    """cost[a, b] = minus the maximised multinomial log-likelihood of months a..b-1"""
    M, K = C.shape
    P = np.vstack([np.zeros((1, K)), np.cumsum(C, 0)])
    cost = np.full((M + 1, M + 1), np.inf)
    for a in range(M):
        n = P[a + 1:] - P[a]
        tot = n.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.where(n > 0, n * np.log(n / tot), 0.0).sum(1)
        cost[a, a + 1:] = -ll
    return cost


def best_partitions(cost, cmax, minlen):
    """for c = 0..cmax boundaries: (total cost, boundaries as month indices), best placement of each count"""
    M = cost.shape[0] - 1
    D = np.full((cmax + 2, M + 1), np.inf)            # D[s, b]: best cost of months 0..b-1 in s periods
    B = np.zeros((cmax + 2, M + 1), int)
    for b in range(minlen, M + 1):
        D[1, b] = cost[0, b]
    for s in range(2, cmax + 2):
        for b in range(s * minlen, M + 1):
            a = np.arange((s - 1) * minlen, b - minlen + 1)
            v = D[s - 1, a] + cost[a, b]
            q = int(np.argmin(v))
            D[s, b], B[s, b] = v[q], a[q]
    out = []
    for c in range(cmax + 1):
        s, b, cps = c + 1, M, []
        if not np.isfinite(D[s, M]):
            break
        while s > 1:
            b = B[s, b]
            cps.append(int(b))
            s -= 1
        out.append((float(D[c + 1, M]), sorted(cps)))
    return out


def shuffle_within(mo, cps, M, rng):
    """permute the release months among the games of each current period"""
    out = mo.copy()
    edges = [0] + list(cps) + [M]
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.nonzero((mo >= a) & (mo < b))[0]
        out[idx] = mo[rng.permutation(idx)]
    return out


def select(mo, ln, M, K, rng, minlen, nshuf, level, cmax):
    """Add boundaries while the gain beats `level` of the within-period shuffles.
    Returns the boundaries (month indices) and one record per boundary tried."""
    parts = best_partitions(seg_cost_table(counts(mo, ln, M, K)), cmax, minlen)
    c, tried = 0, []
    while c + 1 < len(parts):
        gain = parts[c][0] - parts[c + 1][0]
        g0 = np.zeros(nshuf)
        for r in range(nshuf):
            ps = best_partitions(seg_cost_table(counts(shuffle_within(mo, parts[c][1], M, rng), ln, M, K)), c + 1, minlen)
            g0[r] = ps[c][0] - ps[c + 1][0]
        thr, p = float(np.percentile(g0, 100 * level)), float((g0 >= gain).mean())
        tried.append(dict(gain=float(gain), threshold=float(thr), p=p, placement=list(parts[c + 1][1]), added=bool(gain > thr)))
        if gain <= thr:
            break
        c += 1
    return parts[c][1], tried


# ------------------------------------------------------------------ ribbons
def transport(d, cost, eps=1e-12):
    """Cheapest plan that carries the losses of d to its gains (a transportation problem), as a full [K, K] matrix.
    The problem is tiny, so every basis is tried: an optimum of a linear programme lies at a basic solution, here a
    set of sources + sinks - 1 cells whose constraint columns are independent."""
    K = len(d)
    src, dst = [i for i in range(K) if d[i] < -eps], [j for j in range(K) if d[j] > eps]
    P = np.zeros((K, K))
    if not src or not dst:
        return P
    ns, nd = len(src), len(dst)
    cells = [(a, b) for a in range(ns) for b in range(nd)]
    A = np.zeros((ns + nd, len(cells)))
    for q, (a, b) in enumerate(cells):
        A[a, q] = 1
        A[ns + b, q] = 1
    rhs = np.array([-d[i] for i in src] + [d[j] for j in dst])
    A, rhs = A[:-1], rhs[:-1]                          # the last equation follows from the others
    c = np.array([cost[src[a]][dst[b]] for a, b in cells])
    best, arg = np.inf, None
    for basis in combinations(range(len(cells)), ns + nd - 1):
        Bm = A[:, basis]
        if abs(np.linalg.det(Bm)) < 1e-9:
            continue
        x = np.linalg.solve(Bm, rhs)
        if x.min() < -1e-10:
            continue
        v = float(c[list(basis)] @ x)
        if v < best - 1e-15:
            best, arg = v, (basis, x)
    for q, x in zip(*arg):
        a, b = cells[q]
        P[src[a], dst[b]] = max(float(x), 0.0)
    return P


def plans(mo, ln, bounds, M, K, cost):
    """shares of every period and the transport between consecutive periods"""
    edges = [0] + list(bounds) + [M]
    S = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (mo >= a) & (mo < b)
        S.append(np.bincount(ln[m], minlength=K) / max(int(m.sum()), 1))
    return S, [transport(S[t + 1] - S[t], cost) for t in range(len(S) - 1)]


def ribbon_span(w, boundary, start, end, rate, K, mindur):
    """rule 5: (start date, end date, games in the run that resolves a shift of w)"""
    p = 1.0 / K
    n = 8 * p * (1 - p) / w ** 2
    d = min(end - start, max(mindur, n / rate))
    a = min(max(start, boundary - d / 2), end - d)
    return a, a + d, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(rir.DEFAULT_ASSETS))
    ap.add_argument("--data", default=rir.DEFAULT_DATA)
    ap.add_argument("--out", default=None, help="the fit (default <assets>/shift_fit_ice.json)")
    ap.add_argument("--start", type=float, default=2020.0, help="first release date used and left end of the axis")
    ap.add_argument("--end", type=float, default=2024.75, help="right end of the axis")
    ap.add_argument("--minlen", type=int, default=3, help="shortest period, months")
    ap.add_argument("--shuffles", type=int, default=10000, help="shuffles behind each boundary's threshold")
    ap.add_argument("--level", type=float, default=0.95, help="a boundary's gain must beat this share of the shuffles")
    ap.add_argument("--max-boundaries", type=int, default=6)
    ap.add_argument("--min-share", type=float, default=0.01, help="smallest movement drawn as a ribbon")
    ap.add_argument("--mindur", type=float, default=1 / 3, help="shortest ribbon, years")
    ap.add_argument("--bottom", default="Simulation/Management",
                    help="the cluster whose label starts with this is drawn in the bottom row ('' keeps the page's order)")
    ap.add_argument("--test", type=int, default=0, help="shuffles and resamples for the evidence report")
    a = ap.parse_args()
    assets = Path(a.assets).resolve()
    out = Path(a.out).resolve() if a.out else assets / "shift_fit_ice.json"

    page = export_lanes(assets, a.data)
    y0, K = page["year0"], page["k"]
    labels = [c["label"] for c in page["labels"]]
    short = [l.split(" / ")[0] for l in labels]
    cost = np.array(page["dist"], float)
    year_all, lane_all = y0 + np.array(page["tau"], float), np.array(page["lane_of"], int)
    keep = year_all >= a.start - 1e-9
    year, lane = year_all[keep], lane_all[keep]
    N = len(year)
    M = int(np.ceil((year.max() - a.start) * STEP + 1e-9))            # months covered
    month = np.minimum(((year - a.start) * STEP).astype(int), M - 1)
    rate = N / (year.max() - a.start)                                 # releases a year
    date = lambda m: a.start + m / STEP

    rng = np.random.default_rng(SEED)
    bounds, tried = select(month, lane, M, K, rng, a.minlen, a.shuffles, a.level, a.max_boundaries)
    S, P = plans(month, lane, bounds, M, K, cost)
    edges = [0] + bounds + [M]
    # one boundary alone: where it fits best, and the best month at least APART months from there
    c0 = seg_cost_table(counts(month, lane, M, K))
    gain1 = {m: float(c0[0, M] - c0[0, m] - c0[m, M]) for m in range(a.minlen, M - a.minlen + 1)}
    m1 = max(gain1, key=gain1.get)
    away = {m: g for m, g in gain1.items() if abs(m - m1) >= APART}
    m2 = max(away, key=away.get) if away else None
    single = {"best": {"date": round(date(m1), 4), "gain": round(gain1[m1], 3)},
              "runner_up": None if m2 is None else {"date": round(date(m2), 4), "gain": round(gain1[m2], 3)},
              "months_apart_at_least": APART}

    # rule 6: rows
    rows = list(range(K))
    if a.bottom:
        hit = [k for k in range(K) if labels[k].startswith(a.bottom)]
        if len(hit) != 1:
            raise SystemExit("--bottom %r matches %d cluster labels: %s" % (a.bottom, len(hit), labels))
        rest = [k for k in range(K) if k != hit[0]]
        for r, k in enumerate([hit[0]] + rest):
            rows[k] = r

    transfers, left_out = [], []
    drawn = [np.array(S[0], float)]
    for t, b in enumerate(bounds):
        level = drawn[-1].copy()
        for i in range(K):
            for j in range(K):
                w = float(P[t][i, j])
                if w <= 1e-12:
                    continue
                if w < a.min_share:
                    left_out.append({"boundary": round(date(b), 4), "from": i, "to": j, "share": round(w, 5)})
                    continue
                s, e, n = ribbon_span(w, date(b), a.start, a.end, rate, K, a.mindur)
                transfers.append({"from": i, "to": j, "from_label": labels[i], "to_label": labels[j], "share": round(w, 5),
                                  "start": round(s, 4), "end": round(e, 4), "boundary": round(date(b), 4),
                                  "games_to_resolve": int(round(n))})
                level[i] -= round(w, 5)
                level[j] += round(w, 5)
        drawn.append(level)
    transfers.sort(key=lambda x: (x["start"], x["end"]))
    gap = max(float(np.abs(drawn[t] - S[t]).max()) for t in range(len(S)))

    fit = {
        "made_by": "figures/river_shift_fit.py", "method": "shift", "data": a.data, "year0": y0,
        "games_total": int(len(year_all)), "games_used": int(N), "start": a.start, "end": a.end,
        "last_release": round(float(year.max()), 4), "releases_per_year": round(float(rate), 2),
        "labels": labels, "lane_games_used": [int((lane == k).sum()) for k in range(K)], "rows": rows,
        "minlen_months": a.minlen, "shuffles": a.shuffles, "level": a.level, "seed": SEED,
        "min_share": a.min_share, "min_transfer_years": a.mindur,
        "boundaries": [round(date(b), 4) for b in bounds],
        "boundary_tests": [dict(gain=round(x["gain"], 3), threshold=round(x["threshold"], 3), p=x["p"], added=x["added"],
                                placement=[round(date(m), 4) for m in x["placement"]]) for x in tried],
        "single_boundary": single,
        "periods": [{"from": round(date(e0), 4), "to": round(date(e1), 4), "games": int(((month >= e0) & (month < e1)).sum()),
                     "shares": [round(float(v), 5) for v in S[t]]}
                    for t, (e0, e1) in enumerate(zip(edges[:-1], edges[1:]))],
        "cost": [[round(float(v), 5) for v in row] for row in cost],
        "levels_start": [round(float(x), 5) for x in S[0]],
        "levels_end": [round(float(x), 5) for x in drawn[-1]],
        "transfers": transfers, "left_out": left_out, "largest_gap_to_measured": round(gap, 5),
    }
    out.write_text(json.dumps(fit, ensure_ascii=False, indent=1), encoding="utf-8")

    print("games %d of %d released from %.2f on, the last on %.3f: %d months, %.0f releases a year" % (
        N, len(year_all), a.start, year.max(), M, rate))
    print("boundaries (shortest period %d months, %d shuffles, level %.0f%%):" % (a.minlen, a.shuffles, 100 * a.level))
    for q, x in enumerate(tried):
        print("   boundary %d: gain %.2f, threshold %.2f (p = %.4f) -> %s   best placement %s" % (
            q + 1, x["gain"], x["threshold"], x["p"], "added" if x["added"] else "not added",
            ["%.2f" % date(m) for m in x["placement"]]))
    print("   one boundary alone fits best at %.2f (gain %.2f)%s" % (
        date(m1), gain1[m1], "" if m2 is None else "; next best, at least %d months away: %.2f (gain %.2f)" % (APART, date(m2), gain1[m2])))
    print("periods and release shares (points):")
    for pr in fit["periods"]:
        print("   %.2f to %.2f  %4d games   " % (pr["from"], pr["to"], pr["games"])
              + "  ".join("%s %.1f" % (short[k][:4], 100 * pr["shares"][k]) for k in range(K)))
    print("ribbons (source -> target, share moved, drawn from -> to, games in the run that resolves the shift):")
    for t in transfers:
        print("  %d -> %d  %4.1f  %.2f -> %.2f  (%.2f y, %d games)   %s -> %s" % (
            t["from"], t["to"], 100 * t["share"], t["start"], t["end"], t["end"] - t["start"], t["games_to_resolve"],
            short[t["from"]], short[t["to"]]))
    for t in left_out:
        print("  not drawn (under %.1f): %s -> %s %.2f" % (100 * a.min_share, short[t["from"]], short[t["to"]], 100 * t["share"]))
    print("levels drawn stay within %.2f points of the measured shares" % (100 * gap))
    print("rows, bottom to top:", [short[k] for k in sorted(range(K), key=lambda k: rows[k])])
    print("fit ->", out)

    if a.test:
        T = len(bounds)
        near, first = [[] for _ in bounds], np.zeros(a.test, int)
        Pn, Pb = np.zeros((a.test, T, K, K)), np.zeros((a.test, T, K, K))
        for r in range(a.test):
            idx = rng.integers(0, N, N)
            ps = best_partitions(seg_cost_table(counts(month[idx], lane[idx], M, K)), max(T, 1), a.minlen)
            first[r] = ps[1][1][0]
            for q, b in enumerate(bounds):
                near[q].append(min(ps[T][1], key=lambda m: abs(m - b)))
            if T:
                Pb[r] = plans(month[idx], lane[idx], bounds, M, K, cost)[1]
                ms = rng.permutation(month)
                bs = best_partitions(seg_cost_table(counts(ms, lane, M, K)), T, a.minlen)[T][1]
                Pn[r] = plans(ms, lane, bs, M, K, cost)[1]
        yr0 = int(np.floor(a.start))
        by_year = {str(yr0 + y): round(float(((first // STEP) == y).mean()), 4) for y in range(int(np.ceil(M / STEP)))}
        rep = {"made_by": "figures/river_shift_fit.py --test %d" % a.test, "seed": SEED, "runs": a.test, "boundaries": [],
               "single_boundary": {"best": single["best"]["date"], "within_3_months": round(float((np.abs(first - m1) <= 3).mean()), 4),
                                   "runner_up": None if m2 is None else single["runner_up"]["date"],
                                   "runner_up_within_3_months": None if m2 is None else round(float((np.abs(first - m2) <= 3).mean()), 4),
                                   "by_year": by_year},
               "ribbons": []}
        print("evidence, %d shuffles and %d resamples:" % (a.test, a.test))
        for q, b in enumerate(bounds):
            off = np.abs(np.array(near[q]) - b)
            rep["boundaries"].append({"date": round(date(b), 4), "within_3_months": round(float((off <= 3).mean()), 4),
                                      "within_6_months": round(float((off <= 6).mean()), 4)})
            print("   the best %d boundar%s of a resample: one within 3 months of %.2f in %.0f%% of the resamples, within 6 in %.0f%%" % (
                T, "y" if T == 1 else "ies", date(b), 100 * (off <= 3).mean(), 100 * (off <= 6).mean()))
        sb = rep["single_boundary"]
        print("   one boundary alone, in the resamples: within 3 months of %.2f in %.0f%%%s; by calendar year %s" % (
            sb["best"], 100 * sb["within_3_months"],
            "" if m2 is None else ", of %.2f in %.0f%%" % (sb["runner_up"], 100 * sb["runner_up_within_3_months"]), by_year))
        if T:
            top = Pn.reshape(a.test, -1).max(1)
            rep["largest_null_ribbon_95"] = round(float(np.percentile(top, 95)), 5)
            print("   largest ribbon drawn from shuffled dates: under %.1f points in 95%% of the shuffles" % (100 * np.percentile(top, 95)))
            for t in transfers:
                q = bounds.index(int(round((t["boundary"] - a.start) * STEP)))
                i, j, w = t["from"], t["to"], t["share"]
                row = {"from": i, "to": j, "share": w, "boundary": t["boundary"],
                       "p_pair": round(float((Pn[:, q, i, j] >= w).mean()), 4), "p_any": round(float((top >= w).mean()), 4),
                       "present": round(float((Pb[:, q, i, j] >= a.min_share).mean()), 4)}
                rep["ribbons"].append(row)
                print("   %-12s -> %-22s %4.1f   p (this pair) %.3f   p (any ribbon this large) %.3f   present in %3.0f%% of the resamples" % (
                    short[i], short[j], 100 * w, row["p_pair"], row["p_any"], 100 * row["present"]))
        ts = out.with_suffix(".test.json")
        ts.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("evidence ->", ts)


if __name__ == "__main__":
    main()
