"""Paper checkpoint selection from saved projections; no training dependencies."""
import json
from pathlib import Path

import numpy as np

SELECTOR = "val_neutral_noname_sum_exp_minus_rank_v1"


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def query_ranks(queries, gallery, own):
    """Float32 cosine ranks, 1-based; equal similarities share the best rank.

    A multi-pack gallery [K, games, dim] uses max cosine over packs, matching
    the worker. This is distinct from averaging K independent evaluations.
    """
    q = np.asarray(queries, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    own = np.asarray(own)
    if q.ndim != 2 or g.ndim not in (2, 3) or q.shape[-1] != g.shape[-1]:
        raise ValueError("incompatible query/gallery shapes")
    if not q.shape[0] or not g.size or own.shape != (len(q),):
        raise ValueError("empty queries/gallery or incompatible targets")
    if not np.issubdtype(own.dtype, np.integer) or np.any(own < 0) or np.any(own >= g.shape[-2]):
        raise ValueError("targets must be integer gallery indices")
    if not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("non-finite projection values")
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
    sim = q @ g.T if g.ndim == 2 else np.maximum.reduce([q @ pack.T for pack in g])
    return (sim > sim[np.arange(len(own)), own, None]).sum(1) + 1


def rank_score(gallery, articles, targets, rows):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise ValueError("selection rows must be nonempty and unique")
    if len(articles) != len(targets) or np.any(rows < 0) or np.any(rows >= len(articles)):
        raise ValueError("selection rows/targets do not match article projections")
    ranks = query_ranks(articles[rows], gallery, np.asarray(targets)[rows])
    return float(np.exp(-ranks.astype(np.float64)).sum())


def best_epoch(scores):
    """Use only rank scores; prefer the earlier checkpoint on an exact tie."""
    if not scores or not all(np.isfinite(s) for s in scores.values()):
        raise ValueError("no valid checkpoint scores")
    return max(scores, key=lambda ep: (scores[ep], -int(ep)))


def validation_rows(names, article_names, variants, targets, split, fold=None,
                    n_folds=5, cv_seed=20260711):
    """Reconstruct fixed/CV validation membership without loading embeddings."""
    names = list(map(str, names))
    article_names = list(map(str, article_names))
    variants = list(map(str, variants))
    appids = [n.split("_")[0] for n in names]
    if len(set(names)) != len(names) or len(set(appids)) != len(appids):
        raise ValueError("duplicate gallery names/appids")
    n2i = {name: i for i, name in enumerate(names)}
    a2n = dict(zip(appids, names))
    parts = [set(map(str, split[key])) for key in ("train", "val", "test")]
    if any(parts[i] & parts[j] for i in range(3) for j in range(i)):
        raise ValueError("train/validation/test appids overlap")
    universe = sorted(set.union(*parts))
    if not set(universe) <= a2n.keys():
        raise ValueError("split contains unknown gallery appids")
    if fold is None:
        val = parts[1]
    else:
        if n_folds < 3 or not 0 <= fold < n_folds or len(universe) < n_folds:
            raise ValueError("invalid CV fold configuration")
        folds = np.array_split(np.random.default_rng(cv_seed).permutation(len(universe)), n_folds)
        val = {universe[i] for i in folds[(fold + 1) % n_folds]}
    val_names = {a2n[a] for a in val}
    if len(article_names) != len(variants) or len(targets) != len(article_names):
        raise ValueError("query metadata lengths disagree")
    rows = [i for i, (name, var) in enumerate(zip(article_names, variants))
            if name in val_names and var in ("neutral", "noname")]
    pairs = [(article_names[i], variants[i]) for i in rows]
    expected = {(name, var) for name in val_names for var in ("neutral", "noname")}
    if not expected or len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("validation must contain exactly one neutral and noname query per game")
    if any(int(targets[i]) != n2i[article_names[i]] for i in rows):
        raise ValueError("query target indices disagree with gallery name order")
    return np.asarray(rows, dtype=np.int64)


def projection_files(directory, name, max_epoch=None):
    files = {}
    for path in Path(directory).glob(f"tower_{name}_ep*.npz"):
        ep = int(path.stem.rsplit("_ep", 1)[1])
        if ep > 0 and (max_epoch is None or ep <= max_epoch):
            files[ep] = path
    if not files:
        raise ValueError(f"no saved projections for {name}")
    return dict(sorted(files.items()))


def refresh_rank_selection(directory, name, targets, rows, max_epoch=None):
    """Worker integration: upgrade old trajectories from SPg/SPa, never tags.

    Recompute from the current files, so a legacy cvsel/rvsel cache or a stale
    entry without a projection cannot decide the new winner.
    """
    directory = Path(directory)
    traj_path = directory / f"zs_traj_{name}.json"
    traj = json.loads(traj_path.read_text()) if traj_path.exists() else {}
    scores = {}
    for ep, path in projection_files(directory, name, max_epoch).items():
        with np.load(path, allow_pickle=False) as z:
            scores[ep] = rank_score(z["SPg"], z["SPa"], targets, rows)
        traj.setdefault(f"ep{ep}", {}).update(ranksel=scores[ep], selector=SELECTOR)
    ep = best_epoch(scores)
    result = dict(traj[f"ep{ep}"], best_ep=ep)
    atomic_json(traj_path, traj)
    atomic_json(directory / f"zsbest_{name}.json", result)
    return result
