"""Reselect existing w9 checkpoints without training or writing to the source.

Only SPg/SPa arrays and query metadata are read. Results live in --output-dir.
An optional old plan scopes the runs and provides a comparison; its chosen
epochs and scores are never inputs to the new ranking.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

try:
    from .w9_selection import (SELECTOR, atomic_json, best_epoch, projection_files,
                               rank_score, validation_rows)
except ImportError:
    from w9_selection import (SELECTOR, atomic_json, best_epoch, projection_files,
                              rank_score, validation_rows)


def separate_output(output, protected):
    """Resolve junctions/symlinks before enforcing read-only input trees."""
    out = Path(output).resolve()
    for root in protected:
        root = Path(root).resolve()
        if out == root or root in out.parents or out in root.parents:
            raise ValueError(f"output and input trees must be separate: {out} / {root}")
    # Existing redirected result files could otherwise bypass the directory check.
    if out.exists():
        for path in out.rglob("*"):
            resolved = path.resolve()
            if out not in resolved.parents:
                raise ValueError(f"redirected output path: {path}")
    return out


def signature(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True, help="read-only source tree")
    ap.add_argument("--projection-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, action="append", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--reference-plan", type=Path, help="scope runs and compare old epochs")
    ap.add_argument("--run", action="append", help="exact run name without tower_/epoch")
    ap.add_argument("--max-epoch", type=int, default=2000)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--allow-short-runs", action="store_true",
                    help="accept contiguous shorter runs; report their actual horizon")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=20260711)
    ap.add_argument("--expected-gallery-size", type=int, default=2020)
    ap.add_argument("--overwrite", action="store_true", help="ignore per-run score caches")
    args = ap.parse_args(argv)
    if args.max_epoch <= 0 or args.checkpoint_every <= 0:
        ap.error("epoch limits must be positive")
    protected = [args.source_root, args.projection_dir, args.cache_dir, *args.checkpoint_dir]
    if args.reference_plan:
        protected.append(args.reference_plan.parent)
    out = separate_output(args.output_dir, protected)
    with np.load(args.cache_dir / "games.npz", allow_pickle=True) as g:
        names = g["names"]
    with np.load(args.cache_dir / "wiki_eval.npz", allow_pickle=True) as a:
        article_names, variants, targets = a["names"], a["variants"], a["gidx"]
    if len(names) != args.expected_gallery_size:
        raise ValueError(f"gallery has {len(names)} games, expected {args.expected_gallery_size}")
    split = json.loads((args.cache_dir / "wiki_eval_split.json").read_text())
    reference = json.loads(args.reference_plan.read_text()) if args.reference_plan else {}
    runs = {}
    for key, record in reference.items():
        match = re.fullmatch(r"ckpt_(.+)_ep\d+\.pt", record["ckpt"])
        if not match:
            raise ValueError(f"invalid reference checkpoint: {record['ckpt']}")
        name = match[1]
        if name in runs:
            raise ValueError(f"duplicate reference run: {name}")
        runs[name] = (key, record)
    if not reference:
        runs = {p.stem[6:].rsplit("_ep", 1)[0]: (p.stem[6:].rsplit("_ep", 1)[0], {})
                for p in args.projection_dir.glob("tower_*_ep*.npz")}
    if args.run:
        missing = set(args.run) - runs.keys()
        if missing:
            raise ValueError(f"unknown runs: {sorted(missing)}")
        runs = {name: runs[name] for name in args.run}
    if not runs:
        raise ValueError("no runs found")
    plan, summary = {}, {"selector": SELECTOR, "status": "running", "runs": {},
                         "source_root": str(args.source_root.resolve()),
                         "projection_dir": str(args.projection_dir.resolve()),
                         "evaluation": "saved single-draw projections; not K10 table recomputation"}
    start = time.monotonic()
    try:
        for number, (name, (key, old)) in enumerate(sorted(runs.items()), 1):
            fold_match = re.search(r"_fold(\d+)(?:_|$)", name)
            fold = int(fold_match[1]) if fold_match else None
            if name.startswith("w9cv_") and fold is None:
                raise ValueError(f"missing fold in CV run name: {name}")
            if "fold" in old and old["fold"] != fold:
                raise ValueError(f"reference fold disagrees with run name: {name}")
            rows = validation_rows(names, article_names, variants, targets, split,
                                   fold, args.n_folds, args.cv_seed)
            files = projection_files(args.projection_dir, name, args.max_epoch)
            horizon = max(files) if args.allow_short_runs else args.max_epoch
            expected = set(range(args.checkpoint_every, horizon + 1, args.checkpoint_every))
            if set(files) != expected:
                raise ValueError(f"incomplete epoch grid for {name}: missing {sorted(expected - files.keys())}")
            inputs = {str(ep): signature(path) for ep, path in files.items()}
            row_hash = hashlib.sha256(rows.tobytes() + targets[rows].tobytes()).hexdigest()
            identity = {"selector": SELECTOR, "inputs": inputs, "rows_sha256": row_hash,
                        "gallery_size": len(names), "query_count": len(targets)}
            score_path = out / f"rank_traj_{name}.json"
            cached = json.loads(score_path.read_text()) if score_path.exists() and not args.overwrite else {}
            reused = cached.get("identity") == identity
            if reused:
                scores = {int(ep): value for ep, value in cached["scores"].items()}
                if set(scores) != set(files):
                    raise ValueError(f"incomplete score cache: {score_path}; use --overwrite")
            else:
                scores = {}
                for ep, path in files.items():
                    with np.load(path, allow_pickle=False) as z:
                        gallery, articles = z["SPg"], z["SPa"]
                    if gallery.shape[-2] != len(names):
                        raise ValueError(f"gallery size mismatch: {path}")
                    scores[ep] = rank_score(gallery, articles, targets, rows)
                if inputs != {str(ep): signature(path) for ep, path in files.items()}:
                    raise ValueError(f"source projections changed during evaluation: {name}")
                atomic_json(score_path, {"identity": identity, "scores": scores})
            ep = best_epoch(scores)
            ckpt = f"ckpt_{name}_ep{ep}.pt"
            found = [directory / ckpt for directory in args.checkpoint_dir if (directory / ckpt).is_file()]
            if not found:
                raise FileNotFoundError(f"selected checkpoint missing: {ckpt}")
            record = {k: v for k, v in old.items() if k not in ("ep", "ckpt", "checkpoint_path")}
            record.update(ep=ep, ckpt=ckpt, checkpoint_path=str(found[0].resolve()))
            plan[key] = record
            result = {"best_ep": ep, "ranksel": scores[ep], "selector": SELECTOR,
                      "checkpoint_path": record["checkpoint_path"], "validation_queries": len(rows),
                      "candidates": len(scores), "horizon": horizon,
                      "reference_ep": old.get("ep"), "matches_reference": ep == old.get("ep") if old else None}
            atomic_json(out / f"zsbest_{name}.json", result)
            summary["runs"][key] = result
            atomic_json(out / "rank_ckpt_plan.json", plan)
            atomic_json(out / "selection_summary.json", summary)
            print(f"[{number}/{len(runs)}] {name}: ep{ep}, score={scores[ep]:.9f}, "
                  f"reference={old.get('ep')}, candidates={len(scores)}, resumed={reused}", flush=True)
        summary.update(status="complete", elapsed_seconds=round(time.monotonic() - start, 2),
                       run_count=len(plan), candidate_count=sum(r["candidates"] for r in summary["runs"].values()),
                       reference_matches=sum(r["matches_reference"] is True for r in summary["runs"].values()))
    except Exception as exc:
        summary.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - start, 2))
        raise
    finally:
        atomic_json(out / "selection_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
