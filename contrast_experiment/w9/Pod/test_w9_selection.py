"""Small, CPU-only smoke tests; no corpus, CUDA training, or network needed."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

import numpy as np

from w9_reselect import main, separate_output
from w9_selection import (SELECTOR, best_epoch, query_ranks, rank_score,
                           refresh_rank_selection, validation_rows)


class SelectionTests(unittest.TestCase):
    def test_rank_formula_and_ties(self):
        gallery = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        query = np.array([[1., 0.], [1., 0.], [1., 0.]])
        own = np.arange(3)
        np.testing.assert_array_equal(query_ranks(query, gallery, own), [1, 2, 3])
        self.assertAlmostEqual(rank_score(gallery, query, own, [0, 1, 2]),
                               np.exp(-1.) + np.exp(-2.) + np.exp(-3.))
        self.assertEqual(best_epoch({100: 1., 50: 1.}), 50)
        np.testing.assert_array_equal(query_ranks([[1., 0.]], [[1., 0.], [2., 0.]], [1]), [1])

    def test_multi_pack_semantics(self):
        gallery = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
        np.testing.assert_array_equal(query_ranks([[1., 0.]], gallery, [1]), [1])

    def test_reject_corrupt_projections(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rank_score(np.eye(2), np.array([[float('nan'), 0.]]), [0], [0])
        with self.assertRaisesRegex(ValueError, "targets"):
            query_ranks(np.eye(2), np.eye(2), [0, 2])
        with self.assertRaisesRegex(ValueError, "unique"):
            rank_score(np.eye(2), np.eye(2), [0, 1], [0, 0])

    def setUp(self):
        self.names = np.array([f"{i}_game" for i in range(10)])
        self.art_names = np.repeat(self.names, 2)
        self.variants = np.tile(["neutral", "noname"], 10)
        self.targets = np.repeat(np.arange(10), 2)
        self.split = {"train": list(map(str, range(6))), "val": ["6", "7"],
                      "test": ["8", "9"], "seed": 20260711}

    def rows(self, fold=None):
        return validation_rows(self.names, self.art_names, self.variants,
                               self.targets, self.split, fold)

    def test_fixed_and_cv_splits(self):
        np.testing.assert_array_equal(self.rows(), [12, 13, 14, 15])
        # Worker protocol: test fold k, validation fold (k+1)%5.
        folds = np.array_split(np.random.default_rng(20260711).permutation(10), 5)
        for fold in range(5):
            rows = self.rows(fold)
            self.assertEqual(set(self.targets[rows]), set(folds[(fold + 1) % 5]))
            self.assertFalse(set(self.targets[rows]) & set(folds[fold]))
        self.variants[12] = "positive"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.rows()

    def test_reject_split_overlap_and_wrong_gallery_order(self):
        self.targets[12] = 0
        with self.assertRaisesRegex(ValueError, "gallery name order"):
            self.rows()
        self.split["test"].append("6")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.rows()

    def test_test_queries_and_legacy_metrics_cannot_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            gallery = np.eye(10, dtype=np.float32)
            good = gallery[self.targets].copy()
            bad = good.copy()
            bad[self.rows()] = gallery[0]
            name = "w9_example"
            np.savez(directory / f"tower_{name}_ep50.npz", SPg=gallery, SPa=bad)
            np.savez(directory / f"tower_{name}_ep100.npz", SPg=gallery, SPa=good)
            traj = {"ep50": {"cvsel": 999, "rvsel": 999, "val_tag": 1},
                    "ep100": {"cvsel": 0, "rvsel": 0, "val_tag": 0},
                    "ep150": {"ranksel": 10000, "selector": SELECTOR}}
            (directory / f"zs_traj_{name}.json").write_text(json.dumps(traj))
            result = refresh_rank_selection(directory, name, self.targets, self.rows())
            self.assertEqual(result["best_ep"], 100)
            # Replace every non-validation query, including all test queries.
            other = np.setdiff1d(np.arange(20), self.rows())
            good[other] = -gallery[self.targets[other]]
            np.savez(directory / f"tower_{name}_ep100.npz", SPg=gallery, SPa=good)
            updated = refresh_rank_selection(directory, name, self.targets, self.rows())
            self.assertEqual(updated["ranksel"], result["ranksel"])
            self.assertEqual(updated["best_ep"], 100)

    def test_protected_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            for target in (source, source / "nested", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "separate"):
                    separate_output(target, [source])
            self.assertEqual(separate_output(Path(tmp) / "out", [source]), Path(tmp) / "out")

    def test_both_worker_selection_blocks(self):
        # Execute the actual workers' final selection blocks without loading
        # their training machinery or large sentence tensors.
        for filename, fold in (("w9_cv_worker.py", 0), ("w9_a100_worker.py", None)):
            with self.subTest(worker=filename), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                name = "worker_fixture"
                gallery = np.eye(10, dtype=np.float32)
                good = gallery[self.targets]
                bad = good.copy()
                bad[self.rows(fold)] = -bad[self.rows(fold)]
                for ep, articles in ((50, bad), (100, good)):
                    np.savez(directory / f"tower_{name}_ep{ep}.npz", SPg=gallery, SPa=articles)
                (directory / f"zs_traj_{name}.json").write_text(json.dumps({
                    "ep50": {"val_tag": 1, "rvsel": 999},
                    "ep100": {"val_tag": 0, "rvsel": 0}}))
                source = (Path(__file__).parent / filename).read_text(encoding="utf-8")
                start = source.index("    # Refresh historical readout columns")
                end = source.index("    # ---------------- heads (LEGACY", start)
                namespace = dict(__name__="__main__", __package__="", OUT=directory, name=name, np=np, json=json,
                                 names=self.names, art_games=self.art_names, variants=self.variants,
                                 A={"gidx": self.targets}, sp=self.split,
                                 args=SimpleNamespace(fold=0, n_folds=5, cv_seed=20260711))
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(textwrap.dedent(source[start:end]), filename, "exec"), namespace)
                result = json.loads((directory / f"zsbest_{name}.json").read_text())
                self.assertEqual(result["best_ep"], 100)
                self.assertEqual(result["selector"], SELECTOR)
                self.assertEqual(result["rvsel"], 0)

    def test_cli_reselection_resume_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out = root / "source", root / "output"
            source.mkdir()
            np.savez(source / "games.npz", names=self.names)
            np.savez(source / "wiki_eval.npz", names=self.art_names, variants=self.variants,
                     gidx=self.targets)
            (source / "wiki_eval_split.json").write_text(json.dumps(self.split))
            name = "w9cv_wcle_example_fold0_fp"
            good = np.eye(10, dtype=np.float32)[self.targets]
            bad = good.copy()
            bad[self.rows(0)] = -bad[self.rows(0)]
            for ep, articles in ((50, bad), (100, good)):
                np.savez(source / f"tower_{name}_ep{ep}.npz", SPg=np.eye(10), SPa=articles)
                (source / f"ckpt_{name}_ep{ep}.pt").write_bytes(b"fixture checkpoint")
            # The reference intentionally selects the wrong epoch: comparison only.
            (source / "reference.json").write_text(json.dumps({"example": {
                "fold": 0, "ep": 50, "ckpt": f"ckpt_{name}_ep50.pt"}}))
            args = ["--source-root", str(source), "--projection-dir", str(source),
                    "--cache-dir", str(source), "--checkpoint-dir", str(source),
                    "--output-dir", str(out), "--max-epoch", "100",
                    "--expected-gallery-size", "10", "--reference-plan", str(source / "reference.json")]
            def hashes():
                return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            before = hashes()
            with contextlib.redirect_stdout(io.StringIO()) as log:
                main(args)
                main(args)
            result = json.loads((out / "selection_summary.json").read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["runs"]["example"]["best_ep"], 100)
            self.assertFalse(result["runs"]["example"]["matches_reference"])
            self.assertIn("resumed=True", log.getvalue())
            self.assertEqual(hashes(), before)
            (source / f"tower_{name}_ep50.npz").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete epoch grid"):
                main(args)
            self.assertEqual(json.loads((out / "selection_summary.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
