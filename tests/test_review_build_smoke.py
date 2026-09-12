"""Offline release-entry smoke tests (no corpus downloads or training).

Run: python -m unittest discover -s tests -p test_review_build_smoke.py -v
Set LARICE_SMOKE_MODELS=1 to also use already-cached SaT/Qwen models on two games.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "dataset_builder" / "reviews" / "build.py"


class ReviewBuildSmoke(unittest.TestCase):
    def setUp(self):
        parent = os.environ.get("LARICE_SMOKE_ROOT")
        if parent:
            Path(parent).mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="review-smoke-", dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                        HF_DATASETS_OFFLINE="1", PYTHONIOENCODING="utf-8",
                        PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        self.env.pop("PYTHONPATH", None)

    def command(self, args, cwd=None, success=True, timeout=180):
        result = subprocess.run([sys.executable, "-B", *map(str, args)],
                                cwd=cwd or self.work, env=self.env,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def fixture(self):
        data = self.work / "tiny data"
        source = data / "Steam Games Metadata and Player Reviews (2020–2024"
        reviews = source / "Game Reviews"
        reviews.mkdir(parents=True)
        games = {}
        for appid, tag in (("10", "Action"), ("20", "Puzzle")):
            games[appid] = {"name": f"Smoke Game {appid}", "tags": {tag: 10},
                            "short_description": "Explore a small island. Solve its mysteries.",
                            "about_the_game": "Choose your route through the forest.",
                            "detailed_description": "Discover hidden paths and collect useful items.",
                            "release_date": "2024-01-01"}
            with (reviews / f"{appid}_2.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["review", "recommend"])
                writer.writeheader()
                writer.writerows([
                    {"review": "I enjoyed exploring the island. The puzzles were fun.", "recommend": "Recommended"},
                    {"review": "The movement felt slow. I wanted shorter battles.", "recommend": "Not Recommended"}])
        (source / "games.json").write_text(json.dumps(games), encoding="utf-8")
        return data

    def build(self, data, stages, extra=(), module=False, **kw):
        entry = ["-m", "dataset_builder.reviews.build"] if module else [BUILD]
        return self.command([*entry, "--data-dir", data, "--skip-source2",
                             "--only", *stages, "--min-length", "10", "--min-count", "1",
                             "--metadata-workers", "1", "--text-workers", "1", *extra],
                            cwd=ROOT if module else self.work, **kw)

    def test_help_direct_and_module_without_original_checkout(self):
        for script in ("build.py", "embedding_data.py", "embedding_incloud.py"):
            with self.subTest(script=script):
                result = self.command(["-I", ROOT / "dataset_builder/reviews" / script, "--help"])
                self.assertIn("usage:", result.stdout)
        for module in ("dataset_builder.reviews.build", "dataset_builder.reviews.embedding_data"):
            with self.subTest(module=module):
                self.assertIn("usage:", self.command(["-m", module, "--help"], cwd=ROOT).stdout)

    def test_metadata_resume_and_default_tag_h5(self):
        data = self.fixture()
        log = self.work / "build.log"
        self.build(data, ["metadata"], ["--logout-address", log], module=True)
        self.assertIn("build done", log.read_text(encoding="utf-8"))
        metadata = sorted((data / "metadata").glob("*.json"))
        self.assertEqual(len(metadata), 2)
        self.assertEqual(len(json.loads(metadata[0].read_text(encoding="utf-8"))), 5)
        before = {p: p.read_bytes() for p in metadata}
        resumed = self.build(data, ["metadata"])
        self.assertIn("existing metadata files skipped", resumed.stdout)
        self.assertTrue(all(p.read_bytes() == contents for p, contents in before.items()))

        sentences = data / "sentences"
        sentences.mkdir()
        for path in metadata:
            (sentences / path.name).write_text(json.dumps({
                "3": {"sentence_0": {"sentence_text": "Explore the island."},
                      "sentence_1": {"sentence_text": "Solve the puzzles."}},
                "4": {"sentence_0": {"sentence_text": "Movement is slow."},
                      "sentence_1": {"sentence_text": "Battles are long."}}}), encoding="utf-8")
        self.build(data, ["text-h5"])
        with h5py.File(data / "text_h5.h5", "r") as h:
            self.assertEqual(h["texts"].shape, (8,))
            self.assertEqual(h["tag_labels"].shape, (2, 23))
            tag_names = list(h["tag_names"].asstr()[:])
            labels = h["tag_labels"][:]
            self.assertEqual(int(labels[0, tag_names.index("Action/Adventure")]), 1)
            self.assertEqual(int(labels[1, tag_names.index("Puzzle")]), 1)
            np.testing.assert_array_equal(h["game_review_offsets"][:], [0, 2, 4])
            np.testing.assert_array_equal(h["review_offsets"][:], [0, 2, 4, 6, 8])
            np.testing.assert_array_equal(h["positive"][:], [1, 1])
            np.testing.assert_array_equal(h["negative"][:], [1, 1])
        self.assertFalse(list(data.glob("*.tmp")))

    def test_tee_captures_stdout_stderr_and_child_process(self):
        log = self.work / "child.log"
        code = ("import sys,subprocess; "
                f"sys.path.insert(0,{str(ROOT)!r}); "
                "from dataset_builder.logging_tee import maybe_tee; "
                f"ctx=maybe_tee({str(log)!r}); ctx.__enter__(); "
                "print('parent-output',flush=True); print('parent-error',file=sys.stderr,flush=True); "
                "subprocess.run([sys.executable,'-c',\"print('child-output')\"],check=True); "
                "ctx.__exit__(None,None,None)")
        result = self.command(["-c", code])
        for marker in ("parent-output", "parent-error", "child-output"):
            self.assertIn(marker, result.stdout)
            self.assertIn(marker, log.read_text(encoding="utf-8"))

    def test_failure_is_logged_and_returns_nonzero(self):
        log = self.work / "failed.log"
        result = self.command([BUILD, "--data-dir", self.work / "empty", "--skip-source1",
                               "--skip-source2", "--only", "metadata", "--logout-address", log],
                              success=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("No review CSV directories found", log.read_text(encoding="utf-8"))
        self.assertNotIn("ModuleNotFoundError", result.stdout + result.stderr)

    def test_package_cloud_backend_imports_without_network(self):
        # Exercise package imports with explicit synthetic credentials and a
        # mocked HTTP client. No real credential file or service is accessed.
        code = ("import sys; from unittest.mock import patch; "
                f"sys.path.insert(0,{str(ROOT)!r}); "
                "from dataset_builder.reviews.embedding_data import make_embedder; "
                "from dataset_builder.reviews import cloud_embedding; "
                "p=patch.object(cloud_embedding,'load_credentials',return_value={'url':'https://example.invalid','token':'test-only'}); "
                "c=patch.object(cloud_embedding,'EmbeddingClient'); p.start(); c.start(); "
                "embedder,_,backend=make_embedder('cloud',concurrency=1); "
                "assert backend=='cloud'; embedder.close(); c.stop(); p.stop(); print('cloud import OK')")
        self.assertIn("cloud import OK", self.command(["-I", "-c", code]).stdout)

    @unittest.skipUnless(os.environ.get("LARICE_SMOKE_MODELS") == "1", "cached-model smoke is opt-in")
    def test_two_games_with_real_cached_splitter_and_embedder(self):
        data = self.fixture()
        device = os.environ.get("LARICE_SMOKE_DEVICE", "cpu")
        result = self.build(data, ["metadata", "split", "text-h5", "embed-h5"],
                            ["--split-workers", "1", "--split-device", device,
                             "--split-batch-size", "2", "--split-outer-batch-size", "2",
                             "--backend", "local", "--embed-device", device,
                             "--batch-size", "2", "--read-batch-size", "4",
                             "--logout-address", self.work / "real-models.log"], timeout=240)
        self.assertIn("build done", result.stdout)
        with h5py.File(data / "text_h5.h5", "r") as text_h5, h5py.File(data / "embedding_h5.h5", "r") as embedded:
            n = len(text_h5["texts"])
            self.assertGreater(n, 0)
            self.assertLess(n, 100)
            self.assertEqual(embedded["vectors"].shape, (n, 1024))
            vectors = embedded["vectors"][:]
            self.assertTrue(np.isfinite(vectors).all())
            self.assertTrue((np.linalg.norm(vectors.astype(np.float32), axis=1) > 0).all())
            for key in ("game_review_offsets", "review_offsets", "tag_labels", "release_date"):
                np.testing.assert_array_equal(embedded[key][:], text_h5[key][:])
            self.assertEqual(embedded["tag_labels"].shape, (2, 23))
            self.assertEqual(embedded.attrs["embedding_backend"], "local")
            print(f"\nCached-model smoke: 2 games, {n} sentences, vectors={vectors.shape}, tags=(2, 23)")
        self.assertFalse(list(data.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
