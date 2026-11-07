# -*- coding: utf-8 -*-
"""The DATA LAYER owns every location: heavy artefacts under one data root,
outside the code tree. Every location is overridable by environment
variable so an existing local layout can be linked in without copying.

  LARICE_DATA_ROOT   data root            (default: <release>/data)
  LARICE_ASSETS      built training/eval tensors (npz/npy)
  LARICE_RESULTS     checkpoints, projections, per-arm result jsons
  LARICE_CORPORA     text corpora (wiki_*, sp_*)
  LARICE_EMBED_H5    73M-sentence review-embedding h5
  LARICE_TEXT_H5     sentence-index + 23-tag-label h5
"""
import os
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]


def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else default


DATA = _p("LARICE_DATA_ROOT", RELEASE / "data")
ASSETS = _p("LARICE_ASSETS", DATA / "assets")
RESULTS = _p("LARICE_RESULTS", DATA / "results")
CORPORA = _p("LARICE_CORPORA", DATA / "corpora")
EMBED_H5 = _p("LARICE_EMBED_H5", DATA / "reviews" / "embedding_h5.h5")
TEXT_H5 = _p("LARICE_TEXT_H5", DATA / "reviews" / "text_h5.h5")

# The authoritative fixed split (seed 20260711, 204/203/407 over the 814
# clean wiki universe) ships WITH the data layer — it pins the protocol.
SPLIT_JSON = Path(__file__).resolve().parent / "wiki_eval_split.json"

# Shipped with the prebuilt data (Hugging Face chironYu/latenarray-i2ce-data):
# the per-game review / sentence offsets of embedding_h5.h5. With it the
# anchor packs can be rebuilt at any budget from the review pool, no h5.
FULL_POOL_META = "full_pool_meta.npz"
# The 23-tag labels in games.npz order, the same values as embedding_h5.h5's
# tag_labels; read when text_h5.h5 is absent.
TAG_LABELS_NPZ = "tag_labels.npz"


def npz_array_shape(path: Path, key: str) -> tuple:
    """The shape of array `key` in an .npz, read from its header only."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        version = npf.read_magic(f)
        read = npf.read_array_header_1_0 if version == (1, 0) else npf.read_array_header_2_0
        return tuple(read(f)[0])


def gallery_path(cap: int) -> Path:
    """Where the packaged path keeps its anchor packs at `cap` sentences.

    wscan_gal_rev.npz when it was built at that budget (the h5 build writes
    it at the profile's budget); otherwise wscan_gal_rev_pool<cap>.npz, which
    build_assets.py rebuilds from the review pool without the h5. The second
    name keeps a shipped 512-sentence wscan_gal_rev.npz, which the w9
    notebooks read, untouched.
    """
    legacy = ASSETS / "wscan_gal_rev.npz"
    if legacy.exists() and npz_array_shape(legacy, "gal")[1] == cap:
        return legacy
    return ASSETS / f"wscan_gal_rev_pool{cap}.npz"
