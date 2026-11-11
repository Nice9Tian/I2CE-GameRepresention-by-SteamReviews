# -*- coding: utf-8 -*-
"""Where the neighborhood scripts read and write, from three environment variables.

    HOOD_WORK     the working folder (default ./hood_work).  prep.py and cv_hoods.py write their arrays
                  into it; the figure scripts read them and write the numbers they quote beside them
                  (fig_facts_4096k10.json, query_stats_4096k10.json).
    HOOD_OUT      where the figures are written (default <HOOD_WORK>/figures)
    W9_DATA_DIR   the data folder, chironYu/latenarray-i2ce-data (tag_labels.npz gives the genres)
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("HOOD_WORK", "hood_work"))
BIN = WORK / "fixed"                 # prep.py: allgames.npz, frozen.npz, ckpt/ref_anchors_*.npy
FACTS = WORK                         # cv/cv_hoods_4096.json and the JSON files of numbers
OUT = Path(os.environ["HOOD_OUT"]) if os.environ.get("HOOD_OUT") else WORK / "figures"
DATA_DIR = Path(os.environ.get("W9_DATA_DIR", ""))
TITLES = HERE / "all_titles.json"    # the Steam titles of the 2,020 games, in the order of games.npz


def need_data_dir():
    assert os.environ.get("W9_DATA_DIR") and (DATA_DIR / "games.npz").exists(), \
        "set W9_DATA_DIR to the data folder (the one holding games.npz and tag_labels.npz)"
    return DATA_DIR
