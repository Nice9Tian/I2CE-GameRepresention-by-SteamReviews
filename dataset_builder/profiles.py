# -*- coding: utf-8 -*-
"""Reproduction profiles: the anchor budget each one builds and trains at.

This is the single source of truth for the profile -> anchor-budget map.
It lives in the bottom layer on purpose: dataset_builder,
steam_reviews_framework and contrast_experiment all read it, and none of
them has to import upward to find out how big a pack is.

The anchor budget is the one number that decides whether a run fits the
machine, because the gallery is re-encoded with gradient at every step:

    budget   build array (2,020 x cap x 1,024 fp16)   training peak
      512                2.1 GB                        8.4 GiB
    1,024                4.2 GB                        22.3 GiB
    2,048                8.5 GB                       35.0 GiB
    4,096               16.9 GB                        ~61 GiB

A 24 GB desktop GPU covers budgets to 1,024; 2,048 wants a 48 GB card
and 4,096 an 80 GB card.
That is what `litePaperTest` exists to avoid, so the profile has to reach
the data build too, not just the trainer -- assets built at 4,096 would
OOM a desktop card before the first step, and would already have needed
17 GB of host RAM to write.

`w9/Pod/w9_profiles.py` holds the experiment grid these names select
from. It imports LITE_CAP from here when the repository is present and
falls back to a literal when `w9/` has been copied to a pod on its own.
"""

PROFILES = ("fullTest", "paperTest", "litePaperTest")

#: The budget litePaperTest clamps down to. Retrieval has already
#: saturated here (within 0.02 of the full configuration on every
#: reading, Section 4.2 and Table A2), and it fits a 24 GB card.
LITE_CAP = 1024

#: The manuscript's headline budget.
FULL_CAP = 4096

#: The largest anchor budget each profile trains at, which is what the
#: packaged data build has to produce.
ANCHOR_CAP = {
    "fullTest": FULL_CAP,
    "paperTest": FULL_CAP,
    "litePaperTest": LITE_CAP,
}

#: What the packaged entry points use when no profile is named.
DEFAULT_PROFILE = "paperTest"


def check(profile):
    """Return `profile`, or raise with the list of valid names."""
    if profile not in ANCHOR_CAP:
        raise ValueError(f"unknown profile {profile!r}; expected one of "
                         + ", ".join(PROFILES))
    return profile


def anchor_cap(profile=DEFAULT_PROFILE):
    """The anchor budget to build and train at for `profile`."""
    return ANCHOR_CAP[check(profile)]


def clamp(cap, profile=DEFAULT_PROFILE):
    """Clamp one experiment's own budget to what `profile` allows.

    A ceiling, not a setting: a cell the paper ran at 512 stays at 512,
    because rerunning it higher would not reproduce the published row.
    """
    return min(cap, ANCHOR_CAP[check(profile)])
