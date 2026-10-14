import numpy as np
from datetime import datetime
import os
import pandas as pd
import sys
from datetime import datetime

from intervaltree import IntervalTree

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple

def make_non_intersecting_folds(
    df_cv: pd.DataFrame,
    n_folds: int = 3,
    pad: int = 50,
    seed: int | None = None,
):
    """
    Splits barcodes into n_folds such that there are NO overlaps BETWEEN folds
    (taking pad into account) on the same chromosome. Different chromosomes are treated
    independently, i.e., identical coordinates on different chr are NOT considered overlapping.

    Expects exactly one row per barcode in df_cv and columns: 'barcode','chr','start','end'.
    Returns dict[int, set[str]]: sets of barcodes for each fold.

    Algorithm:
      1) For each chr, collect "intersection components" using intervals [start-pad, end+pad).
      2) Components (indivisible blocks) are greedily assigned to folds, balancing by barcode count.
    """
    required = {"barcode", "chr", "start", "end"}
    missing = required - set(df_cv.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # type conversion and basic cleaning
    tmp = df_cv[["barcode", "chr", "start", "end"]].dropna().copy()
    tmp["start"] = tmp["start"].astype(int)
    tmp["end"]   = tmp["end"].astype(int)
    tmp = tmp[tmp["end"] > tmp["start"]]
    # ensure exactly one row per barcode
    if tmp["barcode"].duplicated().any():
        raise ValueError("Expected unique barcode in df_cv (one row per barcode).")

    # === 1) collect intersection components for each chromosome
    components: list[set[str]] = []
    for chrom, sub in tmp.groupby("chr", sort=False):
        sub = sub.copy()
        sub["s"] = (sub["start"] - pad).astype(int)
        sub["e"] = (sub["end"]   + pad).astype(int)
        sub = sub.sort_values("s").reset_index(drop=True)

        cur_comp: set[str] = set()
        cur_end: int | None = None

        for _, r in sub.iterrows():
            s, e, b = int(r["s"]), int(r["e"]), r["barcode"]
            if cur_comp and s <= cur_end:
                # continue current component
                cur_comp.add(b)
                if e > cur_end:
                    cur_end = e
            else:
                # close previous component and start a new one
                if cur_comp:
                    components.append(cur_comp)
                cur_comp = {b}
                cur_end = e
        if cur_comp:
            components.append(cur_comp)

    # if there are no barcodes, return empty folds
    if not components:
        return {i: set() for i in range(n_folds)}

    # === 2) assign components to folds (balanced by barcode count)
    rng = np.random.default_rng(seed)
    order = np.arange(len(components))
    rng.shuffle(order)

    folds: dict[int, set[str]] = {i: set() for i in range(n_folds)}
    sizes = np.zeros(n_folds, dtype=np.int64)  # number of barcodes in each fold

    for idx in order:
        comp = components[idx]
        j = int(np.argmin(sizes))   # assign to the currently smallest fold
        folds[j] |= comp
        sizes[j] += len(comp)

    # sanity check: total number of barcodes matches
    total_barcodes = tmp["barcode"].nunique()
    assigned = sum(len(s) for s in folds.values())
    if assigned != total_barcodes:
        raise RuntimeError(f"assigned {assigned}, {total_barcodes} barcodes expected.")

    print(f"[make_non_intersecting_folds] chr-separation + intersection components with pad={pad}")
    print(f"[make_non_intersecting_folds] n_folds={n_folds}, distribution of barcodes by folds: {sizes.tolist()}")

    return folds


def train_test_val_split_no_overlap(
    df_coords: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float  = 0.15,
    pad: int = 50,
    seed: int | None = None,
    overshoot_tol: float = 0.05,
) -> Dict[str, Set[str]]:
    """
    Splits barcodes into three sets (train/val/test) with target fractions,
    guaranteeing no overlaps between sets on the same chromosome
    (overlaps are defined using intervals [start - pad, end + pad]).

    Parameters
    ----------
    df_coords : DataFrame
        Must contain columns: 'barcode', 'chr', 'start', 'end'.
        Assumes exactly one row per barcode.
    test_frac : float
        Test fraction (0..1).
    val_frac : float
        Validation fraction (0..1).
    pad : int
        Padding used to define coordinate proximity.
    seed : int | None
        If None, each run produces a random split, otherwise it is reproducible.
    overshoot_tol : float
        Allowed fraction overshoot during packing (e.g., 0.05 => allow up to 5% excess).

    Returns
    -------
    dict with keys {'train','val','test'} and values Set[str] (sets of barcodes).
    """

    # -- input validation ---
    need = {"barcode", "chr", "start", "end"}
    miss = need - set(df_coords.columns)
    if miss:
        raise KeyError(f"df_coords is missing required columns: {sorted(miss)}")

    if not (0 <= test_frac < 1 and 0 <= val_frac < 1 and test_frac + val_frac < 1):
        raise ValueError("Invalid fractions: require 0<=test<1, 0<=val<1, and test+val<1.")

    # --- preparation and basic validation ---
    tmp = df_coords[["barcode", "chr", "start", "end"]].dropna().copy()
    tmp["start"] = tmp["start"].astype(int)
    tmp["end"]   = tmp["end"].astype(int)
    tmp = tmp[tmp["end"] > tmp["start"]]

    if tmp["barcode"].duplicated().any():
        # if this data can have several rows per barcode, remove this check
        raise ValueError("Expected unique barcode (exactly one row per barcode)")

    #  1) collect intersection components for each chromosome ---
    components: list[set[str]] = []
    for chrom, sub in tmp.groupby("chr", sort=False):
        sub = sub.copy()
        sub["s"] = (sub["start"] - pad).astype(int)
        sub["e"] = (sub["end"]   + pad).astype(int)
        sub = sub.sort_values("s").reset_index(drop=True)

        cur_comp: set[str] = set()
        cur_end: int | None = None

        for _, r in sub.iterrows():
            s, e, b = int(r["s"]), int(r["e"]), r["barcode"]
            if cur_comp and s <= cur_end:
                # continue current component
                cur_comp.add(b)
                if e > cur_end:
                    cur_end = e
            else:
                # close previous component and start a new one
                if cur_comp:
                    components.append(cur_comp)
                cur_comp = {b}
                cur_end = e
        if cur_comp:
            components.append(cur_comp)

    if not components:
        return {"train": set(), "val": set(), "test": set()}

    # --- 2) randomize component order and greedily pack to target fractions ---
    rng = np.random.default_rng(seed)
    order = np.arange(len(components))
    rng.shuffle(order)

    # component weight = number of barcodes (since 1 row == 1 barcode)
    comp_sizes = [len(components[i]) for i in range(len(components))]
    total = int(np.sum(comp_sizes))
    target_test = int(round(test_frac * total))
    target_val  = int(round(val_frac  * total))

    # accumulators
    sets = {"test": set(), "val": set(), "train": set()}
    sizes = {"test": 0, "val": 0, "train": 0}

    for idx in order:
        comp = components[idx]
        w = comp_sizes[idx]

        # priority: fill test first, then val, the rest goes to train
        if sizes["test"] + w <= int(target_test * (1 + overshoot_tol)):
            bucket = "test"
        elif sizes["val"] + w <= int(target_val * (1 + overshoot_tol)):
            bucket = "val"
        else:
            bucket = "train"

        sets[bucket] |= comp
        sizes[bucket] += w

    # --- 3) sanity checks: no barcode overlaps, fractions are reasonable ---
    assert sets["test"].isdisjoint(sets["val"])
    assert sets["test"].isdisjoint(sets["train"])
    assert sets["val"].isdisjoint(sets["train"])

    got_test = sizes["test"] / total if total else 0.0
    got_val  = sizes["val"]  / total if total else 0.0
    got_train= sizes["train"]/ total if total else 0.0
    print(f"[split_no_neighbor_overlap] "
          f"test≈{got_test*100:.1f}%, val≈{got_val*100:.1f}%, train≈{got_train*100:.1f}% "
          f"(targets: test={test_frac*100:.1f}%, val={val_frac*100:.1f}%)")

    return {"train": sets["train"], "val": sets["val"], "test": sets["test"]}






