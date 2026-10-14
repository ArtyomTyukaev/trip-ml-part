"""
Compute group SHAP for every model of a dataset, then collect the numbers into `OUTPUTS/group_shap/`.

    python group_shap/run_group_shap.py --datasets single_combined --tasks regression
    python group_shap/run_group_shap.py --collect-only

**This submits LSF array jobs and clears OUTPUTS/ for the task it runs.** There is no dry-run mode
that stops short of that. `run_training.py --dry-run` is the only way to look without submitting.

## What it does

1. Writes `group_map.json`, `{dataset: {table_type: {raw feature: group}}}`, from
   `group_definitions.py`. It is passed to the workers as a FILE PATH, not imported:
   `MODELS/XGB_*.py` must not depend on the analysis code.
2. Runs `run_training.py --group-map ... --slot gshap` once per (dataset, task), and runs all six
   CONCURRENTLY. That trains the models exactly as an ordinary run would and adds the
   GSHAP_/BSHAP_/ISHAP_/TSHAP_/GSE_ columns. `--slot` is what makes concurrency safe: it suffixes
   the artifacts folder, the output folder and the LSF job name, so the six runs cannot clear each
   other. Peak occupancy is one GPU per experiment per task - 36 for the three live datasets - and
   the wall clock is the slowest single run, not their sum. Per-run stdout goes to
   `OUTPUTS/group_shap/logs/<dataset>_<task>.log`.
3. Snapshots each merged `results.tsv` into `OUTPUTS/group_shap/raw/`, because the training output
   folder is cleared by the next run and the columns would be unrecoverable.
4. Folds the raw tables into three long tables, one row per (model, group) or (model, feature).

Steps 3 and 4 are `--collect-only`, so a run that died in step 2 can be salvaged.

## Why the outputs are flat and not per-dataset

There are no rounds here: one computation per model, 72 models, so `group_shap.tsv` is about a
thousand rows. Splitting that across three dataset folders would only make a reader open three
files and join them. `dataset` is a column.
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# HERE is group_shap/, the project root is one level up.
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, HERE)

from settings.utils import resolve_path                                    # noqa: E402
from dataset_config import DATASETS, TASKS                                 # noqa: E402
import group_definitions                                                  # noqa: E402
from group_definitions import groups_for_dataset, TABLE_TYPES              # noqa: E402

DATA = os.environ.get("GROUP_SHAP_DATA") or os.path.join(PROJECT, "OUTPUTS", "group_shap")
RAW = os.path.join(DATA, "raw")
LIVE = ["single_combined", "intron_exp", "intron_splicing"]
OUTPUT_OF_TASK = {"regression": "OUTPUTS/XGB_Regressor",
                  "classification": "OUTPUTS/XGB_Classifier"}

# Passed to run_training.py --slot. It suffixes the artifacts folder, the output folder and the LSF
# job name, which is what lets the six (dataset, task) runs sit in the cluster together. `snapshot`
# has to read the SAME suffixed output folder, so the two must never drift - hence one constant.
SLOT = "gshap"
LOG_DIR = os.path.join(DATA, "logs")


def write_group_map(datasets):
    """{dataset: {table_type: {raw feature: group}}} -> data/group_map.json."""
    os.makedirs(DATA, exist_ok=True)
    # MERGE, do not overwrite. A run over one dataset used to replace the whole file, so the map
    # for every other dataset vanished - and the folded tables need all of them to resolve a model
    # column to its group. Rebuilding is cheap, losing it silently is not.
    path = os.path.join(DATA, "group_map.json")
    mapping = {}
    if os.path.exists(path):
        with open(path) as handle:
            mapping = json.load(handle)
    for dataset in datasets:
        per_table = {}
        for table_type in TABLE_TYPES:
            # returns (groups, features, source) - the second and third are for `--check`
            groups, features, _source = groups_for_dataset(dataset, table_type)
            per_table[table_type] = {feature: group
                                     for group, members in groups.items()
                                     for feature in members}
            # Every feature the model will see must land in a group, or `build_col_group` raises
            # inside the worker after the training has already been paid for.
            missing = [f for f in features if f not in per_table[table_type]]
            if missing:
                raise ValueError("%s / %s: %d features belong to no group: %s"
                                 % (dataset, table_type, len(missing), missing[:5]))
        mapping[dataset] = per_table
        sizes = {t: len(set(m.values())) for t, m in per_table.items()}
        print("  %-16s groups per table: %s" % (dataset, sizes))
    path = os.path.join(DATA, "group_map.json")
    with open(path, "w") as handle:
        json.dump(mapping, handle, indent=1, sort_keys=True)
    print("  wrote %s" % path)
    return path, mapping


def per_dataset_map(mapping, dataset):
    """The worker wants {table_type: {...}} for its own dataset only."""
    path = os.path.join(DATA, "group_map_%s.json" % dataset)
    with open(path, "w") as handle:
        json.dump(mapping[dataset], handle, indent=1, sort_keys=True)
    return path


def drop_list_for(dataset, table_type):
    """Path to the drop list for this dataset, or None if it is the baseline one.

    The table type's baseline list plus `extra_drop` from `dataset_config`. The file is rebuilt on
    every run from one source rather than kept as a second copy: a copy would drift from the
    declaration sooner or later, and the drift would be silent because the drop is applied with
    `errors='ignore'`.

    It has to be a SEPARATE file rather than an edit of `settings/columns_to_drop_in_*.txt`: that
    list is shared by the whole project, and a crashed run would leave it trimmed for every run
    after it.
    """
    extra = group_definitions.extra_drop(dataset, table_type)
    if not extra:
        return None
    baseline = resolve_path(group_definitions.BASELINE_LISTS[table_type])
    lines = [l.strip() for l in open(baseline) if l.strip()]
    for name in extra:
        if name not in lines:
            lines.append(name)
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "drop_%s_%s.txt" % (table_type, dataset))
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("  %-16s %-11s also dropped: %s"
          % (dataset, table_type, ", ".join(extra)))
    return path


def train(dataset, task, map_path, args):
    """Launch one (dataset, task) run and return the handle WITHOUT waiting for it.

    Non-blocking on purpose. The six combinations used to run one after another, and since the
    group SHAP is about 95% of a run and does not shrink, that serialisation - not the arithmetic -
    was what made a full sweep take hours: six stages of 8 workers instead of one of 36.

    `--slot` is what makes concurrency safe, and it is not new: `run_training.py` grew it for
    exactly this ("two runs of the SAME task can be in the cluster at once"). It suffixes all three
    things a concurrent run would collide on - the artifacts folder, the output folder and the LSF
    job name. Without it the six runs share one artifacts folder and each clears the others at
    startup, which loses results silently rather than failing.

    stdout goes to a per-run log instead of the terminal: six pollers printing `bjobs` summaries
    into one stream is unreadable, and the caller reports each run's outcome as it ends.
    """
    command = [sys.executable, os.path.join(HERE, "run_training.py"),
               "--dataset", dataset, "--task", task,
               "--group-map", map_path,
               "--slot", SLOT,
               "--group-shap-n-explain", str(args.n_explain),
               "--group-shap-n-background", str(args.n_background),
               "--group-shap-seed", str(args.seed)]
    # Per-dataset drop. `group_definitions.resolve_features` gets the same list, so the group
    # membership and the trained model come from one declaration.
    for table_type, flag in (("intragenic", "--drop-intragenic"),
                             ("intergenic", "--drop-intergenic")):
        path = drop_list_for(dataset, table_type)
        if path:
            command += [flag, path]
    if args.no_plots:
        command.append("--no-plots")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "%s_%s.log" % (dataset, task))
    handle = open(log_path, "w")
    print("  launched %-16s %-14s -> %s" % (dataset, task, log_path), flush=True)
    process = subprocess.Popen(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT)
    return {"process": process, "handle": handle, "log": log_path,
            "dataset": dataset, "task": task}


def wait_for(runs):
    """Wait for every launched run. Returns the ones that failed, having reported each as it ends."""
    failed = []
    for run in runs:
        code = run["process"].wait()
        run["handle"].close()
        status = "ok" if code == 0 else "FAILED (exit %d)" % code
        print("  %-16s %-14s %s" % (run["dataset"], run["task"], status), flush=True)
        if code != 0:
            failed.append(run)
    return failed


def snapshot(dataset, task):
    """Copy the merged results.tsv out of OUTPUTS/ before anything else clears it."""
    source = resolve_path("%s/%s_%s/results.tsv" % (OUTPUT_OF_TASK[task], dataset, SLOT))
    if not os.path.exists(source):
        print("  no results.tsv for %s / %s - skipping" % (dataset, task))
        return None
    os.makedirs(RAW, exist_ok=True)
    target = os.path.join(RAW, "%s_%s_results.tsv" % (dataset, task))
    frame = pd.read_csv(source, sep="\t")
    frame.to_csv(target, sep="\t", index=False)
    print("  %s  (%d rows, %d columns)" % (target, len(frame), frame.shape[1]))
    return target


def fold(mapping):
    """Turn every raw table in data/raw/ into the three long tables."""
    group_rows, feature_rows, info_rows, broken = [], [], [], []
    for name in sorted(os.listdir(RAW)) if os.path.isdir(RAW) else []:
        if not name.endswith("_results.tsv"):
            continue
        dataset, task = name[:-len("_results.tsv")].rsplit("_", 1)
        frame = pd.read_csv(os.path.join(RAW, name), sep="\t")

        # The worker records a group-SHAP failure in a column instead of raising, so that one bad
        # model does not discard an entire array element and trigger three retrainings. That makes
        # it this function's job to refuse to publish a partial run - otherwise a soft failure
        # becomes a silent one, which is worse than the crash it replaced.
        if "gshap_error" in frame.columns:
            for _, bad in frame[frame["gshap_error"].notna()].iterrows():
                broken.append("%s/%s %s/%s: %s" % (dataset, task, bad["experiment"],
                                                   bad["table_type"], bad["gshap_error"]))
        for _, row in frame.iterrows():
            key = dict(dataset=dataset, task=task, table_type=row["table_type"],
                       experiment=row["experiment"])
            feature_to_group = mapping[dataset][row["table_type"]]

            # A results.tsv covers BOTH table types, so its GSHAP_ columns are the union over
            # them - and a group that exists in only one leaves NaN in the rows of the other.
            # `intron_annotations` is exactly that case: all of `DistToIntron5`, `DistToIntron3`,
            # `RelIntronPos` and `MANE_intron_length` sit on
            # `settings/columns_to_drop_in_intergenic.txt`, so the group owns no column there,
            # `build_col_group` does not make it a player, and it never had a value to report.
            #
            # Skipping on NaN alone would also swallow a genuine failure, so the test is the
            # PLAYER LIST the worker recorded, not the value. A group absent from `gshap_groups`
            # was never in the game. A group present in it whose value is NaN is a defect and
            # must survive into the table where the checks below can see it.
            players = set(str(row.get("gshap_groups", "")).split(","))
            for column in frame.columns:
                if column.startswith("GSHAP_"):
                    group = column[len("GSHAP_"):]
                    if group not in players:
                        continue
                    group_rows.append(dict(key, group=group, gshap=row[column],
                                           bshap=row.get("BSHAP_%s" % group),
                                           gshap_se=row.get("GSE_%s" % group)))
                elif column.startswith("ISHAP_"):
                    level = column[len("ISHAP_"):]
                    feature_rows.append(dict(
                        key, level=level,
                        group=_group_of_level(level, feature_to_group),
                        shap=row.get("SHAP_%s" % level),
                        tshap=row.get("TSHAP_%s" % level),
                        ishap=row[column]))

            info = {k[len("gshap_"):]: v for k, v in row.items() if k.startswith("gshap_")}
            if info:
                info_rows.append(dict(key, **info))

    if broken:
        raise SystemExit("%d model(s) failed group SHAP, nothing was written:\n  %s"
                         % (len(broken), "\n  ".join(broken)))

    os.makedirs(DATA, exist_ok=True)
    for rows, filename in ((group_rows, "group_shap.tsv"),
                           (feature_rows, "feature_shap.tsv"),
                           (info_rows, "run_info.tsv")):
        if not rows:
            print("  nothing to write for %s" % filename)
            continue
        table = pd.DataFrame(rows)
        path = os.path.join(DATA, filename)
        table.to_csv(path, sep="\t", index=False)
        print("  %s  (%d rows)" % (path, len(table)))


def _group_of_level(level, feature_to_group):
    """Model column -> group, by exact name first and by one-hot prefix second."""
    if level in feature_to_group:
        return feature_to_group[level]
    owners = {g for feature, g in feature_to_group.items() if level.startswith(feature + "_")}
    return owners.pop() if len(owners) == 1 else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=LIVE, choices=sorted(DATASETS))
    parser.add_argument("--tasks", nargs="+", default=sorted(TASKS), choices=sorted(TASKS))
    parser.add_argument("--n-explain", type=int, default=200)
    parser.add_argument("--n-background", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2005)
    parser.add_argument("--no-plots", action="store_true", default=True)
    parser.add_argument("--collect-only", action="store_true",
                        help="skip training, just snapshot and fold what is already in OUTPUTS/")
    args = parser.parse_args()

    print("group map:")
    _, mapping = write_group_map(args.datasets)

    if not args.collect_only:
        # All (dataset, task) combinations at once. Each is an independent LSF array, isolated by
        # SLOT, so the wall clock is the SLOWEST run rather than their sum.
        print("\nlaunching %d runs concurrently:" % (len(args.datasets) * len(args.tasks)))
        runs = [train(dataset, task, per_dataset_map(mapping, dataset), args)
                for dataset in args.datasets
                for task in args.tasks]

        print("\nwaiting. Live progress: tail -f %s/*.log" % LOG_DIR, flush=True)
        failed = wait_for(runs)

        # Snapshot whatever succeeded before raising: OUTPUTS/ is cleared by the next run of
        # anything, so a failure in one combination must not cost the other five their results.
        print("\nsnapshotting:")
        for run in runs:
            if run not in failed:
                snapshot(run["dataset"], run["task"])
        if failed:
            raise SystemExit("%d of %d runs failed: %s"
                             % (len(failed), len(runs),
                                ", ".join("%s/%s (see %s)" % (r["dataset"], r["task"], r["log"])
                                          for r in failed)))
    else:
        print("\nsnapshotting what is in OUTPUTS/:")
        for dataset in args.datasets:
            for task in args.tasks:
                snapshot(dataset, task)

    print("\nfolding into data/:")
    fold(mapping)
    return 0


if __name__ == "__main__":
    sys.exit(main())
