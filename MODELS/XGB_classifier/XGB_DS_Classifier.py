import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from intervaltree import IntervalTree
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import shap
import sys
from xgboost import DMatrix
import xgboost as xgb
import copy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
sys.path.append(str(PROJECT_ROOT / "settings" / "custom_functions"))

from get_replics_dicts import get_replics_dicts
from get_groups_of_features import get_groups_of_features
from make_folds import make_non_intersecting_folds
from make_folds import train_test_val_split_no_overlap
from check_train_test_split import check_train_test_split
from get_gene_tables import make_intra_and_inter_genic_tables
from reg_to_classification import devide_into_intervals, devide_into_intervals_intron, devide_into_binary
from group_shap import build_feature_of_column, shap_columns

# Replicate maps and feature-group definitions (from project settings)
replics_dicts = get_replics_dicts()
groups_to_features = get_groups_of_features()

# --- Path helpers & config from environment ---


def rel(p: str) -> str:
    """Project-relative -> absolute string path."""
    return str((PROJECT_ROOT / p).resolve())


def require_env_path(name: str) -> str:
    return os.environ[name]


def getenv_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def getenv_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def booster_params(n_classes):
    """Fixed booster block. Each regularisation knob can be overridden by an environment variable.

    THE DEFAULTS ARE THE VALUES THAT USED TO BE HARDCODED HERE, so a run that sets nothing is
    bit-identical to any run before 2026-09-05. That is the whole point of writing it this way:
    none of the other four datasets changes. The regressor already has the same construction
    (`XGB_basic_DS_regressor.py:booster_params`), it is repeated here rather than invented.

    Why it was needed. The classifier is regularised much harder than the regressor on the same
    data: `reg_lambda` 12 vs 1, `min_child_weight` 25 vs 5, `gamma` 2 vs 0.1. With balanced
    classes this is merely conservative. On SON, where the positive class is about 6 %, it was
    fatal: the `multi:softprob` hessian for a class with p~0.06 is small, a leaf never collects
    a weight of 25, no split passes `gamma`, and the booster returns a CONSTANT. Measured on
    2026-09-05: `chosen_best_iteration` = 1 in all ten models, the sum of mean|SHAP| exactly zero
    in nine of ten, AUC exactly 0.500. That is not "poor quality" but no model at all, and the
    metric alone cannot tell the two apart. Only the zero SHAP and the stop at the first
    iteration can.
    """
    return {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": ["mlogloss", "merror"],
        "tree_method": "hist",             # good default for GPU
        "device": "cuda",
        "learning_rate": getenv_float("LEARNING_RATE", 0.03),
        "max_depth": 6,                    # baseline, grid search overrides
        "subsample": getenv_float("SUBSAMPLE", 0.6),
        "colsample_bytree": getenv_float("COLSAMPLE_BYTREE", 0.5),
        "reg_lambda": getenv_float("REG_LAMBDA", 12),
        "min_child_weight": getenv_float("MIN_CHILD_WEIGHT", 25),
        "gamma": getenv_float("GAMMA", 2),  # minimum loss reduction for a split
        "seed": 5673,
    }


def getenv_list_int(name: str, default_list):
    raw = os.getenv(name, "")
    if raw.strip() == "":
        return list(default_list)
    # expected format: "1,2,3,4"
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# env paths (already absolute from notebook)
input_dir = require_env_path("INPUT_DIR")
exp_list = require_env_path("EXP_LIST")
barcode_preds_path = require_env_path("BARCODE_PREDS_PATH")
path_to_file_annot_table_cols_to_drop = require_env_path("ANNOT_DROP_COLS")
path_to_file_columns_to_drop_in_intragenic = require_env_path(
    "INTRA_DROP_COLS")

# internal paths (make them absolute via PROJECT_ROOT)
#
# ARTIFACTS_DIR lets two runs of the SAME task work side by side. It used to be hardcoded, so
# every classification run on the machine wrote into one folder and cleared it on startup - which
# is why "two datasets cannot run at once" was a rule. run_training.py --slot now points each run
# at its own folder, and the default keeps the old path for anything that does not pass it.
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "MODELS/XGB_classifier/artifacts")
output_dir_path = rel(f"{ARTIFACTS_DIR}/output_splitted")
grid_out_path = rel(f"{ARTIFACTS_DIR}/grid_selection")

categorical_columns = os.getenv("CATEGORICAL_COLUMNS", "ChromHMM")
categorical_columns = [c.strip()
                       for c in categorical_columns.split(",") if c.strip()]

DEPTH_GRID = getenv_list_int("DEPTH_GRID", [1, 2, 3, 4, 5, 6])
NUM_BOOST_ROUND = getenv_int("NUM_BOOST_ROUND", 4000)
EARLY_STOP = getenv_int("EARLY_STOP", 1000)
# Seeds the CV fold draw. Unseeded, two rounds of a feature-removal chain differed by a fresh
# fold split as well as by the removed feature, and the split feeds the depth choice and the
# early-stopping iteration. The holdout split already used 2005.
FOLD_SEED = getenv_int("FOLD_SEED", 2005)
lines_per_job = getenv_int("LINES_PER_JOB", 3)

# Group SHAP. OFF unless GROUP_MAP_JSON points at a file, so an ordinary run is byte-identical to
# what it was. The mapping arrives as a file rather than an import on purpose: importing
# group_shap/group_definitions.py here would make this training script depend on the analysis code.
GROUP_MAP_JSON = os.getenv("GROUP_MAP_JSON", "").strip()
GROUP_SHAP_N_EXPLAIN = getenv_int("GROUP_SHAP_N_EXPLAIN", 200)
GROUP_SHAP_N_BG = getenv_int("GROUP_SHAP_N_BG", 50)
GROUP_SHAP_SEED = getenv_int("GROUP_SHAP_SEED", 2005)
group_map_all = {}
if GROUP_MAP_JSON:
    with open(GROUP_MAP_JSON) as handle:
        group_map_all = json.load(handle)
    print(f"group SHAP on: {GROUP_MAP_JSON}, "
          f"{GROUP_SHAP_N_EXPLAIN} explained x {GROUP_SHAP_N_BG} background", flush=True)

# Binary mode: the target becomes 1 if value > BINARY_THRESHOLD, else 0.
# When the variable is not set the script keeps its original 3-class behaviour.
BINARY_THRESHOLD = os.getenv("BINARY_THRESHOLD", "").strip()
BINARY_THRESHOLD = float(BINARY_THRESHOLD) if BINARY_THRESHOLD else None

# CLASS_PASSTHROUGH: the target column ALREADY holds integer labels 0..K-1, so no binning is
# needed. Added 2026-09-06 for the three-class SON model (down / unchanged / up).
#
# Why it was needed. None of the three existing branches can pass ready labels through.
# `BINARY_THRESHOLD` makes exactly two classes. The tertile branch needs the folder to be named
# exactly `intron_splicing`. The median branch computes its threshold FROM THE DATA, so whatever
# goes in, its own split comes out. Binary SON got by with a threshold of 0.5 as a pass-through,
# but there is no such trick for three classes.
#
# Off by default, so the other four datasets are not affected.
CLASS_PASSTHROUGH = os.getenv("CLASS_PASSTHROUGH", "").strip() == "1"
N_CLASSES = getenv_int("N_CLASSES", 3) if CLASS_PASSTHROUGH else (
    2 if BINARY_THRESHOLD is not None else 3)
if CLASS_PASSTHROUGH and BINARY_THRESHOLD is not None:
    raise ValueError("CLASS_PASSTHROUGH and BINARY_THRESHOLD together make no sense: the first "
                     "says the labels are ready, the second says build them with a threshold. "
                     "Keep one.")

job_index = int(os.environ.get("LSB_JOBINDEX", 1))

log_path = rel(f"{ARTIFACTS_DIR}/log_training/XGB_{job_index}.log")
err_path = rel(f"{ARTIFACTS_DIR}/log_training_errors/XGB_{job_index}.err")

# --- Ensure artifact directories exist ---

os.makedirs(grid_out_path, exist_ok=True)
os.makedirs(barcode_preds_path, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(os.path.dirname(err_path), exist_ok=True)
os.makedirs(output_dir_path, exist_ok=True)

# LSF array job index: read experiment list and slice the batch for this worker
with open(exp_list, 'r') as file:
    exp_names = [line.strip() for line in file.readlines()]

# Line range in EXP_LIST for this job (LINES_PER_JOB experiments per task)
start_LSB_JOBINDEX = (job_index - 1) * lines_per_job
end_LSB_JOBINDEX = start_LSB_JOBINDEX + lines_per_job
batch = exp_names[start_LSB_JOBINDEX:end_LSB_JOBINDEX]

if not batch:
    print(
        f"No experiments for job_index={job_index} (start={start_LSB_JOBINDEX}, end={end_LSB_JOBINDEX})")
    sys.exit(0)

with open(path_to_file_annot_table_cols_to_drop, 'r') as file_:
    annot_table_cols_to_drop = [j.strip() for j in file_.readlines()]

with open(path_to_file_columns_to_drop_in_intragenic, 'r') as file_:
    columns_to_drop_in_intragenic = [j.strip() for j in file_.readlines()]

# Redirect stdout/stderr to per-job log files under artifacts/log_training*
os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(os.path.dirname(err_path), exist_ok=True)

if os.path.exists(log_path):
    os.remove(log_path)
sys.stdout = open(log_path, "w", buffering=1)
sys.stderr = open(err_path, "w", buffering=1)

print(f"Job index: {job_index}")
print(f"start index: {start_LSB_JOBINDEX}")
print(f"end index: {end_LSB_JOBINDEX}")

target_col = "exp.log10"


def ovr_auc_per_class(y_true, proba, labels):
    """Per-class binary AUC in one-vs-rest mode. NaN if a class has no positive or no negative."""
    aucs = {}
    for k in labels:
        y_bin = (y_true == k).astype(int)
        # roc_auc_score needs both classes present in y_bin
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            aucs[k] = np.nan
        else:
            aucs[k] = roc_auc_score(y_bin, proba[:, k])
    return aucs


def auc_any(y_true, proba, average):
    """ROC-AUC for both modes. With 2 classes sklearn wants a single column of scores,
    not the full probability matrix."""
    if proba.shape[1] == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average=average)


all_results = []
# Whether this run's target is a splicing fraction, and therefore gets the tertile binning of
# `reg_to_classification.SPLICING_LOW/HIGH` instead of the median rule.
#
# Decided from the DATASET, not from the experiment name. It used to be a hardcoded list of the
# 16 pool names, which worked only while the two intron datasets named their experiments
# differently. Since the 2026 rebuild both name them after the intron - A1, B1, B3-6, ... - so
# 'A1' is an expression readout in intron_exp and a splicing fraction in intron_splicing, and
# no list of names can tell them apart. Getting it wrong is silent in one direction: RPM values
# put through the splicing bins collapse into one class.
#
# It also ends the maintenance debt of that list, which had to be edited by hand every time an
# intron or a pool was added.
is_splicing = os.path.basename(os.path.normpath(input_dir)) == 'intron_splicing'

for exp in batch:
    out = make_intra_and_inter_genic_tables(
        input_dir, exp, replics_dicts, columns_to_drop_in_intragenic, annot_table_cols_to_drop, filename_prefix='.alldata.tab')
    if out is None:
        print(f'Output of make_intra_and_inter_genic_tables() is None')
        continue
    intrAgenic_alltable, intErgenic_alltable = out

    # Guarded because a feature-removal round can drop these two, and they are rewritten here
    # by name.
    for _col in ["MANE_all", "MANE_exonic"]:
        if _col in intErgenic_alltable.columns:
            intErgenic_alltable[_col] = intErgenic_alltable[_col].apply(
                lambda x: 0 if pd.isna(x) or x == 0 else 1)



    # A splicing fraction runs from 0 to 1 and has its own three-class rule at 0.6 and 0.8.
    # BINARY_THRESHOLD is checked first below, so leaving it set from an earlier run on
    # another dataset would silently binarise those fractions instead. Fail loudly rather
    # than train on quietly wrong labels.
    if BINARY_THRESHOLD is not None and is_splicing:
        raise ValueError(
            f'{exp} is a splicing experiment and uses the tertile binning, but '
            f'BINARY_THRESHOLD is set to {BINARY_THRESHOLD}. Clear it for this dataset.')

    # Map continuous target to classes. Both rules here use fixed thresholds, so they can be
    # applied to the whole table safely. The median-based rule cannot - its threshold is
    # learned from the data, so it is applied further down, after the holdout is split off.
    if CLASS_PASSTHROUGH:
        # The labels are already in the column. Check that they really are 0..N_CLASSES-1 and
        # nothing else. Silently training on garbage is easiest right here, and `multi:softprob`
        # on an out-of-range label fails deep inside xgboost, where the cause is no longer visible.
        for table in [intrAgenic_alltable, intErgenic_alltable]:
            if table.empty:
                continue
            vals = sorted(table[target_col].dropna().unique())
            if any(float(v) != int(v) for v in vals) or min(vals) < 0 or max(vals) >= N_CLASSES:
                raise ValueError(f'{exp}/{target_col}: CLASS_PASSTHROUGH expects integer labels '
                                 f'0..{N_CLASSES - 1}, got {vals}')
            table[target_col] = table[target_col].astype(int)
    elif BINARY_THRESHOLD is not None:
        for table in [intrAgenic_alltable, intErgenic_alltable]:
            table[target_col] = table[target_col].apply(
                lambda x: devide_into_binary(x, BINARY_THRESHOLD))
    elif is_splicing:
        for table in [intrAgenic_alltable, intErgenic_alltable]:
            table[target_col] = table[target_col].apply(
                lambda x: devide_into_intervals_intron(x))
    # The median branch used to sit here and computed the threshold on the whole sample,
    # holdout included, which leaked the labels. Moved below.
    # else:
    #     for table in [intrAgenic_alltable, intErgenic_alltable]:
    #         median_exp = table[table[target_col] > 0][target_col].median()
    #         table[target_col] = table[target_col].apply(
    #             lambda x: devide_into_intervals(x, median_exp))

    # Intragenic vs intergenic: same XGBoost template, depth is tuned per CV below
    tables = {
        'intragenic': (intrAgenic_alltable, booster_params(N_CLASSES)),
        'intergenic': (intErgenic_alltable, booster_params(N_CLASSES)),
    }

    for table_type, (table, params) in tables.items():
        if table.empty:
            print(
                f"{datetime.now():[%Y-%m-%d %H:%M:%S]} Skipping empty {table_type} table for {exp}")
            continue

        # Only the categorical columns this table actually has. get_dummies raises KeyError on
        # a missing one, so once a feature-removal round drops a categorical - ChromHMM, SPIN,
        # Chr and the rest - every worker would die instead of training without it.
        present_categorical = [c for c in categorical_columns if c in table.columns]
        columns_before_encoding = list(table.columns)
        df_cv = pd.get_dummies(table, columns=present_categorical)
        df_cv = df_cv.reset_index().rename(columns={"index": "barcode"})
        df_cv = shuffle(df_cv, random_state=448)
        df_cv = shuffle(df_cv, random_state=4348)

        # Barcode coordinates (kept for downstream, not used as model features)
        df_coords = df_cv[['barcode', 'chr', 'start', 'end']].copy()

        holdout_and_train = train_test_val_split_no_overlap(
            df_cv, test_frac=0.15, val_frac=0, seed=2005)

        # Median-based binning, applied here rather than before the split. The threshold is
        # learned from the data, so it must come from the training pool only - taking it from
        # the whole sample would let the holdout distribution shape its own labels.
        if BINARY_THRESHOLD is None and not is_splicing and not CLASS_PASSTHROUGH:
            train_target = df_cv.loc[
                df_cv['barcode'].isin(holdout_and_train['train']), target_col]
            median_exp = train_target[train_target > 0].median()
            print(f'{table_type}: class threshold from train pool only, median = {median_exp:.4f}')
            df_cv[target_col] = df_cv[target_col].apply(
                lambda x: devide_into_intervals(x, median_exp))

        holdout_df = df_cv[df_cv['barcode'].isin(
            holdout_and_train['test'])].copy()
        train_pool_df = df_cv[df_cv['barcode'].isin(
            holdout_and_train['train'])].copy()

        print(
            f'Train size {len(train_pool_df)} | Holdout size {len(holdout_df)}')

        print(
            f"Target stats - Train: mean={train_pool_df[target_col].mean():.3f}, std={train_pool_df[target_col].std():.3f}")
        print(
            f"Target stats - Holdout: mean={holdout_df[target_col].mean():.3f}, std={holdout_df[target_col].std():.3f}")

        exclude_cols = ["barcode", "chr", "start", "end", "strand", target_col]
        feature_cols = [c for c in df_cv.columns if c not in exclude_cols]

        print(f"=== Diagnostics for {exp} {table_type} ===")
        print(f"Feature count: {len(feature_cols)}")
        print(
            f"Train pool size: {len(train_pool_df)}, holdout: {len(holdout_df)}")

        holdout_df = holdout_df.replace(
            [np.inf, -np.inf], np.nan).dropna(subset=[target_col]).fillna(0)
        train_pool_df = train_pool_df.replace(
            [np.inf, -np.inf], np.nan).dropna(subset=[target_col]).fillna(0)

        # === Assign barcodes to 3 folds with distance check ===
        folds = make_non_intersecting_folds(train_pool_df, n_folds=3, seed=FOLD_SEED)

        pred_rows_exp = []
        APPEND_TRAIN = True
        grid_val_rows = []
        grid_val_rows_all = []

        for fold_idx in range(3):

            val_barcodes = folds[fold_idx]
            train_barcodes = set(train_pool_df['barcode']) - val_barcodes

            train_df = train_pool_df[train_pool_df['barcode'].isin(
                train_barcodes)].copy()
            val_df = train_pool_df[train_pool_df['barcode'].isin(
                val_barcodes)].copy()

            train_df = shuffle(train_df, random_state=2003)
            val_df = shuffle(val_df, random_state=2000)

            train_df.fillna(0, inplace=True)
            val_df.fillna(0, inplace=True)

            train_df = (
                train_df
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[target_col])
                .fillna(0)
                .reset_index(drop=True)
            )
            val_df = (
                val_df
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[target_col])
                .fillna(0)
                .reset_index(drop=True)
            )

            out = check_train_test_split(
                train_df, val_df, feature_cols, target_col)
            if out is None:
                continue
            X_train, X_val, y_train, y_val, feature_cols_fold, train_df_clean, val_df_clean = out

            assert len(train_df_clean) == X_train.shape[0]
            assert len(val_df_clean) == X_val.shape[0]
            # Labels must match the cleaned frames (integer class indices)
            assert np.array_equal(
                y_train, train_df_clean[target_col].astype(int).values)
            assert np.array_equal(
                y_val,       val_df_clean[target_col].astype(int).values)

            y_train = y_train.astype(int)
            y_val = y_val.astype(int)

            dtr = xgb.DMatrix(X_train,  label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # --- Depth grid search on validation fold only (holdout never used here) ---
            grid_rows = []

            for depth in DEPTH_GRID:
                # Copy params so we do not mutate the shared template dict
                params_cur = dict(params)
                params_cur["max_depth"] = depth
                # Small tables: use full row subsample (no stochastic sampling)
                if len(table) < 4000:
                    params_cur['subsample'] = 1

                dtr = xgb.DMatrix(X_train,  label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)

                bst = xgb.train(
                    params=params_cur,
                    dtrain=dtr,
                    num_boost_round=NUM_BOOST_ROUND,
                    evals=[(dtr, "train"), (dval, "val")],
                    early_stopping_rounds=EARLY_STOP,
                    verbose_eval=False
                )

                best_it = (bst.best_iteration or 0) + 1
                # Early stopping metric on val (typically val-mlogloss for multi:softprob)
                val_best_mlogloss = float(bst.best_score)

                # Validation macro-F1 at best_iteration (argmax over classes)
                proba_val = bst.predict(dval, iteration_range=(0, best_it))
                y_pred_val = proba_val.argmax(axis=1)
                val_f1_macro = f1_score(y_val, y_pred_val, average="macro")

                grid_rows.append({
                    "dataset": os.path.basename(input_dir),
                    "experiment": exp,
                    "table_type": table_type,
                    "fold": fold_idx + 1,
                    "max_depth": depth,
                    "best_iteration": best_it,
                    "val_best_mlogloss": val_best_mlogloss,  # lower is better
                    "val_f1_macro": val_f1_macro             # higher is better
                })

                print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} [{table_type.upper()}] {exp} | "
                      f"Fold {fold_idx + 1} | depth={depth} | best_it={best_it} | "
                      f"val_mlogloss={val_best_mlogloss:.6f} | val_F1_macro={val_f1_macro:.4f}")

            grid_df = pd.DataFrame(grid_rows)
            grid_df_fold = grid_df.copy()
            grid_df_fold["fold"] = fold_idx + 1
            grid_df_fold["experiment"] = exp
            grid_df_fold["table_type"] = table_type
            grid_val_rows.append(grid_df_fold)

        # --- Pick best depth/iteration by validation macro-F1 (use idxmin on val_best_mlogloss for mlogloss) ---
        all_grid_results = pd.concat(grid_val_rows, ignore_index=True)
        best_row_idx = all_grid_results["val_f1_macro"].idxmax()
        best_row = all_grid_results.loc[best_row_idx]

        chosen_depth = int(best_row["max_depth"])
        chosen_best_it = int(best_row["best_iteration"])

        print(f"\nCV selection: depth={chosen_depth}, n_estimators={chosen_best_it}, "
              f"val_F1_macro={best_row['val_f1_macro']:.4f}, val_mlogloss={best_row['val_best_mlogloss']:.6f}")

        # --- Final model: full train pool vs holdout (fold column FINAL in saved tables, like regressor) ---
        params_final = dict(params)
        params_final["max_depth"] = chosen_depth
        if len(table) < 4000:
            # match grid-search branch for small tables
            params_final['subsample'] = 1

        out = check_train_test_split(
            train_pool_df, holdout_df, feature_cols, target_col)
        if out is None:
            continue
        X_train, X_holdout, y_train, y_holdout, feature_cols_fold, train_df_clean, holdout_df_clean = out

        dtrain_full = xgb.DMatrix(X_train, label=y_train)
        dhold = xgb.DMatrix(X_holdout,       label=y_holdout)

        final = xgb.train(params_final, dtrain_full,
                          num_boost_round=chosen_best_it, verbose_eval=False)

        # --- Metrics on full train (same split as check_train_test_split returned) ---
        proba_train = final.predict(dtrain_full)
        y_pred_train = proba_train.argmax(axis=1)

        n_classes = proba_train.shape[1]
        labels = list(range(n_classes))

        auc_tr_dict = ovr_auc_per_class(y_train, proba_train, labels)

        # Train prediction table (metadata + class probabilities)
        if APPEND_TRAIN:
            # Prefer the cleaned frame from check_train_test_split, the fallback would be invalid for DMatrix
            train_meta_df = train_df_clean if 'train_df_clean' in locals(
            ) else dtrain_full.loc[np.isfinite(dtrain_full[target_col])].reset_index(drop=True).copy()

            train_meta_cols = ["barcode", "chr", "start", "end"]
            if "strand" in train_meta_df.columns:
                train_meta_cols.append("strand")

            tmp_train = train_meta_df[train_meta_cols + [target_col]].copy()
            tmp_train.rename(columns={target_col: "target_true"}, inplace=True)
            tmp_train["target_pred"] = y_pred_train.astype(int)
            tmp_train["split"] = "train_full"
            tmp_train["fold"] = "FINAL"
            tmp_train["experiment"] = exp

            # Per-class probabilities from softmax output
            if proba_train.ndim == 2:
                for k in range(proba_train.shape[1]):
                    tmp_train[f"proba_class{k}"] = proba_train[:, k]

            pred_rows_exp.append(tmp_train)

        b_acc_train = balanced_accuracy_score(y_train, y_pred_train)
        acc_train = accuracy_score(y_train, y_pred_train)
        f1_macro_train = f1_score(y_train, y_pred_train, average="macro")
        f1_weighted_train = f1_score(y_train, y_pred_train, average="weighted")
        try:
            roc_auc_macro_train = auc_any(y_train, proba_train, "macro")
            roc_auc_weighted_train = auc_any(y_train, proba_train, "weighted")
        except ValueError:
            roc_auc_macro_train = np.nan
            roc_auc_weighted_train = np.nan

        prec_tr, rec_tr, f1_tr, supp_tr = precision_recall_fscore_support(
            y_train, y_pred_train, average=None, labels=[0, 1, 2], zero_division=0
        )

        # --- Metrics on holdout (test split) ---
        proba_hold = final.predict(dhold)
        y_pred_hold = proba_hold.argmax(axis=1)

        auc_te_dict = ovr_auc_per_class(y_holdout, proba_hold, labels)

        # Holdout prediction table (same FINAL label as train rows above)
        holdout_meta_df = holdout_df_clean if 'holdout_df_clean' in locals(
        ) else test_df.loc[np.isfinite(test_df[target_col])].reset_index(drop=True).copy()

        test_meta_cols = ["barcode", "chr", "start", "end"]
        if "strand" in holdout_meta_df.columns:
            test_meta_cols.append("strand")

        tmp_test = holdout_meta_df[test_meta_cols + [target_col]].copy()
        tmp_test.rename(columns={target_col: "target_true"}, inplace=True)
        tmp_test["target_pred"] = y_pred_hold.astype(int)
        tmp_test["split"] = "holdout"
        tmp_test["fold"] = "FINAL"
        tmp_test["experiment"] = exp

        if proba_hold.ndim == 2:
            for k in range(proba_hold.shape[1]):
                tmp_test[f"proba_class{k}"] = proba_hold[:, k]

        pred_rows_exp.append(tmp_test)

        b_acc_test = balanced_accuracy_score(y_holdout, y_pred_hold)
        acc_test = accuracy_score(y_holdout, y_pred_hold)
        f1_macro_test = f1_score(y_holdout, y_pred_hold, average="macro")
        f1_weighted_test = f1_score(y_holdout, y_pred_hold, average="weighted")
        try:
            roc_auc_macro_test = auc_any(y_holdout, proba_hold, "macro")
            roc_auc_weighted_test = auc_any(y_holdout, proba_hold, "weighted")
        except ValueError:
            roc_auc_macro_test = np.nan
            roc_auc_weighted_test = np.nan

        prec_te, rec_te, f1_te, supp_te = precision_recall_fscore_support(
            y_holdout, y_pred_hold, average=None, labels=[0, 1, 2], zero_division=0
        )

        print(
            f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} [{table_type.upper()}] {exp} | Holdout ")
        print(
            f"TEST: b_Acc={b_acc_test:.4f}, Acc={acc_test:.4f}, F1_macro={f1_macro_test:.4f}, ROC-AUC_macro={roc_auc_macro_test:.4f}")
        print(
            f"TRAIN: b_Acc={b_acc_train:.4f}, Acc={acc_train:.4f}, F1_macro={f1_macro_train:.4f}, ROC-AUC_macro={roc_auc_macro_train:.4f}")

        # One row per (experiment, table_type): final model after CV (fold = FINAL, like regressor)
        result = {
            "experiment": exp,
            "table_type": table_type,
            "fold": "FINAL",

            # Hyperparameters chosen from CV
            "chosen_max_depth": chosen_depth,
            "chosen_best_iteration": chosen_best_it,

            # Validation metrics used for selection
            "VAL_best_mlogloss": float(best_row["val_best_mlogloss"]),
            "VAL_f1_macro_selected": float(best_row["val_f1_macro"]),

            # TRAIN
            "TRAIN_b_accuracy": b_acc_train,
            'TRAIN_accuracy': acc_train,
            "TRAIN_f1_macro": f1_macro_train,
            "TRAIN_f1_weighted": f1_weighted_train,
            "TRAIN_roc_auc_macro": roc_auc_macro_train,
            "TRAIN_roc_auc_weighted": roc_auc_weighted_train,

            # TEST
            "TEST_b_accuracy": b_acc_test,
            'TEST_accuracy': acc_test,
            "TEST_f1_macro": f1_macro_test,
            "TEST_f1_weighted": f1_weighted_test,
            "TEST_roc_auc_macro": roc_auc_macro_test,
            "TEST_roc_auc_weighted": roc_auc_weighted_test,
        }

#         # per-class (TRAIN)
#         for i, (p, r, f, s) in enumerate(zip(prec_tr, rec_tr, f1_tr, supp_tr)):
#             result[f"TRAIN_class{i}_precision"] = float(p)
#             result[f"TRAIN_class{i}_recall"]    = float(r)
#             result[f"TRAIN_class{i}_f1"]        = float(f)
#             result[f"TRAIN_class{i}_support"]   = int(s)
#             result[f"TRAIN_class{i}_roc_auc_ovr"] = float(auc_tr_dict.get(i, np.nan)
#                                                   if not np.isnan(auc_tr_dict.get(i, np.nan)) else np.nan)

#         # per-class (TEST)
#         for i, (p, r, f, s) in enumerate(zip(prec_te, rec_te, f1_te, supp_te)):
#             result[f"TEST_class{i}_precision"] = float(p)
#             result[f"TEST_class{i}_recall"]    = float(r)
#             result[f"TEST_class{i}_f1"]        = float(f)
#             result[f"TEST_class{i}_support"]   = int(s)
#             result[f"TEST_class{i}_roc_auc_ovr"] = float(auc_te_dict.get(i, np.nan)
#                                                  if not np.isnan(auc_te_dict.get(i, np.nan)) else np.nan)

        # --- SHAP values ---
        # Take the feature columns from the cleaned frame rather than from X_train:
        # X_train comes from .values on a mixed bool/float frame, so numpy upcasts it
        # to object and XGBoost rejects it. feature_cols_fold already has the constant
        # features removed, and train_df_clean holds the same rows as X_train.
        X_train_full_df = train_df_clean[feature_cols_fold]

        explainer = shap.Explainer(final)
        shap_values = explainer(X_train_full_df)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        # A multiclass model returns (rows, features, classes), so the line above leaves
        # one value per (feature, class). Keep the positive class in binary mode,
        # average over classes otherwise.
        if mean_abs_shap.ndim == 2:
            mean_abs_shap = (mean_abs_shap[:, 1] if mean_abs_shap.shape[1] == 2
                             else mean_abs_shap.mean(axis=1))

        # add SHAP importances as separate fields
        for feat, shap_val in zip(X_train_full_df.columns, mean_abs_shap):
            result[f"SHAP_{feat}"] = shap_val

        # --- group SHAP, only when a group map was supplied ---
        # Adds GSHAP_<group>, BSHAP_<group> and ISHAP_<feature>. Nothing above is touched, so a run
        # without GROUP_MAP_JSON produces exactly the columns it always did.
        if group_map_all:
            # Wrapped, and the failure is recorded in a column rather than raised. Unwrapped, one
            # bad model discards the whole array element - including table types that already
            # finished - and run_training.py then RETRAINS it three times before giving up. The
            # error must stay visible, so run_group_shap.py refuses to publish a run that has any
            # gshap_error set.
            try:
                group_of_feature = group_map_all[table_type]
                feature_of_column = build_feature_of_column(
                    columns_before_encoding, list(X_train_full_df.columns), present_categorical)
                extra, gs_info = shap_columns(
                    final, X_train_full_df, feature_of_column, group_of_feature,
                    n_explain=GROUP_SHAP_N_EXPLAIN, n_background=GROUP_SHAP_N_BG,
                    seed=GROUP_SHAP_SEED)
                result.update(extra)
                for key, value in gs_info.items():
                    result[f"gshap_{key}"] = value
                print(f"  group SHAP {exp} {table_type}: {gs_info['n_groups']} groups, "
                      f"additivity {gs_info['additivity']:.2e}", flush=True)
            except Exception as exc:
                result["gshap_error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [group SHAP FAILED] {exp} {table_type}: {exc}", flush=True)

        all_results.append(result)

        # Write barcode-level predictions and depth-grid TSV for this experiment and table_type
        if pred_rows_exp:
            pred_df = pd.concat(pred_rows_exp, ignore_index=True)

            # If proba columns exist, add argmax class and its probability
            proba_cols = [
                c for c in pred_df.columns if c.startswith("proba_class")]
            if proba_cols:
                P = pred_df[proba_cols].to_numpy()
                idx = P.argmax(axis=1)
                # pred_df["pred_class"] = idx.astype(int)
                pred_df["pred_class_prob"] = P[np.arange(P.shape[0]), idx]
            else:
                # pred_df["pred_class"] = pred_df["target_pred"]
                pred_df["pred_class_prob"] = np.nan

            # Stable column order for train and holdout rows in one file
            base_cols = ["experiment", "barcode", "chr", "start", "end"]
            if "strand" in pred_df.columns:
                base_cols.append("strand")
            base_cols += ["target_true", "target_pred",
                          "pred_class_prob", "split", "fold"]

            other_cols = [c for c in pred_df.columns if c not in set(
                base_cols + proba_cols)]
            pred_df = pred_df[[
                c for c in base_cols if c in pred_df.columns] + proba_cols + other_cols]

            pred_df.to_csv(
                f'{barcode_preds_path}/{exp}_{table_type}_predictions_train_and_test_allfolds.tsv', sep="\t", index=False)
            print(
                f"Saved: {barcode_preds_path}/{exp}_{table_type}_predictions_train_and_test_allfolds.tsv")
        else:
            print("No prediction rows to save for this experiment.")

        if grid_val_rows:
            grid_all = pd.concat(grid_val_rows, ignore_index=True)
            grid_val_rows_all.append(grid_all)

        if grid_val_rows_all:
            grid_all_both = pd.concat(grid_val_rows_all, ignore_index=True)
            grid_all_path = os.path.join(
                grid_out_path, f"{exp}_{table_type}_xgb_depth_grid_val.tsv")
            grid_all_both.to_csv(grid_all_path, sep='\t', index=False)
            print(f"Saved: {grid_all_path}")

# Aggregate results for this LSF batch job
#
# run_token stamps WHICH run produced this file. Without it the driver had no way to tell a fresh
# output from one left behind by the previous round: on 2026-08-23 a worker of round 4 outlived
# clear_artifacts() and wrote its file into the freshly emptied folder, the round-5 worker for the
# same experiment was then killed, and run_training.py merged round 4's row into round 5 because
# "the file exists". run_training.py drops this column again after checking it.
_out = pd.DataFrame(all_results)
_out["run_token"] = os.getenv("RUN_TOKEN", "")
_out.to_csv(
    f"{output_dir_path}/all_xgboost_results[{start_LSB_JOBINDEX}:{end_LSB_JOBINDEX}].tsv",
    sep='\t', index=False
)
for stream in (sys.stdout, sys.stderr):
    try:
        stream.flush()
    except OSError:
        pass
# try:
#     sys.stdout.flush()
#     sys.stderr.flush()
# finally:
#     sys.stdout.close()
#     sys.stderr.close()