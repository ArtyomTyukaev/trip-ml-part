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
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
sys.path.append(str(PROJECT_ROOT / "settings" / "custom_functions"))

from get_replics_dicts import get_replics_dicts
from get_groups_of_features import get_groups_of_features
from make_folds import make_non_intersecting_folds
from make_folds import train_test_val_split_no_overlap
from check_train_test_split import check_train_test_split
from get_gene_tables import make_intra_and_inter_genic_tables
from reg_to_classification import devide_into_intervals
from group_shap import build_feature_of_column, shap_columns

# === Define dictionaries ===

replics_dicts = get_replics_dicts()
groups_to_features = get_groups_of_features()

##########################################################

def rel(p: str) -> str:
    """Project-relative -> absolute string path."""
    return str((PROJECT_ROOT / p).resolve())

def require_env_path(name: str) -> str:
    return os.environ[name]

def getenv_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def getenv_list_int(name: str, default_list):
    raw = os.getenv(name, "")
    if raw.strip() == "":
        return list(default_list)
    # expected format: "1,2,3,4"
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def getenv_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    return float(raw) if raw.strip() != "" else float(default)


def booster_params():
    """The fixed booster block, with every regularisation knob overridable by an env var.

    The DEFAULTS ARE THE VALUES THAT WERE HARDCODED HERE, so a run that sets nothing is
    bit-identical to every run before 2026-08-27 - that is the point of writing it this way rather
    than editing the numbers.

    Why it is worth having a handle at all. The regressor is barely regularised - `reg_lambda` at
    its default 1, `min_child_weight` 5, `gamma` 0.1 - while the classifier on the SAME data runs
    `reg_lambda` 12, `min_child_weight` 25, `gamma` 2. Measured on intron_splicing, the regressor's
    early stopping fires at a median of 119-140 rounds out of 3000: it is not running out of
    capacity, it starts memorising and the stopper cuts it off. That asymmetry looks unintended and
    is the one lever the depth grid could not test - widening max_depth to 2..7 moved holdout R2 by
    at most 0.008.
    """
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",             # optimal for GPU
        "device": "cuda",
        "learning_rate": getenv_float("LEARNING_RATE", 0.03),
        "max_depth": 6,                    # overridden per grid point by DEPTH_GRID
        "subsample": getenv_float("SUBSAMPLE", 0.8),
        "colsample_bytree": getenv_float("COLSAMPLE_BYTREE", 0.8),
        "reg_lambda": getenv_float("REG_LAMBDA", 1),
        "min_child_weight": getenv_float("MIN_CHILD_WEIGHT", 5),
        "gamma": getenv_float("GAMMA", 0.1),
        "seed": 5673,
        "max_bin": 300,
    }


# env paths (already absolute from notebook)
input_dir = require_env_path("INPUT_DIR")
exp_list = require_env_path("EXP_LIST")
barcode_preds_path = require_env_path("BARCODE_PREDS_PATH")
path_to_file_annot_table_cols_to_drop = require_env_path("ANNOT_DROP_COLS")
path_to_file_columns_to_drop_in_intragenic = require_env_path("INTRA_DROP_COLS")

# internal paths (make them absolute via PROJECT_ROOT)
#
# ARTIFACTS_DIR lets two runs of the SAME task work side by side. It used to be hardcoded, so
# every regression run on the machine wrote into one folder and cleared it on startup - which is
# why "two datasets cannot run at once" was a rule. run_training.py --slot now points each run at
# its own folder, and the default keeps the old path for anything that does not pass it.
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "MODELS/XGB_regressor/artifacts")
output_dir_path = rel(f"{ARTIFACTS_DIR}/output_splitted")
grid_out_path   = rel(f"{ARTIFACTS_DIR}/grid_selection")

categorical_columns = os.getenv("CATEGORICAL_COLUMNS", "ChromHMM")
categorical_columns = [c.strip() for c in categorical_columns.split(",") if c.strip()]

DEPTH_GRID = getenv_list_int("DEPTH_GRID", [3,4,5,6,7])
NUM_BOOST_ROUND = getenv_int("NUM_BOOST_ROUND", 15000)
EARLY_STOP = getenv_int("EARLY_STOP", 2000)
# Seeds the CV fold draw. Unseeded, two rounds of a feature-removal chain differed by a fresh
# fold split as well as by the removed feature, and the split feeds the depth choice and the
# early-stopping iteration. The holdout split already used 2005.
FOLD_SEED = getenv_int("FOLD_SEED", 2005)
lines_per_job = getenv_int("LINES_PER_JOB", 3)

# Group SHAP. OFF unless GROUP_MAP_JSON points at a file, so an ordinary run is byte-identical to
# what it was. The mapping arrives as a file rather than an import on purpose: importing
# group_shap/group_definitions.py here would make this training script depend on the analysis code.
# The file holds {table_type: {raw feature: group}} and is written by group_shap/run_group_shap.py.
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

# LOG_TARGET=1 replaces the target with log10(1 + target). Needed when the column holds
# raw RPM: a few huge values would otherwise dominate the squared error.
# Unset keeps the original behaviour.
LOG_TARGET = os.getenv("LOG_TARGET", "").strip() == "1"

job_index = int(os.environ.get("LSB_JOBINDEX", 1))

log_path = rel(f"{ARTIFACTS_DIR}/log_training/XGB_{job_index}.log")
err_path = rel(f"{ARTIFACTS_DIR}/log_training_errors/XGB_{job_index}.err")

##########################################################

os.makedirs(grid_out_path, exist_ok=True)
os.makedirs(barcode_preds_path, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(os.path.dirname(err_path), exist_ok=True)
os.makedirs(output_dir_path, exist_ok=True)



# index of the current job in the LSF array
with open(exp_list, 'r') as file:
    exp_names = [line.strip() for line in file.readlines()]

# compute the range of lines assigned to this job
start_LSB_JOBINDEX = (job_index - 1) * lines_per_job
end_LSB_JOBINDEX = start_LSB_JOBINDEX + lines_per_job
batch = exp_names[start_LSB_JOBINDEX:end_LSB_JOBINDEX]

if not batch:
    print(f"No experiments for job_index={job_index} (start={start_LSB_JOBINDEX}, end={end_LSB_JOBINDEX})")
    sys.exit(0)


with open(path_to_file_annot_table_cols_to_drop,'r') as file_:
    annot_table_cols_to_drop = [j.strip() for j in file_.readlines()]

with open(path_to_file_columns_to_drop_in_intragenic,'r') as file_:
    columns_to_drop_in_intragenic = [j.strip() for j in file_.readlines()]

# === Logging ===
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

all_results = []

for exp in batch:
    out = make_intra_and_inter_genic_tables(input_dir, exp, replics_dicts, columns_to_drop_in_intragenic, annot_table_cols_to_drop, filename_prefix='.alldata.tab')
    if out is None:
        print(f'Output of make_intra_and_inter_genic_tables() is None')
        continue
    intrAgenic_alltable, intErgenic_alltable = out

    # Guarded because a feature-removal round can drop these two, and they are rewritten here
    # by name.
    for _col in ["MANE_all", "MANE_exonic"]:
        if _col in intErgenic_alltable.columns:
            intErgenic_alltable[_col] = intErgenic_alltable[_col].apply(
                lambda x: np.nan if pd.isna(x) else (0 if x == 0 else 1))



    if LOG_TARGET:
        for table in [intrAgenic_alltable, intErgenic_alltable]:
            table[target_col] = np.log10(1 + table[target_col])

    # One source for both tables. They carried two hand-written copies of the same twelve numbers,
    # so a change to one silently trained the two table types on different boosters.
    tables = {
        'intragenic': (intrAgenic_alltable, booster_params()),
        'intergenic': (intErgenic_alltable, booster_params()),
    }

    for table_type, (table, params) in tables.items():
        if table.empty:
            print(f"{datetime.now():[%Y-%m-%d %H:%M:%S]} Skipping empty {table_type} table for {exp}")
            continue

        # Only the categorical columns this table actually has. get_dummies raises KeyError on
        # a missing one, so once a feature-removal round drops a categorical - ChromHMM, SPIN,
        # Chr and the rest - every worker would die instead of training without it.
        present_categorical = [c for c in categorical_columns if c in table.columns]
        columns_before_encoding = list(table.columns)
        df_cv = pd.get_dummies(table, columns=present_categorical)
        df_cv = df_cv.reset_index().rename(columns={"index": "barcode"})
        df_cv = shuffle(df_cv, random_state=448)

        df_coords = df_cv[['barcode', 'chr', 'start', 'end']].copy()

        holdout_and_train = train_test_val_split_no_overlap(df_cv, test_frac=0.15, val_frac=0, seed=2005)

        holdout_df = df_cv[df_cv['barcode'].isin(holdout_and_train['test'])].copy()
        train_pool_df = df_cv[df_cv['barcode'].isin(holdout_and_train['train'])].copy()

        print(f"Target stats - Train: mean={train_pool_df[target_col].mean():.3f}, std={train_pool_df[target_col].std():.3f}")
        print(f"Target stats - Holdout: mean={holdout_df[target_col].mean():.3f}, std={holdout_df[target_col].std():.3f}")



        exclude_cols = ["barcode", "chr", "start", "end", "strand", target_col]
        feature_cols = [c for c in df_cv.columns if c not in exclude_cols]

        print(f"=== DIAGNOSTICS FOR {exp} {table_type} ===")
        print(f"Total features: {len(feature_cols)}")
        print(f"Train pool size: {len(train_pool_df)}, holdout size: {len(holdout_df)}")

        holdout_df = holdout_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).fillna(0)

        # === Assign barcodes to 3 folds with distance check ===
        folds = make_non_intersecting_folds(train_pool_df, seed=FOLD_SEED)

        # accumulate prediction rows across all folds
        pred_rows_exp = []
        grid_val_rows = []
        grid_val_rows_all = []

        for fold_idx in range(3):
            val_barcodes = folds[fold_idx]
            train_barcodes = set(train_pool_df['barcode']) - val_barcodes

            train_df = train_pool_df[train_pool_df['barcode'].isin(train_barcodes)].copy()
            val_df = train_pool_df[train_pool_df['barcode'].isin(val_barcodes)].copy()

            train_df = shuffle(train_df, random_state=2003)
            val_df = shuffle(val_df, random_state=2000)

            train_df_clean = (
                train_df
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[target_col])
                .fillna(0)
                .reset_index(drop=True)
            )
            val_df_clean = (
                val_df
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=[target_col])
                .fillna(0)
                .reset_index(drop=True)
            )

            # --- build feature matrices as DataFrames (DO NOT convert to .to_numpy at this stage) ---
            X_tr_df  = train_df_clean[feature_cols].copy()
            X_val_df = val_df_clean[feature_cols].copy()
            y_tr = train_df_clean[target_col].to_numpy().astype(np.float32)
            y_val = val_df_clean[target_col].to_numpy().astype(np.float32)

            # --- 1) constant / invalid features detected on TRAIN ---
            EPS = 1e-12
            # column-wise std, numeric_only in case of an unexpected non-numeric column
            stds = X_tr_df.std(axis=0, ddof=0, numeric_only=True)
            bad_mask = (stds <= EPS) | (~np.isfinite(stds.values))
            const_cols = stds.index[bad_mask].tolist()

            if const_cols:
                print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} "
                      f"WARNING: dropping constant/invalid feature (TRAIN fold): {len(const_cols)} columns -> {const_cols[:10]}{' ...' if len(const_cols)>10 else ''}")
                X_tr_df.drop(columns=const_cols, inplace=True, errors="ignore")
                X_val_df.drop(columns=const_cols, inplace=True, errors="ignore")

            # --- 2) final conversion to numpy (or DataFrame can be passed directly to DMatrix) ---
            X_tr = X_tr_df.to_numpy()
            X_val = X_val_df.to_numpy()


            dtr  = xgb.DMatrix(X_tr,  label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)



            grid_rows = []

            for depth in DEPTH_GRID:
                params_cur = dict(params)           # copy to avoid modifying the original dictionary
                params_cur["max_depth"] = int(depth)

                bst = xgb.train(
                        params=params_cur,
                        dtrain=dtr,
                        num_boost_round=NUM_BOOST_ROUND,
                        evals=[(dtr, "train"), (dval, "val")],
                        early_stopping_rounds=EARLY_STOP,
                        verbose_eval=False
                    )

                best_it = (bst.best_iteration or 0) + 1
                val_best_rmse = float(bst.best_score)

                # additional metrics
                y_val_pred = bst.predict(dval, iteration_range=(0, best_it))
                val_rmse = float(np.sqrt(np.mean((y_val - y_val_pred) ** 2)))
                val_spearman = float(spearmanr(y_val, y_val_pred)[0])
                val_r2 = r2_score(y_val, y_val_pred)

                grid_rows.append({
                    "dataset": os.path.basename(input_dir),
                    "experiment": exp,
                    "table_type": table_type,
                    "fold": fold_idx + 1,
                    "max_depth": int(depth),
                    "best_iteration": best_it,
                    "val_best_rmse": val_best_rmse,
                    "val_rmse": val_rmse,
                    "val_spearman": val_spearman,
                    'val_r2':val_r2
                })

                print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} [{table_type.upper()}] {exp} | "
                      f"Fold {fold_idx + 1} | depth={depth} | best_it={best_it} | "
                      f"val_R2={val_r2:.4f} | val_RMSE={val_best_rmse:.6f} | val_Spearman={val_spearman:.4f}")


            grid_df = pd.DataFrame(grid_rows)

            # accumulate grid results for later saving
            grid_df_fold = grid_df.copy()
            grid_df_fold["fold"] = fold_idx + 1
            grid_df_fold["experiment"] = exp
            grid_df_fold["table_type"] = table_type
            grid_val_rows.append(grid_df_fold)

          # end of fold loop
        # =========================================================
        # # SELECT BEST CONFIGURATION BASED ON VALIDATION
        # =========================================================



        all_grid_results = pd.concat(grid_val_rows, ignore_index=True)
        best_row = all_grid_results.loc[all_grid_results["val_best_rmse"].idxmin()]

        chosen_depth_global   = int(best_row["max_depth"])
        chosen_nrounds_global = int(best_row["best_iteration"])

        print(f"[CV-SELECT] depth={chosen_depth_global}, iters={chosen_nrounds_global}, "
              f"val_best_rmse={best_row['val_best_rmse']:.6f}, "
              f"val_spearman={best_row.get('val_spearman', float('nan')):.4f}")

        # # ===== final training on full train_pool_df and evaluation on holdout =====
        # # 0) prepare full training set
        train_full_df = (
            train_pool_df
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=[target_col])
            .fillna(0)
            .reset_index(drop=True)
        )

        X_train_full_df = train_full_df[feature_cols].copy()
        y_train_full = train_full_df[target_col].to_numpy().astype(np.float32)

        # 1) drop constant features on FULL train and align holdout to the same feature set
        EPS = 1e-12
        stds_full = X_train_full_df.std(axis=0, ddof=0, numeric_only=True)
        const_cols_full = stds_full.index[(stds_full <= EPS) | (~np.isfinite(stds_full.values))].tolist()
        if const_cols_full:
            print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} [FINAL] drop const feats: {len(const_cols_full)}\n")
            X_train_full_df.drop(columns=const_cols_full, inplace=True, errors="ignore")


        y_hold = holdout_df[target_col].to_numpy().astype(np.float32)

        X_hold_df = holdout_df[feature_cols].copy()
        X_hold_df.drop(columns=const_cols_full, inplace=True, errors="ignore")

        # if dummy columns are missing in holdout, add zero-filled columns
        missing_in_hold = [c for c in X_train_full_df.columns if c not in X_hold_df.columns]
        for c in missing_in_hold:
            X_hold_df[c] = 0.0

        X_hold_df = X_hold_df[X_train_full_df.columns]

        # 2) create DMatrix and train WITHOUT early stopping (use num_boost_round selected from CV folds)
        dtrain_full = xgb.DMatrix(X_train_full_df.to_numpy(), label=y_train_full)
        dhold       = xgb.DMatrix(X_hold_df.to_numpy(),       label=y_hold)

        params_final = dict(params)
        params_final["max_depth"] = int(chosen_depth_global)

        # =========================================================
        # FINAL TRAINING ON FULL TRAIN SET
        # =========================================================

        final = xgb.train(
            params=params_final,
            dtrain=dtrain_full,
            num_boost_round=int(chosen_nrounds_global),
            verbose_eval=False
        )

        # 3) compute final metrics
        y_pred_train_full = final.predict(dtrain_full)
        train_full_r2 = r2_score(y_train_full, y_pred_train_full)
        train_full_spearman = float(spearmanr(y_train_full, y_pred_train_full)[0])

        y_pred_hold = final.predict(dhold)
        hold_r2 = r2_score(y_hold, y_pred_hold)
        hold_spearman = float(spearmanr(y_hold, y_pred_hold)[0])

        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} [FINAL] {table_type.upper()} {exp} | "
              f"TRAIN(full): R2={train_full_r2:.4f}, Spearman={train_full_spearman:.4f} | "
              f"HOLDOUT: R2={hold_r2:.4f}, Spearman={hold_spearman:.4f}")

        result = {
                            "experiment": exp,
                            "table_type": table_type,
                            "fold": "FINAL",
                            "chosen_max_depth": int(chosen_depth_global),
                            "chosen_best_iteration": int(chosen_nrounds_global),
                            "TRAIN_full_R2": float(train_full_r2),
                            "TRAIN_full_Spearman": float(train_full_spearman),
                            "HOLDOUT_R2": float(hold_r2),
                            "HOLDOUT_Spearman": float(hold_spearman),
                        }

        # --- SHAP values ---
        explainer = shap.Explainer(final)  # XGBoost detects the GPU by itself
        shap_values = explainer(X_train_full_df)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        shap_feature_importance = dict(zip(list(X_train_full_df.columns), mean_abs_shap))


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

        # === TRAIN(full) predictions table ===
        train_meta_cols = ["barcode", "chr", "start", "end"]
        if "strand" in train_full_df.columns:
            train_meta_cols.append("strand")

        tmp_train = train_full_df[train_meta_cols + [target_col]].copy()
        tmp_train.rename(columns={target_col: "target_true"}, inplace=True)
        tmp_train["target_pred"] = y_pred_train_full.astype(float)
        tmp_train["split"] = "train_full"
        tmp_train["fold"] = "FINAL"
        tmp_train["experiment"] = exp
        pred_rows_exp.append(tmp_train)

        # === HOLDOUT predictions table ===
        holdout_meta_cols = ["barcode", "chr", "start", "end"]
        if "strand" in holdout_df.columns:
            holdout_meta_cols.append("strand")

        tmp_hold = holdout_df[holdout_meta_cols + [target_col]].copy()
        tmp_hold.rename(columns={target_col: "target_true"}, inplace=True)
        tmp_hold["target_pred"] = y_pred_hold.astype(float)
        tmp_hold["split"] = "holdout"
        tmp_hold["fold"] = "FINAL"
        tmp_hold["experiment"] = exp
        pred_rows_exp.append(tmp_hold)


        # after fold loop
        # === save per-experiment predictions (train+test) ===
        if pred_rows_exp:
            pred_df = pd.concat(pred_rows_exp, ignore_index=True)

            # put the columns in order (train and test go into one file)
            base_cols = ["experiment", "barcode", "chr", "start", "end"]
            if "strand" in pred_df.columns:
                base_cols.append("strand")
            base_cols += ["target_true", "target_pred", "split", "fold"]

            pred_df = pred_df[
                [c for c in base_cols if c in pred_df.columns] +
                [c for c in pred_df.columns if c not in set(base_cols)]
            ]

            pred_df.to_csv(
                f"{barcode_preds_path}/{exp}_{table_type}_predictions_trainfull_and_holdout.tsv",
                sep="\t", index=False
            )
            print()
            print(f"Saved: {barcode_preds_path}/{exp}_{table_type}_predictions_trainfull_and_holdout.tsv")

        else:
            print("No prediction rows to save for this experiment.")

        if grid_val_rows:
            grid_all = pd.concat(grid_val_rows, ignore_index=True)
            grid_val_rows_all.append(grid_all)

        if grid_val_rows_all:
            grid_all_both = pd.concat(grid_val_rows_all, ignore_index=True)
            grid_all_path = os.path.join(grid_out_path, f"{exp}_{table_type}_xgb_depth_grid_val.tsv")
            grid_all_both.to_csv(grid_all_path, sep='\t', index=False)
            print(f"Saved: {grid_all_path}")
            print()
            print()
            print()

# === Save final dataframe ===
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