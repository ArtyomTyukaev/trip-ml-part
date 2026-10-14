import numpy as np
from datetime import datetime
import os
import pandas as pd
import sys
from datetime import datetime

def check_train_test_split(train_df, val_df, feature_cols: list, target_col='exp.log10'):
    # === Drop rows with NaN/inf in y ===
    initial_train_size = len(train_df)
    initial_val_size = len(val_df)

    train_df = train_df[np.isfinite(train_df[target_col])]
    val_df = val_df[np.isfinite(val_df[target_col])]

    if len(train_df) < initial_train_size:
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Removed {initial_train_size - len(train_df)} rows with NaN/inf in y_train")

    if len(val_df) < initial_val_size:
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Removed {initial_val_size - len(val_df)} rows with NaN/inf in y_val")

    # === Drop features with zero std on train ===
    X_train_df = train_df[feature_cols].copy()
    X_val_df = val_df[feature_cols].copy()

    stds = X_train_df.std()
    non_const_cols = stds[stds > 0].index.tolist()

    removed_features = list(set(feature_cols) - set(non_const_cols))
    if removed_features:
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Removed constant features: {removed_features}")

    feature_cols = non_const_cols  # update the feature list
    if not feature_cols:
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} All features are constant, fold skipped")
        return None

    X_train = X_train_df[feature_cols].values
    X_val = X_val_df[feature_cols].values
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values

    if val_df.empty or train_df.empty:
        print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Empty train/val after the split, skipping the fold")
        return None
                                    # === Check for NaN/inf ===
    def report_invalid(arr, name):
        try:
            arr = arr.astype(float)
            if not np.all(np.isfinite(arr)):
                print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} WARNING: NaN or inf found in {name}, fold skipped")
                return False
        except Exception as e:
            print(f"{datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')} Error while checking {name}: {e}, fold skipped")
            return False
        return True

    if not all([report_invalid(X_train, 'X_train'),
                report_invalid(X_val, 'X_val'),
                report_invalid(y_train, 'y_train'),
                report_invalid(y_val, 'y_val')]):
        return None
    
    train_df_clean = train_df.reset_index(drop=True).copy()
    val_df_clean   = val_df.reset_index(drop=True).copy()

        
    return X_train, X_val, y_train, y_val, feature_cols, train_df_clean, val_df_clean
