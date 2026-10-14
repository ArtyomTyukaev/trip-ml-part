import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def prepare_agg_df_for_plots(
    DATA: pd.DataFrame,
    metrics=None,
) -> pd.DataFrame:
    """
    Prepare agg_df for plotting.

    Input DATA must contain:
      - 'experiment'
      - 'table_type'  (e.g. 'intragenic' / 'intergenic')
      - regression metric columns (TRAIN_full_R2, HOLDOUT_R2, etc.)
      - optional SHAP_* columns

    Output:
      - agg_df: mean-aggregated over (experiment, table_type)
    """

    print(f"[INFO] Input DATA shape: {DATA.shape}")

    # Default regression metrics
    if metrics is None:
        metrics = [
            "TRAIN_full_R2",
            "TRAIN_full_Spearman",
            "HOLDOUT_R2",
            "HOLDOUT_Spearman",
        ]

    df = DATA.copy()

    # Columns to aggregate: selected metrics + all SHAP_* columns
    shap_cols = [c for c in df.columns if c.startswith("SHAP_")]
    metric_cols_present = [c for c in metrics if c in df.columns]
    cols_to_use = metric_cols_present + shap_cols

    print(f"[INFO] Metrics used: {metric_cols_present}")
    print(f"[INFO] SHAP columns found: {len(shap_cols)}")

    # Aggregate to one row per (experiment, table_type)
    agg_df = (
        df.groupby(["experiment", "table_type"], as_index=False)[cols_to_use]
          .mean()
    )

    print(f"[INFO] Output agg_df shape: {agg_df.shape}")

    return agg_df

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _experiment_label_map(experiment_order, experiment_labels):
    """Code -> drawn text. Empty dict when the caller asked for no relabelling.

    `experiment_labels` may be a mapping keyed on the code, or a sequence parallel to
    `experiment_order`. Either way the code stays the index and the join key and only the
    tick text changes, and a code with no entry is drawn unchanged - which is what lets a
    non-intron experiment (`5ss_noP`) share an axis with the introns without a special case.
    """
    if experiment_labels is None:
        return {}
    if isinstance(experiment_labels, dict):
        return dict(experiment_labels)
    if experiment_order is None:
        raise ValueError(
            "experiment_labels given as a sequence needs experiment_order to pair it with"
        )
    return dict(zip(experiment_order, experiment_labels))


def plot_metric_clustermap(
    agg_df: pd.DataFrame,
    dataset_name: str,
    table_type: str,
    model: str = "XGBoost",
    split: str = "both",

    # === experiment axis ===
    # Both None reproduce today's behaviour exactly: sorted(), then clustering on top.
    # experiment_order is the list of codes left to right, experiment_labels the text to
    # draw for them. This module must not depend on the analysis code, so the caller supplies both.
    experiment_order=None,
    experiment_labels=None,

    # === simple size control ===
    width_per_exp: float = 0.95,
    height: float = 7.0,

    # === font sizes ===
    title_font_size: int = 28,
    axis_font_size: int = 20,
    cell_font_size: int = 16,

    # === other ===
    save: bool = True,
    outdir: str = "plots",
    dropnan: bool = True,
):
    """
    Size logic:
        width  = width_per_exp * n_experiments
        height = fixed (height)

    Experiments are on X-axis.
    """

    table_type = table_type.lower().strip()
    split = split.lower().strip()

    if table_type not in {"intragenic", "intergenic"}:
        raise ValueError("table_type must be 'intragenic' or 'intergenic'")

    if split not in {"train", "test", "both"}:
        raise ValueError("split must be 'train', 'test', or 'both'")

    # metric selection

    if split == "train":
        metrics = ["TRAIN_full_R2", "TRAIN_full_Spearman"]
    elif split == "test":
        metrics = ["HOLDOUT_R2", "HOLDOUT_Spearman"]
    else:
        metrics = [
            "HOLDOUT_R2", "HOLDOUT_Spearman",
            "TRAIN_full_R2", "TRAIN_full_Spearman"
        ]

    sub = agg_df.copy()

    if "table_type" not in sub.columns:
        raise ValueError("agg_df must contain 'table_type'")

    sub = sub[sub["table_type"].str.lower() == table_type].copy()

    if dropnan:
        sub.dropna(axis=1, how="all", inplace=True)

    if sub.empty:
        print(f"No data for table_type={table_type}")
        return

    if "experiment" not in sub.columns:
        raise ValueError("agg_df must contain 'experiment'")

    mat = sub.groupby("experiment")[metrics].mean()

    if experiment_order is None:
        mat = mat.reindex(sorted(mat.index))
    else:
        # Restricted to the experiments this panel actually holds, so a run with fewer of
        # them does not gain an all-NaN column for every one it is missing.
        mat = mat.reindex([e for e in experiment_order if e in mat.index])

    metric_data = mat.T

    if dropnan:
        metric_data.dropna(axis=0, how="all", inplace=True)
        metric_data.dropna(axis=1, how="all", inplace=True)

    if metric_data.empty:
        print("metric_data empty")
        return

    n_exps = metric_data.shape[1]

    # === simple figsize logic ===

    figsize = (
        width_per_exp * n_exps,
        height
    )

    print(f"FIGSIZE: {figsize}")
    print(f"[INFO] experiments={n_exps}")

    g = sns.clustermap(
        metric_data,
        cmap="coolwarm",
        linewidths=4,
        linecolor="white",
        figsize=figsize,
        # Clustering re-orders the columns and so throws the reindex above away - that is
        # why the two panels of one figure used to put the same experiment in different
        # columns. Off exactly when the caller has fixed the order.
        col_cluster=experiment_order is None,
        row_cluster=False,
        cbar=False,
        dendrogram_ratio=(0.1, 0.1),
        annot=True,
        annot_kws={"size": cell_font_size},
        fmt=".2f",
        vmin=0,
        vmax=1,
    )

    g.ax_col_dendrogram.set_visible(False)
    g.cax.set_visible(False)

    title_split = {
        "train": "TRAIN",
        "test": "TEST",
        "both": "TRAIN+TEST"
    }[split]

    g.ax_heatmap.set_title(
        f"{model} | {dataset_name} | {table_type} | {title_split}",
        fontsize=title_font_size,
        pad=20
    )

    g.ax_heatmap.set_xlabel(
        "Experiment",
        fontsize=axis_font_size,
        labelpad=15
    )

    g.ax_heatmap.set_ylabel(
        "Metric",
        fontsize=axis_font_size,
        labelpad=15
    )

    label_map = _experiment_label_map(experiment_order, experiment_labels)
    if label_map:
        # Looked up on the drawn code, not by position: with col_cluster the tick order is
        # not the matrix order, and a panel holding a subset is shorter than
        # experiment_order, so pairing by index would put a gene name on the wrong column.
        g.ax_heatmap.set_xticklabels(
            [label_map.get(t.get_text(), t.get_text())
             for t in g.ax_heatmap.get_xticklabels()]
        )

    for lbl in g.ax_heatmap.get_xticklabels():
        lbl.set_fontsize(axis_font_size)
        lbl.set_rotation(90)

    for lbl in g.ax_heatmap.get_yticklabels():
        lbl.set_fontsize(axis_font_size)
        lbl.set_rotation(0)

    g.fig.subplots_adjust(left=0.30, bottom=0.20)

    if save:

        os.makedirs(outdir, exist_ok=True)

        fname = f"reg_metrics_heatmap_{dataset_name}_{table_type}_{title_split}.png"

        path = os.path.join(outdir, fname)

        plt.savefig(path, dpi=300, bbox_inches="tight")

        print(f"Saved: {path}")

    plt.show()


# The two SHAP heatmaps used to live here, built on sns.clustermap. They now come from
# shap_heatmaps.py, which exposes the same names and the same call signature, so callers
# and notebook cells do not change. The originals are kept, fully commented out, in
# archive/old_shap_plots.py together with the reasons they were replaced.
from shap_heatmaps import plot_shap_clustermap_pct, plot_shap_clustermap_log_pct
