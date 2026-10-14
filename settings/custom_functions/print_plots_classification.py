import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_agg_df_cls(DATA: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare agg_df for classification plotting.

    Input DATA must contain:
      - 'experiment'
      - 'table_type'  ('intragenic' / 'intergenic')
      - metric columns (TEST_b_accuracy, TEST_accuracy, etc.)

    Output:
      - agg_df: mean-aggregated over (experiment, table_type)
    """
    metrics = [
        "TEST_b_accuracy",
        "TEST_accuracy",
        "TEST_roc_auc_weighted",
        "TRAIN_b_accuracy",
        "TRAIN_accuracy",
        "TRAIN_roc_auc_weighted",
    ]

    metrics_present = [m for m in metrics if m in DATA.columns]
    if not metrics_present:
        raise ValueError(f"None of the expected metric columns found. Columns: {DATA.columns.tolist()}")

    print(f"[INFO] Input DATA shape: {DATA.shape}")
    print(f"[INFO] Metrics used: {metrics_present}")

    agg_df = (
        DATA.groupby(["experiment", "table_type"], as_index=False)[metrics_present]
        .mean()
    )

    print(f"[INFO] Output agg_df shape: {agg_df.shape}")

    return agg_df

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


def plot_metric_clustermap_cls(
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
    # === size control ===
    width_per_exp: float = 2,
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
    split = split.lower().strip()
    table_type = table_type.lower().strip()

    if table_type not in {"intragenic", "intergenic"}:
        raise ValueError("table_type must be 'intragenic' or 'intergenic'")

    if split not in {"train", "test", "both"}:
        raise ValueError("split must be 'train', 'test', or 'both'")

    if split == "train":
        metrics = ["TRAIN_b_accuracy", "TRAIN_accuracy", "TRAIN_roc_auc_weighted"]
    elif split == "test":
        metrics = ["TEST_b_accuracy", "TEST_accuracy", "TEST_roc_auc_weighted"]
    else:
        metrics = [
            "TEST_b_accuracy", "TEST_accuracy", "TEST_roc_auc_weighted",
            "TRAIN_b_accuracy", "TRAIN_accuracy", "TRAIN_roc_auc_weighted",
        ]

    if "table_type" not in agg_df.columns:
        raise ValueError("agg_df must contain 'table_type'")
    if "experiment" not in agg_df.columns:
        raise ValueError("agg_df must contain 'experiment'")

    sub = agg_df[agg_df["table_type"].str.lower() == table_type].copy()

    if dropnan:
        sub.dropna(axis=1, how="all", inplace=True)

    if sub.empty:
        print(f"No data for table_type={table_type}")
        return

    available_metrics = [m for m in metrics if m in sub.columns]
    if not available_metrics:
        print(f"No matching metric columns found for table_type={table_type}")
        return

    mat = sub.groupby("experiment")[available_metrics].mean()
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
        print("metric_data is empty after cleaning.")
        return

    n_exps = metric_data.shape[1]
    figsize = (width_per_exp * n_exps, height)
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

    title_split = {"train": "TRAIN", "test": "TEST", "both": "TRAIN+TEST"}[split]

    g.ax_heatmap.set_title(
        f"{model} | {dataset_name} | {table_type} | {title_split}",
        fontsize=title_font_size,
        pad=20,
    )
    g.ax_heatmap.set_xlabel("Experiment", labelpad=15, fontsize=axis_font_size)
    g.ax_heatmap.set_ylabel("Metric", labelpad=15, fontsize=axis_font_size)

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
        fname = f"cls_metrics_heatmap_{model}_{dataset_name}_{table_type}_{title_split}.png"
        path = os.path.join(outdir, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.show()
