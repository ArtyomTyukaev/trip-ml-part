"""
SHAP heatmaps built from matplotlib primitives (no seaborn.clustermap).

Why this module exists
---------------------
`sns.clustermap` lays out four axes (row dendrogram, column dendrogram, heatmap,
colorbar) as *fractions of the figure*. The heatmap therefore gets whatever space is
left over, so the cell aspect ratio changes with the number of rows/columns, and
`cbar=False` / `row_cluster=False` still reserve their slots (the empty white 0/1 box
in the corner of the old figures).

Here the geometry runs the other way round: the caller gives the cell size in inches
(`cell_w_in`, `cell_h_in`) and the figure size is derived from the data shape. Every
axes is placed with an explicit `fig.add_axes`, so an axes that is turned off costs
nothing. A 100-column heatmap is readable by construction.

Semantics kept from `print_plots_regression.plot_shap_clustermap_pct` /
`plot_shap_clustermap_log_pct`:
  * rows = experiments (alphabetical unless the caller passes `experiment_order`),
    columns = features
  * features sorted by descending total |SHAP| (this is now also the drawing order,
    columns are hierarchically clustered exactly as sns.clustermap did)
  * NaN cells (feature absent from that model) are black
  * values at or below the `white_pct` percentile are white
  * x tick labels coloured by feature group (`groups_to_features` / `group_colors`)
  * title `"{model} | {dataset_name} | {table_type}"`.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy.cluster.hierarchy import linkage, dendrogram

# Fallback width of one character as a fraction of the font size, used only if the
# text cannot be measured. Margins do not have to be exact, only large enough.
_CHAR_W = 0.62


def _text_width_in(labels, fontsize: float) -> float:
    """
    Width in inches of the longest of `labels` when rendered at `fontsize`.

    Measured with TextPath (no figure needed) so the margins fit the actual labels
    instead of a guessed character width.
    """
    labels = [str(s) for s in labels] or [""]
    try:
        fp = FontProperties(size=fontsize)
        widths = [TextPath((0, 0), s, prop=fp).get_extents().width for s in labels]
        return max(widths) / 72.0
    except Exception:                                   # pragma: no cover
        longest = max(len(s) for s in labels)
        return longest * fontsize * _CHAR_W / 72.0


# ----------------------------------------------------------------------------------
# data preparation
# ----------------------------------------------------------------------------------

def build_shap_matrix(
    agg_df: pd.DataFrame,
    table_type: str,
    dropnan: bool = True,
) -> pd.DataFrame:
    """
    Extract the SHAP matrix for one table_type.

    Returns a DataFrame: rows = experiments (alphabetical), columns = features
    without the 'SHAP_' prefix, sorted by descending sum of |SHAP| over experiments.

    `dropnan=True` drops features that are NaN for *every* experiment shown
    (i.e. the feature was absent from all of these models).
    """
    for col in ("experiment", "table_type"):
        if col not in agg_df.columns:
            raise ValueError(f"agg_df must contain '{col}'")

    table_type = table_type.lower().strip()
    sub = agg_df[agg_df["table_type"].str.lower() == table_type]

    shap_cols = [c for c in sub.columns if c.startswith("SHAP_")]
    if not shap_cols:
        raise ValueError("no SHAP_* columns in agg_df")

    mat = sub.groupby("experiment")[shap_cols].mean()
    mat.columns = [c.replace("SHAP_", "") for c in mat.columns]

    if dropnan:
        mat = mat.dropna(axis=1, how="all")

    # Descending total importance. kind="mergesort" is a stable sort, so the many
    # features with total importance exactly 0 keep their original column order
    # instead of being shuffled arbitrarily (the old code used the default quicksort,
    # whose tie order is not reproducible).
    order = (mat.abs().sum(axis=0)
                .sort_values(ascending=False, kind="mergesort").index)
    return mat[order]


def column_order(mat: pd.DataFrame, cluster_on: pd.DataFrame = None):
    """
    Column order for drawing: hierarchical clustering of the columns, average linkage
    on euclidean distance, which is exactly what sns.clustermap did by default. This
    keeps the new figures directly comparable with the old ones.

    `cluster_on` is the matrix actually clustered. Pass the matrix that gets DRAWN:
    sns.clustermap clustered whatever array it was handed, so the linear version has
    to cluster the NaN->0 filled values and the log version the vmin-substituted ones,
    otherwise the two would come out in different orders. Defaults to `mat.fillna(0)`.

    Returns the list of columns.
    """
    if mat.shape[1] < 2:
        return list(mat.columns)

    src = mat.fillna(0.0) if cluster_on is None else cluster_on
    Z = linkage(src.T.values, method="average", metric="euclidean")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    return [mat.columns[i] for i in leaves]


def row_order_and_labels(mat: pd.DataFrame, experiment_order, experiment_labels):
    """`(mat with its rows in the caller's order, the text to draw for them)`.

    The counterpart of `column_order` for the experiment axis, except that the order is
    not derived from the data. It is handed in, because this module must not depend on the
    analysis code. Both arguments None leaves everything as it was: the alphabetical index
    `build_shap_matrix` produced, drawn as the codes themselves.

    The reindex is restricted to the experiments actually present, so a panel holding a
    subset does not gain a row of NaN for every one it is missing - and because nothing is
    dropped, the feature clustering downstream is unaffected: permuting the rows permutes
    the coordinates of every column vector alike and cannot change a euclidean distance.

    `experiment_labels` may be a mapping keyed on the code or a sequence parallel to
    `experiment_order`. It is applied by looking the code up, never by position, and a code
    it does not know is drawn unchanged - which is what lets a non-intron experiment
    (`5ss_noP`) share an axis with the introns without a special case. The index stays the
    code either way, because it is the join key of every table underneath the figure.
    """
    if experiment_order is not None:
        mat = mat.reindex([e for e in experiment_order if e in mat.index])

    if experiment_labels is None:
        return mat, list(mat.index)
    if isinstance(experiment_labels, dict):
        mapping = experiment_labels
    elif experiment_order is None:
        raise ValueError(
            "experiment_labels given as a sequence needs experiment_order to pair it with"
        )
    else:
        mapping = dict(zip(experiment_order, experiment_labels))
    return mat, [mapping.get(e, e) for e in mat.index]


# ----------------------------------------------------------------------------------
# generic renderer
# ----------------------------------------------------------------------------------

def _render_heatmap(
    values: np.ndarray,          # already ordered, may contain NaN
    row_labels,
    col_labels,
    cmap,
    norm,
    title: str,
    cbar_label: str,
    cell_w_in: float,
    cell_h_in: float,
    show_colorbar: bool,
    colorbar_w_in: float,
    title_font_size: int,
    x_tick_font_size: int,
    y_tick_font_size: int,
    grid_lw: float,
    x_label_colors,              # list of colors, one per column
):
    """
    Draw one heatmap with explicitly placed axes. All sizes are in inches, so the
    cell aspect ratio is exactly `cell_w_in : cell_h_in` in the saved file.
    """
    n_rows, n_cols = values.shape

    heat_w = n_cols * cell_w_in
    heat_h = n_rows * cell_h_in

    # margins reserved for tick labels / title / colorbar. There are no axis labels
    # and no dendrogram, so nothing else needs space.
    left_in = 0.12 + _text_width_in(row_labels, y_tick_font_size)
    bottom_in = 0.12 + _text_width_in(col_labels, x_tick_font_size)
    top_in = 0.30 + title_font_size * 1.7 / 72.0
    right_in = 0.25
    if show_colorbar:
        # colorbar bar + its tick labels
        right_in = 0.45 + colorbar_w_in + 0.85

    # The title is centred over the heatmap. Widen the margins if it would stick out
    # of the figure (we do not use bbox_inches="tight", so it would simply be cut).
    title_w = _text_width_in([title], title_font_size)
    centre = left_in + heat_w / 2.0
    left_in += max(0.0, title_w / 2.0 + 0.1 - centre)
    centre = left_in + heat_w / 2.0
    right_in += max(0.0,
                    centre + title_w / 2.0 + 0.1 - (left_in + heat_w + right_in))

    fig_w = left_in + heat_w + right_in
    fig_h = bottom_in + heat_h + top_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([
        left_in / fig_w,
        bottom_in / fig_h,
        heat_w / fig_w,
        heat_h / fig_h,
    ])

    masked = np.ma.masked_invalid(values)   # NaN -> cmap "bad" colour (black)
    ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        aspect="auto",          # the axes box already has the right shape
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, ha="center",
                       fontsize=x_tick_font_size)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, rotation=0, fontsize=y_tick_font_size)

    for lbl, color in zip(ax.get_xticklabels(), x_label_colors):
        lbl.set_color(color)

    # white cell separators (the old figures used linewidths=5 on clustermap)
    if grid_lw > 0:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=grid_lw)
        ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    # colorbar in its own axes: cannot distort the heatmap
    if show_colorbar:
        cbar_h = min(heat_h, 3.0)
        cax = fig.add_axes([
            (left_in + heat_w + 0.45) / fig_w,
            (bottom_in + (heat_h - cbar_h) / 2.0) / fig_h,
            colorbar_w_in / fig_w,
            cbar_h / fig_h,
        ])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, cax=cax, extend="min")
        cb.set_label(cbar_label, fontsize=y_tick_font_size)
        cb.ax.tick_params(labelsize=y_tick_font_size - 1)

    # title centred over the heatmap
    fig.text(
        (left_in + heat_w / 2.0) / fig_w,
        (fig_h - 0.25) / fig_h,
        title,
        ha="center", va="top", fontsize=title_font_size,
    )

    return fig, ax


def _x_label_colors(columns, groups_to_features, group_colors):
    """One colour per column. Ungrouped features stay black."""
    if not groups_to_features or not group_colors:
        return ["black"] * len(columns)

    feature_to_group = {
        f: g for g, flist in groups_to_features.items() for f in flist
    }
    return [
        group_colors.get(feature_to_group.get(c), "black")
        for c in columns
    ]


def _save(fig, save, outdir, fname, dpi):
    if not save:
        return None
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    # no bbox_inches="tight": the figure box is already exactly the right size, and
    # trimming it would silently change the cell aspect ratio.
    fig.savefig(path, dpi=dpi, facecolor="white")
    print(f"Saved: {path}")
    return path


# ----------------------------------------------------------------------------------
# public plots
# ----------------------------------------------------------------------------------

def plot_shap_clustermap_pct(
    agg_df: pd.DataFrame,
    dataset_name: str,
    table_type: str,
    model: str = "XGBoost",

    # Experiment (row) axis. Both None keep today's behaviour: the alphabetical index of
    # build_shap_matrix, drawn as the codes. See row_order_and_labels.
    experiment_order=None,
    experiment_labels=None,

    # Old-style figure size, kept so existing notebook cells work unchanged.
    # width is the total heatmap width in inches and is divided by the number of
    # columns to get the cell width. height_per_exp is the cell height.
    width: float = 40.0,
    height_per_exp: float = 0.63,

    # Direct cell geometry. If given, these win over width / height_per_exp.
    cell_w_in: float = None,
    cell_h_in: float = None,

    # optional decorations
    show_colorbar: bool = True,
    colorbar_w_in: float = 0.18,

    # fonts
    title_font_size: int = 20,
    x_tick_font_size: int = 9,
    y_tick_font_size: int = 11,
    grid_lw: float = 0.5,

    # behaviour
    white_pct: float = 0.0,
    dropnan: bool = True,
    save: bool = True,
    outdir: str = "plots",
    dpi: int = 200,
    show: bool = True,

    groups_to_features: dict | None = None,
    group_colors: dict | None = None,
):
    """
    SHAP heatmap, linear colour scale. Replacement for `plot_shap_clustermap_pct`.

    Rows = experiments (alphabetical unless `experiment_order` fixes them), columns =
    features sorted by descending total |SHAP|.

    Colour rules
      * NaN                    -> black  (feature absent from that model)
      * value <= eps           -> white, where eps is the `white_pct` percentile of
                                  all finite |SHAP| values (eps = 0 if white_pct = 0,
                                  so exact zeros are white too)
      * eps .. max             -> Reds

    Figure size is derived: width  = margins + n_features * cell_w_in
                            height = margins + n_experiments * cell_h_in

    Returns the matplotlib Figure.
    """
    table_type = table_type.lower().strip()
    mat = build_shap_matrix(agg_df, table_type, dropnan=dropnan)
    if mat.empty:
        print(f"No data for {dataset_name} - {table_type}")
        return None

    mat, row_labels = row_order_and_labels(mat, experiment_order, experiment_labels)

    # sns.clustermap was handed the NaN->0 filled matrix, so cluster the same thing
    mat = mat[column_order(mat, cluster_on=mat.fillna(0.0))]
    values = mat.values.astype(float)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print("All values are NaN, nothing to plot.")
        return None

    white_pct = float(np.clip(white_pct, 0.0, 100.0))
    eps = np.percentile(np.abs(finite), white_pct) if white_pct > 0 else 0.0

    vmax = float(finite.max())
    if not np.isfinite(vmax) or vmax <= eps:
        vmax = eps * 10 if eps > 0 else 1.0

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_under("white")   # <= eps
    cmap.set_bad("black")     # NaN
    # nextafter so that a value exactly equal to eps also falls under vmin -> white
    norm = Normalize(vmin=np.nextafter(eps, np.inf), vmax=vmax)

    n_nan = int(np.isnan(values).sum())
    n_finite = int(finite.size)
    frac_white = (np.abs(finite) <= eps).sum() / n_finite * 100.0
    print(
        f"[INFO] shape={values.shape}, white_pct={white_pct:.4g}%, "
        f"eps={eps:.6g}, vmax={vmax:.6g}, white={frac_white:.4f}%, "
        f"NaN(black) cells={n_nan}"
    )

    # resolve geometry: explicit cell sizes win, otherwise derive from the old-style
    # width / height_per_exp so the figure comes out the size the caller expected
    cw = cell_w_in if cell_w_in is not None else width / max(len(mat.columns), 1)
    ch = cell_h_in if cell_h_in is not None else height_per_exp

    fig, _ = _render_heatmap(
        values=values,
        row_labels=row_labels,
        col_labels=list(mat.columns),
        cmap=cmap,
        norm=norm,
        title=f"{model} | {dataset_name} | {table_type}",
        cbar_label="mean |SHAP|",
        cell_w_in=cw,
        cell_h_in=ch,
        show_colorbar=show_colorbar,
        colorbar_w_in=colorbar_w_in,
        title_font_size=title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        grid_lw=grid_lw,
        x_label_colors=_x_label_colors(mat.columns, groups_to_features, group_colors),
    )

    _save(fig, save, outdir,
          f"shap_heatmap_{model}_{dataset_name}_{table_type}.png", dpi)

    if show:
        plt.show()
        return None      # old behaviour: notebooks must not re-display
    return fig


def plot_shap_clustermap_log_pct(
    agg_df: pd.DataFrame,
    dataset_name: str,
    table_type: str,
    model: str = "XGBoost",

    # Experiment (row) axis. Both None keep today's behaviour: the alphabetical index of
    # build_shap_matrix, drawn as the codes. See row_order_and_labels.
    experiment_order=None,
    experiment_labels=None,

    # Old-style figure size, kept so existing notebook cells work unchanged.
    # width is the total heatmap width in inches and is divided by the number of
    # columns to get the cell width. height_per_exp is the cell height.
    width: float = 40.0,
    height_per_exp: float = 0.63,

    # Direct cell geometry. If given, these win over width / height_per_exp.
    cell_w_in: float = None,
    cell_h_in: float = None,

    # optional decorations
    show_colorbar: bool = True,
    colorbar_w_in: float = 0.18,

    # fonts
    title_font_size: int = 20,
    x_tick_font_size: int = 9,
    y_tick_font_size: int = 11,
    grid_lw: float = 0.5,

    # behaviour
    white_pct: float = 0.0,
    dropnan: bool = True,
    save: bool = True,
    outdir: str = "plots",
    dpi: int = 200,
    show: bool = True,

    groups_to_features: dict | None = None,
    group_colors: dict | None = None,
):
    """
    SHAP heatmap, logarithmic colour scale. Replacement for
    `plot_shap_clustermap_log_pct`.

    Handling of values a log scale cannot take (documented masking):
      * NaN  -> masked with `np.ma.masked_invalid` and drawn black. It never reaches
                LogNorm, so no warning and no log(NaN).
      * <= 0  (mostly exact zeros: the feature was in the model but never used)
             -> replaced by the sentinel `vmin / 10` before drawing. The sentinel is
                positive, so LogNorm is safe, and it is below vmin, so the cell gets
                the "under" colour, white. log(0) is therefore never evaluated.
      * 0 < value <= thr, thr = the `white_pct` percentile of the positive values
             -> also below vmin, hence white.

    Colour range: vmin = just above thr, vmax = max positive value.

    Returns the matplotlib Figure.
    """
    table_type = table_type.lower().strip()
    mat = build_shap_matrix(agg_df, table_type, dropnan=dropnan)
    if mat.empty:
        print(f"No data for {dataset_name} - {table_type}")
        return None

    # Before the thresholds, so `values` and the clustering source below are built from the
    # rows in their final order. None of thr/vmin/vmax depends on the row order anyway.
    mat, row_labels = row_order_and_labels(mat, experiment_order, experiment_labels)

    values = mat.values.astype(float)

    finite = values[np.isfinite(values)]
    pos = finite[finite > 0]
    if pos.size == 0:
        print("No positive values for log-scale")
        return None

    white_pct = float(np.clip(white_pct, 0.0, 100.0))
    thr = np.percentile(pos, white_pct) if white_pct > 0 else pos.min()

    vmin = np.nextafter(max(thr, pos.min()), np.inf)
    vmax = float(pos.max())
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nextafter(vmin, np.inf))

    # The old log figure clustered the matrix it actually drew, in which NaN and
    # non-positive cells had been replaced by vmin. Cluster the same thing, so the
    # column order comes out identical, and only then reorder.
    cluster_src = mat.copy()
    cluster_src[mat.isna() | (mat <= 0)] = vmin
    mat = mat[column_order(mat, cluster_on=cluster_src)]
    values = mat.values.astype(float)

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_under("white")   # <= thr, and the zero/negative sentinel
    cmap.set_bad("black")     # NaN
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # positive sentinel for non-positive values: keeps LogNorm away from log(0)
    display = values.copy()
    nonpositive = np.isfinite(display) & (display <= 0)
    display[nonpositive] = vmin / 10.0

    n_nan = int(np.isnan(values).sum())
    n_finite = int(finite.size)
    frac_under = (finite <= thr).sum() / n_finite * 100.0
    print(
        f"[INFO] shape={values.shape}, white_pct={white_pct:.4g}%, "
        f"thr={thr:.6g}, vmin={vmin:.6g}, vmax={vmax:.6g}, "
        f"under(white)={frac_under:.4f}%, non-positive cells={int(nonpositive.sum())}, "
        f"NaN(black) cells={n_nan}"
    )

    # resolve geometry: explicit cell sizes win, otherwise derive from the old-style
    # width / height_per_exp so the figure comes out the size the caller expected
    cw = cell_w_in if cell_w_in is not None else width / max(len(mat.columns), 1)
    ch = cell_h_in if cell_h_in is not None else height_per_exp

    fig, _ = _render_heatmap(
        values=display,
        row_labels=row_labels,
        col_labels=list(mat.columns),
        cmap=cmap,
        norm=norm,
        title=(f"{model} | {dataset_name} | {table_type}"
               f"  (log scale, white_pct={white_pct:.3g}%, thr={thr:.3g})"),
        cbar_label="mean |SHAP| (log)",
        cell_w_in=cw,
        cell_h_in=ch,
        show_colorbar=show_colorbar,
        colorbar_w_in=colorbar_w_in,
        title_font_size=title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        grid_lw=grid_lw,
        x_label_colors=_x_label_colors(mat.columns, groups_to_features, group_colors),
    )

    _save(fig, save, outdir,
          f"log_shap_heatmap_{model}_{dataset_name}_{table_type}"
          f"_pct{white_pct:.3g}.png", dpi)

    if show:
        plt.show()
        return None      # old behaviour: notebooks must not re-display
    return fig
