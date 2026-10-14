"""
Vertical SHAP heatmap: features down the Y axis, experiments across the X axis.

Differences from the horizontal pair in shap_heatmaps.py, all requested for readability:

  * features are rows, so a long feature list grows downwards and the names read
    horizontally, no rotated head needed
  * features are ordered by decreasing importance, top to bottom. No clustering here, so
    the order is exactly the ranking
  * a feature missing from a model is GREY, not black, so it does not draw the eye
  * rows that would show nothing are dropped: a feature missing from every experiment, and
    a feature whose every value falls at or below the white threshold. What is left is
    only rows with at least one visible cell
  * the figure has a fixed total height and the row height is derived from it, so the
    number of features does not change how tall the picture is.

One function does both colour scales, `log_scale=False` gives the linear one.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

from shap_heatmaps import build_shap_matrix, _text_width_in, _x_label_colors, _save

MISSING_COLOR = "0.97"     # very light grey for features absent from the model
# Note: white cells (below the threshold) are pure white, so pushing this any closer to
# 1.0 makes "missing" and "contributed nothing" indistinguishable on screen.


def plot_shap_heatmap_vertical(
    agg_df: pd.DataFrame,
    dataset_name: str,
    table_type: str,
    model: str = "XGBoost",
    log_scale: bool = False,

    # Height of the heatmap itself in inches. The row height is height / number of
    # features, so the rows stretch to fill it and every figure comes out the same height
    # whatever the feature count. min_cell_h_in is a floor on the row height: with very
    # many features the rows would become unreadably thin, and past that point the figure
    # grows taller instead. cell_h_in overrides both and fixes the row height outright.
    height: float = 11.0,
    min_cell_h_in: float = 0.18,
    cell_h_in: float = None,
    cell_w_in: float = 0.85,

    show_colorbar: bool = True,
    colorbar_w_in: float = 0.18,

    title_font_size: int = 15,
    x_tick_font_size: int = 11,
    y_tick_font_size: int = 8,
    grid_lw: float = 0.64,

    white_pct: float = 20.0,
    drop_invisible: bool = True,
    experiment_order=None,

    save: bool = True,
    outdir: str = "plots",
    dpi: int = 200,
    show: bool = True,

    groups_to_features: dict = None,
    group_colors: dict = None,
):
    """
    Rows are features, ordered by decreasing total |SHAP|. Columns are experiments.

    Colours
      * missing feature        -> light grey
      * value <= threshold     -> white
      * everything else        -> Reds, linear or logarithmic

    The threshold is the `white_pct` percentile of the finite values, computed on the full
    matrix BEFORE any row is dropped, so dropping rows never shifts the colour scale.

    `drop_invisible=True` removes rows in which no cell clears that threshold, so no row
    is entirely white. Weak but non-zero features disappear with it, which is the point.

    `experiment_order` fixes the left to right order of the columns. Default is
    alphabetical, which is what `groupby` produced before and is stable across figures.
    """
    table_type = table_type.lower().strip()

    mat = build_shap_matrix(agg_df, table_type, dropnan=True)   # drops all-NaN features
    if mat.empty:
        print(f"No data for {dataset_name} - {table_type}")
        return None

    mat = mat.T                                   # features down, experiments across
    order = list(experiment_order) if experiment_order else sorted(mat.columns)
    missing = [e for e in order if e not in mat.columns]
    if missing:
        raise ValueError(f"experiment_order lists unknown experiments: {missing}")
    mat = mat[order]

    values = mat.values.astype(float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        print("All values are NaN, nothing to plot.")
        return None

    white_pct = float(np.clip(white_pct, 0.0, 100.0))

    # threshold first, on the full matrix
    if log_scale:
        pos = finite[finite > 0]
        if pos.size == 0:
            print("No positive values for a log scale.")
            return None
        threshold = np.percentile(pos, white_pct) if white_pct > 0 else pos.min()
    else:
        threshold = np.percentile(np.abs(finite), white_pct) if white_pct > 0 else 0.0

    if drop_invisible:
        n_before = mat.shape[0]
        visible = np.nan_to_num(values, nan=-np.inf) > threshold
        mat = mat.loc[visible.any(axis=1)]
        if mat.empty:
            print("Every row is below the white threshold, nothing to plot.")
            return None
        values = mat.values.astype(float)
        finite = values[np.isfinite(values)]
        print(f"[INFO] dropped {n_before - mat.shape[0]} rows with no cell above the "
              f"threshold, {mat.shape[0]} left")

    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(MISSING_COLOR)       # NaN, feature absent from that model
    cmap.set_under("white")

    if log_scale:
        pos = finite[finite > 0]
        vmin = np.nextafter(max(threshold, pos.min()), np.inf)
        vmax = float(pos.max())
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nextafter(vmin, np.inf))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        # positive sentinel keeps LogNorm away from log(0), it lands under vmin -> white
        display = values.copy()
        display[np.isfinite(display) & (display <= 0)] = vmin / 10.0
        cbar_label = "mean |SHAP| (log)"
        scale_name = "log scale"
    else:
        vmax = float(finite.max())
        if not np.isfinite(vmax) or vmax <= threshold:
            vmax = threshold * 10 if threshold > 0 else 1.0
        norm = Normalize(vmin=np.nextafter(threshold, np.inf), vmax=vmax)
        display = values
        cbar_label = "mean |SHAP|"
        scale_name = "linear scale"

    print(f"[INFO] shape={values.shape} (features x experiments), "
          f"white_pct={white_pct:.4g}%, thr={threshold:.6g}, vmax={vmax:.6g}, "
          f"missing(grey) cells={int(np.isnan(values).sum())}")

    n_rows, n_cols = values.shape
    ch = cell_h_in if cell_h_in is not None else max(height / n_rows, min_cell_h_in)
    heat_w = n_cols * cell_w_in
    heat_h = n_rows * ch

    # Two header lines: the title, and a smaller line with the plotting settings.
    title = f"{model} | {dataset_name} | {table_type}"
    subtitle = (f"{scale_name}, white_pct={white_pct:.3g}%, "
                f"thr={threshold:.3g}, max={vmax:.3g}")
    sub_font = max(6.0, title_font_size * 0.6)
    title_h = title_font_size * 1.6 / 72.0
    sub_h = sub_font * 1.6 / 72.0

    left_in = 0.12 + _text_width_in(list(mat.index), y_tick_font_size)
    # the experiment names sit above the heatmap, so the top margin holds them and both
    # header lines
    top_in = 0.55 + title_h + sub_h
    bottom_in = 0.25
    right_in = 0.25 + (0.45 + colorbar_w_in + 0.85 if show_colorbar else 0.0)

    # keep the header inside the canvas, there is no bbox_inches="tight" to save it
    head_w = max(_text_width_in([title], title_font_size),
                 _text_width_in([subtitle], sub_font))
    centre = left_in + heat_w / 2.0
    left_in += max(0.0, head_w / 2.0 + 0.1 - centre)
    centre = left_in + heat_w / 2.0
    right_in += max(0.0, centre + head_w / 2.0 + 0.1 - (left_in + heat_w + right_in))

    fig_w = left_in + heat_w + right_in
    fig_h = bottom_in + heat_h + top_in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([left_in / fig_w, bottom_in / fig_h,
                       heat_w / fig_w, heat_h / fig_h])

    ax.imshow(np.ma.masked_invalid(display), cmap=cmap, norm=norm,
              aspect="auto", interpolation="nearest", origin="upper")

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(mat.index), fontsize=y_tick_font_size)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(list(mat.columns), fontsize=x_tick_font_size)
    ax.xaxis.set_ticks_position("top")     # experiment names above the heatmap

    for lbl, color in zip(ax.get_yticklabels(),
                          _x_label_colors(mat.index, groups_to_features, group_colors)):
        lbl.set_color(color)

    if grid_lw > 0:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=grid_lw)
        ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2)

    if show_colorbar:
        cbar_h = min(heat_h, 3.0)
        cax = fig.add_axes([(left_in + heat_w + 0.45) / fig_w,
                            (bottom_in + (heat_h - cbar_h) / 2.0) / fig_h,
                            colorbar_w_in / fig_w, cbar_h / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                          cax=cax, extend="min")
        cb.set_label(cbar_label, fontsize=x_tick_font_size)
        cb.ax.tick_params(labelsize=x_tick_font_size - 1)

    x_mid = (left_in + heat_w / 2.0) / fig_w
    fig.text(x_mid, (fig_h - 0.15) / fig_h, title,
             ha="center", va="top", fontsize=title_font_size)
    fig.text(x_mid, (fig_h - 0.15 - title_h) / fig_h, subtitle,
             ha="center", va="top", fontsize=sub_font, color="0.35")

    kind = "log_shap" if log_scale else "shap"
    _save(fig, save, outdir,
          f"{kind}_vertical_{model}_{dataset_name}_{table_type}.png", dpi)

    if show:
        plt.show()
        return None
    return fig
