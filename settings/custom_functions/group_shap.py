"""
Shapley values for FEATURE GROUPS, by exact enumeration of all coalitions.

Imported by both training scripts. It lives here rather than in `group_shap/` because the numbers
can only be made where the fitted booster and the training matrix are both in memory (inside the
LSF worker), and `MODELS/XGB_*.py` must not depend on the analysis code.
The scripts that consume the output are in `group_shap/`.

THE GAME. For one explained row `x` and a background sample `B`:

    v(S)   = mean over b in B of  f(row: columns of groups in S from x, all others from b)
    phi_g  = sum over T (subsets of the other groups) of  w(|T|) * [ v(T + g) - v(T) ]
    w(t)   = t!(k-t-1)!/k!  =  1 / (k * C(k-1, t))

`w(t)` is the fraction of the k! arrival orders that put exactly the crowd T in the room before g.

WHAT THE CHECKS CAN AND CANNOT SEE - read this before trusting a residual.

`sum_g phi_g == v(full) - v(empty)` is an ALGEBRAIC IDENTITY of the Shapley value of any set
function. Comparing `phi` against `values[-1] - values[0]`, both taken from the same array, is
therefore a tautology: it catches a wrong weight, a wrong coalition selection or a wrong popcount,
and it is deaf to everything that makes `values` the wrong numbers. Measured, all PASSING at ~1e-15
while phi was wrong by 0.4 to 1.9: probabilities instead of margins, one donor per column instead
of per row, a permuted `col_group`, an understated `n_groups`.

So this module runs four checks that are NOT tautological, each against something computed by a
different route:

    1. `v(empty)` must be identical for every explained row - it is `mean_b f(b)`, which cannot
       depend on x. Correct build: spread 0.0. Transposed donors: 5.61.
    2. `v(full)` must equal `f(x)` row by row, from a separate predict call.
       Correct: 3e-08. Transposed donors: 3.56.
    3. `sum_g phi_g` against `f(x) - mean_b f(b)`, both from separate predict calls rather than
       from `values`. Catches the margin flag (1.98), the donor construction (0.55), a bit-convention
       shift (0.24) and an understated `n_groups` (0.24).
    4. the margin flag itself, against the other XGBoost API where it is spelled differently.

What no additivity check can see is a permuted `col_group`, a pure labelling bug. The only cover
for that is a toy model whose groups have distinct known roles.

FOUR TRAPS.

1. MARGIN, not probability. Additivity holds in margin space only. The two entry points spell the
   flag differently - `predict(..., output_margin=True)` versus
   `inplace_predict(..., predict_type="margin")` - and `inplace_predict` defaults to `"value"`.
   Note this is a CLASSIFIER-only hazard: for regression `value` IS the margin.
   Do NOT try to detect it by checking whether a row sums to 1. Softmax constrains the
   PROBABILITIES. The raw margins are free, and measured row sums on a 2-class softprob model at
   300 rounds run 0.49 to 1.64 with the production `subsample=0.6, colsample_bytree=0.5`, and
   -2.72 to 4.81 with subsampling alone. They come out at exactly 1.0 only when NOTHING breaks the
   symmetry between the two classes - no row or column sampling - because the per-class gradients
   are then exact negatives, the trees mirror, and the two `base_score` halves of 0.5 survive. The
   mean stays near 1.0 even when individual rows scatter, so a mean will not reveal it either.
   The reliable check is `_verify`'s comparison against the other predict API, where the flag is
   spelled explicitly.
2. ONE DONOR PER ROW, not per column, or a one-hot block breaks and the within-group correlation
   structure is destroyed.
3. EMPTY GROUPS ARE DUMMY PLAYERS - dropped by `build_col_group`, since keeping one would double
   the coalition count for a guaranteed zero.
4. `col_group` MUST COME FROM THE FITTED MODEL'S COLUMNS, because constant features are dropped
   per model.

THE SOFTMAX GAUGE. Only differences between class margins are identified: adding the same function
to every class leaves the prediction unchanged, so a per-class `phi_k` is defined only up to that
shift and `mean_k |phi_k|` is NOT invariant to it (measured `max |sum_k phi_k| = 1.945` on a
3-class model). Everything here is therefore centred across classes before `|.|`. For two classes
that centring makes `|phi_0| == |phi_1|` exactly, which removes the 6 % class-choice arbitrariness
that `subsample`/`colsample_bytree` otherwise introduce in the production classifier.

COST. `2^k * n_explain * n_background` rows of inference per model, and that is the whole cost.
"""

import math

import numpy as np

__all__ = ["shapley_weights", "build_col_group", "build_feature_of_column",
           "group_shapley", "signed_group_sum", "gauge_centre", "shap_columns"]


def shapley_weights(k):
    """w[t] for a coalition of size t, t = 0..k-1.

    Uses 1/(k*C(k-1,t)) rather than the factorial form: 14! is 8.7e10 and dividing two huge numbers
    loses precision for nothing. Identical to the factorial form for k = 1..20.
    """
    if k < 1:
        raise ValueError("k must be at least 1, got %r" % (k,))
    return np.array([1.0 / (k * math.comb(k - 1, t)) for t in range(k)])


def build_feature_of_column(original_columns, encoded_columns, categorical_columns):
    """Which raw column each model column came from, after `pd.get_dummies`.

    A plain column maps to itself, and a one-hot level `ChromHMM_Quies` maps to `ChromHMM`. Built where
    the encoding happens, with the categorical list in hand, so the prefix match can be CHECKED -
    it raises if two categoricals could claim the same level rather than picking one. Note the
    `column in plain` branch runs first, so a real column that happens to look like a level wins.
    """
    plain = set(original_columns) - set(categorical_columns)
    mapping = {}
    for column in encoded_columns:
        if column in plain:
            mapping[column] = column
            continue
        owners = [c for c in categorical_columns if column.startswith(c + "_")]
        if len(owners) != 1:
            raise ValueError("column %r matches %d categorical prefixes: %s"
                             % (column, len(owners), owners))
        mapping[column] = owners[0]
    return mapping


def build_col_group(model_columns, feature_of_column, group_of_feature):
    """Map each model column to a group index. Returns (col_group, group_names).

    Only groups owning at least one column are returned, so there are no dummy players.
    """
    unmapped = [c for c in model_columns
                if c not in feature_of_column or feature_of_column[c] not in group_of_feature]
    if unmapped:
        raise ValueError("%d model columns belong to no group, first few: %s"
                         % (len(unmapped), unmapped[:5]))

    group_per_column = [group_of_feature[feature_of_column[c]] for c in model_columns]
    group_names = sorted(set(group_per_column))
    position = {name: i for i, name in enumerate(group_names)}
    col_group = np.array([position[g] for g in group_per_column])
    return col_group, group_names


def popcount(values):
    """How many bits are set in each value. Non-negative input only."""
    values = np.asarray(values)
    if (values < 0).any():
        raise ValueError("popcount on a negative value would loop forever (arithmetic shift)")
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    counted = np.zeros_like(values)
    rest = values.copy()
    while rest.any():
        counted += rest & 1
        rest >>= 1
    return counted


def predict_margin(booster, rows, keep_class=None):
    """Raw margins, always 2-D. See trap 1 about the flag name."""
    margins = booster.inplace_predict(rows, predict_type="margin")
    if margins.ndim == 1:
        margins = margins[:, None]
    return margins if keep_class is None else margins[:, keep_class]


def gauge_centre(per_class):
    """Remove the softmax gauge freedom by centring across the class axis (the last one).

    Only differences between class margins are identified, so an uncentred per-class attribution is
    not a well-defined quantity. One output is left alone. Two become exact mirrors, so |phi| stops
    depending on which class is read.
    """
    if per_class.shape[-1] == 1:
        return per_class
    return per_class - per_class.mean(axis=-1, keepdims=True)


def coalition_values(booster, x_explain, x_background, col_group, n_groups, keep_class=None,
                     verbose=False):
    """v[mask] for every one of the 2^k coalitions: (2^k, n_explain, n_outputs).

    `mask` is a bit per group. Bit g set means group g is taken from the explained row.
    """
    n_explain = len(x_explain)
    n_background = len(x_background)

    # Row (i * n_background + j) is explained row i seen against donor j. Both blocks are built
    # once. Each coalition then only chooses, per column, which block to read, and it takes the
    # whole column from one side, which is trap 2.
    from_x = np.repeat(x_explain, n_background, axis=0)
    from_donor = np.tile(x_background, (n_explain, 1))
    row_buffer = np.empty_like(from_x)

    n_coalitions = 2 ** n_groups
    values = None
    for mask in range(n_coalitions):
        take_from_x = ((mask >> col_group) & 1).astype(bool)     # one bool per column
        np.copyto(row_buffer, from_donor)
        np.copyto(row_buffer, from_x, where=take_from_x)         # the (p,) mask broadcasts over rows

        margins = predict_margin(booster, row_buffer, keep_class)
        # float64 for the average over donors: in float32 this alone costs ~8e-07, which is the
        # size of the accuracy figures this module reports, and it does not cancel in the gains.
        averaged = margins.reshape(n_explain, n_background, -1).mean(axis=1, dtype=np.float64)

        if values is None:
            values = np.empty((n_coalitions, n_explain, averaged.shape[1]))
        values[mask] = averaged

        if verbose and mask % 2048 == 0:
            print("      coalition %d / %d" % (mask, n_coalitions), flush=True)
    return values


def group_shapley(booster, x_explain, x_background, col_group, n_groups, keep_class=None,
                  rtol=1e-4, verbose=False):
    """Exact group Shapley values. Returns (phi, checks).

    phi     (n_groups, n_explain, n_outputs), signed, gauge-centred across classes
    checks  a dict of the four non-tautological residuals, for `run_info`
    """
    if not 1 <= n_groups <= 22:
        raise ValueError("n_groups must be 1..22, got %r (2^k stops being an enumeration)"
                         % (n_groups,))
    if len(col_group) != x_explain.shape[1]:
        raise ValueError("col_group has %d entries for %d columns"
                         % (len(col_group), x_explain.shape[1]))
    if col_group.max() >= n_groups or len(np.unique(col_group)) != n_groups:
        raise ValueError("col_group indexes %d distinct groups, n_groups says %d - an unused index "
                         "is a dummy player that doubles the work for a guaranteed zero"
                         % (len(np.unique(col_group)), n_groups))
    if len(x_background) < 2:
        raise ValueError("background has %d rows. The mean over donors would be meaningless, and "
                         "an empty one silently yields NaN that the residual check cannot see"
                         % len(x_background))

    values = coalition_values(booster, x_explain, x_background, col_group, n_groups,
                              keep_class, verbose)

    weight = shapley_weights(n_groups)
    all_masks = np.arange(2 ** n_groups)
    coalition_size = popcount(all_masks)

    phi = np.zeros((n_groups, values.shape[1], values.shape[2]))
    for g in range(n_groups):
        bit = 1 << g
        without_g = all_masks[(all_masks & bit) == 0]
        # Chunked so the difference array stays small. This changes the result only by rounding
        # (measured 9e-16), not structurally.
        for chunk in np.array_split(without_g, 16):
            gain = values[chunk | bit] - values[chunk]
            phi[g] += np.tensordot(weight[coalition_size[chunk]], gain, axes=(0, 0))

    checks = _verify(booster, x_explain, x_background, values, phi, keep_class, rtol)
    return gauge_centre(phi), checks


def _verify(booster, x_explain, x_background, values, phi, keep_class, rtol):
    """The four checks described in the module docstring. Each uses a route other than `values`."""
    on_x = predict_margin(booster, x_explain, keep_class).astype(np.float64)
    on_background = predict_margin(booster, x_background, keep_class).astype(np.float64)
    scale = max(1.0, float(np.abs(on_x).max()))
    tol = rtol * scale

    # Spread across EXPLAINED ROWS, per class, then the worst class. Taking max and min over both
    # axes at once measures the gap BETWEEN classes instead, which for a two-class model is about
    # 2.2 - a false alarm that only shows up on a classifier, never on the single-output toy.
    empty_room_spread = float((values[0].max(axis=0) - values[0].min(axis=0)).max())
    full_room_gap = float(np.abs(values[-1] - on_x).max())
    additivity = float(np.abs(phi.sum(axis=0) - (on_x - on_background.mean(axis=0))).max())

    # The margin flag, against the other API where it is spelled explicitly. Only this one can see
    # a probabilities-instead-of-margins mistake, and only on a classifier.
    import xgboost as xgb
    other_api = booster.predict(xgb.DMatrix(x_explain), output_margin=True)
    if other_api.ndim == 1:
        other_api = other_api[:, None]
    if keep_class is not None:
        other_api = other_api[:, keep_class]
    margin_flag = float(np.abs(on_x - other_api).max())

    failures = []
    # NaN first, and explicitly. Every comparison below is `> tol`, and `nan > tol` is False, so a
    # NaN anywhere would switch the whole safety net off silently instead of tripping it.
    if not np.isfinite(phi).all():
        failures.append("phi contains %d non-finite values" % int((~np.isfinite(phi)).sum()))
    if not np.isfinite(values).all():
        failures.append("the coalition table contains non-finite values")
    if empty_room_spread > tol:
        failures.append("v(empty) varies by %.3e across explained rows. It is mean_b f(b) and "
                        "cannot depend on x - the donor construction is transposed" % empty_room_spread)
    if full_room_gap > tol:
        failures.append("v(full) differs from f(x) by %.3e - the mask or the donor layout is wrong"
                        % full_room_gap)
    if additivity > tol:
        failures.append("sum_g phi_g differs from f(x) - mean_b f(b) by %.3e" % additivity)
    if margin_flag > tol:
        failures.append("the two predict APIs disagree by %.3e - the margin flag is wrong"
                        % margin_flag)
    if failures:
        raise AssertionError("group SHAP verification failed (tolerance %.1e):\n  %s"
                             % (tol, "\n  ".join(failures)))

    return dict(empty_room_spread=empty_room_spread, full_room_gap=full_room_gap,
                additivity=additivity, margin_flag=margin_flag, tolerance=tol)


def signed_group_sum(per_row_shap, col_group, n_groups):
    """Sum the SIGNED individual SHAP inside each group, per row. (n_groups, n_rows, n_out).

    This is what "just summing the shap values of a group" means, and it is NOT the existing
    `SHAP_` aggregation, which takes |.| of each feature BEFORE summing, so features pushing in
    opposite directions on the same row add up instead of cancelling.
    """
    shap_values = np.asarray(per_row_shap)
    if shap_values.ndim == 2:
        shap_values = shap_values[:, :, None]

    sums = np.zeros((n_groups, shap_values.shape[0], shap_values.shape[2]))
    for g in range(n_groups):
        columns = np.flatnonzero(col_group == g)
        if len(columns):
            sums[g] = shap_values[:, columns, :].sum(axis=1)
    return sums


def _collapse(centred_per_class, row_axis):
    """mean over rows of |value|, then over classes. Input must already be gauge-centred."""
    per_class = np.abs(centred_per_class).mean(axis=row_axis)
    return per_class.mean(axis=-1)


def shap_columns(booster, x_frame, feature_of_column, group_of_feature,
                 n_explain=200, n_background=50, seed=2005, bootstrap=200, verbose=False):
    """Everything the training script adds to its `result` row, as {column name: value}.

    Five quantities, all on the SAME sampled rows so every comparison between them is paired:

        TSHAP_<feature>  tree-path individual - the existing value function, this row sample
        ISHAP_<feature>  interventional individual
        BSHAP_<group>    interventional, signed sum inside the group
        GSHAP_<group>    interventional, the group as one player
        GSE_<group>      bootstrap standard error of GSHAP over the explained rows

    `TSHAP` exists because the pre-existing `SHAP_` columns are computed over ALL training rows,
    so `SHAP` against `ISHAP` would differ by the value function AND by the row sample - measured,
    the row-sample effect is about half the size of the value-function effect. `TSHAP` against
    `ISHAP` is the paired comparison that isolates the value function.

    Returns `(columns, info)`.
    """
    import shap                                            # only needed on this path

    columns = list(x_frame.columns)
    col_group, group_names = build_col_group(columns, feature_of_column, group_of_feature)

    matrix = x_frame.to_numpy(dtype=np.float32)            # materialised once, not twice
    rng = np.random.default_rng(seed)
    # The background size is fixed FIRST and the explained rows come from what is left. The old
    # order was the other way round, and on a small matrix it gave an EMPTY background: `take` is
    # never longer than the matrix, so with `len(matrix) <= n_explain` the slice `take[n_explain:]`
    # is empty and `group_shapley` failed with "background has 0 rows". Failing was right (a silent
    # NaN would have been worse), but the cause was how the sample was split, not the data. The
    # son_sig*_reg models hold 28-201 rows with `n_explain = 200`, and ALL twelve failed although
    # there were enough rows for a background.
    #
    # On large matrices nothing changes: with `len(matrix) >= n_explain + n_background` both sizes
    # equal the requested ones, so the three main datasets of the project (19,561 to 267,962 rows)
    # get the same sample as before. Checked on them separately.
    n_bg = min(n_background, max(2, len(matrix) // 4))
    n_ex = min(n_explain, len(matrix) - n_bg)
    if n_ex < 1:
        raise ValueError("matrix has %d rows, too few to split into explained and background"
                         % len(matrix))
    # The slice order is the OLD one, explained rows first, then background. It must not change:
    # with `len(matrix) >= n_explain + n_background` `take` has the same length as before, so
    # `take[:n_ex]` and `take[n_ex:]` hold THE SAME rows as before the fix. Swap them and the
    # models already computed would no longer reproduce, although both splits are equally valid.
    take = rng.choice(len(matrix), size=n_ex + n_bg, replace=False)
    explain_rows = matrix[take[:n_ex]]
    background_rows = matrix[take[n_ex:]]

    phi, checks = group_shapley(booster, explain_rows, background_rows, col_group,
                                len(group_names), verbose=verbose)

    out = {}
    for name, value in zip(group_names, _collapse(phi, 1)):
        out["GSHAP_%s" % name] = float(value)

    # Bootstrap over the explained rows. `phi` is already in hand, so this is arithmetic only, and
    # it matters: the round-0 group ranking is not separated at p<0.05 in 9 of 12 chains, so a
    # point estimate without a spread invites conclusions the data cannot carry.
    boot_rng = np.random.default_rng(seed + 1)
    draws = np.empty((bootstrap, len(group_names)))
    for b in range(bootstrap):
        pick = boot_rng.integers(0, phi.shape[1], phi.shape[1])
        draws[b] = _collapse(phi[:, pick, :], 1)
    for name, value in zip(group_names, draws.std(axis=0, ddof=1)):
        out["GSE_%s" % name] = float(value)

    interventional = shap.TreeExplainer(booster, data=background_rows,
                                        feature_perturbation="interventional")
    per_row = gauge_centre(_as3d(interventional(explain_rows, check_additivity=False).values))
    for name, value in zip(columns, _collapse(per_row, 0)):
        out["ISHAP_%s" % name] = float(value)

    signed = signed_group_sum(per_row, col_group, len(group_names))
    for name, value in zip(group_names, _collapse(signed, 1)):
        out["BSHAP_%s" % name] = float(value)

    tree_path = gauge_centre(_as3d(shap.Explainer(booster)(explain_rows).values))
    for name, value in zip(columns, _collapse(tree_path, 0)):
        out["TSHAP_%s" % name] = float(value)

    # The interventional individual path and the group path decompose the SAME total, so their
    # signed sums must agree row by row. Without this, BSHAP - half of the headline comparison -
    # has no check at all.
    cross = float(np.abs(per_row.sum(axis=1) - phi.sum(axis=0)).max())
    if cross > checks["tolerance"]:
        raise AssertionError("individual and group attributions disagree on the total by %.3e "
                             "(tolerance %.1e)" % (cross, checks["tolerance"]))

    info = dict(n_explain=len(explain_rows), n_background=len(background_rows), seed=seed,
                n_groups=len(group_names), groups=",".join(group_names),
                n_outputs=phi.shape[2], individual_vs_group=cross, **checks)
    return out, info


def _as3d(values):
    return values[:, :, None] if values.ndim == 2 else values
