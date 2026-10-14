"""How an experiment is NAMED and ORDERED on a figure. Display only, never in the data.

    from experiment_labels import label, labels, order, ordered

The eight IntronTRIP experiments are called after the intron construct (`A1`, `B3-9`, `H1`), and
that letter code is an identity, not a label. It names the files on disk
(`tables_for_training/<dataset>/{intron}.alldata.tab`), it is the line content of
`DATA/<dataset>/names_all_clean.txt`, and it is the join key in every `results.tsv` and
`group_shap.tsv`. **Nothing here renames any of that.** The map is applied at the last possible
moment, on the axis, so a figure can say `TTC17` while every table underneath it still says
`B3-6`.

`H1` and `B3-9` are two different constructs in the same gene: `H1` is `PACRGL` and `B3-9` is
`PACRGL Control`. That is why the map is keyed on the intron code, never on the gene.

The order is the one the introns are discussed in. It cannot be derived from the data, so it is a
constant here.

Names the module does not know pass through: `label` returns them unchanged and `order` appends
them after the introns in sorted order. So `single_combined` (`5ss_noP`, `CPA_noP`) needs no
special case.
"""

# --------------------------------------------------------------------------- the map
INTRON_GENE = {
    "A1": "NBEAL2",
    "B1": "TONSL",
    "B3-6": "TTC17",
    "B3-9": "PACRGL Control",
    "C1": "USP21",
    "D1": "KPTN",
    "E1": "DDX51",
    "H1": "PACRGL",
}

# Left to right, top to bottom, on every axis that carries introns.
INTRON_ORDER = ["B3-9", "B3-6", "H1", "E1", "A1", "B1", "C1", "D1"]

assert set(INTRON_ORDER) == set(INTRON_GENE), "the order and the map must name the same introns"

# `single_combined`'s two libraries. Same rule as the introns: the code
# stays the identity of the files and of the `experiment` column, the name lives only on the axis.
# `order()` leaves them alone - they are not introns, so they keep their own relative order and
# follow the introns when both appear, which on this project's figures never happens.
LIBRARY_NAME = {
    "5ss_noP": "Regular",
    "CPA_noP": "PCPA",
}

assert not set(LIBRARY_NAME) & set(INTRON_GENE), "an experiment must not be named twice"


def label(experiment):
    """The display name for an experiment code, or the code unchanged if it has none.

    Covers both maps: intron codes become gene names, `single_combined`'s two libraries become
    `Regular` and `PCPA`. Anything else passes through, so an added experiment shows its raw name
    on the axis rather than silently borrowing someone else's.
    """
    if experiment in INTRON_GENE:
        return INTRON_GENE[experiment]
    return LIBRARY_NAME.get(experiment, experiment)


def labels(experiments):
    """`label` over a sequence, order preserved."""
    return [label(e) for e in experiments]


def order(experiments):
    """`experiments` in the fixed intron order, with anything else appended sorted.

    Only the names actually present come back - a panel holding six of the eight introns gets six
    - so this is safe to apply to a subset, and the returned list is always a permutation of the
    input. Duplicates are collapsed, because every caller uses this as an axis and an axis cannot
    carry the same experiment twice.
    """
    present = set(experiments)
    known = [e for e in INTRON_ORDER if e in present]
    rest = sorted(present - set(known))
    return known + rest


def ordered(experiments):
    """`(codes, labels)` - the fixed order and the gene names for it, in step.

    The pair is what most callers want and getting it in one call is what stops the two from
    drifting: the code list stays the key for looking values up, the label list is what is drawn.
    """
    codes = order(experiments)
    return codes, labels(codes)
