"""
Group the model features, so the SHAP heatmaps can colour them and the group SHAP knows its players.

`groups_for()` looks at the SHAP columns that are actually present and assigns each one to a
group. Anything it cannot place ends up in `other`, which is visible on the plot and therefore
easy to notice.

It works on two kinds of column name:

  * **Model columns** (`ChromHMM_Quies`, `Chr_chr7`), what a trained model actually has, so what
    the `SHAP_*` columns of `results.tsv` are named after.
  * **Raw features** (`ChromHMM`, `Chr`), after replicate averaging and before one-hot encoding.
    This is the namespace the drop lists are applied in, and `group_definitions.py` uses it.

Every rule names the raw column as well as its one-hot levels, so both namespaces are covered.
"""

import sys


def _one_hot(features, prefix):
    """One-hot levels of a categorical feature, e.g. ChromHMM_EnhA1 for prefix ChromHMM_.

    The trailing underscore matters. Without it `Chr_` would also collect every `ChromHMM_`
    column, because `Chr` is a prefix of `ChromHMM`.
    """
    return [f for f in features if f.startswith(prefix)]


# One colour per group, and they must all differ: the previous table reused four of them
# ('mane_expression' and 'histone_marks' were both tab:olive, 'chromhmm' and 'pol2' both
# tab:cyan, 'transcriptomics' and 'accessibility' both tab:orange, 'annotations' and
# 'conservation' both tab:brown), and those pairs appear on the same axis, so the colour said
# nothing where it mattered most.
#
# These are used as TEXT on a white surface, so the gate is readability first: every one clears
# 4:1 against white. Found by search rather than picked - worst pair dE 13.3, hues spread round
# the wheel with a minimum gap of 16 degrees.
GROUP_COLORS = {
    # The ten below are exactly the colours the published SHAP heatmaps use
    # (trip-pipeline/ML/single_combined_pos_features/figures/) - the matplotlib tab: cycle.
    # Do not change them: a reader who learned that ChromHMM is cyan there must find it cyan
    # here.
    # `annotations` was split into `gene_annotations` and `intron_annotations` on
    # 2026-08-25. The key stays so that figures redrawn from tables written
    # BEFORE the split still find a colour. `groups_for` no longer emits it.
    'annotations':     '#8c564b',   # tab:brown, historical
    'gene_annotations': '#8c564b',  # inherits the brown - it holds 9 of the old group's 12
    'mane_expression': '#bcbd22',   # tab:olive
    'chromhmm':        '#17becf',   # tab:cyan
    'nuclear_org':     '#9467bd',   # tab:purple
    'gc_content':      '#e377c2',   # tab:pink
    'repeats':         '#d62728',   # tab:red
    'chromosome':      '#7f7f7f',   # tab:gray
    'transcriptomics': '#ff7f0e',   # tab:orange
    'global_norm':     '#1f77b4',   # tab:blue
    'nmd':             '#2ca02c',   # tab:green
    # These five never appear in that published figure - it is the track-free dataset - and four
    # of them used to REUSE a colour from the ten above ('histone_marks' was olive like
    # mane_expression, 'pol2' cyan like chromhmm, 'accessibility' orange like transcriptomics,
    # 'conservation' brown like annotations). Those pairs do share an axis in the chromatin
    # datasets, so the colour said nothing exactly where it was needed. Given the ten fixed
    # above, these five were chosen by search to keep every pair apart: worst pair over all
    # fifteen is dE 11.3, and each clears 4:1 against white, which is the gate for text.
    'histone_marks':   '#a41fd0',
    'pol2':            '#2c4d7d',
    'accessibility':   '#50762d',
    'conservation':    '#59304a',
    'rbp':             '#4159f3',
    # Three added 2026-08-25 with the split. Chosen by the same search the five above were:
    # over a 36x4x4 HSV grid, keeping every candidate at least 11.3 dE (OKLab) from all
    # existing colours and maximising the worst pair among themselves. The worst pair of the
    # whole 19-colour palette stays 11.3 - identical to before the three were added, so the
    # split costs nothing in separability. Contrast against white: 8.6, 11.2 and 5.2 to 1.
    # Was '#085908', a dark green 12.0 dE from 'accessibility' '#50762d', so on the legend the two
    # looked like the same green. The replacement was picked by the same search as the others:
    # the minimum distance to ALL other colours went up to 19.4 dE (OKLab), and it holds under
    # deuteranopia (19.4) and protanopia (17.5). Contrast against white is 18.3:1. No green has
    # that margin, the palette already has three greens ('accessibility', 'nmd' and this one), so
    # the colour moved to a dark blue. Its nearest neighbour, 'cage' '#1313bf', is a bright
    # saturated blue, so the pair is separated by lightness as well.
    'intron_annotations': '#000059',
    'cage':            '#1313bf',
    'icshape':         '#bf2f8f',
    'other':           '#1a1a1a',
}


def groups_for(agg_df):
    """Return (groups_to_features, group_colors) for whatever this frame contains."""
    features = [c[5:] for c in agg_df.columns if c.startswith('SHAP_')]

    groups = {
        # The raw categorical names are here as well as their one-hot levels: the SHAP columns
        # of a trained model are one-hot, but the feature-removal summaries carry the raw column,
        # and both go through this function.
        # Split out of one `annotations` group on 2026-08-25: what the
        # host GENE is and where in it the integration sits, against where inside an INTRON it
        # sits. The two answer different questions, and the intron three only mean anything for
        # an intronic integration.
        'gene_annotations': [f for f in [
            'MANE_all', 'MANE_exonic', 'MANE_antisense', 'gencode_noncoding', 'MANElength',
            'DistToTSS', 'DistToTES', 'RelPos', 'annot_rna_type'] if f in features]
            + _one_hot(features, 'annot_rna_type_'),
        # `MANE_intron_length` is the length of the host intron, so it is filed here. It exists
        # only in an older dataset that is not part of this repository.
        'intron_annotations': [f for f in [
            'DistToIntron5', 'DistToIntron3', 'RelIntronPos', 'MANE_intron_length']
            if f in features],
        'mane_expression': [f for f in ['MANE_all_exp.norm.log10'] if f in features],
        'chromhmm': [f for f in ['ChromHMM'] if f in features] + _one_hot(features, 'ChromHMM_'),
        # Both the raw replicate names (GSE81553_SON_TSA-Seq_Condition1) and the averaged ones
        # that get_gene_tables.py collapses them into (SON_TSA-Seq). The averaging happens at
        # training time, so which of the two reaches the SHAP columns depends on the dataset.
        # GSE81553 covers both the TSA-Seq/HiC tracks, which are nuclear organisation, and the
        # binned GRO-Seq and RNA-Seq, which are transcription. Splitting on the accession alone
        # filed the two strongest transcription tracks in the wrong group.
        # TSA-Seq measures distance from a nuclear body, so it is organisation whatever the
        # antibody was against - Pol2_TSA-Seq included. GSE81553_K562_Smoothed_Pol2_ChIP-Seq is
        # the exception carved out below: a ChIP occupancy track that happens to sit in the same
        # accession, and occupancy is what `pol2` means.
        'nuclear_org': [f for f in ['SPIN'] if f in features] + _one_hot(features, 'SPIN_')
                       + [f for f in features if f.startswith('GSE81553_')
                          and 'GRO-Seq' not in f and 'RNA-Seq' not in f
                          and 'Pol2_ChIP-Seq' not in f]
                       + [f for f in features
                          if f.endswith('TSA-Seq') or f == 'Speckle_distance'],
        'gc_content': [f for f in ['GC', 'GC_category'] if f in features]
                      + _one_hot(features, 'GC_category_'),
        'repeats': [f for f in ['RepeatGroup'] if f in features]
                   + _one_hot(features, 'RepeatGroup_'),
        # 'Chr' exactly, never by prefix: `Chr_` would also swallow every `ChromHMM_` column.
        'chromosome': [f for f in ['Chr'] if f in features] + _one_hot(features, 'Chr_'),
        # K562.Cell is RNA-seq of the whole cell, a track of an older dataset that is not part
        # of this repository. No rule named it, so it was the one ungrouped raw feature there.
        # `closestCAGEdist` used to live here and became a category of its own on 2026-08-25.
        # It is a distance to the nearest annotated transcription start, not a
        # measurement of how much transcription runs through this locus, which is what the rest
        # of this group is.
        'cage': [f for f in ['closestCAGEdist'] if f in features],
        'transcriptomics': [f for f in ['K562.Cell'] if f in features]
                           + _one_hot(features, 'RNAseq') + _one_hot(features, 'GRO')
                           + _one_hot(features, 'NET_') + _one_hot(features, 'PRO_')
                           + [f for f in features if f.startswith('GSE81553_')
                              and ('GRO-Seq' in f or 'RNA-Seq' in f)],
        # RNA-binding proteins mapped by CUT&RUN / CUT&Tag. Only in an older dataset that is not
        # part of this repository.
        #
        # Keyed on WHAT was mapped, not on the method. The previous rule was `endswith
        # ('_CUTnRUN')` and it filed RNAPIIS2P_CUTnRUN and RNAPIIS5P_CUTnTAG here, so "remove the
        # rbp group" would have taken Pol II occupancy out with the splicing factors and no
        # single answer would have been readable. They go to `pol2`. H3K36me3_CUTnRUN was already
        # being pulled out by the H3K prefix, which is the same principle applied by accident.
        'rbp': [f for f in features
                if (f.endswith('_CUTnRUN') or f.endswith('_CUTnTAG'))
                and not f.startswith(('H3K', 'RNAPII'))],
        'global_norm': [f for f in ['meanNormRPM', 'meanMapRPM', 'norm'] if f in features],
        # For IntronTRIP `NMD_expected` becomes a four-level factor (no NMD / only if spliced /
        # only if unspliced / either way), so the one-hot levels have to be caught too.
        'nmd': [f for f in ['NMD_expected'] if f in features]
               + _one_hot(features, 'NMD_expected_'),
        'histone_marks': [f for f in features if f.startswith(('H2A', 'H3K', 'H4K'))]
                         + _one_hot(features, 'spKAS'),
        'pol2': [f for f in features
                 if 'POLR2A' in f or f.startswith(('ENCSR', 'RNAPII'))
                 or 'Pol2_ChIP-Seq' in f],
        'accessibility': [f for f in ['ATAC', 'DNAse'] if f in features],
        # icSHAPE came out of `conservation` on 2026-08-25. It measures RNA
        # secondary structure in the cell, which is not evolutionary conservation at all - the
        # two were together only because both were single leftover tracks.
        'conservation': [f for f in ['phyloP'] if f in features],
        'icshape': [f for f in ['icSHAPE'] if f in features],
    }

    # Anything not placed above. An empty `other` is the normal case - a non-empty one means a
    # new kind of feature appeared and this function should learn about it.
    placed = {f for members in groups.values() for f in members}
    groups['other'] = [f for f in features if f not in placed]

    groups = {name: members for name, members in groups.items() if members}
    if groups.get('other'):
        print(f'[feature_groups] ungrouped features: {groups["other"]}', file=sys.stderr)

    return groups, {name: GROUP_COLORS[name] for name in groups}
