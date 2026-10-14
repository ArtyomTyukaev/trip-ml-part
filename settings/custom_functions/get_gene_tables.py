import numpy as np
from datetime import datetime
import os
import pandas as pd
import sys

# What to do with ChromHMM states that have fewer than MIN_CHROMHMM_BARCODES barcodes.
#
#   "drop"  (the default, the old behaviour): drop THOSE ROWS entirely
#   "merge": rename the state to 'rare' and keep the rows
#
# The rule was written for the project's datasets of 20,000 to 270,000 rows, where a state with
# nine barcodes really is negligible. On a small dataset it is destructive. The son_splicing
# tables of 80 to 700 rows are spread over 15 to 17 states, so MOST states naturally have fewer
# than ten barcodes and the filter throws away 23 to 76 % of the data. A barcode is dropped with
# all its 76 features because of one categorical value.
#
# The point of the rule is to leave no one-hot column without support and to keep the column sets
# of the two tables consistent. Merging the rare states into one category does the same and loses
# nothing.
#
# The mode comes from the environment, NOT from an argument: both functions are called from the
# training scripts, and an argument would mean changing their signatures in four places. The
# default is the old behaviour, so the three main datasets of the project are not affected.
MIN_CHROMHMM_BARCODES = 10


def _rare_chromhmm_mode():
    mode = os.environ.get("RARE_CHROMHMM", "drop").strip().lower() or "drop"
    if mode not in ("drop", "merge"):
        raise ValueError("RARE_CHROMHMM must be 'drop' or 'merge', got %r" % mode)
    return mode


def _rare_states(df, intrAgenic_rows):
    """States that are rare in at least one of the two tables. A union, so that the one-hot
    column sets of intragenic and intergenic stay consistent."""
    rare = set()
    for rows in (intrAgenic_rows, ~intrAgenic_rows):
        counts = df.loc[rows, 'ChromHMM'].value_counts()
        rare |= set(counts[counts < MIN_CHROMHMM_BARCODES].index)
    return rare


def make_intra_and_inter_genic_tables(output_dir, exp, replics_dicts, columns_to_drop_in_intragenic, annot_table_cols_to_drop, filename_prefix='.alldata.tab'):
    log_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{log_time} Processing {output_dir} | Experiment: {exp}")

    try:
        df = pd.read_csv(f"{output_dir}/{exp}{filename_prefix}", sep='\t', index_col=0)
        print(f'Dataset shape: {df.shape}')
    except FileNotFoundError:
        print(f"{log_time} Skipping missing file: {output_dir}/{exp}.alldata.tab")
        return None

    # Aggregate replicates
    for family, group in replics_dicts.items():
        for key, val in group.items():
            val = [c for c in val if c in df.columns]
            if not val:
                continue
            target = key
            if key in df.columns and key not in val:
                # An older dataset carries RNAseqPolyA, RNAseqPolyAminus and RNAseqPolyAnucleolus
                # twice: once as an hg38 histone-family track and once as the stranded pair
                # <name>_1 / <name>_2. Writing the replicate mean under the bare name silently
                # overwrote the histone track - the two correlate at only 0.59, 0.45 and 0.13,
                # so they are different measurements - and the model then trained on the
                # stranded mean under the histone name, with SHAP_<name> meaning neither.
                # Give the aggregate its family suffix and keep both columns.
                target = f'{key}_{family}'
                print(f'  [!] replicate group {key!r} collides with an existing column, '
                      f'writing the {family} mean as {target!r} and keeping the original')
            # mean() skips NaN and divides by the replicates that were measured. sum()/len(val)
            # counted an unmeasured replicate as a zero and halved the value. All-NaN stays NaN.
            df[target] = df[val].mean(axis=1)
            df.drop(columns=val, inplace=True)

    df.drop(columns=['score', 'strand'], inplace=True, errors='ignore')


    df = df[df["ChromHMM"].notna()].copy()

    intrAgenic_rows = df["MANElength"].notna()

    # 'merge' mode: rare states are renamed BEFORE the split and the rows stay. It has to happen
    # right here, before fillna and one-hot, so that both tables inherit one category.
    if _rare_chromhmm_mode() == "merge":
        _rare = _rare_states(df, intrAgenic_rows)
        if _rare:
            _n = int(df['ChromHMM'].isin(_rare).sum())
            df.loc[df['ChromHMM'].isin(_rare), 'ChromHMM'] = 'rare'
            print(f'  [i] RARE_CHROMHMM=merge: {len(_rare)} rare states, {_n} barcodes '
                  f'relabelled to "rare" instead of being dropped')

    # Stripping the ChromHMM states that carry fewer than 10 barcodes is a ROW filter, but it
    # used to be computed from the two finished tables - after the column drops - so it needed
    # ChromHMM to still be a column there. That made ChromHMM the one feature that could
    # not be dropped: putting it on a drop list raised KeyError in the worker.
    #
    # Computed here instead, from the frame that still has the column, over exactly the same two
    # row sets. Column drops do not change rows and fillna(0) does not touch a string column with
    # no NaN left in it, so the states selected are identical to before. Only the moment changed.
    rare_intrA = df.loc[intrAgenic_rows, 'ChromHMM'].value_counts()
    rare_intrA = rare_intrA[rare_intrA < 10].index.tolist()

    # Over the rows the intergenic table will actually hold. It used to count over the whole frame,
    # which was right while that table WAS the whole frame. Now it would select rare states from a
    # population the table does not contain, and drop barcodes for a scarcity that is not there.
    rare_intEr = df.loc[~intrAgenic_rows, 'ChromHMM'].value_counts()
    rare_intEr = rare_intEr[rare_intEr < 10].index.tolist()

    union_rare = set(rare_intrA) | set(rare_intEr)
    # Held as barcodes rather than as a mask, so it survives the column drops and the row
    # subsetting below without depending on ChromHMM being present any more.
    rare_barcodes = df.index[df['ChromHMM'].isin(union_rare)]

    intrAgenic_alltable = df[intrAgenic_rows].copy()
    intrAgenic_alltable.fillna(0, inplace=True)
    intrAgenic_alltable.drop(columns=columns_to_drop_in_intragenic, inplace=True, errors='ignore')

    # Integrations OUTSIDE a MANE gene, and only those. Until 2026-08-27 this was `df.copy()`,
    # every row with the gene-body features hidden, so the table contained the intragenic rows
    # as a subset. The pair now compares two DIFFERENT row sets, not one row set seen twice.
    #
    # The criterion is `MANElength`: outside a MANE gene. Integrations inside a gencode NON-coding
    # gene (lncRNA, pseudogene, 13 % of these rows) stay here, which is why `DistToTSS`,
    # `DistToTES` and `RelPos` remain on the intergenic drop list. They measure a position inside
    # the containing gene, so here they would mean "inside the lncRNA" while in the intragenic
    # table they mean "inside the MANE gene". Same name, different quantity. Verified:
    # `DistToTSS + DistToTES` equals `MANElength` for 100 % of rows that have one.
    intErgenic_alltable = df[~intrAgenic_rows].copy()
    intErgenic_alltable.drop(columns=annot_table_cols_to_drop, inplace=True, errors='ignore')

    # In 'merge' mode NOTHING is dropped, and that has to be set explicitly rather than relying on
    # `union_rare` coming out empty. It does not come out empty: the renaming creates a 'rare'
    # category, and if fewer than ten barcodes were merged, that category itself passes the rarity
    # test and is dropped by the very next step. This is exactly what happened on `single_combined`:
    # 9 barcodes were merged into 'rare' and then removed, so the mode did nothing.
    if _rare_chromhmm_mode() == "merge":
        rare_barcodes = df.index[[]]
    intrAgenic_alltable = intrAgenic_alltable[~intrAgenic_alltable.index.isin(rare_barcodes)].copy()
    intErgenic_alltable = intErgenic_alltable[~intErgenic_alltable.index.isin(rare_barcodes)].copy()

    return intrAgenic_alltable, intErgenic_alltable




def make_intra_and_inter_genic_tables_with_barcodes(output_dir, exp, replics_dicts, columns_to_drop_in_intragenic, annot_table_cols_to_drop, filename_prefix='.alldata.tab'):
    log_time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{log_time} Processing {output_dir} | Experiment: {exp}")

    try:
        df = pd.read_csv(f"{output_dir}/{exp}{filename_prefix}", sep='\t', index_col=0)
        print(f'Dataset shape: {df.shape}')
    except FileNotFoundError:
        print(f"{log_time} Skipping missing file: {output_dir}/{exp}.alldata.tab")
        return None

    # Aggregate replicates
    for family, group in replics_dicts.items():
        for key, val in group.items():
            val = [c for c in val if c in df.columns]
            if not val:
                continue
            target = key
            if key in df.columns and key not in val:
                # An older dataset carries RNAseqPolyA, RNAseqPolyAminus and RNAseqPolyAnucleolus
                # twice: once as an hg38 histone-family track and once as the stranded pair
                # <name>_1 / <name>_2. Writing the replicate mean under the bare name silently
                # overwrote the histone track - the two correlate at only 0.59, 0.45 and 0.13,
                # so they are different measurements - and the model then trained on the
                # stranded mean under the histone name, with SHAP_<name> meaning neither.
                # Give the aggregate its family suffix and keep both columns.
                target = f'{key}_{family}'
                print(f'  [!] replicate group {key!r} collides with an existing column, '
                      f'writing the {family} mean as {target!r} and keeping the original')
            # mean() skips NaN and divides by the replicates that were measured. sum()/len(val)
            # counted an unmeasured replicate as a zero and halved the value. All-NaN stays NaN.
            df[target] = df[val].mean(axis=1)
            df.drop(columns=val, inplace=True)

    df.drop(columns=['score'], inplace=True, errors='ignore')


    df = df[df["ChromHMM"].notna()].copy()

    intrAgenic_rows = df["MANElength"].notna()

    # 'merge' mode: rare states are renamed BEFORE the split and the rows stay. It has to happen
    # right here, before fillna and one-hot, so that both tables inherit one category.
    if _rare_chromhmm_mode() == "merge":
        _rare = _rare_states(df, intrAgenic_rows)
        if _rare:
            _n = int(df['ChromHMM'].isin(_rare).sum())
            df.loc[df['ChromHMM'].isin(_rare), 'ChromHMM'] = 'rare'
            print(f'  [i] RARE_CHROMHMM=merge: {len(_rare)} rare states, {_n} barcodes '
                  f'relabelled to "rare" instead of being dropped')

    # Stripping the ChromHMM states that carry fewer than 10 barcodes is a ROW filter, but it
    # used to be computed from the two finished tables - after the column drops - so it needed
    # ChromHMM to still be a column there. That made ChromHMM the one feature that could
    # not be dropped: putting it on a drop list raised KeyError in the worker.
    #
    # Computed here instead, from the frame that still has the column, over exactly the same two
    # row sets. Column drops do not change rows and fillna(0) does not touch a string column with
    # no NaN left in it, so the states selected are identical to before. Only the moment changed.
    rare_intrA = df.loc[intrAgenic_rows, 'ChromHMM'].value_counts()
    rare_intrA = rare_intrA[rare_intrA < 10].index.tolist()

    # Over the rows the intergenic table will actually hold. It used to count over the whole frame,
    # which was right while that table WAS the whole frame. Now it would select rare states from a
    # population the table does not contain, and drop barcodes for a scarcity that is not there.
    rare_intEr = df.loc[~intrAgenic_rows, 'ChromHMM'].value_counts()
    rare_intEr = rare_intEr[rare_intEr < 10].index.tolist()

    union_rare = set(rare_intrA) | set(rare_intEr)
    # Held as barcodes rather than as a mask, so it survives the column drops and the row
    # subsetting below without depending on ChromHMM being present any more.
    rare_barcodes = df.index[df['ChromHMM'].isin(union_rare)]

    intrAgenic_alltable = df[intrAgenic_rows].copy()
    intrAgenic_alltable.fillna(0, inplace=True)
    intrAgenic_alltable.drop(columns=columns_to_drop_in_intragenic, inplace=True, errors='ignore')

    # Integrations OUTSIDE a MANE gene, and only those. Until 2026-08-27 this was `df.copy()`,
    # every row with the gene-body features hidden, so the table contained the intragenic rows
    # as a subset. The pair now compares two DIFFERENT row sets, not one row set seen twice.
    #
    # The criterion is `MANElength`: outside a MANE gene. Integrations inside a gencode NON-coding
    # gene (lncRNA, pseudogene, 13 % of these rows) stay here, which is why `DistToTSS`,
    # `DistToTES` and `RelPos` remain on the intergenic drop list. They measure a position inside
    # the containing gene, so here they would mean "inside the lncRNA" while in the intragenic
    # table they mean "inside the MANE gene". Same name, different quantity. Verified:
    # `DistToTSS + DistToTES` equals `MANElength` for 100 % of rows that have one.
    intErgenic_alltable = df[~intrAgenic_rows].copy()
    intErgenic_alltable.drop(columns=annot_table_cols_to_drop, inplace=True, errors='ignore')

    # In 'merge' mode NOTHING is dropped, and that has to be set explicitly rather than relying on
    # `union_rare` coming out empty. It does not come out empty: the renaming creates a 'rare'
    # category, and if fewer than ten barcodes were merged, that category itself passes the rarity
    # test and is dropped by the very next step. This is exactly what happened on `single_combined`:
    # 9 barcodes were merged into 'rare' and then removed, so the mode did nothing.
    if _rare_chromhmm_mode() == "merge":
        rare_barcodes = df.index[[]]
    intrAgenic_alltable = intrAgenic_alltable[~intrAgenic_alltable.index.isin(rare_barcodes)].copy()
    intErgenic_alltable = intErgenic_alltable[~intErgenic_alltable.index.isin(rare_barcodes)].copy()

    return intrAgenic_alltable, intErgenic_alltable