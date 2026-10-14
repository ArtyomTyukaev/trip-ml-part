"""
Which raw columns each functional group owns, per dataset and per table type.

    python group_shap/group_definitions.py --check
    python group_shap/group_definitions.py --dataset single_combined

The grouping rules themselves live in `feature_groups.py`. This file applies them in the
namespace the drop lists and the workers see, which is neither the table header nor the model's
SHAP columns.

## The namespace, derived the way the pipeline derives it

`make_intra_and_inter_genic_tables()` does, in this order:

  1. average replicates. A replicate mean whose name collides with an existing column gets its
     family suffix, so both columns survive.
  2. drop `score` and `strand`.
  3. drop the table type's own baseline list (`settings/columns_to_drop_in_*.txt`).
  4. one-hot encode the categorical columns, at training time, after all of the above.

`resolve_features()` reproduces exactly steps 1-3 from the table header. Getting this wrong fails
silently, because `get_gene_tables.py` drops with `errors='ignore'`.

## What `--check` is for

It asserts that every dataset's feature list is fully covered: no member of a group missing from
the table, and nothing in the table left in `other`.
"""

import argparse
import glob
import os
import sys

# Everything this module imports sits next to it in group_shap/ or under settings/.
ANALYSIS = os.path.dirname(os.path.abspath(__file__))
HERE = ANALYSIS
PROJECT = os.path.dirname(ANALYSIS)
sys.path.insert(0, PROJECT)
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, os.path.join(PROJECT, 'settings', 'custom_functions'))

from settings.utils import resolve_path              # noqa: E402
from dataset_config import DATASETS                  # noqa: E402
from feature_groups import groups_for                # noqa: E402
from get_replics_dicts import get_replics_dicts      # noqa: E402

import pandas as pd                                  # noqa: E402

# Coordinates the training scripts exclude from the feature matrix, plus the target. `chr`,
# `start` and `end` are lower case in the built tables and that is what the scripts key on. The
# feature `Chr` with a capital C is a different column and IS a feature.
NON_FEATURES = {'chr', 'start', 'end', 'score', 'strand', 'exp.log10'}

TABLE_TYPES = ['intragenic', 'intergenic']

BASELINE_LISTS = {
    'intragenic': 'settings/columns_to_drop_in_intragenic.txt',
    'intergenic': 'settings/columns_to_drop_in_intergenic.txt',
}

# Groups that can never be dropped, with the reason. Empty: every group can be measured.
UNDROPPABLE = {}


def read_list(path):
    return [line.strip() for line in open(path) if line.strip()]


def table_header(dataset):
    """The columns of one built training table. Every experiment of a dataset shares them."""
    pattern = str(resolve_path(f'tables_for_training/{dataset}')) + '/*.alldata.tab'
    tables = sorted(glob.glob(pattern))
    if not tables:
        raise FileNotFoundError(
            f'no *.alldata.tab under tables_for_training/{dataset} - has notebook 1 been run '
            f'for this dataset?')
    return list(pd.read_csv(tables[0], sep='\t', nrows=0).columns), os.path.basename(tables[0])


def apply_replicate_averaging(columns):
    """Rename replicate sets to their aggregate, mirroring get_gene_tables.py.

    Returns the column list as it stands after averaging. A replicate mean whose name collides with
    an existing column is written as `<key>_<family>`, so both columns survive.
    """
    out = list(columns)
    for family, group in get_replics_dicts().items():
        for key, members in group.items():
            present = [c for c in members if c in out]
            if not present:
                continue
            target = key if not (key in out and key not in present) else f'{key}_{family}'
            for c in present:
                out.remove(c)
            out.append(target)
    return out


def extra_drop(dataset, table_type):
    """Features dropped on top of the baseline list, for this dataset only.

    The single source is `dataset_config.DATASETS[dataset]['extra_drop']`. Empty for
    `single_combined`, `NMD_expected` for the intron and SON datasets.
    """
    from dataset_config import DATASETS
    return list(DATASETS.get(dataset, {}).get('extra_drop', {}).get(table_type, []))


def resolve_features(dataset, table_type):
    """The raw feature names a model of this dataset and table type is actually trained on."""
    header, source = table_header(dataset)
    features = apply_replicate_averaging([c for c in header
                                          if c not in NON_FEATURES
                                          and not c.startswith('Unnamed')])
    dropped = set(read_list(resolve_path(BASELINE_LISTS[table_type])))
    # Plus the per-dataset drop from `dataset_config`. Training gets the same list, so the group
    # membership and the trained model cannot drift apart.
    dropped |= set(extra_drop(dataset, table_type))
    return sorted(f for f in features if f not in dropped), source


def groups_for_dataset(dataset, table_type):
    """{group: [raw feature, ...]} for this dataset and table type, plus the ungrouped rest.

    `groups_for()` wants a frame with SHAP_* columns because that is how every caller reaches it.
    A bare frame with no rows is enough, since it only ever reads the column names.
    """
    features, source = resolve_features(dataset, table_type)
    groups, _ = groups_for(pd.DataFrame(columns=[f'SHAP_{f}' for f in features]))
    return groups, features, source


def check(datasets):
    """Assert full coverage in both directions, for every dataset and table type."""
    problems = []
    for dataset in datasets:
        for table_type in TABLE_TYPES:
            try:
                groups, features, source = groups_for_dataset(dataset, table_type)
            except FileNotFoundError as exc:
                problems.append(f'{dataset}: {exc}')
                continue

            ungrouped = groups.get('other', [])
            placed = [f for name, members in groups.items() if name != 'other' for f in members]
            duplicated = sorted({f for f in placed if placed.count(f) > 1})
            missing = sorted(set(features) - set(placed) - set(ungrouped))

            print(f'{dataset:18s} {table_type:11s} {len(features):3d} features in '
                  f'{len([g for g in groups if g != "other"]):2d} groups   ({source})')
            for name, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                if name == 'other':
                    continue
                print(f'    {name:16s} {len(members):3d}  {" ".join(sorted(members))}')

            if ungrouped:
                problems.append(f'{dataset}/{table_type}: {len(ungrouped)} features in no group: '
                                f'{sorted(ungrouped)}')
            if duplicated:
                problems.append(f'{dataset}/{table_type}: features in two groups at once, so '
                                f'removing one group would half-remove another: {duplicated}')
            if missing:
                problems.append(f'{dataset}/{table_type}: unaccounted features {missing}')
            print()

    if problems:
        print('PROBLEMS')
        for p in problems:
            print(f'  - {p}')
        return 1
    print('every feature of every dataset sits in exactly one group')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default=None,
                        help='one dataset. Default is all of them')
    parser.add_argument('--check', action='store_true',
                        help='exit non-zero if any feature is ungrouped or double-grouped')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else sorted(DATASETS)
    code = check(datasets)
    raise SystemExit(code if args.check else 0)


if __name__ == '__main__':
    main()
