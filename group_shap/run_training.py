"""
Run one training job from the command line, without touching a notebook.

The notebooks `2) Train XGB Regressor.ipynb` and `3) Train XGB Classifier.ipynb` do exactly
what this script does, but they keep the dataset in a parameter cell. There is one copy of
each notebook, so two runs cannot use different settings at the same time and a loop over
rounds cannot change them without editing the file. This script takes the settings as
arguments instead.

    python group_shap/run_training.py --dataset single_combined --task regression

Steps, in order: clear the artifact folders, submit an LSF array job, wait for it, report any
error logs, merge the per-job outputs into results.tsv, and draw the plots.
"""

import argparse
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings.utils import resolve_path, PROJECT_ROOT

sys.path.insert(0, str(resolve_path('settings/custom_functions')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import DATASETS, GRID, TASKS


# How many times an array element may be resubmitted before the run is called broken. Three is
# enough for a dispatch failure that hits about one element in fifty, and small enough that a
# real, deterministic error still surfaces within a couple of minutes.
MAX_ATTEMPTS = 3


# The booster knobs `MODELS/XGB_regressor/XGB_basic_DS_regressor.py:booster_params()` reads from
# the environment. Listed here so a misspelt --booster-param fails at submission instead of
# quietly doing nothing inside the worker.
BOOSTER_KNOBS = {'LEARNING_RATE', 'SUBSAMPLE', 'COLSAMPLE_BYTREE',
                 'REG_LAMBDA', 'MIN_CHILD_WEIGHT', 'GAMMA'}


def drop_list_for(dataset, table_type, baseline_path):
    """Path to the drop list: the baseline list plus `extra_drop` from `dataset_config`.

    Added 2026-09-11 to fix a divergence. Before that only `run_group_shap.py`, which builds its
    command itself, read `extra_drop`. A plain run did not see it, so for `intron_splicing` group
    SHAP was computed WITHOUT `NMD_expected` while the metrics of the same dataset were computed
    WITH it. The two could drift apart silently, because the drop is applied with
    `errors='ignore'`.

    A separate file rather than an edit of `settings/columns_to_drop_in_*.txt`: that list is shared
    by the whole project, and a crashed run would leave it trimmed for every run after it.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import group_definitions
    extra = group_definitions.extra_drop(dataset, table_type)
    if not extra:
        return baseline_path
    lines = [l.strip() for l in open(baseline_path) if l.strip()]
    added = [n for n in extra if n not in lines]
    lines += added
    out_dir = resolve_path('OUTPUTS/drop_lists')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'drop_%s_%s.txt' % (table_type, dataset))
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    if added:
        print('drop extra  %-11s %s  (from dataset_config)' % (table_type, ', '.join(added)))
    return path


def worker_defaults(worker_path):
    """Booster defaults READ FROM THE WORKER SOURCE: {NAME: value}.

    Parsed, not copied. A copy would drift from the worker silently, and these are exactly the
    numbers one needs later: `min_child_weight` is 25 in the classifier and 5 in the regressor,
    and neither was written down anywhere.

    Exactly one pattern is parsed, `getenv_float("NAME", NUMBER)`, which is how both workers set
    all six knobs. If the pattern changes, an empty dict comes back and `run_info.txt` honestly
    says "not readable". That is better than confidently recording a stale number.
    """
    try:
        text = open(worker_path, encoding='utf-8').read()
    except OSError:
        return {}
    found = re.findall(r'getenv_float\(\s*"([A-Z_]+)"\s*,\s*([0-9.]+)\s*\)', text)
    return {name: value for name, value in found if name in BOOSTER_KNOBS}


def write_run_info(output_dir, args, env, experiments, run_token, worker_path):
    """`run_info.txt` next to `results.tsv`: how exactly this run was produced.

    Added 2026-09-10. The hyperparameters reached the worker through the environment and none of
    them was written to disk, so there was no way to tell which `min_child_weight` the current
    numbers were computed with.

    The EFFECTIVE values of the knobs are written, not only the overrides. A default nobody
    touched is part of the answer too, and usually the part that matters.
    """
    defaults = worker_defaults(worker_path)
    lines = [
        '# how this run was produced. Written by group_shap/run_training.py at submission time.',
        'written        %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset        %s' % args.dataset,
        'task           %s' % args.task,
        'slot           %s' % (args.slot or '(none)'),
        'run_token      %s' % run_token,
        'experiments    %d  (%s)' % (len(experiments), ', '.join(experiments)),
        'depth_grid     %s' % args.depth_grid,
        'worker         %s' % worker_path,
        '',
        '# booster knobs, EFFECTIVE values. "default" means nothing overrode the worker.',
    ]
    for knob in sorted(BOOSTER_KNOBS):
        if knob in env:
            lines.append('%-14s %s   (overridden)' % (knob.lower(), env[knob]))
        elif knob in defaults:
            lines.append('%-14s %s   (default)' % (knob.lower(), defaults[knob]))
        else:
            lines.append('%-14s not readable from %s' % (knob.lower(), os.path.basename(worker_path)))
    lines += ['', '# everything else handed to the worker through the environment.']
    for key in sorted(env):
        if key not in BOOSTER_KNOBS:
            lines.append('%-22s %s' % (key, env[key]))

    path = output_dir / 'run_info.txt'
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('run_info    %s' % path)


def clear_artifacts(artifacts_rel, output_dir, preds_dir):
    """Remove what the previous run left behind, then recreate the empty folders.

    The training script appends nothing - it writes one file per array job - so stale files
    from an earlier run would be merged into the new results.tsv.

    Takes the artifacts path rather than the task config, because with --slot the two are no
    longer the same thing: the folder is the task's plus the slot suffix.
    """
    artifacts = resolve_path(artifacts_rel)
    for name in ['output_splitted', 'grid_selection', 'log_err_LSF',
                 'log_out_LSF', 'log_training', 'log_training_errors']:
        shutil.rmtree(artifacts / name, ignore_errors=True)
        os.makedirs(artifacts / name, exist_ok=True)

    shutil.rmtree(output_dir / 'plots', ignore_errors=True)
    shutil.rmtree(preds_dir, ignore_errors=True)
    if (output_dir / 'results.tsv').exists():
        os.remove(output_dir / 'results.tsv')

    os.makedirs(output_dir / 'plots', exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)


def submit_and_wait(cmd, poll_seconds=30):
    """Submit the array job, block until every element has left the queue, return the statuses.

    Does not raise on EXIT. The caller decides, because a failing element is not always a
    failing run: the cluster drops roughly one element in fifty at dispatch - same second in
    and out, exit code 255, zero CPU time, empty logs, a different host each time - and without a
    retry that one element would fail the whole run.
    """
    submit = subprocess.run(cmd, shell=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(submit.stdout)
    if submit.returncode != 0:
        raise RuntimeError(f'bsub failed with code {submit.returncode}')

    match = re.search(r'Job <(\d+)>', submit.stdout)
    if not match:
        raise RuntimeError('cannot parse the LSF job id out of the bsub output')
    job_id = match.group(1)

    print(f'waiting for LSF job {job_id}')
    silent_polls = 0
    while True:
        out = subprocess.run(f'bjobs -noheader -o stat {job_id}', shell=True,
                             text=True, stdout=subprocess.PIPE).stdout
        statuses = [s.split()[0] for s in out.splitlines() if s.strip()]
        # bjobs lists the array elements as DONE for an hour after they finish, so silence here
        # is not "the job is over" - it is bjobs having failed. Treating it as the end of the
        # wait would merge whatever happens to be on disk at that moment, which is the notebook's
        # behaviour and how a round can be built from a half-finished array. Retry, then give up.
        if not statuses:
            silent_polls += 1
            if silent_polls > 4:
                raise RuntimeError(f'bjobs reported nothing for job {job_id} five times in a '
                                   f'row - cannot tell whether the array finished')
            time.sleep(poll_seconds)
            continue
        silent_polls = 0
        # Every state that means "not finished". SSUSP and USUSP were missing, and the omission is
        # not cosmetic: LSF suspends jobs when a host is loaded - seen 2026-08-27, six of seventeen
        # jobs suspended at once on hgn39 - and a suspended element is very much still coming back.
        # With only {PEND, RUN, SUSP} listed, an array whose last unfinished element was SSUSP read
        # as finished, and the caller then resubmitted that element while the original was still
        # alive. Both carry the SAME run_token, so `is_fresh` cannot tell them apart, and the two
        # would race to write one output file - possibly while merge_results was reading it.
        if not any(s in {'PEND', 'RUN', 'SUSP', 'SSUSP', 'USUSP', 'PSUSP', 'WAIT',
                         'PROV', 'UNKWN'} for s in statuses):
            break
        time.sleep(poll_seconds)

    final = subprocess.run(f'bjobs -a -noheader -o stat {job_id}', shell=True,
                           text=True, stdout=subprocess.PIPE).stdout
    statuses = [s.split()[0] for s in final.splitlines() if s.strip()]
    print(f'final statuses: {", ".join(sorted(set(statuses)))}')
    return statuses


def output_of(artifacts, job_index, lines_per_job):
    """The file this array element writes, derived the same way the worker derives it.

    XGB_*.py builds the name from its own LSB_JOBINDEX as
    all_xgboost_results[{start}:{start + lines_per_job}].tsv, without clipping the end to the
    number of experiments, so the name is predictable from the index alone.
    """
    start = (job_index - 1) * lines_per_job
    return (resolve_path(artifacts) / 'output_splitted'
            / f'all_xgboost_results[{start}:{start + lines_per_job}].tsv')


def is_fresh(path, token):
    """True only if this file exists AND was written by the run identified by `token`.

    Existence alone is not proof of authorship, and treating it as proof corrupted a round on
    2026-08-23. clear_artifacts() empties output_splitted/ at the start of every round, but LSF
    reporting an array as finished does not mean its processes are gone: a worker of the previous
    round outlived the cleanup and wrote its file into the emptied folder. The current round's
    worker for that same experiment was then killed, the driver saw a file where it expected one,
    retried nothing, and merged the previous round's row - which still carried the feature this
    round had removed. Only check_dropped_gone() downstream noticed, and only because the stale
    row happened to hold a dropped feature.
    """
    if not path.exists():
        return False
    import pandas as pd
    try:
        head = pd.read_csv(path, sep='\t', nrows=1)
    except Exception:
        return False
    if 'run_token' not in head.columns or head.empty:
        return False
    return bool((head['run_token'].astype(str) == token).all())


def report_errors(artifacts_rel):
    """Print anything the workers wrote to their error logs. Empty is what we want."""
    artifacts = resolve_path(artifacts_rel)
    found = False
    for folder in ['log_training_errors', 'log_err_LSF']:
        d = artifacts / folder
        for name in sorted(os.listdir(d)) if d.is_dir() else []:
            path = d / name
            if path.is_file() and path.stat().st_size > 0:
                found = True
                print(f'\n--- {folder}/{name} ---')
                print(path.read_text(errors='replace'))
    if not found:
        print('error logs are empty')
    return found


def merge_results(artifacts_rel, output_dir, n_experiments, run_token=None):
    """Concatenate the per-job .tsv files into one results.tsv.

    Each array element writes its own file, so a missing one means an experiment was skipped.
    Two rows are expected per experiment, one for the intragenic table and one for intergenic.
    """
    import pandas as pd

    parts = sorted(p for p in (resolve_path(artifacts_rel) / 'output_splitted').iterdir()
                   if p.is_file() and p.suffix == '.tsv' and not p.name.startswith('.'))
    if not parts:
        raise FileNotFoundError('no per-job .tsv files to merge - every array element failed')

    data = pd.concat([pd.read_csv(p, sep='\t') for p in parts], ignore_index=True)

    # Belt and braces on top of is_fresh(): refuse to merge a row this run did not produce.
    # is_fresh() should already have caught it and retried the element, so reaching this raise
    # means a file appeared between that check and this merge - exactly the race that corrupted
    # intron_exp/classification round 5 on 2026-08-23.
    if run_token is not None:
        if 'run_token' not in data.columns:
            raise RuntimeError(
                'the per-job outputs carry no run_token column. The worker scripts under MODELS/ '
                'are older than this driver - a stale row cannot be detected, so refusing to '
                'merge rather than risk mixing rounds.')
        alien = data[data['run_token'].astype(str) != run_token]
        if not alien.empty:
            raise RuntimeError(
                f'{len(alien)} of {len(data)} merged rows were written by another run '
                f'{sorted(set(alien["run_token"].astype(str)))} instead of {run_token}: '
                f'{sorted(set(zip(alien["experiment"], alien["table_type"])))}. A previous '
                f'round left files behind. Nothing was written.')
        data = data.drop(columns=['run_token'])

    out = output_dir / 'results.tsv'
    data.to_csv(out, sep='\t', index=False)

    expected = 2 * n_experiments
    if len(data) != expected:
        print(f'[WARNING] results.tsv has {len(data)} rows, expected {expected}. '
              f'Some experiments did not finish.')
    else:
        print(f'[OK] {len(data)} rows from {len(parts)} job outputs')
    return out


# The classifier reports a different set of metrics, and prepare_agg_df_for_plots defaults to
# the regression ones, so they have to be named explicitly.
CLS_METRICS = ['TEST_b_accuracy', 'TEST_accuracy', 'TEST_roc_auc_weighted',
               'TRAIN_b_accuracy', 'TRAIN_accuracy', 'TRAIN_roc_auc_weighted']


def draw_plots(task, dataset, output_dir):
    """Draw the metric and SHAP heatmaps, the same ones the notebooks draw.

    The SHAP plots come from the regression module for both tasks - it is the only place that
    keeps the SHAP_* columns. prepare_agg_df_cls() drops them, so the classifier needs two
    aggregated frames, one for metrics and one for SHAP.
    """
    import pandas as pd
    from experiment_labels import ordered
    from feature_groups import groups_for
    from print_plots_regression import (prepare_agg_df_for_plots, plot_shap_clustermap_pct,
                                        plot_shap_clustermap_log_pct)

    data = pd.read_csv(output_dir / 'results.tsv', sep='\t')
    plots_dir = output_dir / 'plots'

    # One axis order and one set of names for all six figures, derived from the codes this round
    # actually produced. The codes stay in the frame - they are the join key of results.tsv and of
    # every summary built from it - and only the tick text becomes the gene name. Passing the order
    # also switches the metric heatmaps' own clustering off, which is what used to put the same
    # intron in a different column in each of the two panels of one figure.
    exp_codes, exp_names = ordered(data['experiment'].unique())

    for table_type in ['intragenic', 'intergenic']:
        if task == 'regression':
            from print_plots_regression import plot_metric_clustermap
            plot_metric_clustermap(
                agg_df=prepare_agg_df_for_plots(data), dataset_name=dataset,
                table_type=table_type, model='XGBoost', split='both',
                experiment_order=exp_codes, experiment_labels=exp_names,
                width_per_exp=5, height=7, title_font_size=32, axis_font_size=20,
                cell_font_size=18, save=True, outdir=plots_dir)
        else:
            from print_plots_classification import prepare_agg_df_cls, plot_metric_clustermap_cls
            plot_metric_clustermap_cls(
                agg_df=prepare_agg_df_cls(data), dataset_name=dataset,
                table_type=table_type, model='XGBoost', split='both',
                experiment_order=exp_codes, experiment_labels=exp_names,
                width_per_exp=4, height=7, save=True, outdir=plots_dir)

    metrics = None if task == 'regression' else CLS_METRICS
    agg_shap = prepare_agg_df_for_plots(data, metrics=metrics)
    groups, colors = groups_for(agg_shap)

    for table_type in ['intragenic', 'intergenic']:
        for plot in [plot_shap_clustermap_pct, plot_shap_clustermap_log_pct]:
            plot(agg_df=agg_shap, dataset_name=dataset, table_type=table_type,
                 model='XGBoost', experiment_order=exp_codes, experiment_labels=exp_names,
                 width=40, height_per_exp=0.63, title_font_size=32,
                 x_tick_font_size=12, y_tick_font_size=12,
                 white_pct=20.0, dropnan=True, save=True, outdir=plots_dir,
                 groups_to_features=groups, group_colors=colors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    parser.add_argument('--task', required=True, choices=sorted(TASKS))
    parser.add_argument('--exp-list', default=None,
                        help='experiment list file, relative to the project root. '
                             'Default is DATA/<dataset>/names_all_clean.txt')
    parser.add_argument('--booster-param', action='append', default=[], dest='booster_params',
                        metavar='NAME=VALUE',
                        help='override one fixed booster parameter for THIS run, repeatable: '
                             '--booster-param REG_LAMBDA=12 --booster-param GAMMA=2. Reaches the '
                             'worker as an environment variable of that name, and the worker\'s '
                             'default is the value that was hardcoded, so omitting it changes '
                             'nothing. Known to the regressor: LEARNING_RATE, SUBSAMPLE, '
                             'COLSAMPLE_BYTREE, REG_LAMBDA, MIN_CHILD_WEIGHT, GAMMA')
    parser.add_argument('--num-boost-round', default=None, dest='num_boost_round',
                        help='override the boosting-round cap for this run. Worth raising together '
                             'with a smaller LEARNING_RATE, which needs more rounds to get there')
    parser.add_argument('--depth-grid', default=GRID['depth_grid'], dest='depth_grid',
                        help='comma-separated max_depth values for the grid search. Overrides '
                             'GRID["depth_grid"] for THIS run only - editing the shared constant '
                             'would silently change every other run and is easy to forget to undo. '
                             'Pair it with --slot so the trial does not overwrite the baseline')
    parser.add_argument('--no-plots', action='store_true',
                        help='skip the per-run figures')
    parser.add_argument('--drop-intragenic', default='settings/columns_to_drop_in_intragenic.txt',
                        help='the two drop lists, project-relative. Giving each task its own '
                             'pair is what lets regression and classification run at the same '
                             'time - the training scripts read them through ANNOT_DROP_COLS '
                             'and INTRA_DROP_COLS, never by a fixed path')
    parser.add_argument('--drop-intergenic', default='settings/columns_to_drop_in_intergenic.txt')
    parser.add_argument('--slot', default=None,
                        help='isolate this run from every other one. Pass a SHORT name for what '
                             'the run is, and the dataset is added for you. '
                             'The artifacts folder, the output folder and the LSF job name all '
                             'get it, so two runs of the SAME task (two datasets, or two '
                             'configurations) can be in the cluster at once. '
                             'Without it everything shares one artifacts folder and a second run '
                             'wipes the first at startup')
    # Group SHAP. Off unless --group-map is given. It has to be an explicit argument rather than an
    # exported variable: `bsub` propagates the launching shell's environment, so a stray
    # `export GROUP_MAP_JSON=...` would silently switch group SHAP on for every later run too,
    # and each of them would then pay about 40 GPU-minutes per model.
    parser.add_argument('--group-map', default=None,
                        help='JSON of {table_type: {raw feature: group}}. Switches on the '
                             'GSHAP_/BSHAP_/ISHAP_/TSHAP_ columns')
    parser.add_argument('--group-shap-n-explain', type=int, default=200)
    parser.add_argument('--group-shap-n-background', type=int, default=50)
    parser.add_argument('--group-shap-seed', type=int, default=2005)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the paths and the bsub command, then exit WITHOUT clearing '
                             'anything and without submitting. There was no such flag until '
                             '2026-08-23, and a "does it even start" probe therefore wiped '
                             'OUTPUTS/ and put a real array in the queue')
    args = parser.parse_args()

    # Validated HERE, before the --dry-run branch, so a dry run checks the same thing a real one
    # does. A typo in the NAME would otherwise reach the worker as a variable nothing reads, and
    # the trial would come back looking like a clean test of a setting that was never applied -
    # indistinguishable from "this parameter does nothing".
    booster_env = {}
    for item in args.booster_params:
        if '=' not in item:
            raise SystemExit(f'--booster-param wants NAME=VALUE, got {item!r}')
        name, value = item.split('=', 1)
        name = name.strip().upper()
        if name not in BOOSTER_KNOBS:
            raise SystemExit(f'--booster-param {name}: the worker reads none of that name. '
                             f'Known: {", ".join(sorted(BOOSTER_KNOBS))}')
        float(value)                       # raises here rather than inside the LSF worker
        booster_env[name] = value.strip()

    ds_cfg = DATASETS[args.dataset]
    task_cfg = TASKS[args.task]

    # Per-dataset booster settings FROM THE CONFIG, under the same names as --booster-param.
    # The command line overrides the config, not the other way round, so `--booster-param` stays
    # a way to try something once without touching the config.
    #
    # Added 2026-09-09. `run_group_shap.py` builds its command itself and does NOT pass
    # `--booster-param`, so a group SHAP run on a SON dataset would have trained the models with
    # the hardcoded `min_child_weight = 25`, the value at which models on 300-650 rows come out
    # CONSTANT. The numbers would have drifted from the main run silently. A setting the model's
    # existence depends on has to live where every caller sees it.
    for _name, _value in (ds_cfg.get('booster_params') or {}).items():
        _name = _name.strip().upper()
        if _name not in BOOSTER_KNOBS:
            raise SystemExit(f'dataset_config booster_params: {_name} is not a knob the worker '
                             f'reads. Known: {", ".join(sorted(BOOSTER_KNOBS))}')
        booster_env.setdefault(_name, str(_value))

    # The same for the depth grid. A wider grid LEAVES the choice to cross-validation instead of
    # imposing a value, and for SON it halves the overfitting of the regression (median
    # train-holdout gap 0.120 against 0.198). Overridden only when the flag is set explicitly, so
    # the comparison with GRID stays fair.
    if ds_cfg.get('depth_grid') and args.depth_grid == GRID['depth_grid']:
        args.depth_grid = ds_cfg['depth_grid']

    exp_list = resolve_path(args.exp_list or f'DATA/{args.dataset}/names_all_clean.txt')
    experiments = [l.strip() for l in open(exp_list) if l.strip()]

    # A slot separates all three of the things a concurrent run would otherwise collide on. The
    # dataset is folded in here rather than by the caller, so the slot itself stays short and the
    # paths do not repeat it: artifacts_intron_exp_chain, not
    # artifacts_chain_intron_exp_regression. The artifacts path reaches the worker through
    # ARTIFACTS_DIR and must therefore stay project-relative, not absolute.
    if args.slot:
        artifacts_rel = f'{task_cfg["artifacts"]}_{args.dataset}_{args.slot}'
        output_dir = resolve_path(f'{task_cfg["output"]}/{args.dataset}_{args.slot}')
        job_name = f'{task_cfg["job_name"]}_{args.dataset}_{args.slot}'
    else:
        artifacts_rel = task_cfg['artifacts']
        output_dir = resolve_path(f'{task_cfg["output"]}/{args.dataset}')
        job_name = task_cfg['job_name']
    preds_dir = output_dir / 'predictions_for_barcodes'
    artifacts = resolve_path(artifacts_rel)

    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} {args.dataset} / {args.task} / '
          f'{len(experiments)} experiments'
          + (f' / slot {args.slot}' if args.slot else ''))

    n_jobs = max(1, math.ceil(len(experiments) / GRID['lines_per_job']))

    if args.dry_run:
        print('\n--- DRY RUN: nothing cleared, nothing submitted ---')
        print(f'artifacts   {artifacts}')
        print(f'output      {output_dir}')
        print(f'predictions {preds_dir}')
        print(f'job name    {job_name}[1-{n_jobs}]')
        print(f'drop intra  {resolve_path(args.drop_intragenic)}')
        print(f'drop inter  {resolve_path(args.drop_intergenic)}')
        print(f'would clear {artifacts} (6 subfolders), {output_dir}/results.tsv, '
              f'{output_dir}/plots, {preds_dir}')
        # The group-SHAP switch is worth showing here: it is the difference between a 40-minute
        # round and a 2-hour one, and a silently-off run looks exactly like a normal one.
        if args.group_map:
            print(f'group SHAP  ON, {resolve_path(args.group_map)}, '
                  f'{args.group_shap_n_explain} explained x {args.group_shap_n_background} '
                  f'background, seed {args.group_shap_seed}')
        else:
            print('group SHAP  off (no --group-map)')
        return

    clear_artifacts(artifacts_rel, output_dir, preds_dir)

    # Identifies THIS run in every file its workers write, so a leftover from the previous round
    # can be recognised instead of merged. Must be unique per invocation, hence the pid and the
    # clock. The slot, dataset and task are in there only to make a stale-file message readable.
    run_token = (f'{args.slot or "noslot"}_{args.dataset}_{args.task}'
                 f'_{os.getpid()}_{int(time.time())}')
    print(f'run_token {run_token}')

    env = {
        'RUN_TOKEN': run_token,
        'EXP_LIST': str(exp_list),
        'LINES_PER_JOB': str(GRID['lines_per_job']),
        'INPUT_DIR': str(resolve_path(f'tables_for_training/{args.dataset}')),
        'OUTPUT_DIR_PATH': str(output_dir),
        'BARCODE_PREDS_PATH': str(preds_dir),
        # `extra_drop` from the config is merged in HERE, so a plain run and a group SHAP run
        # train the same model. Without it the baseline path is used.
        'ANNOT_DROP_COLS': drop_list_for(args.dataset, 'intergenic',
                                         str(resolve_path(args.drop_intergenic))),
        'INTRA_DROP_COLS': drop_list_for(args.dataset, 'intragenic',
                                         str(resolve_path(args.drop_intragenic))),
        'DEPTH_GRID': args.depth_grid,
        'CATEGORICAL_COLUMNS': ds_cfg['categorical'],
        'NUM_BOOST_ROUND': args.num_boost_round or GRID['num_boost_round'][args.task],
        'EARLY_STOP': GRID['early_stop'],
        'FOLD_SEED': GRID['fold_seed'],
        'LOG_TARGET': ds_cfg['log_target'],
        'PROJECT_ROOT': str(PROJECT_ROOT),
        # Project-relative, because the worker resolves it through PROJECT_ROOT itself. This is
        # what makes --slot work: without it the worker would write into the task's one shared
        # artifacts folder no matter which folder this script had prepared and is watching.
        'ARTIFACTS_DIR': artifacts_rel,
    }
    env.update(booster_env)                # validated above, before the --dry-run branch
    # Only the classifier reads this one, and only a non-empty value switches it to binary.
    if args.task == 'classification':
        env['BINARY_THRESHOLD'] = ds_cfg['binary_threshold']

    # What to do with rare ChromHMM states: 'drop' (the old behaviour, the default) or 'merge',
    # which puts them into one 'rare' category without losing rows. Meant for small datasets.
    # On SON the "state with fewer than 10 barcodes" filter threw away up to 76 % of rows,
    # because tables of a few hundred rows are spread over 15-17 states. Read by
    # `settings/custom_functions/get_gene_tables.py`.
    if ds_cfg.get('rare_chromhmm'):
        env['RARE_CHROMHMM'] = ds_cfg['rare_chromhmm']
        # Ready labels 0..K-1 in the target column. Only for a dataset that put them there.
        # The others have no such key and the variable is not set.
        if ds_cfg.get('class_passthrough'):
            env['CLASS_PASSTHROUGH'] = '1'
            env['N_CLASSES'] = str(ds_cfg['n_classes'])

    if args.group_map:
        env['GROUP_MAP_JSON'] = str(resolve_path(args.group_map))
        env['GROUP_SHAP_N_EXPLAIN'] = str(args.group_shap_n_explain)
        env['GROUP_SHAP_N_BG'] = str(args.group_shap_n_background)
        env['GROUP_SHAP_SEED'] = str(args.group_shap_seed)
        print(f'group SHAP ON: {env["GROUP_MAP_JSON"]}, '
              f'{args.group_shap_n_explain} explained x {args.group_shap_n_background} background, '
              f'seed {args.group_shap_seed}')

    write_run_info(output_dir, args, env, experiments, run_token,
                   str(resolve_path(task_cfg['script'])))

    env_str = ' '.join(f'{k}="{v}"' for k, v in env.items())

    def command(indices):
        # LSF takes a comma-separated list as an array spec, so a retry submits only the
        # elements that came back empty instead of recomputing the whole array.
        spec = f'1-{n_jobs}' if indices is None else ','.join(str(i) for i in indices)
        return (f'{env_str} bsub -J "{job_name}[{spec}]" '
                f'-q {GRID["queue"]} -gpu {shlex.quote(GRID["gpu_spec"])} '
                f'-R rusage[mem={GRID["mem_mb"]}] -R affinity[{GRID["affinity"]}] '
                f'-oo "{artifacts}/log_out_LSF/out_%I.txt" '
                f'-eo "{artifacts}/log_err_LSF/err_%I.txt" '
                f'{sys.executable} "{resolve_path(task_cfg["script"])}"')

    # An element whose output is missing OR STALE is the only failure worth retrying, and it is a
    # stronger test than the LSF status: a worker that exits cleanly without writing is caught
    # too, and so is one whose file was actually written by the previous round.
    # clear_artifacts() is deliberately not called again - the elements that did finish keep
    # their files.
    pending = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        submit_and_wait(command(pending))
        report_errors(artifacts_rel)
        candidates = list(range(1, n_jobs + 1)) if pending is None else pending
        missing = [i for i in candidates
                   if not is_fresh(output_of(artifacts_rel, i, GRID['lines_per_job']), run_token)]
        if not missing:
            break
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(
                f'array elements {missing} produced no output of their own after '
                f'{MAX_ATTEMPTS} attempts. This is no longer the cluster dropping an element at '
                f'dispatch - read {artifacts}/log_training_errors and '
                f'log_out_LSF/out_<index>.txt.')
        print(f'[retry {attempt}/{MAX_ATTEMPTS - 1}] no fresh output from array elements '
              f'{missing}, resubmitting just those')
        # Assigning this was missing until 2026-08-23, so every retry silently resubmitted the
        # WHOLE array while the message above claimed otherwise. Wasteful, and it widened the
        # window in which a straggler from the previous round could write into the cleared folder.
        pending = missing

    merge_results(artifacts_rel, output_dir, len(experiments), run_token)

    if not args.no_plots:
        draw_plots(args.task, args.dataset, output_dir)

    print(f'{datetime.now():%Y-%m-%d %H:%M:%S} done, results in {output_dir}')


if __name__ == '__main__':
    main()
