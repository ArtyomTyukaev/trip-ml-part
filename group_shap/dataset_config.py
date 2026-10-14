"""
Per-dataset settings for the scripted training runs (`run_training.py`, `run_group_shap.py`).

The notebooks keep the same values in their parameter cell. Only the datasets used in the paper
are listed.
"""

FULL_CATEGORICAL = "ChromHMM,SPIN,GC_category,RepeatGroup,annot_rna_type,Chr"

DATASETS = {
    'single_combined': {
        # The target column holds raw meanRPM, so the regressor takes log10(1 + x) itself.
        'log_target': '1',
        # Classification is binary: expressed above 0.1 RPM or not.
        'binary_threshold': '0.1',
        'categorical': FULL_CATEGORICAL,
    },
    'intron_exp': {
        # The same kind of readout as single_combined: raw meanRPM, one experiment per intron.
        'log_target': '1',
        'binary_threshold': '0.1',
        'categorical': FULL_CATEGORICAL,
        # In the intron datasets NMD_expected is identical to "insertion inside a CDS" (on 100 %
        # of 151,260 barcodes), so it is dropped. It is already on the intergenic drop list, so
        # only the intragenic table changes.
        'extra_drop': {'intragenic': ['NMD_expected']},
    },
    'intron_splicing': {
        # The target is a splicing fraction between 0 and 1, so no log.
        'log_target': '',
        # Must stay empty. The classifier bins this dataset at the tertiles 0.288 / 0.692
        # (reg_to_classification.py) and recognises it by the name of the input folder.
        'binary_threshold': '',
        'categorical': FULL_CATEGORICAL,
        'extra_drop': {'intragenic': ['NMD_expected']},
    },
    'son_th15_cls': {
        # SON classification. The target column already holds the 0/1 label, and a threshold of
        # 0.5 just passes it through.
        'log_target': '',
        'binary_threshold': '0.5',
        'categorical': FULL_CATEGORICAL,
        'extra_drop': {'intragenic': ['NMD_expected'], 'intergenic': ['NMD_expected']},
        # With the default min_child_weight of 25 these models (a few hundred rows per fold)
        # never split and come out constant.
        'booster_params': {'MIN_CHILD_WEIGHT': 3},
    },
    'son_sig15_reg': {
        # SON regression. The target is the raw negative meanL2FC, already a log2, so no log.
        'log_target': '',
        'binary_threshold': '',
        'categorical': FULL_CATEGORICAL,
        'extra_drop': {'intragenic': ['NMD_expected'], 'intergenic': ['NMD_expected']},
        'depth_grid': '3,4,5,6',
    },
}

GRID = {
    'depth_grid': '5,6',
    # The regressor gets 3000 boosting rounds and the classifier 1000, as in the notebooks.
    'num_boost_round': {'regression': '3000', 'classification': '1000'},
    'early_stop': '500',
    # Seeds the CV folds. The holdout split is seeded with 2005 inside the workers.
    'fold_seed': '2005',
    'lines_per_job': 1,
    # LSF settings. Memory changes only the queue time, not the numbers.
    'queue': 'short-gpu',
    'gpu_spec': 'num=1:j_exclusive=yes:gmem=17G:gmodel=NVIDIAA40',
    'mem_mb': 16000,
    'affinity': 'thread*1',
}

TASKS = {
    'regression': {
        'script': 'MODELS/XGB_regressor/XGB_basic_DS_regressor.py',
        'artifacts': 'MODELS/XGB_regressor/artifacts',
        'output': 'OUTPUTS/XGB_Regressor',
        'job_name': 'regJob',
    },
    'classification': {
        'script': 'MODELS/XGB_classifier/XGB_DS_Classifier.py',
        'artifacts': 'MODELS/XGB_classifier/artifacts',
        'output': 'OUTPUTS/XGB_Classifier',
        'job_name': 'clsJob',
    },
}
