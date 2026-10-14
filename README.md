# TRIP machine learning

Machine learning code of the paper "Genomic context tunes sequence-encoded RNA processing".

This repository contains scripts for training machine learning models to predict expression levels and splicing efficiency of TRIP reporters integrated into the genome.

The main goal is to learn how genomic context (chromatin features, protein tracks, etc.) influences reporter expression and splicing.

The core models used in this project are:

- **XGBoost Regressor**
- **XGBoost Classifier**

The methods are described in [METHODS.md](METHODS.md).

# Conda Environment Setup

This repository includes an `environment.yml` file that specifies all required package versions. Create the environment before running any notebook or script.

1) From the repository root, run:

   ```
   conda env create -f environment.yml
   ```

2) Install a Jupyter kernel:

   ```
   conda activate TRIPml
   python -m ipykernel install --user --name tripml --display-name "TRIPml"
   ```

The notebooks use the kernel `tripml`, in lower case. Only the conda environment is called `TRIPml`. Check with `jupyter kernelspec list`.

To run all code of a notebook at once:

```
jupyter execute "2) Train XGB Regressor.ipynb" --kernel_name tripml
```

Run all notebooks and scripts in this environment and from the repository root.

# Project

<pre>
trip-ml-part/
│
├── DATA/                          # input files of each dataset, provided by the user
├── MODELS/
│   ├── XGB_classifier/            # XGB_DS_Classifier.py, the classification training script
│   └── XGB_regressor/             # XGB_basic_DS_regressor.py, the regression training script
├── NMD/                           # prediction of the NMD_expected feature
├── OUTPUTS/                       # results, created at run time
├── group_shap/                    # training of the paper's models with group SHAP
├── plots/                         # figure panels of the paper
├── settings/                      # configuration files and shared functions
├── tables_for_training/           # training tables, written by notebook 1 or provided by the user
├── 1) Data Preprocessing.ipynb
├── 2) Train XGB Regressor.ipynb
├── 3) Train XGB Classifier.ipynb
├── METHODS.md
├── README.md
└── environment.yml
</pre>

All paths are relative to the repository root, which `settings/utils.py` finds from its own location.

For each task, every experiment gets two models: one trained on the integrations inside MANE genes and one on those outside. In the code and the outputs the two tables are called `intragenic` (inside) and `intergenic` (outside).

The notebooks train one dataset at a time with the settings typed into their parameter cell. The models of the paper come from the scripts in `group_shap/`.

# settings/

Contains configuration files and helper utilities.

Structure:
<pre>
settings/
├── utils.py                            # finds the project root and resolves paths
├── custom_functions/                   # custom Python functions
├── columns_to_drop_in_intergenic.txt   # columns removed from the table outside MANE genes
└── columns_to_drop_in_intragenic.txt   # columns removed from the table inside MANE genes
</pre>


The main files in `custom_functions/`:

- `get_gene_tables.py` builds the two tables of an experiment, inside and outside MANE genes, and averages the replicate tracks listed in `get_replics_dicts.py`
- `make_folds.py` makes the leakage-safe holdout and cross-validation splits, which keep nearby integrations on the same side
- `check_train_test_split.py` checks a split and drops features that are constant on the training rows
- `reg_to_classification.py` turns the target into class labels
- `group_shap.py` computes the exact group SHAP
- `print_plots_*.py` and `shap_heatmaps*.py` draw the metric and SHAP plots

# 1) Data Preprocessing.ipynb

Takes the input files of one dataset from `DATA/<dataset>/` (annotated barcode tables, chromatin feature tables, barcode order files and BED files) and merges them by barcode into one table per experiment that can be directly used by the training scripts.

# 2) Train XGB Regressor.ipynb

This notebook is used to train the XGBoost regressor. It submits `MODELS/XGB_regressor/XGB_basic_DS_regressor.py` as an LSF array job on a GPU, waits for it and merges the results. It allows:

- Training the model on the prepared tables in `tables_for_training/<dataset_name>/`
- Summarizing the metrics and the SHAP values of every experiment in `results.tsv`
- Generating predictions per barcode
- Plotting evaluation metric cluster maps
- Plotting SHAP value cluster maps

After training, results are saved to: `OUTPUTS/XGB_Regressor/<dataset_name>/`, with `results.tsv`, the per-barcode predictions in `predictions_for_barcodes/` and the figures in `plots/`.

The dataset and the settings are chosen in the parameter cell. The list of experiments is read from `DATA/<dataset_name>/names_all_clean.txt`, so this file is needed even if the training tables are provided directly. `LOG_TARGET = 1` trains on `log10(1 + x)` and is used for the expression datasets. For `intron_splicing` it is left empty. The values for each dataset are in `group_shap/dataset_config.py`. The LSF queue and the GPU are set by `QUEUE` and `GPU_SPEC`, and the logs go to `MODELS/XGB_regressor/artifacts/`.

Detailed explanation of outputs and workflow is provided directly inside the notebook.

# 3) Train XGB Classifier.ipynb

This notebook is used to train the XGBoost classifier. It allows the same things as the XGB Regressor notebook, but for the classification task. It submits `MODELS/XGB_classifier/XGB_DS_Classifier.py`, and results are saved to `OUTPUTS/XGB_Classifier/<dataset_name>/`.

The classes:

- With `BINARY_THRESHOLD` set (0.1 for the expression datasets), there are two classes, above the threshold and the rest.
- For `intron_splicing`, there are three classes of splicing efficiency, cut at 0.288 and 0.692. The input folder name `intron_splicing` is what switches this on, so the folder must keep this name, and `BINARY_THRESHOLD` must be empty.
- With neither, there are three classes: zero, below the median of the non-zero values, and the rest. The median is taken on the training pool.

# group_shap/

Scripts that train the models of the paper with exact group SHAP switched on (see [METHODS.md](METHODS.md#group-shap)).

<pre>
group_shap/
├── run_group_shap.py      # entry point: trains the models and collects the group SHAP
├── run_training.py        # one training run from the command line, called by run_group_shap.py
├── dataset_config.py      # settings of the five datasets used in the paper
├── group_definitions.py   # which features belong to which group, per dataset and table
├── feature_groups.py      # the grouping rules and the group colors
└── experiment_labels.py   # names and order of the experiments on the figures
</pre>


# NMD/

Predicts which integrations make the host transcript an NMD substrate (see [METHODS.md](METHODS.md#the-nmd-feature)). The result is the `NMD_expected` feature.

<pre>
NMD/
├── 01_build_transcripts.py   # the MANE transcripts in transcript coordinates
├── 02_map_barcodes.py        # places every barcode on a MANE transcript
├── 03_call_nmd.py            # the NMD call for every barcode
├── 04_make_table.py          # placement, NMD call and CHX result in one table per library
├── common.py                 # argument parsing and shared helpers
├── genuine_barcodes/         # input, provided by the user
├── logFC/                    # input, provided by the user
├── work/                     # intermediate tables, created at run time
└── output/                   # final tables, created at run time
</pre>

The four steps run in order:

```
python NMD/01_build_transcripts.py
python NMD/02_map_barcodes.py
python NMD/03_call_nmd.py
python NMD/04_make_table.py
```

# plots/

The figure panels of the paper that belong to the machine learning part, as PNG.

<pre>
plots/
├── Fig2A.png
├── Fig2E_classification_inside_MANE.png
├── Fig2E_classification_outside_MANE.png
├── Fig2E_regression_inside_MANE.png
├── Fig2E_regression_outside_MANE.png
├── Fig6C.png
├── Fig6E.png
├── Fig6F_classification.png
└── Fig6F_regression.png
</pre>
