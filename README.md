# Working Memory Planning Code

This repository contains the analysis and computational modeling code for the working-memory planning manuscript.

## Overview

### Analysis

The `analysis/` directory contains the R pipeline for cleaning behavioral and model-output CSV files, generating manuscript figures, and running statistical models. Preprocessing scripts write standardized datasets to `analysis/data/processed_data/`; figure scripts write PDFs to `analysis/figures/`; stats scripts write model summaries to `analysis/stats/`.

### Model

The `model/` directory contains the Python implementation of the variational recurrent neural network model, training loop, simulation code, and Python package requirements. The model entry point is `model/src/main.py`.

## Repository Structure

```text
.
├── analysis/
│   ├── data/
│   │   ├── raw_data/
│   │   └── processed_data/
│   ├── preprocessing/
│   ├── scripts/
│   │   ├── figures/
│   │   └── stats/
│   ├── figures/
│   ├── stats/
│   └── requirements.R
├── model/
│   ├── requirements.txt
│   └── src/
│       ├── main.py
│       ├── model.py
│       ├── train.py
│       ├── simulate.py
│       ├── config.py
│       └── helper.py
└── README.md
```

## Data Download

Download the raw data from OSF:

https://osf.io/38gqz/files/osfstorage

Place the downloaded files directly in:

```text
analysis/data/raw_data/
```

## R Analysis Setup

From the repository root, install the R dependencies with:

```r
source("analysis/requirements.R")
```

The R code assumes this layout: the repository root contains `analysis/`, and analysis paths are built by `analysis/scripts/utils.R` through `analysis_path <- function(...) here::here("analysis", ...)`. Run R from the repository root, or open the repository root as the RStudio project/root. If you reorganize the repository, update that `analysis_path()` helper once so it points to the correct `analysis/` directory.

## R Analysis Pipeline

Run scripts in this order:

1. Preprocess raw data. This script runs all preprocessing scripts:

```r
source("analysis/preprocessing/run_all_preprocessing.R")
```

2. Generate figures. This script runs all figure scripts:

```r
source("analysis/scripts/figures/run_all_figures.R")
```

3. Run statistics:

```r
source("analysis/scripts/stats/generate_stats.r")
```

## Python Model Setup

Create and activate a virtual environment, then install dependencies:

```bash
cd model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

## Python Model Usage

Command format:

```bash
python model/src/main.py <lambda_values> <alpha_values> <model_dir> <simulation_dir> <epochs> <seed> <tree_size> <mode> <tree_order>
```

Arguments:

- `<lambda_values>`: comma-separated information-cost weights, for example `0.1,1,10`.
- `<alpha_values>`: comma-separated reconstruction-loss weights.
- `<model_dir>`: directory for trained model weights.
- `<simulation_dir>`: directory for simulation CSV outputs.
- `<epochs>`: number of training epochs, for example `120`.
- `<seed>`: random seed.
- `<tree_size>`: root-inclusive tree size: `3`, `7`, `13`, `31`, or `40`.
- `<mode>`: `train` to train then simulate, or another mode value to skip training and run simulation from existing weights.
- `<tree_order>`: tree topology/order label. For 31-node simulations use `deep_breadth`, `deep_depth`, `wide_breadth`, or `wide_depth`; for smaller trees this argument is still required but does not change the tree.

The action/reward loss uses a default weight of `1.0`. The information-cost,
reconstruction, and action/reward weights are scaled internally by the model's
tree-size-specific parameter scaler.

Example:

```bash
python model/src/main.py 1 1 model/weights model/simulations 120 1 7 train deep_depth
```

## Outputs

- Cleaned analysis data: `analysis/data/processed_data/`
- Manuscript figures: `analysis/figures/`
- Statistical summaries: `analysis/stats/`
- Model weights: the `<model_dir>` passed to `model/src/main.py`
- Model simulations: the `<simulation_dir>` passed to `model/src/main.py`

## Expected runtimes

Runtimes are approximate and were tested on a MacBook Pro, 13-inch, M2, 2022, running macOS Monterey 12.4 with 8 GB memory. No non-standard hardware is required.

### Installation

The R analysis code was tested using R 4.3.0. The Python model code was tested using Python 3.12.

Assuming R and Python are already installed, installing the R dependencies typically takes approximately 10 minutes, and installing the Python dependencies typically takes approximately 20 minutes.

To install the R dependencies from the repository root, run:

```r
source("analysis/requirements.R")
```

To install the Python dependencies, run:

```bash
cd model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
```

### R analysis pipeline

The de-identified raw data available on OSF serve as the real example dataset for running the R analysis pipeline. After the raw data have been downloaded from OSF and placed in `analysis/data/raw_data/`, the expected runtimes are:

- Preprocessing: approximately 5 minutes.
- Figure generation: approximately 10 minutes.
- Statistical analyses: approximately 5 minutes.
- Full R analysis pipeline: approximately 20 minutes.

To run the full R analysis pipeline from the repository root, run:

```r
source("analysis/preprocessing/run_all_preprocessing.R")
source("analysis/scripts/figures/run_all_figures.R")
source("analysis/scripts/stats/generate_stats.r")
```


### Python model demo

The following demo is intended only to verify that the model code runs and produces simulation output. It is not intended to reproduce the manuscript model results. The Python model demo does not require an external input dataset; the model generates the simulated task data/output internally.

From the repository root, run:

```bash
mkdir -p outputs/models outputs/simulations
python model/src/main.py 1 1 outputs/models outputs/simulations 10 1 7 train deep_depth
```

Expected output:

- trained model files written to `outputs/models/`;
- simulation CSV files written to `outputs/simulations/`.

Expected runtime for this demo is approximately 5 minutes.

### Full Python model runtime

A full 120-epoch model training run for one parameter setting takes approximately 40 minutes on the test machine described above. Runtime scales with the number of lambda values, alpha values, seeds, tree sizes, and tree orders.
