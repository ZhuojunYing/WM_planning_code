# Working Memory Planning Raw Data

This OSF repository contains the raw behavioral datasets and model-output
datasets for the working-memory planning manuscript.

The analysis code is not included in this OSF data deposit. To reproduce the
processed data, figures, and statistical analyses, download the accompanying
code repository separately and place these CSV files in the code repository at:

```text
analysis/data/raw_data/
```

## File Inventory

### Behavioral Data

- `exp1_beh.csv`: Experiment 1 participant data.
- `exp2_beh.csv`: Experiment 2 participant data.
- `exp2_40n_beh.csv`: 40-node participant data.
- `exp3_beh.csv`: Experiment 3 participant data.

### Model Outputs

- `exp1_model.csv`: Experiment 1 model simulations for the planning condition.
- `exp1_model_exp.csv`: Experiment 1 model simulations for the exposure condition.
- `exp1_model_preregistered.csv`: Experiment 1 preregistered model simulations.
- `exp2_model.csv`: Experiment 2 model simulations for the planning condition.
- `exp2_model_exp.csv`: Experiment 2 model simulations for the exposure condition.
- `exp3_model_deep_breadth.csv`: Experiment 3 model simulations, deep tree with breadth-first order.
- `exp3_model_deep_depth.csv`: Experiment 3 model simulations, deep tree with depth-first order.
- `exp3_model_wide_breadth.csv`: Experiment 3 model simulations, wide tree with breadth-first order.
- `exp3_model_wide_depth.csv`: Experiment 3 model simulations, wide tree with depth-first order.

### Supplementary Model Files

- `rd_2n_model.csv`: Rate-distortion summary for the 2-node model.
- `rd_6n_model.csv`: Rate-distortion summary for the 6-node model.

## Reproducing Processed Data

After downloading the separate analysis code repository and placing these files
in `analysis/data/raw_data/`, run the preprocessing pipeline from the code
repository root:

```r
source("analysis/preprocessing/run_all_preprocessing.R")
```

This runs the preprocessing scripts in `analysis/preprocessing/` and creates the
processed CSV files used by the figure and statistics scripts.
