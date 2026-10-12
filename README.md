# Working Memory Planning

Code for *Resource Rational Encoding of Reward Information During Planning*.
Run the commands below from the repository root.

## Workflows

The Python scripts train models, fit participant parameters, generate fitted
simulations, and perform parameter recovery. The R scripts preprocess behavior,
fit statistical analyses, and export tables, in-text statistics, and figures.

To reproduce results, download the inputs, preprocess behavior, obtain matching
fit/recovery outputs, then generate statistics before plots. Fit/recovery
outputs can be downloaded or generated; generating them requires matching
trained weights. Bash/Slurm wrappers are not required for the Python modules.

## Storage

Configure locations in `project_paths.conf`. `ProjectRoot: .` uses this
repository's directory; an absolute path or the `WM_PLANNING_ROOT` environment
variable can also be used. `StorageRoot` controls generated model artifacts.
The analysis runners use the current R installation and `python3`; set
`WM_PLANNING_PYTHON` to select a different Python executable.

## Data download

Download the data from [OSF](https://osf.io/38gqz) and extract into:

| Input | Location |
| --- | --- |
| Raw behavior | `analysis/data/raw_beh_data/exp1_beh.csv`, `exp2_beh.csv`, `exp3_beh.csv`, `exp2_40n_beh.csv` |
| Fitted simulations | `analysis/data/model_simulations/exp{n}_{model}.csv` |
| Selected parameter fits | `analysis/data/model_simulations/exp{n}_{model}_parameters.csv` |
| Recovery results and configuration | `analysis/data/model_simulations/recovery_results/exp{n}_full_recovery.csv` and `.json` |

For Exp1 and Exp2, model names are `full`, `exposure`, `simultaneous`, and
`independent`; Exp3 uses `full`. `simulation_inputs.csv` records the matching
model configurations. `ParameterFits` and `RecoveryInputs` in
`project_paths.conf` control the fit and recovery locations. Keep participant
IDs consistent across behavior, simulations, and selected fits. 

## Python and R environments

Install Python and R dependencies in your chosen environments:

```bash
python -m pip install -r requirements.txt
Rscript --vanilla analysis/requirements.R
Rscript --vanilla -e 'install.packages(c("mgcv", "jsonlite"), repos="https://cloud.r-project.org")'
```
Assuming R and Python are already installed, installing the R dependencies typically takes approximately 10 minutes, and installing the Python dependencies typically takes approximately 20 minutes.


## Python model usage

### Train models

The training interface is:

```text
python -m MODULE LAMBDA_VALUES ALPHA_VALUES WEIGHT_DIR SIMULATION_DIR \
  EPOCHS SEED TREE_SIZE train TREE_TYPE [SIMULATION_TRIALS] [options]
```

`default` selects the configured directories. Training takes comma-separated
lambda and **alpha** lists and evaluates their Cartesian product. For a
lambda/rho grid, train each pair using `alpha = lambda * rho`.

Example: one full-model planning parameter pair for a 7-node tree:

```bash
python -m model.src.main_vae.main \
  '0.01' '0.001' default default 320 1 7 train standard 2000 \
  --nonlinear-scale --rnn-units 128 --reconstruction-hidden-units 128 \
  --reconstruction-loss-scope final_timestep --max-action-gradient-scale 1 \
  --final-learning-rate 0.0003 \
  --late-lr-plateau-start-epoch 200 --late-lr-plateau-end-epoch 300
```

Use tree size `13` for Exp2. For Exp3 use `31`, train each of `deep_breadth`,
`deep_depth`, `wide_breadth`, and `wide_depth`, and add
`--action-entropy-bonus 0.05` for the released configuration. Repeat for all
parameter cells and seeds needed by fitting.

Variant training modules are
`model.src.alternative_models.simultaneous_channel.main` and
`model.src.alternative_models.independent_channel.main`. The independent model
uses `--rnn-units 8 --latent-dim 1 --channel-kl-aggregation sum
--channel-kl-time-weighting remaining_steps`. Full-model exposure training uses
alpha `1` and `--condition exposure --no-critic-loss`.

Training generates task rewards internally, saves weights, and runs grid
simulations afterward. Architecture, loss, and schedule settings must match
those used for fitting. See `model/src/main_vae/config.py` for training options.


### Fit participants and generate fitted simulations

Fitters select parameters from trained grid cells; they do not train missing
weights. Example matching the single pair above:

```bash
python -m model.src.main_vae.fit_participant_parameters \
  7 '0.01' '0.1' --rho --seed-values '1' \
  --behavior-csv analysis/data/processed_data/exp1_beh_new.csv \
  --condition planning --mc-samples 1 --batch-size 128 \
  --reward-nll-scale 0.5 --model-epochs 320 \
  --nonlinear-scale --rnn-units 128 --reconstruction-hidden-units 128 \
  --reconstruction-loss-scope final_timestep --max-action-gradient-scale 1 \
  --final-learning-rate 0.0003 \
  --late-lr-plateau-start-epoch 200 --late-lr-plateau-end-epoch 300 \
  --simulation-trials-per-seed 100 --simulation-random-seed 2027 \
  --skip-fitted-plots
```

With `--rho`, the third positional argument is alpha/lambda; otherwise it is
alpha. Use tree `13` and `exp2_beh_new.csv` for Exp2. Expand the grids and seed
list after training the corresponding weights. For exposure fits, omit
`--rho`, pass alpha `1`, and use `--condition exposure --no-critic-loss
--reward-nll-scale 1`. The 7-/13-node fitter saves parameter fits and fitted
simulations. `--skip-fitted-plots` leaves figure generation to the R workflow.

Use the corresponding variant fitting modules for alternative models:

```bash
python -m model.src.alternative_models.simultaneous_channel.fit_participant_parameters --help
python -m model.src.alternative_models.independent_channel.fit_participant_parameters --help
```

For Exp3, use `model.src.main_vae.fit_participant_parameters_31` with positional
arguments `31 LAMBDA_VALUES ALPHA_OR_RHO_VALUES`, `--rho` when appropriate,
`--behavior-csv analysis/data/processed_data/exp3_beh.csv`, and matching trained
model options. Then simulate the completed selected-fit CSV:

```bash
python -m model.src.main_vae.simulate_fitted_participants_31 \
  --fit-csv /path/to/participant_parameter_fits_31_CONFIG.csv \
  --simulation-seeds '1,2,3,4,5' --trials-per-seed-tree 25 \
  --batch-size 128 --simulation-random-seed 2027
```

For parameter recovery, `model.src.main_vae.parameter_recovery` generates
synthetic responses with known parameters and refits them. Inspect its options:

```bash
python -m model.src.main_vae.fit_participant_parameters --help
python -m model.src.main_vae.fit_participant_parameters_31 --help
python -m model.src.main_vae.parameter_recovery --help
```

## R setup and analysis

Process the raw behavior, fit the exposure-memory mixture, then generate
statistics and plots:

```bash
Rscript --vanilla analysis/preprocessing/run_all_preprocessing.R

for experiment in exp1 exp2; do
  python -m model.src.mixture_model.fit_exposure_memory_mixture "$experiment"
done

Rscript --vanilla analysis/scripts/stats/generate_tables.r
Rscript --vanilla analysis/scripts/figures/generate_all_plots.r
```

Preprocessing runs only the four behavioral scripts and creates
`exp1_beh_new.csv`, `exp2_beh_new.csv`, `exp3_beh.csv`, and `exp2_40n_beh.csv`
in `analysis/data/processed_data/`. Final fitted simulations are read directly.
The mixture fitter and statistics runner use
`analysis/data/model_simulations/exp1_exposure_memory_mixture.csv` and
`exp2_exposure_memory_mixture.csv` (the configured `ParameterFits` folder).


`generate_tables.r` first fits all selected analyses, then exports tables and
in-text statistics. 

Outputs are organized under `analysis/results/`:

- `stats/fitted/` and `tests/`: saved statistical results.
- `tables/` and `intext_stats/`: LaTeX tables and text fragments.
- `figures/`: experiment-specific plot folders.
- `logs/`: per-step output and error logs.

To export existing results without refitting:

```bash
Rscript --vanilla analysis/scripts/stats/fitted/export_tables.R \
  --table-set all --experiments exp1,exp2,exp3
python3 analysis/scripts/stats/fitted/export_intext_stats.py \
  --experiments exp1,exp2,exp3
```

Processed CSVs need not be downloaded. Selected parameter fits and recovery
outputs can be downloaded **or generated** using the model scripts above.

## Approximate runtimes

| Task | Reported runtime |
| --- | --- |
| Behavioral preprocessing | 5 minutes |
| Figures | 30 minutes |
| Statistics excluding the current `glmer` analyses | 20 minutes |
| `glmer` analyses | Approximately 5 hours per experiment |
| Model training | 100 minutes per 320-epoch model setting (original report) |

These are approximate reports, not newly measured benchmarks. Runtime depends on hardware, dataset size, parameter-grid
size, seeds, tree sizes, and tree orders.
