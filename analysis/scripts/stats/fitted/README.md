# Statistical analyses

Run from the repository root after behavioral preprocessing and preparing the
model inputs described in the [main README](../../../../README.md).

```bash
Rscript --vanilla analysis/scripts/stats/generate_tables.r
```

This runs the selected analyses before exporting all tables and in-text
statistics. Append `--dry-run` to preview the commands. Outputs are under
`analysis/results/stats/fitted/`, `analysis/results/tests/`,
`analysis/results/tables/`, and `analysis/results/intext_stats/`; logs are in
`analysis/results/logs/`.

For individual experiments, use `exp1.R`, `exp2.R`, `exp3.R`, or `exp2_40n.R`
in this directory. They share `common.R` and the central `project_paths.conf`.
Selected parameter fits are read from
`analysis/data/model_simulations/exp{n}_{model}_parameters.csv` by default.
Recovery inputs are used by the plotting scripts, not refitted by this workflow.

To export completed statistics without refitting:

```bash
Rscript --vanilla analysis/scripts/stats/fitted/export_tables.R \
  --table-set all --experiments exp1,exp2,exp3
python3 analysis/scripts/stats/fitted/export_intext_stats.py \
  --experiments exp1,exp2,exp3
```
