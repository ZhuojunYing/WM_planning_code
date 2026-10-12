# Resolve the shared path configuration before loading project code.
local({
  starts <- c(Sys.getenv("WM_PLANNING_ROOT", ""),
    dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))),
    dirname(vapply(sys.frames(), function(f) if (is.null(f$ofile)) "" else as.character(f$ofile), character(1))),
    getwd())
  found <- NULL
  for (start in starts[nzchar(starts)]) {
    directory <- normalizePath(path.expand(start), mustWork = FALSE)
    repeat {
      if (file.exists(file.path(directory, "project_paths.conf"))) {
        found <- directory
        break
      }
      parent <- dirname(directory)
      if (identical(parent, directory)) break
      directory <- parent
    }
    if (!is.null(found)) break
  }
  if (is.null(found)) stop("Set WM_PLANNING_ROOT to the absolute repository folder containing project_paths.conf.")
  source(file.path(found, "analysis/scripts/project_paths.R"), local = FALSE)
})

# Generate all final statistics, then tables, then in-text fragments.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 2L && args[1] == "--supplemental") {
  experiment <- match.arg(args[2], c("exp1", "exp2"))
  source(project_path("analysis/scripts/stats/fitted/common.R"))
  behavior <- project_behavior_path(experiment)
  mixture <- file.path(PARAMETER_FIT_DIR, paste0(experiment, "_exposure_memory_mixture.csv"))
  planning <- analysis_selected_fit(simulation_input_path(paste0(experiment, "_full.csv")))
  run_exposure_associations(experiment, behavior, mixture,
    file.path(ANALYSIS_RESULTS_DIR, "tests/exposure_memory_performance", experiment))
  run_lambda_associations(planning, mixture,
    file.path(ANALYSIS_RESULTS_DIR, "tests/exposure_memory_lambda", experiment))
  quadratic_node_performance(behavior,
    file.path(ANALYSIS_RESULTS_DIR, "tests/quadratic_node_reward_performance", experiment, "participants"))
  quit(status = 0L)
}
if (length(args) && !identical(args, "--dry-run")) stop("Usage: Rscript generate_tables.r [--dry-run]")
dry_run <- identical(args, "--dry-run")
Sys.setenv(WM_PLANNING_ROOT = REPO_ROOT)
setwd(REPO_ROOT)
rscript <- analysis_rscript()
fitted <- project_path("analysis/scripts/stats/fitted")
steps <- list()
add <- function(name, script, args = character(), command = rscript) {
  steps[[length(steps) + 1L]] <<- list(name = name, command = command,
    args = c(if (identical(command, rscript)) "--vanilla", script, args))
}
for (experiment in c("exp1", "exp2", "exp3")) {
  add(paste0("stats_", experiment), file.path(fitted, paste0(experiment, ".R")),
    c("--output-dir", STATS_ROOT))
}
add("stats_exp2_40n", file.path(fitted, "exp2_40n.R"),
  c("--stats-root", STATS_ROOT, "--stats-only"))
add("stats_cross_experiment", file.path(fitted, "common.R"),
  c("--cross-experiment", "--source", "participants,sequential", "--output-dir", STATS_ROOT))
for (experiment in c("exp1", "exp2")) {
  add(paste0("stats_supplemental_", experiment), project_path("analysis/scripts/stats/generate_tables.r"),
    c("--supplemental", experiment))
}
add("export_tables", file.path(fitted, "export_tables.R"),
  c("--table-set", "all", "--experiments", "exp1,exp2,exp3", "--stats-root", STATS_ROOT,
    "--output-dir", TABLE_DIR))
add("export_intext_stats", file.path(fitted, "export_intext_stats.py"),
  c("--experiments", "exp1,exp2,exp3", "--stats-root", STATS_ROOT,
    "--tables-dir", TABLE_DIR, "--output-dir", INTEXT_DIR, "--legacy-stats-dir", LEGACY_STATS_DIR),
  command = analysis_python())
if (!dry_run) {
  simulations <- file.path(SIMULATION_INPUT_DIR, .simulation_inputs$public_file)
  analysis_require_files(c(vapply(c("exp1", "exp2", "exp3", "exp2_40n"), project_behavior_path, character(1)),
    simulations, vapply(simulations, analysis_selected_fit, character(1)),
    file.path(PARAMETER_FIT_DIR, paste0(c("exp1", "exp2"), "_exposure_memory_mixture.csv"))))
}
analysis_run_steps(steps, dry_run)
message(if (dry_run) "Dry run only; no statistics generated." else "Completed statistics, tables and in-text exports under ",
  if (!dry_run) ANALYSIS_RESULTS_DIR)
