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

# Run the selected final plots in isolated R processes.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) && !identical(args, "--dry-run")) stop("Usage: Rscript generate_all_plots.r [--dry-run]")
dry_run <- identical(args, "--dry-run")
Sys.setenv(WM_PLANNING_ROOT = REPO_ROOT)
setwd(REPO_ROOT)
figures <- project_path("analysis/scripts/figures")
steps <- list()
add <- function(experiment, args = character()) {
  script <- if (experiment == "exp3") "exp3.R" else paste0(experiment, ".r")
  steps[[length(steps) + 1L]] <<- list(name = paste0("plots_", experiment),
    command = analysis_rscript(), args = c("--vanilla", file.path(figures, script), args))
}
add("exp1")
add("exp2")
simulation <- simulation_input_path("exp3_full.csv")
add("exp3", c(analysis_selected_fit(simulation), project_behavior_path("exp3"), "1,2,3,4,5", simulation))
add("exp2_40n")
if (!dry_run) {
  simulations <- file.path(SIMULATION_INPUT_DIR, .simulation_inputs$public_file)
  # Coefficient plots and the quadratic plot consume statistics from generate_tables.r.
  context <- unlist(lapply(c("exp1", "exp2"), function(e) unlist(lapply(
    c("fixed_effects.rds", "all_coefficients.csv"), function(f) file.path(STATS_ROOT, e,
      c("participants", "sequential"), "context_joint_conditions", f)))))
  quadratic <- file.path(ANALYSIS_RESULTS_DIR, "tests/quadratic_node_reward_performance",
    c("exp1", "exp2"), "participants", "participant_coefficients_and_performance.csv")
  recovery <- file.path(RECOVERY_INPUT_DIR, c(
    paste0(c("exp1", "exp2", "exp3"), "_full_recovery.csv"),
    paste0(c("exp1", "exp2", "exp3"), "_full_recovery.json")))
  analysis_require_files(c(vapply(c("exp1", "exp2", "exp3", "exp2_40n"), project_behavior_path, character(1)),
    simulations, vapply(simulations, analysis_selected_fit, character(1)), context, quadratic, recovery))
}
analysis_run_steps(steps, dry_run)
message(if (dry_run) "Dry run only; no plots generated." else "Plots written under ",
  if (!dry_run) ANALYSIS_PLOT_DIR)
