# Shared path configuration; resolution only, no directory creation or analysis.
.path_frames <- vapply(sys.frames(), function(f) if (is.null(f$ofile)) "" else as.character(f$ofile), character(1))
.path_starts <- c(Sys.getenv("WM_PLANNING_ROOT", ""), dirname(rev(.path_frames[nzchar(.path_frames)])), getwd())
.path_find_root <- function(starts) {
  for (start in starts[nzchar(starts)]) {
    candidate <- normalizePath(path.expand(start), mustWork = FALSE)
    repeat {
      if (file.exists(file.path(candidate, "project_paths.conf"))) return(candidate)
      parent <- dirname(candidate)
      if (identical(parent, candidate)) break
      candidate <- parent
    }
  }
  stop("Cannot find project_paths.conf; set WM_PLANNING_ROOT to the absolute public repository path.")
}
.path_config_root <- .path_find_root(.path_starts)
.path_config <- as.list(read.dcf(file.path(.path_config_root, "project_paths.conf"))[1, ])
.path_absolute <- function(path, base) {
  path <- path.expand(path)
  if (!grepl("^(/|[A-Za-z]:[/\\\\])", path)) path <- file.path(base, path)
  normalizePath(path, mustWork = FALSE)
}
REPO_ROOT <- .path_absolute(Sys.getenv("WM_PLANNING_ROOT", .path_config$ProjectRoot), .path_config_root)
project_path <- function(...) file.path(REPO_ROOT, ...)
.path_setting <- function(key) .path_absolute(.path_config[[key]], REPO_ROOT)
get_storage_root <- function() .path_absolute(
  Sys.getenv("WM_PLANNING_STORAGE_ROOT", Sys.getenv("WM_PLANNING_COMPARISON_STORAGE_ROOT",
    .path_config$StorageRoot)), REPO_ROOT)
STORAGE_ROOT <- get_storage_root()
MODEL_DIR <- file.path(STORAGE_ROOT, "models")
WEIGHT_DIR <- file.path(MODEL_DIR, "weights")
CHECKPOINT_DIR <- file.path(MODEL_DIR, "checkpoints")
DATA_DIR <- file.path(STORAGE_ROOT, "data")
SIMULATION_DIR <- file.path(STORAGE_ROOT, "simulations")
PLOT_DIR <- file.path(STORAGE_ROOT, "plots")
LOG_DIR <- file.path(STORAGE_ROOT, "logs")
TRAINING_LOG_DIR <- file.path(LOG_DIR, "training")
ANALYSIS_LOG_DIR <- file.path(.path_setting("AnalysisResults"), "logs")
SLURM_LOG_DIR <- file.path(LOG_DIR, "slurm")
SUMMARY_DIR <- file.path(STORAGE_ROOT, "summaries")
RESULT_DIR <- file.path(STORAGE_ROOT, "results")
CACHE_DIR <- file.path(STORAGE_ROOT, "caches")
TEMP_DIR <- file.path(STORAGE_ROOT, "tmp")
ANALYSIS_DATA_DIR <- .path_setting("AnalysisData")
RAW_DATA_DIR <- .path_setting("RawData")
PROCESSED_DATA_DIR <- .path_setting("ProcessedData")
SIMULATION_INPUT_DIR <- .path_setting("SimulationInputs")
PARAMETER_FIT_DIR <- .path_setting("ParameterFits")
RECOVERY_INPUT_DIR <- .path_setting("RecoveryInputs")
ANALYSIS_PLOT_DIR <- .path_setting("Figures")
ANALYSIS_SUMMARY_DIR <- .path_setting("AnalysisResults")
STATS_ROOT <- file.path(ANALYSIS_SUMMARY_DIR, "stats", "fitted")
ANALYSIS_RESULTS_DIR <- ANALYSIS_SUMMARY_DIR
TABLE_DIR <- .path_setting("Tables")
INTEXT_DIR <- .path_setting("Intext")
LEGACY_STATS_DIR <- .path_setting("LegacyStats")

# Shared orchestration only: the runners execute each analysis in a fresh process.
analysis_rscript <- function() file.path(R.home("bin"), "Rscript")
analysis_python <- function() Sys.getenv("WM_PLANNING_PYTHON", "python3")
analysis_selected_fit <- function(simulation, root = STORAGE_ROOT) {
  canonical <- canonical_simulation_basename(simulation)
  index <- match(canonical, .simulation_inputs$canonical_file)
  if (identical(normalizePath(root, mustWork = FALSE), STORAGE_ROOT) && !is.na(index))
    return(file.path(PARAMETER_FIT_DIR,
      sub("[.]csv$", "_parameters.csv", .simulation_inputs$public_file[index])))
  name <- sub("^participant_fitted_simulations_", "participant_parameter_fits_",
    canonical_simulation_basename(simulation))
  name <- sub("_n[0-9]+(_per_seed_tree)?_simseed[0-9]+(_new)?\\.csv$", "\\2.csv", name)
  file.path(root, "summaries", "participant_fits", name)
}
analysis_require_files <- function(paths) {
  missing <- unique(paths[!file.exists(paths)])
  if (length(missing)) stop("Missing inputs (no analyses started):\n", paste(missing, collapse = "\n"))
}
analysis_run_steps <- function(steps, dry_run = FALSE) {
  for (step in steps) {
    message("\n", step$name, "\n", paste(shQuote(c(step$command, step$args)), collapse = " "))
    if (dry_run) next
    log_dir <- file.path(ANALYSIS_RESULTS_DIR, "logs")
    dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
    out <- file.path(log_dir, paste0(step$name, ".out"))
    err <- file.path(log_dir, paste0(step$name, ".err"))
    message("Logs: ", out, " and ", err)
    status <- system2(step$command, shQuote(step$args), stdout = out, stderr = err)
    if (status != 0L) {
      if (file.exists(err)) cat(tail(readLines(err, warn = FALSE), 40), sep = "\n")
      stop(step$name, " failed (exit ", status, "); subsequent steps were not run.")
    }
  }
  invisible(NULL)
}
project_behavior_path <- function(experiment) file.path(PROCESSED_DATA_DIR,
  paste0(experiment, if (experiment %in% c("exp1", "exp2")) "_beh_new.csv" else "_beh.csv"))

.simulation_inputs <- read.csv(project_path("simulation_inputs.csv"), stringsAsFactors = FALSE)
canonical_simulation_basename <- function(path) {
  index <- match(basename(path), .simulation_inputs$public_file)
  if (is.na(index)) basename(path) else .simulation_inputs$canonical_file[index]
}
simulation_input_path <- function(filename, root = STORAGE_ROOT) {
  index <- match(basename(filename), .simulation_inputs$canonical_file)
  if (is.na(index)) index <- match(basename(filename), .simulation_inputs$public_file)
  if (!is.na(index) && identical(normalizePath(root, mustWork = FALSE), STORAGE_ROOT))
    return(file.path(SIMULATION_INPUT_DIR, .simulation_inputs$public_file[index]))
  file.path(root, "simulations", "participant_fits", filename)
}

ensure_directory <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(path)
}

ensure_parent_directory <- function(path) {
  ensure_directory(dirname(path))
  invisible(path)
}

require_existing_path <- function(path, description = "Required path") {
  if (!file.exists(path) && !dir.exists(path)) {
    stop(sprintf("%s does not exist: %s", description, path))
  }
  path
}

format_local_scp_command <- function(
    remote_paths,
    local_directory,
    recursive = FALSE,
    remote_host = "dtn") {
  paths <- as.character(remote_paths)
  if (length(paths) == 0L || any(!nzchar(paths))) {
    stop("At least one remote path is required.", call. = FALSE)
  }
  local_directory <- sub("/+$", "", as.character(local_directory))
  if (!nzchar(local_directory)) {
    stop("The local destination directory cannot be empty.", call. = FALSE)
  }

  flags <- if (isTRUE(recursive)) "-pr" else "-p"
  remote_arguments <- vapply(
    paths,
    function(path) shQuote(paste0(remote_host, ":", path)),
    character(1)
  )
  local_argument <- shQuote(paste0(local_directory, "/"))
  scp_command <- if (length(remote_arguments) == 1L) {
    sprintf(
      "scp %s %s %s",
      flags,
      remote_arguments[[1]],
      local_argument
    )
  } else {
    paste(
      c(
        sprintf("scp %s \\", flags),
        paste0("  ", remote_arguments, " \\"),
        paste0("  ", local_argument)
      ),
      collapse = "\n"
    )
  }
  paste(
    sprintf("mkdir -p %s", shQuote(local_directory)),
    scp_command,
    sep = "\n"
  )
}

print_local_scp_command <- function(
    remote_paths,
    local_directory,
    recursive = FALSE,
    remote_host = "dtn") {
  cat("\nRun this on your local computer from the repository root:\n")
  cat(
    format_local_scp_command(
      remote_paths,
      local_directory,
      recursive = recursive,
      remote_host = remote_host
    ),
    "\n",
    sep = ""
  )
  invisible(NULL)
}
