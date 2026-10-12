#!/usr/bin/env Rscript
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

source(project_path("analysis/scripts/stats/fitted/common.R"))

# Keep the regular input paths, source selection, and exports; use all trials/nodes.
# Accuracy uses binomial-logit BAM; continuous outcomes use Gaussian BAM.
# Shared estimators are defined in common.R; no benchmark scripts are sourced.
# Preserve the response checks previously inherited through the benchmark chain.
exp3_response_family <- function(data, response) {
  if (response == "accuracy") {
    if (any(!is.finite(data[[response]]) | !data[[response]] %in% c(0, 1)))
      fail("Logistic accuracy requires binary 0/1 observations")
    return(binomial(link = "logit"))
  }
  if (response != "norm_reward") fail("Unsupported Exp3 response: %s", response)
  gaussian()
}

add_exp3_odds_ratios <- function(table, family) {
  if (family$family == "binomial") table <- mutate(table,
    odds_ratio = exp(beta), odds_ratio_ci_low = exp(ci_low), odds_ratio_ci_high = exp(ci_high))
  table
}

run_exp3_stats_bam <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (!requireNamespace("mgcv", quietly = TRUE)) fail("Install mgcv in the Torch R environment")
  threads <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1"))
  if (is.na(threads) || threads < 1L) fail("SLURM_CPUS_PER_TASK must be a positive integer")
  runner <- function(data, response, rhs, directory, focal = NULL) {
    random_rhs <- rhs
    if (basename(directory) == "rank_change") {
      data$tree_c <- ifelse(data$tree == "wide", .5, -.5)
      data$order_c <- ifelse(data$order == "DFS", .5, -.5)
      data$final_rank_better_than_encoding_rank <- as.numeric(data$final_rank < data$encoding_rank)
      rhs <- "final_rank_better_than_encoding_rank * tree_c * order_c + final_rank * tree_c * order_c"
      random_rhs <- rhs
      focal <- colnames(model.matrix(as.formula(paste("~", rhs)), data))
    }
    fit_exp3_bam(data, response, rhs, directory, focal, threads, random_rhs = random_rhs,
      sampling_note = "Regular Exp3 stats: Sampling NONE for participants and simulations; all trials and nodes before analysis-specific exclusions.",
      workflow_note = "Regular Exp3 manuscript statistics: binomial-logit BAM for binary accuracy and Gaussian BAM for normalized reward, discrete=TRUE.")
  }
  dispatch <- run_source_stats
  environment(dispatch) <- list2env(list(run_stats_model = runner), parent = environment(run_source_stats))
  run <- run_experiment_stats
  environment(run) <- list2env(list(run_source_stats = dispatch), parent = environment(run_experiment_stats))
  run("exp3", args)
}

if (sys.nframe() == 0L) run_exp3_stats_bam()
