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

# Participant-only 40-node pilot context model; sourcing runs no analyses.
source(project_path("analysis/scripts/stats/fitted/common.R"))

pilot40_predictors <- c("node_reward_z", "node_reward_squared_z",
  "sibling_reward_z", "sibling_reward_squared_z", "preceding_sum_reward_z", "aunt_reward_z")
pilot40_rhs <- paste0("(", paste(pilot40_predictors, collapse = " + "), ") * condition")

prepare_pilot40 <- function(raw) {
  require_columns(raw, c("subject", "condition", "accuracy", "actual_reward",
    "sibling_reward", "preceding_sum_reward", "aunt_reward", "is_leaf"), "40-node pilot")
  leaf <- tolower(as.character(raw$is_leaf))
  if (anyNA(leaf) || any(!leaf %in% c("true", "false", "1", "0")))
    stop("Invalid or missing is_leaf in pilot data")
  leaf <- leaf %in% c("true", "1")
  condition <- tolower(trimws(as.character(raw$condition)))
  if (anyNA(condition) || !setequal(unique(condition), c("planning", "exposure")))
    stop("Pilot requires planning and exposure only")
  if (anyNA(raw$subject) || any(!nzchar(trimws(as.character(raw$subject)))) ||
      any(!is.na(raw$accuracy) & !raw$accuracy %in% c(0, 1)))
    stop("Invalid participant IDs or nonbinary accuracy")
  result <- data.frame(participant = factor(raw$subject),
    condition = factor(condition, levels = c("planning", "exposure")),
    accuracy = raw$accuracy, node_reward = raw$actual_reward,
    sibling_reward = raw$sibling_reward,
    preceding_sum_reward = ifelse(leaf, raw$preceding_sum_reward, 0),
    aunt_reward = ifelse(leaf, raw$aunt_reward, 0))
  numeric_columns <- setdiff(names(result), c("participant", "condition"))
  if (any(!vapply(result[numeric_columns], is.numeric, logical(1))))
    stop("Pilot accuracy/reward columns must be numeric")
  if (any(vapply(result[numeric_columns], function(x) any(is.infinite(x)), logical(1))))
    stop("Infinite values in pilot data")
  result
}

write_pilot40_descriptives <- function(raw, root) {
  require_columns(raw, c("subject", "condition", "graph", "accuracy", "norm_reward"), "40-node descriptives")
  raw$condition <- tolower(trimws(as.character(raw$condition)))
  if (anyNA(raw$subject) || anyNA(raw$graph)) stop("Missing pilot participant/trial identifiers")
  accuracy <- raw %>% group_by(subject, condition) %>%
    summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop") %>% rename(participant = subject)
  trials <- raw %>% filter(condition == "planning") %>% distinct(subject, graph, norm_reward)
  if (anyDuplicated(trials[c("subject", "graph")])) stop("Inconsistent reward within pilot trial")
  reward <- trials %>% group_by(subject) %>%
    summarise(norm_reward = mean(norm_reward, na.rm = TRUE), .groups = "drop") %>% rename(participant = subject)
  if (any(!is.finite(accuracy$accuracy)) || any(!is.finite(reward$norm_reward))) stop("Missing participant means")
  directory <- file.path(root, "exp2_40n", "participants")
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write.csv(accuracy, file.path(directory, "participant_accuracy.csv"), row.names = FALSE)
  write.csv(reward, file.path(directory, "participant_norm_reward.csv"), row.names = FALSE)
}

pilot40_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  storage <- STORAGE_ROOT
  input <- project_behavior_path("exp2_40n")
  root <- STATS_ROOT
  table_dir <- NULL
  descriptive_only <- FALSE
  stats_only <- FALSE
  while (length(args)) {
    if (args[1] == "--stats-only") { stats_only <- TRUE; args <- args[-1]; next }
    if (args[1] == "--descriptive-only") {
      descriptive_only <- TRUE
      args <- args[-1]
      next
    }
    if (length(args) < 2 || !args[1] %in% c("--input", "--stats-root", "--tables-dir"))
      stop("Options: --input CSV --stats-root DIRECTORY --tables-dir DIRECTORY")
    if (args[1] == "--input") input <- args[2]
    if (args[1] == "--stats-root") root <- args[2]
    if (args[1] == "--tables-dir") table_dir <- args[2]
    args <- args[-c(1, 2)]
  }
  if (is.null(table_dir)) table_dir <- TABLE_DIR
  message("Participant-only pilot input: ", input)
  raw <- read.csv(input)
  data <- prepare_pilot40(raw)
  write_pilot40_descriptives(raw, root)
  if (descriptive_only) {
    message("Saved pilot participant-level descriptives; no model refitted.")
    return(invisible(NULL))
  }
  directory <- file.path(root, "exp2_40n", "participants", "context_joint_conditions")
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write.csv(data.frame(path = normalizePath(input), md5 = unname(tools::md5sum(input)),
    rows = nrow(data), participants = nlevels(data$participant)),
    file.path(directory, "input_provenance.csv"), row.names = FALSE)
  writeLines(c("Participant-only 40-node pilot; no participant-level simulations fitted.",
    "No participant-level performance exclusion or subsampling.",
    "All probes; preceding-sum and aunt predictors set to zero for non-leaves.",
    "Incomplete model rows excluded before pooled planning/exposure standardization.",
    "Independent participant intercept and slopes for all fixed contrasts, including interactions."),
    file.path(directory, "scope.txt"))
  fit_exp3_bam(data, "accuracy", pilot40_rhs, directory,
    threads = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "1")),
    workflow_note = "Participant-only 40-node pilot; manuscript table uses these BAM coefficients.")
  if (stats_only) return(invisible(NULL))
  source(project_path("analysis/scripts/stats/fitted/export_tables.R"))
  export_fitted_tables(root, table_dir, table_set = "pilot_40n")
}

if (sys.nframe() == 0L) pilot40_main()
