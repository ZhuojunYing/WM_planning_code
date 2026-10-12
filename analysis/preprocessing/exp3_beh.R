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

# ==============================================================================
# Experiment 3 Behavioral Preprocessing
# Author: WM Planning manuscript team
# Purpose: Clean tree-search behavioral data, derive task structure variables,
#          and remove the bottom 5% of subjects on recall and choice metrics.
# ==============================================================================

# ==============================================================================
# Library Imports & Source Files
# ==============================================================================

library(dplyr)
library(here)
library(readr)

source(project_path("analysis", "scripts", "figures", "utils.R"))

# ==============================================================================
# Data Loading
# ==============================================================================

df_raw <- read_analysis_csv("data", "raw_data", "exp3_beh.csv")

reward_columns <- paste0("n", seq_len(30), "_reward")
required_columns <- c(
  "subject", "graph", "tree_type", "show_arrow", "est_best_path",
  "node", "actual_reward", "estimated_reward", "V",
  "min_path_reward", "max_path_reward", reward_columns
)
missing_columns <- setdiff(required_columns, names(df_raw))

if (length(missing_columns) > 0L) {
  stop(
    "Experiment 3 raw data are missing required columns: ",
    paste(missing_columns, collapse = ", "),
    call. = FALSE
  )
}

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df <- df_raw %>%
  convert_boolean_like_columns() %>%
  # The updated export has one action row followed by two recall rows per
  # participant trial. The full reward vector and chosen path are repeated on
  # the recall rows, so retaining only those rows preserves the historical
  # two-row-per-trial processed-data structure without losing fitter inputs.
  dplyr::filter(
    show_arrow == TRUE,
    !is.na(node),
    !is.na(actual_reward),
    !is.na(estimated_reward)
  ) %>%
  dplyr::mutate(
    condition = "planning",
    est_best_path = trimws(as.character(est_best_path)),
    dplyr::across(
      dplyr::all_of(reward_columns),
      ~ suppressWarnings(as.numeric(.x))
    ),
    accuracy = ifelse(estimated_reward == actual_reward, 1, 0),
    correct_choice = ifelse((max_path_reward - V) < 0.1, 1, 0),
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward),
    tree = ifelse(tree_type %in% c("deep_depth", "deep_breadth"), "deep", "wide"),
    order = ifelse(tree_type %in% c("deep_depth", "wide_depth"), "depth", "breadth")
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

subject_performance <- df %>%
  dplyr::group_by(subject) %>%
  dplyr::summarize(
    mean_accuracy = mean(accuracy, na.rm = TRUE),
    mean_accuracy_wide = mean(accuracy[tree == "wide"], na.rm = TRUE),
    mean_accuracy_deep = mean(accuracy[tree == "deep"], na.rm = TRUE),
    mean_choice = mean(correct_choice, na.rm = TRUE),
    mean_choice_wide = mean(correct_choice[tree == "wide"], na.rm = TRUE),
    mean_choice_deep = mean(correct_choice[tree == "deep"], na.rm = TRUE),
    .groups = "drop"
  )

cutoff_percentile <- 0.05
accuracy_cutoff <- stats::quantile(
  subject_performance$mean_accuracy,
  cutoff_percentile,
  na.rm = TRUE
)
choice_cutoff <- stats::quantile(
  subject_performance$mean_choice,
  cutoff_percentile,
  na.rm = TRUE
)

cat("\n--- Performance Cutoffs (Dropping worst 5%) ---\n")
cat("Min Accuracy threshold (Keep >=): ", round(accuracy_cutoff, 3), "\n")
cat("Min Choice threshold (Keep >=): ", round(choice_cutoff, 3), "\n")

kept_subjects_list <- subject_performance %>%
  dplyr::filter(mean_accuracy >= accuracy_cutoff & mean_choice >= choice_cutoff)

print_retention_summary(
  title = "Percentile Filtering Results",
  n_original = nrow(subject_performance),
  n_kept = nrow(kept_subjects_list)
)

df_filtered <- df %>%
  dplyr::semi_join(kept_subjects_list, by = "subject") %>%
  dplyr::select(
    subject, graph, condition, tree, order, tree_type, est_best_path,
    dplyr::all_of(reward_columns), accuracy, correct_choice, norm_reward,
    everything()
  )

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_filtered, "data", "processed_data", "exp3_beh.csv")
