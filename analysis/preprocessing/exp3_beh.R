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

source(here::here("analysis", "scripts", "utils.R"))

# ==============================================================================
# Data Loading
# ==============================================================================

df_raw <- read_analysis_csv("data", "raw_data", "exp3_beh.csv")

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df <- df_raw %>%
  convert_boolean_like_columns() %>%
  dplyr::filter(show_arrow == TRUE) %>%
  dplyr::mutate(
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
    subject, tree, order, tree_type, accuracy, correct_choice,
    norm_reward, everything()
  )

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_filtered, "data", "processed_data", "exp3_beh.csv")
