# ==============================================================================
# Experiment 2 Behavioral Preprocessing
# Author: WM Planning manuscript team
# Purpose: Clean raw behavioral data and retain subjects who exceed chance on
#          recall accuracy and planning choices.
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

df_raw <- read_analysis_csv("data", "raw_data", "exp2_beh.csv")

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df <- df_raw %>%
  convert_true_false_strings() %>%
  dplyr::mutate(
    is_leaf = node %in% c(2, 3, 4, 6, 7, 8, 10, 11, 12),
    condition = ifelse(show_arrow == FALSE, "exposure", "planning"),
    accuracy = ifelse(abs(actual_reward - estimated_reward) < 1, 1, 0),
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward),
    correct_choice = ifelse(abs(V - max_path_reward) < 0.1, 1, 0),
    parent_reward = path_reward - actual_reward
  ) %>%
  dplyr::select(
    subject, condition,  accuracy, correct_choice,
    norm_reward, parent_reward, everything()
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

subject_significance <- df %>%
  dplyr::group_by(subject) %>%
  dplyr::summarize(
    n_trials_acc = sum(!is.na(accuracy)),
    n_correct_acc = sum(accuracy == 1, na.rm = TRUE),
    n_trials_plan = sum(condition == "planning" & !is.na(correct_choice)),
    n_correct_plan = sum(condition == "planning" & correct_choice == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    p_val_acc = ifelse(
      n_trials_acc > 0,
      binom.test(n_correct_acc, n_trials_acc, p = 1 / 9, alternative = "greater")$p.value,
      1
    ),
    p_val_plan = ifelse(
      n_trials_plan > 0,
      binom.test(n_correct_plan, n_trials_plan, p = 1 / 9, alternative = "greater")$p.value,
      1
    )
  ) %>%
  dplyr::ungroup()

kept_subjects_list <- subject_significance %>%
  dplyr::filter(p_val_acc < 0.05 & p_val_plan < 0.05)

print_retention_summary(
  title = "Binomial Filtering Results",
  n_original = nrow(subject_significance),
  n_kept = nrow(kept_subjects_list),
  extra_lines = sprintf("Total unique subjects in raw data:  %s", dplyr::n_distinct(df$subject))
)

df_clean <- df %>%
  dplyr::semi_join(kept_subjects_list, by = "subject") %>%
  dplyr::select(where(~ !is.list(.)))

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_clean, "data", "processed_data", "exp2_beh.csv")
