# ==============================================================================
# Experiment 2 40-Node Behavioral Preprocessing
# Author: WM Planning manuscript team
# Purpose: Clean raw behavioral data and calculate recall, choice, and reward
#          metrics for the 40-node task.
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

df_raw <- read_analysis_csv("data", "raw_data", "exp2_40n_beh.csv")

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df_clean <- df_raw %>%
  convert_true_false_strings() %>%
  dplyr::mutate(
    condition = ifelse(show_arrow == FALSE, "exposure", "planning"),
    accuracy = ifelse(estimated_reward == actual_reward, 1, 0),
    preceding_sum_reward = ifelse(is_leaf == TRUE, path_reward - actual_reward, 0),
    aunt_reward = ifelse(is_leaf == TRUE, aunt_reward, 0),
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward),
    correct_choice = ifelse(abs(V - max_path_reward) < 0.1, 1, 0)
  ) %>%
  dplyr::select(
    condition,
    accuracy,
    correct_choice,
    norm_reward,
    preceding_sum_reward,
    everything()
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

# No subject-level exclusion is applied for this dataset.

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_clean, "data", "processed_data", "exp2_40n_beh.csv")
