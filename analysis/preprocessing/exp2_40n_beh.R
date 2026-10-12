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

source(project_path("analysis", "scripts", "figures", "utils.R"))

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
