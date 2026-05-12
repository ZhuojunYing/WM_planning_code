# ==============================================================================
# Experiment 3 Simulation Preprocessing
# Author: WM Planning manuscript team
# Purpose: Batch-load condition-specific simulation files and derive tree,
#          order, reward, and recall metrics.
# ==============================================================================

# ==============================================================================
# Library Imports & Source Files
# ==============================================================================

library(dplyr)
library(here)
library(purrr)
library(readr)
library(stringr)

source(here::here("analysis", "scripts", "utils.R"))

# ==============================================================================
# Data Loading
# ==============================================================================

file_list <- list.files(
  path = analysis_path("data", "raw_data"),
  pattern = "^exp3_model_.*\\.csv$",
  full.names = TRUE
)

if (length(file_list) == 0) {
  stop("No simulation files found in the raw_data directory.")
}

df_raw <- purrr::map_dfr(file_list, ~ {
  condition_name <- stringr::str_replace_all(basename(.x), "exp3_model_|\\.csv", "")

  readr::read_csv(.x, show_col_types = FALSE) %>%
    dplyr::mutate(tree_type = condition_name)
})

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df_clean <- df_raw %>%
  dplyr::mutate(
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward),
    accuracy = ifelse(round(estimated_reward) == round(actual_reward), 1, 0),
    tree = ifelse(tree_type %in% c("deep_depth", "deep_breadth"), "deep", "wide"),
    order = ifelse(tree_type %in% c("deep_depth", "wide_depth"), "depth", "breadth")
  ) %>%
  dplyr::select(
    tree_type, tree, order, accuracy, norm_reward,
    estimated_reward, actual_reward, everything()
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

# Metrics are calculated at the trial level for downstream model comparisons.

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_clean, "data", "processed_data", "exp3_model.csv")


