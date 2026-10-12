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

source(project_path("analysis", "scripts", "figures", "utils.R"))

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
