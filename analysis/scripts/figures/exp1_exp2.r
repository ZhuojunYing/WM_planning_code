# ==============================================================================
# Script: Plotting and Model Fitting (Experiment 1 & 2)
# Description: Generates binned behavioral vs model plots (bars/shaded lines),
#              fits mixed-effects models, and plots coefficient comparisons.
# ==============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(grid)
library(rlang)
library(lmerTest) # Used for generating p-values in mixed models
library(here)     # For safe, relative pathing

source(here::here("analysis", "scripts", "utils.R"))
source(here::here("analysis", "scripts", "plot_utils.R"))

# ==============================================================================
# Execution Wrapper
# ==============================================================================

# Function to process all plots for a specific version/experiment
process_version <- function(version) {
  
  # Dynamic Data Loading via here()
  beh_file <- analysis_path("data", "processed_data", paste0(version, "_beh.csv"))
  model_file <- analysis_path("data", "processed_data", paste0(version, "_model.csv"))
  
  df_beh <- read.csv(beh_file)
  df_model <- read.csv(model_file)
  
  # FIX APPLIED: Standardize 'subject' to character to prevent bind_rows() type mismatch
  df_beh$subject <- as.character(df_beh$subject)
  
  if (!"subject" %in% names(df_model)) {
    df_model$subject <- "model_agent"
  } else {
    df_model$subject <- as.character(df_model$subject)
  }
  
  # Set up axis boundaries based on experiment version
  if (version == "exp1") {
    y_min <- 0.3
    y_max <- 0.7
  } else {
    y_min <- 0.1
    y_max <- 0.5
  }
  
  save_dir <- version

  # Generate Plots
  plot_combined_coefs(df_beh, df_model, y_var = "accuracy", x_lab = "Coefficient", figure_name = "coef_by_source", save_dir = save_dir)

  plot_binned_data_shade(df_beh, df_model, dependent_var = "actual_reward", x_lab = "Probed Reward", y_lab = "Accuracy", figure_name = "node_reward", y_min = y_min, y_max = y_max, x_step = 2, y_var = "accuracy", save_dir = save_dir)

  plot_binned_data_shade(df_beh %>% filter(is_leaf == TRUE), df_model %>% filter(is_leaf == TRUE), dependent_var = "aunt_reward", x_lab = "Average Aunt Reward", y_lab = "Accuracy", figure_name = "aunt_reward", y_min = y_min, y_max = y_max, x_step = 2, y_var = "accuracy", save_dir = save_dir)

  plot_binned_data_shade(df_beh, df_model, dependent_var = "sibling_reward", x_lab = "Average Sibling Reward", y_lab = "Accuracy", figure_name = "sibling_reward", y_min = y_min, y_max = y_max, x_step = 2, y_var = "accuracy", save_dir = save_dir)

  plot_binned_data_shade(df_beh %>% filter(is_leaf == TRUE), df_model %>% filter(is_leaf == TRUE), dependent_var = "parent_reward", x_lab = "Parent Reward", y_lab = "Accuracy", figure_name = "parent_reward", y_min = y_min, y_max = y_max, y_var = "accuracy", x_step = 2, save_dir = save_dir)

  # Generate and Plot Rankings
  if (version == "exp1") {
    df_beh_rank <- calculate_path_rank_exp1_beh(df_beh %>% filter(condition == "planning"))
    df_model_rank <- calculate_path_rank_exp1_model(df_model %>% filter(condition == "planning"))
    x_max_val <- 4.5
    x_step_val <- NULL
    y_min <- 0.3
    y_max <- 0.7
  } else {
    df_beh_rank <- calculate_path_rank_exp2_beh(df_beh %>% filter(condition == "planning"))
    df_model_rank <- calculate_path_rank_exp2_model(df_model %>% filter(condition == "planning"))
    x_max_val <- 9.5
    x_step_val <- 1
    y_min <- 0.1
    y_max <- 0.6
  }

  # Standardize rank column names
  df_beh_rank$path_rank <- df_beh_rank$p_rank
  df_model_rank$path_rank <- df_model_rank$p_rank

  # FIX APPLIED: Removed 'path' from group_by
  df_beh_rank_grouped <- df_beh_rank %>%
    group_by(subject, condition, graph, path_rank) %>%
    summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")

  # FIX APPLIED: Removed 'path' from group_by
  df_model_rank_grouped <- df_model_rank %>%
    group_by(subject, condition, graph, path_rank) %>%
    summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")

  plot_binned_data_bar(df_beh_rank_grouped, df_model_rank_grouped, dependent_var = "path_rank", x_lab = "Path Rank", y_lab = "Accuracy", figure_name = "path_rank", y_min = y_min, y_max = y_max, y_var = "accuracy", x_min = 1, x_max = x_max_val, x_step = x_step_val, save_dir = save_dir)
  }

# ------------------------------------------------------------------------------
# Execute Script
# ------------------------------------------------------------------------------
# process_version("exp1")
process_version("exp2")





