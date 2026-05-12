# ==============================================================================
# Script: Plotting Behavioral Data (40-Node Network)
# Description: Generates binned behavioral plots (bars/shaded lines/boxplots),
#              fits mixed-effects models, and ranks 39 paths.
# ==============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(grid)
library(rlang) 
library(lmerTest)
library(here)

source(here::here("analysis", "scripts", "utils.R"))
source(here::here("analysis", "scripts", "plot_utils.R"))

# ==============================================================================
# Execution Wrapper
# ==============================================================================

process_version <- function(version) {
  
  # Load data dynamically based on the version passed in
  beh_file <- analysis_path("data", "processed_data", paste0(version, "_beh.csv"))
  df_beh <- read.csv(beh_file) 
  
  # Ensure accuracy exists (using your custom mapping logic)
  if("true_reward" %in% colnames(df_beh) && "estimated_reward" %in% colnames(df_beh)) {
    df_beh$accuracy <- ifelse(df_beh$estimated_reward == df_beh$true_reward, 1, 0)
  }
  
  save_dir <- version
  y_min <- 0; y_max <- 0.6
  
  # 1. Plot Combined Coefs
  plot_combined_coefs(df_beh = df_beh, x_lab = "Coefficient", y_lab = "", figure_name = "coef_by_source", save_dir = save_dir)
  # 
  # # 2. Node Reward Shade
  # plot_binned_data_shade(df_beh = df_beh, dependent_var = "actual_reward", x_lab = "Probed Reward", y_lab = "Accuracy", 
  #                        figure_name = "node_reward", y_min = y_min, y_max = y_max, x_size = 33, y_size = 33, x_step = 2, 
  #                        show_exposure = FALSE, save_dir = save_dir)
  # 
  # # 3. Sibling Reward
  # plot_binned_data_shade(df_beh = df_beh, dependent_var = "sibling_reward", x_lab = "Average\nSibling Reward", y_lab = "Accuracy", 
  #                        figure_name = "sibling_reward", y_min = y_min, y_max = y_max, x_size = 33, y_size = 33, x_step = 2, 
  #                        show_exposure = FALSE, save_dir = save_dir)
  # 
  # # 4. Aunt Reward
  # plot_binned_data_shade(df_beh = df_beh %>% filter(is_leaf == TRUE), dependent_var = "aunt_reward", x_lab = "Average\nAunt Reward", y_lab = "Accuracy", 
  #                        figure_name = "aunt_reward", y_min = y_min, y_max = y_max, x_size = 33, y_size = 33, x_step = 2, 
  #                        show_exposure = FALSE, save_dir = save_dir)
  # 
  # # 5. Other Reward
  # plot_binned_data_shade(df_beh = df_beh %>% filter(is_leaf == TRUE), dependent_var = "preceding_sum_reward", x_lab = "Preceding Reward\nSum", y_lab = "Accuracy", 
  #                        figure_name = "preceding_sum_reward", y_min = y_min, y_max = y_max, x_step = 2, x_min = -7, x_max = 7, x_size = 33, y_size = 33, 
  #                        show_exposure = FALSE, save_dir = save_dir)
  # 
  # # 6. Path Rank Plotting
  # # 6. Path Rank Plotting
  # df_beh_rank <- calculate_path_rank_exp2_40n_beh(
  #   df_beh %>% dplyr::filter(condition == "planning")
  # )
  # 
  # df_beh_rank_grouped <- df_beh_rank %>% 
  #   mutate(path_rank = p_rank) %>%
  #   group_by(subject, condition, graph, path_rank) %>% # <--- Removed `path`
  #   summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")
  # 
  # plot_binned_data_bar(
  #   df_beh = df_beh_rank_grouped, 
  #   dependent_var = "path_rank", 
  #   x_lab = "Path Rank", 
  #   y_lab = "Accuracy", 
  #   figure_name = "path_rank", 
  #   y_min = 0, 
  #   y_max = 0.6, 
  #   x_step = 1, 
  #   x_label_step = 9, 
  #   x_min = 1, 
  #   x_max = 27, 
  #   save_dir = save_dir
  # )
}

# Run the function on the specific exp2_40n dataset
process_version("exp2_40n")
