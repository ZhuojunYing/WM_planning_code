# ==============================================================================
# Experiment 1 Preregistered Simulation Preprocessing
# Author: WM Planning manuscript team
# Purpose: Merge preregistered planning simulations with exposure simulations and
#          attach relational rewards for the 6-node network.
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

sibling_mapping <- c("1" = 4, "2" = 3, "3" = 2, "4" = 1, "5" = 6, "6" = 5)
aunt_mapping <- c("1" = 4, "2" = 4, "3" = 4, "4" = 1, "5" = 1, "6" = 1)

df_plan <- read_analysis_csv("data", "raw_data", "exp1_model_preregistered.csv") %>%
  dplyr::mutate(condition = "planning")

df_exp <- read_analysis_csv("data", "raw_data", "exp1_model_exp.csv") %>%
  dplyr::mutate(condition = "exposure")

df <- dplyr::bind_rows(df_plan, df_exp)

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df_clean <- df %>%
  dplyr::mutate(
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward + 1e-06),
    node_reward = actual_reward,
    is_leaf = node %in% c(2, 3, 5, 6),
    accuracy = ifelse(estimated_reward == actual_reward, 1, 0),
    error = abs(actual_reward - estimated_reward),
    correct_choice = ifelse(V == max_path_reward, 1, 0),
    parent_reward = path_reward - node_reward,
    sibling_node = as.numeric(sibling_mapping[as.character(node)]),
    aunt_node = as.numeric(aunt_mapping[as.character(node)])
  ) %>%
  dplyr::group_by(condition, graph) %>%
  dplyr::mutate(
    sibling_reward = actual_reward[match(sibling_node, node)],
    aunt_reward = actual_reward[match(aunt_node, node)]
  ) %>%
  dplyr::ungroup() %>%
  dplyr::select(
    is_leaf, graph, condition, node, norm_reward, actual_reward,
    estimated_reward, parent_reward, sibling_reward, aunt_reward,
    accuracy, correct_choice, error
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

# Relational rewards are computed within each graph and condition above.

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_clean, "data", "processed_data", "exp1_model_preregistered.csv")
print(mean(df_clean$accuracy))