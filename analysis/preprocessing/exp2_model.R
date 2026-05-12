# ==============================================================================
# Experiment 2 Simulation Preprocessing
# Author: WM Planning manuscript team
# Purpose: Merge planning and exposure simulations, clean boolean fields, and
#          attach relational rewards for the 12-node network.
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

mother_map <- c("2" = 1, "3" = 1, "4" = 1, "6" = 5, "7" = 5, "8" = 5, "10" = 9, "11" = 9, "12" = 9)
sib1_map <- c("1" = 5, "2" = 3, "3" = 2, "4" = 2, "5" = 1, "6" = 7, "7" = 6, "8" = 6, "9" = 1, "10" = 11, "11" = 10, "12" = 10)
sib2_map <- c("1" = 9, "2" = 4, "3" = 4, "4" = 3, "5" = 9, "6" = 8, "7" = 8, "8" = 7, "9" = 5, "10" = 12, "11" = 12, "12" = 11)
aunt1_map <- c("2" = 5, "3" = 5, "4" = 5, "6" = 1, "7" = 1, "8" = 1, "10" = 1, "11" = 1, "12" = 1)
aunt2_map <- c("2" = 9, "3" = 9, "4" = 9, "6" = 9, "7" = 9, "8" = 9, "10" = 5, "11" = 5, "12" = 5)

df_plan <- read_analysis_csv("data", "raw_data", "exp2_model.csv") %>%
  dplyr::mutate(condition = "planning")

df_exp <- read_analysis_csv("data", "raw_data", "exp2_model_exp.csv") %>%
  dplyr::mutate(condition = "exposure")

df_raw <- dplyr::bind_rows(df_plan, df_exp)

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df_base <- df_raw %>%
  convert_true_false_strings() %>%
  dplyr::distinct(condition, graph, node, .keep_all = TRUE) %>%
  dplyr::mutate(
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward + 1e-06),
    is_leaf = node %in% c(2, 3, 4, 6, 7, 8, 10, 11, 12),
    error = abs(actual_reward - estimated_reward),
    accuracy = ifelse(estimated_reward == actual_reward, 1, 0),
    parent_reward = ifelse(is_leaf, path_reward - actual_reward, 0),
    correct_choice = ifelse(V == max_path_reward, 1, 0),
    mother_node = as.numeric(mother_map[as.character(node)]),
    sib1_node = as.numeric(sib1_map[as.character(node)]),
    sib2_node = as.numeric(sib2_map[as.character(node)]),
    aunt1_node = as.numeric(aunt1_map[as.character(node)]),
    aunt2_node = as.numeric(aunt2_map[as.character(node)])
  )

df_clean <- df_base %>%
  dplyr::group_by(condition, graph) %>%
  dplyr::mutate(
    mother_reward_raw = actual_reward[match(mother_node, node)],
    sib1_reward = actual_reward[match(sib1_node, node)],
    sib2_reward = actual_reward[match(sib2_node, node)],
    aunt1_reward = actual_reward[match(aunt1_node, node)],
    aunt2_reward = actual_reward[match(aunt2_node, node)]
  ) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    mother_reward = ifelse(is_leaf, mother_reward_raw, NA_real_),
    sibling_reward = rowMeans(cbind(sib1_reward, sib2_reward), na.rm = TRUE),
    aunt_reward_raw = rowMeans(cbind(aunt1_reward, aunt2_reward), na.rm = TRUE),
    aunt_reward = ifelse(is_leaf, aunt_reward_raw, NA_real_),
    sibling_reward = ifelse(is.nan(sibling_reward), NA_real_, sibling_reward),
    aunt_reward = ifelse(is.nan(aunt_reward), NA_real_, aunt_reward)
  ) %>%
  dplyr::select(
    is_leaf, graph, condition, node, norm_reward, actual_reward,
    estimated_reward, parent_reward, mother_reward, sibling_reward,
    aunt_reward, accuracy, correct_choice
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

# Relational rewards are computed within each graph and condition above.

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_clean, "data", "processed_data", "exp2_model.csv")
