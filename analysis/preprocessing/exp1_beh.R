# ==============================================================================
# Experiment 1 Behavioral Preprocessing
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

df_raw <- read_analysis_csv("data", "raw_data", "exp1_beh.csv")

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df <- df_raw %>%
  dplyr::mutate(
    condition = ifelse(show_arrow == FALSE, "exposure", "planning"),
    is_leaf = node %in% c(2, 3, 5, 6),
    graph = as.numeric(graph),
    node = as.numeric(node),
    parent_reward = path_reward - actual_reward,
    correct_choice = ifelse(V == max_path_reward, 1, 0),
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward),
    accuracy = ifelse(estimated_reward == actual_reward, 1, 0),
    node_reward = actual_reward
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

subject_significance <- df %>%
  dplyr::group_by(subject) %>%
  dplyr::summarize(
    n_trials_acc = sum(!is.na(accuracy)),
    n_correct_acc = sum(accuracy == 1, na.rm = TRUE),
    n_trials_choice_plan = sum(condition == "planning" & !is.na(correct_choice)),
    n_correct_choice_plan = sum(
      condition == "planning" & correct_choice == 1,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    p_val_acc = ifelse(
      n_trials_acc > 0,
      binom.test(n_correct_acc, n_trials_acc, p = 1 / 9, alternative = "greater")$p.value,
      1
    ),
    p_val_choice_plan = ifelse(
      n_trials_choice_plan > 0,
      binom.test(
        n_correct_choice_plan,
        n_trials_choice_plan,
        p = 1 / 4,
        alternative = "greater"
      )$p.value,
      1
    )
  ) %>%
  dplyr::ungroup()

kept_subjects_list <- subject_significance %>%
  dplyr::filter(p_val_acc < 0.05 & p_val_choice_plan < 0.05)

print_retention_summary(
  title = "Strict Binomial Filtering Results",
  n_original = nrow(subject_significance),
  n_kept = nrow(kept_subjects_list)
)

df_clean <- df %>%
  dplyr::semi_join(kept_subjects_list, by = "subject")

# ==============================================================================
# Saving Outputs
# ==============================================================================

write_analysis_csv(df_clean, "data", "processed_data", "exp1_beh.csv")


print(max(df$subject))
df_clean <- df_clean %>% mutate(
  parent_reward = ifelse(is_leaf == TRUE, parent_reward, 0),
  aunt_reward = ifelse(is_leaf == TRUE, aunt_reward, 0),
)
print(max(df_clean$subject))
model <- lmer( accuracy ~ (scale(node_reward) + scale(node_reward^2) + scale(sibling_reward) + scale(sibling_reward^2) + scale(aunt_reward) + scale(parent_reward))   + ( 1 | subject), data = df_clean %>% filter(condition == "planning")   )
summary(model)
