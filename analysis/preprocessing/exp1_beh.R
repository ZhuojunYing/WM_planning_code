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

source(project_path("analysis", "scripts", "figures", "utils.R"))

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
      binom.test(n_correct_acc, n_trials_acc, p = 1 / 11, alternative = "greater")$p.value,
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

write_analysis_csv(df_clean, "data", "processed_data", "exp1_beh_new.csv")
