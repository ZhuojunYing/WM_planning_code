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
# Experiment 2 Behavioral Preprocessing
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

df_raw <- readr::read_csv(
  file.path(RAW_DATA_DIR, "exp2_beh.csv"),
  show_col_types = FALSE
)

required_raw_columns <- c("est_best_path")
missing_raw_columns <- setdiff(required_raw_columns, names(df_raw))
if (length(missing_raw_columns) > 0L) {
  stop(
    sprintf(
      "Experiment 2 raw behavioral data are missing required columns: %s",
      paste(missing_raw_columns, collapse = ", ")
    ),
    call. = FALSE
  )
}

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

df <- df_raw %>%
  convert_true_false_strings() %>%
  dplyr::mutate(
    est_best_path = trimws(as.character(est_best_path)),
    est_best_path = dplyr::if_else(
      is.na(est_best_path) |
        est_best_path %in% c("", "None", "None, None, None"),
      NA_character_,
      est_best_path
    ),
    is_leaf = node %in% c(2, 3, 4, 6, 7, 8, 10, 11, 12),
    condition = ifelse(show_arrow == FALSE, "exposure", "planning"),
    accuracy = ifelse(abs(actual_reward - estimated_reward) < 1, 1, 0),
    norm_reward = (V - min_path_reward) / (max_path_reward - min_path_reward),
    correct_choice = ifelse(abs(V - max_path_reward) < 0.1, 1, 0),
    parent_reward = path_reward - actual_reward
  ) %>%
  dplyr::select(
    subject, condition, est_best_path, accuracy, correct_choice,
    norm_reward, parent_reward, everything()
  )

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

subject_significance <- df %>%
  dplyr::group_by(subject) %>%
  dplyr::summarize(
    n_trials_acc = sum(!is.na(accuracy)),
    n_correct_acc = sum(accuracy == 1, na.rm = TRUE),
    n_trials_plan = sum(condition == "planning" & !is.na(correct_choice)),
    n_correct_plan = sum(condition == "planning" & correct_choice == 1, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    p_val_acc = ifelse(
      n_trials_acc > 0,
      binom.test(n_correct_acc, n_trials_acc, p = 1 / 11, alternative = "greater")$p.value,
      1
    ),
    p_val_plan = ifelse(
      n_trials_plan > 0,
      binom.test(n_correct_plan, n_trials_plan, p = 1 / 9, alternative = "greater")$p.value,
      1
    )
  ) %>%
  dplyr::ungroup()

kept_subjects_list <- subject_significance %>%
  dplyr::filter(p_val_acc < 0.05 & p_val_plan < 0.05)

print_retention_summary(
  title = "Binomial Filtering Results",
  n_original = nrow(subject_significance),
  n_kept = nrow(kept_subjects_list),
  extra_lines = sprintf("Total unique subjects in raw data:  %s", dplyr::n_distinct(df$subject))
)

df_clean <- df %>%
  dplyr::semi_join(kept_subjects_list, by = "subject") %>%
  dplyr::select(where(~ !is.list(.)))

# ==============================================================================
# Saving Outputs
# ==============================================================================

processed_output_path <- file.path(PROCESSED_DATA_DIR, "exp2_beh_new.csv")
dir.create(
  dirname(processed_output_path),
  recursive = TRUE,
  showWarnings = FALSE
)
readr::write_csv(df_clean, processed_output_path)
