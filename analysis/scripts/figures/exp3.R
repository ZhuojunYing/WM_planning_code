#!/usr/bin/env Rscript
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


# Experiment 3 interaction plots for human behavior and the participant-fitted
# sequential VAE. The best-fit table records the four topology-specific weight
# paths used to score each participant. The preferred input is the standalone
# participant-fitted simulation CSV, containing new trials for every
# participant, model seed, and topology. A legacy fallback can still summarize
# the original grid simulations when that fourth input is omitted.

suppressPackageStartupMessages({
  library(cowplot)
  library(dplyr)
  library(ggplot2)
  library(here)
  library(purrr)
  library(readr)
  library(tidyr)
})

source(project_path("analysis", "scripts", "project_paths.R"))
source(project_path("analysis", "scripts", "figures", "plot_utils_fitted_simulation.R"))

TREE_TYPES <- c(
  "deep_breadth",
  "deep_depth",
  "wide_breadth",
  "wide_depth"
)
INTERACTION_COLORS <- c(
  "deep" = "#755374",
  "wide" = "#5b705e"
)
INTERACTION_OUTLINE_COLORS <- c(
  "deep_outline" = "#523A51",
  "wide_outline" = "#404E42"
)

fail <- function(...) {
  stop(sprintf(...), call. = FALSE)
}

usage <- function() {
  cat(paste0(
    "Usage:\n",
    "  Rscript analysis/scripts/figures/exp3_fitted_interactions.R \\\n",
    "    PARTICIPANT_FITS_CSV [PROCESSED_BEHAVIOR_CSV] \\\n",
    "    [SIMULATION_SEEDS] [FITTED_SIMULATION_CSV]\n\n",
    "PARTICIPANT_FITS_CSV must be the summed 31-node best-fit table. ",
    "The behavioral CSV defaults to the centralized processed exp3_beh.csv. ",
    "SIMULATION_SEEDS defaults to 1,2,3,4,5. When FITTED_SIMULATION_CSV ",
    "is provided, model plots use its newly generated participant-specific ",
    "trials.\n"
  ))
}

require_columns <- function(data, columns, label) {
  missing_columns <- setdiff(columns, names(data))
  if (length(missing_columns) > 0L) {
    fail(
      "%s is missing required columns: %s",
      label,
      paste(missing_columns, collapse = ", ")
    )
  }
}

tree_labels <- function(data) {
  data %>%
    mutate(
      tree = ifelse(startsWith(tree_type, "deep_"), "deep", "wide"),
      order = ifelse(endsWith(tree_type, "_depth"), "DFS", "BFS"),
      tree = factor(tree, levels = c("deep", "wide")),
      order = factor(order, levels = c("BFS", "DFS"))
    )
}

parse_seed_list <- function(text) {
  tokens <- trimws(strsplit(text, ",", fixed = TRUE)[[1]])
  if (
    length(tokens) == 0L ||
      any(!grepl("^[0-9]+$", tokens)) ||
      anyDuplicated(tokens)
  ) {
    fail("SIMULATION_SEEDS must contain unique non-negative integers.")
  }
  seeds <- suppressWarnings(as.integer(tokens))
  if (any(is.na(seeds))) {
    fail("A simulation seed is outside R's integer range.")
  }
  seeds
}

exp3_tree_paths <- function(tree_type) {
  children <- switch(
    tree_type,
    deep_breadth = list(
      `0` = c(1L, 2L), `1` = c(3L, 4L), `2` = c(5L, 6L),
      `3` = c(7L, 8L), `4` = c(9L, 10L), `5` = c(11L, 12L),
      `6` = c(13L, 14L), `7` = c(15L, 16L),
      `8` = c(17L, 18L), `9` = c(19L, 20L),
      `10` = c(21L, 22L), `11` = c(23L, 24L),
      `12` = c(25L, 26L), `13` = c(27L, 28L),
      `14` = c(29L, 30L)
    ),
    deep_depth = list(
      `0` = c(1L, 16L), `1` = c(2L, 9L), `2` = c(3L, 6L),
      `3` = c(4L, 5L), `6` = c(7L, 8L), `9` = c(10L, 13L),
      `10` = c(11L, 12L), `13` = c(14L, 15L),
      `16` = c(17L, 24L), `17` = c(18L, 21L),
      `18` = c(19L, 20L), `21` = c(22L, 23L),
      `24` = c(25L, 28L), `25` = c(26L, 27L),
      `28` = c(29L, 30L)
    ),
    wide_breadth = list(
      `0` = 1:5, `1` = 6:10, `2` = 11:15,
      `3` = 16:20, `4` = 21:25, `5` = 26:30
    ),
    wide_depth = list(
      `0` = c(1L, 7L, 13L, 19L, 25L), `1` = 2:6,
      `7` = 8:12, `13` = 14:18, `19` = 20:24,
      `25` = 26:30
    ),
    fail("Unsupported Experiment 3 tree type: %s", tree_type)
  )

  walk_paths <- function(node, path) {
    next_nodes <- children[[as.character(node)]]
    if (is.null(next_nodes) || length(next_nodes) == 0L) {
      return(list(as.integer(path)))
    }
    unlist(
      lapply(
        next_nodes,
        function(next_node) {
          walk_paths(next_node, c(path, as.integer(next_node)))
        }
      ),
      recursive = FALSE
    )
  }
  paths <- walk_paths(0L, integer())
  if (!identical(sort(unique(unlist(paths))), 1:30)) {
    fail("The %s paths do not cover reward nodes 1 through 30.", tree_type)
  }
  paths
}

exp3_tree_order_labels <- function(tree_type) {
  factor(
    as.character(tree_type),
    levels = TREE_TYPES,
    labels = c("Deep - BFS", "Deep - DFS", "Wide - BFS", "Wide - DFS")
  )
}

prepare_exp3_human_rank_rows <- function(behavior) {
  reward_columns <- paste0("n", seq_len(30L), "_reward")
  require_columns(
    behavior,
    c(
      "subject", "graph", "tree_type", "node", "actual_reward",
      "estimated_reward", "accuracy", reward_columns
    ),
    "Experiment 3 behavioral rank data"
  )
  behavior <- behavior %>%
    mutate(
      subject = as.character(subject),
      graph = as.character(graph),
      tree_type = as.character(tree_type),
      node = as.integer(node),
      accuracy = as.numeric(accuracy),
      actual_reward = as.numeric(actual_reward),
      estimated_reward = as.numeric(estimated_reward),
      across(all_of(reward_columns), as.numeric)
    ) %>%
    filter(
      tree_type %in% TREE_TYPES,
      node %in% seq_len(30L),
      is.finite(accuracy),
      is.finite(actual_reward),
      is.finite(estimated_reward),
      if_all(all_of(reward_columns), is.finite)
    )
  trial_keys <- c("subject", "graph", "tree_type", reward_columns)
  if (anyDuplicated(behavior[c(trial_keys, "node")])) {
    fail("Experiment 3 behavior contains duplicate trial/node recall rows.")
  }

  purrr::map_dfr(TREE_TYPES, function(current_tree_type) {
    tree_rows <- behavior %>% filter(tree_type == current_tree_type)
    trials <- tree_rows %>%
      distinct(across(all_of(trial_keys))) %>%
      mutate(.rank_trial_id = row_number())
    rank_lookup <- fitted_observation_final_rank_lookup(
      trials,
      reward_columns,
      exp3_tree_paths(current_tree_type)
    )
    tree_rows %>%
      left_join(
        trials,
        by = trial_keys,
        relationship = "many-to-one"
      ) %>%
      left_join(
        rank_lookup,
        by = c(".rank_trial_id", "node"),
        relationship = "many-to-one"
      ) %>%
      # The full reward vector still determines final rank. Only the final
      # presentation is omitted because its rank cannot change afterward.
      filter(node < 30L) %>%
      transmute(
        source = "participants",
        participant_id = subject,
        tree_type,
        tree_order = exp3_tree_order_labels(tree_type),
        trial_id = paste0(current_tree_type, "_human_", .rank_trial_id),
        node,
        node_reward = actual_reward,
        accuracy,
        observation_rank,
        final_rank,
        rank_change
      )
  })
}

prepare_exp3_model_rank_rows <- function(simulation) {
  require_columns(
    simulation,
    c(
      "subject", "tree_type", "simulation_model_seed",
      "simulation_trial", "graph", "node", "actual_reward",
      "estimated_reward", "accuracy"
    ),
    "Experiment 3 fitted-simulation rank data"
  )
  trial_keys <- c(
    "subject", "tree_type", "simulation_model_seed",
    "simulation_trial", "graph"
  )
  if (anyDuplicated(simulation[c(trial_keys, "node")])) {
    fail("Experiment 3 fitted simulations contain duplicate trial/node rows.")
  }

  purrr::map_dfr(TREE_TYPES, function(current_tree_type) {
    tree_rows <- simulation %>% filter(tree_type == current_tree_type)
    trials <- tree_rows %>%
      select(all_of(trial_keys), node, actual_reward) %>%
      tidyr::pivot_wider(
        names_from = node,
        values_from = actual_reward,
        names_prefix = "reward_"
      ) %>%
      mutate(.rank_trial_id = row_number())
    reward_columns <- paste0("reward_", seq_len(30L))
    require_columns(
      trials,
      reward_columns,
      sprintf("%s fitted-simulation trials", current_tree_type)
    )
    rank_lookup <- fitted_observation_final_rank_lookup(
      trials,
      reward_columns,
      exp3_tree_paths(current_tree_type)
    )
    tree_rows %>%
      left_join(
        trials %>% select(all_of(trial_keys), .rank_trial_id),
        by = trial_keys,
        relationship = "many-to-one"
      ) %>%
      left_join(
        rank_lookup,
        by = c(".rank_trial_id", "node"),
        relationship = "many-to-one"
      ) %>%
      # The full reward vector still determines final rank. Only the final
      # presentation is omitted because its rank cannot change afterward.
      filter(node < 30L) %>%
      transmute(
        source = "model",
        participant_id = as.character(subject),
        tree_type,
        tree_order = exp3_tree_order_labels(tree_type),
        trial_id = paste0(current_tree_type, "_model_", .rank_trial_id),
        node,
        node_reward = actual_reward,
        accuracy,
        observation_rank,
        final_rank,
        rank_change
      )
  })
}

parse_fit_routing <- function(fits, simulation_seeds) {
  require_columns(
    fits,
    c(
      "subject",
      "best_lambda",
      "best_alpha",
      "nll_model_seed",
      "weight_files"
    ),
    "31-node participant-fit CSV"
  )
  if (anyDuplicated(fits$subject)) {
    fail("The participant-fit CSV contains more than one row per participant.")
  }

  routing <- fits %>%
    transmute(
      subject = as.character(subject),
      best_lambda = as.numeric(best_lambda),
      best_alpha = as.numeric(best_alpha),
      nll_model_seed = as.integer(nll_model_seed),
      tree_weight = strsplit(as.character(weight_files), "|", fixed = TRUE)
    ) %>%
    tidyr::unnest_longer(tree_weight) %>%
    tidyr::separate_wider_delim(
      tree_weight,
      delim = "=",
      names = c("tree_type", "weight_file"),
      too_many = "merge"
    ) %>%
    tidyr::crossing(simulation_model_seed = simulation_seeds) %>%
    mutate(
      tree_type = trimws(tree_type),
      weight_file = trimws(weight_file),
      simulation_basename = purrr::map2_chr(
        weight_file,
        simulation_model_seed,
        function(path, seed) {
          sub(
            "_seed_[0-9]+_tree_",
            paste0("_seed_", seed, "_tree_"),
            basename(path)
          )
        }
      ),
      simulation_file = file.path(
        SIMULATION_DIR,
        sub(
          "[.]weights[.]h5$",
          ".csv",
          simulation_basename
        )
      )
    )

  if (any(!routing$tree_type %in% TREE_TYPES)) {
    fail("The fit table contains an unsupported 31-node tree type.")
  }
  if (
    any(!is.finite(routing$best_lambda)) ||
      any(!is.finite(routing$best_alpha)) ||
      any(is.na(routing$nll_model_seed)) ||
      any(is.na(routing$simulation_model_seed))
  ) {
    fail("The fit table contains an invalid parameter or model seed.")
  }
  routing_counts <- routing %>%
    distinct(subject, tree_type) %>%
    count(subject, name = "tree_count")
  if (any(routing_counts$tree_count != length(TREE_TYPES))) {
    fail("Every participant must map to exactly four topology simulations.")
  }
  duplicate_tree <- routing %>%
    count(subject, tree_type, simulation_model_seed, name = "rows") %>%
    filter(rows != 1L)
  if (nrow(duplicate_tree) > 0L) {
    fail("A participant has duplicate tree-type/model-seed routing.")
  }
  missing_files <- unique(routing$simulation_file[!file.exists(
    routing$simulation_file
  )])
  if (length(missing_files) > 0L) {
    fail(
      "Matching model simulation files do not exist: %s",
      paste(missing_files, collapse = "; ")
    )
  }
  routing
}

summarize_simulation_file <- function(path) {
  required_columns <- c(
    "graph",
    "node",
    "actual_reward",
    "estimated_reward",
    "V",
    "min_path_reward",
    "max_path_reward"
  )
  simulation <- readr::read_csv(
    path,
    col_select = dplyr::all_of(required_columns),
    show_col_types = FALSE,
    progress = FALSE
  )
  require_columns(simulation, required_columns, basename(path))
  simulation <- simulation %>%
    transmute(
      graph = as.character(graph),
      node = as.integer(node),
      actual_reward = as.numeric(actual_reward),
      estimated_reward = as.numeric(estimated_reward),
      chosen_path_reward = as.numeric(V),
      min_path_reward = as.numeric(min_path_reward),
      max_path_reward = as.numeric(max_path_reward)
    )
  if (
    any(!is.finite(simulation$actual_reward)) ||
      any(!is.finite(simulation$estimated_reward))
  ) {
    fail("Simulation contains non-finite reward values: %s", path)
  }
  if (!identical(sort(unique(simulation$node)), 1:30)) {
    fail("Simulation does not contain reward nodes 1 through 30: %s", path)
  }
  if (anyDuplicated(simulation[c("graph", "node")])) {
    fail("Simulation contains duplicate graph/node rows: %s", path)
  }

  accuracy <- mean(
    round(simulation$estimated_reward) == round(simulation$actual_reward)
  )
  trial_rewards <- simulation %>%
    distinct(
      graph,
      chosen_path_reward,
      min_path_reward,
      max_path_reward
    ) %>%
    mutate(
      reward_range = max_path_reward - min_path_reward,
      norm_reward = ifelse(
        reward_range > 0,
        (chosen_path_reward - min_path_reward) / reward_range,
        NA_real_
      )
    )
  if (any(!is.finite(trial_rewards$norm_reward))) {
    fail("Simulation has an undefined normalized reward: %s", path)
  }

  tibble(
    simulation_file = path,
    accuracy = accuracy,
    norm_reward = mean(trial_rewards$norm_reward),
    simulation_trials = nrow(trial_rewards),
    simulation_recall_rows = nrow(simulation)
  )
}

prepare_fitted_simulation_summary <- function(
    path,
    fits,
    simulation_seeds) {
  required_columns <- c(
    "subject",
    "tree_type",
    "simulation_model_seed",
    "simulation_trial",
    "graph",
    "node",
    "actual_reward",
    "estimated_reward",
    "V",
    "min_path_reward",
    "max_path_reward"
  )
  simulation <- readr::read_csv(
    path,
    col_select = dplyr::all_of(required_columns),
    show_col_types = FALSE,
    progress = FALSE
  )
  require_columns(simulation, required_columns, "Fitted simulation CSV")
  simulation <- simulation %>%
    transmute(
      subject = as.character(subject),
      tree_type = as.character(tree_type),
      simulation_model_seed = as.integer(simulation_model_seed),
      simulation_trial = as.integer(simulation_trial),
      graph = as.character(graph),
      node = as.integer(node),
      actual_reward = as.numeric(actual_reward),
      estimated_reward = as.numeric(estimated_reward),
      chosen_path_reward = as.numeric(V),
      min_path_reward = as.numeric(min_path_reward),
      max_path_reward = as.numeric(max_path_reward),
      accuracy = as.numeric(
        round(estimated_reward) == round(actual_reward)
      )
    )
  if (nrow(simulation) == 0L) {
    fail("Fitted simulation CSV contains no rows: %s", path)
  }
  if (any(!simulation$tree_type %in% TREE_TYPES)) {
    fail("Fitted simulation CSV contains an unsupported tree type.")
  }
  if (!setequal(unique(simulation$simulation_model_seed), simulation_seeds)) {
    fail(
      "Fitted simulation seeds do not match requested seeds. Found %s; expected %s.",
      paste(sort(unique(simulation$simulation_model_seed)), collapse = ","),
      paste(sort(simulation_seeds), collapse = ",")
    )
  }
  if (!identical(sort(unique(simulation$node)), 1:30)) {
    fail("Fitted simulation CSV does not contain reward nodes 1 through 30.")
  }
  if (anyDuplicated(simulation[c(
    "subject",
    "tree_type",
    "simulation_model_seed",
    "simulation_trial",
    "node"
  )])) {
    fail(
      paste0(
        "Fitted simulation CSV contains duplicate participant/tree/model-",
        "seed/trial/node rows."
      )
    )
  }

  fit_subjects <- fits %>%
    transmute(subject = as.character(subject)) %>%
    distinct()
  if (!setequal(unique(simulation$subject), fit_subjects$subject)) {
    fail("Fitted simulation participants do not match the best-fit CSV.")
  }
  complete_design <- simulation %>%
    distinct(subject, tree_type, simulation_model_seed) %>%
    count(subject, name = "design_cells")
  expected_design_cells <- length(TREE_TYPES) * length(simulation_seeds)
  if (any(complete_design$design_cells != expected_design_cells)) {
    fail(
      "At least one participant lacks a topology/model-seed simulation cell."
    )
  }
  trial_counts <- simulation %>%
    distinct(
      subject,
      tree_type,
      simulation_model_seed,
      simulation_trial
    ) %>%
    count(subject, tree_type, simulation_model_seed, name = "trials")
  if (
    any(trial_counts$trials < 1L) ||
      n_distinct(trial_counts$trials) != 1L
  ) {
    fail(
      paste0(
        "Every participant/tree/model-seed cell must contain the same ",
        "positive number of trials."
      )
    )
  }

  accuracy_summary <- simulation %>%
    group_by(subject, tree_type, simulation_model_seed) %>%
    summarise(accuracy = mean(accuracy), .groups = "drop")
  reward_summary <- simulation %>%
    distinct(
      subject,
      tree_type,
      simulation_model_seed,
      graph,
      chosen_path_reward,
      min_path_reward,
      max_path_reward
    ) %>%
    mutate(
      reward_range = max_path_reward - min_path_reward,
      norm_reward = ifelse(
        reward_range > 0,
        (chosen_path_reward - min_path_reward) / reward_range,
        NA_real_
      )
    ) %>%
    group_by(subject, tree_type, simulation_model_seed) %>%
    summarise(norm_reward = mean(norm_reward), .groups = "drop")

  participant_summary <- accuracy_summary %>%
    inner_join(
      reward_summary,
      by = c("subject", "tree_type", "simulation_model_seed"),
      relationship = "one-to-one"
    ) %>%
    group_by(subject, tree_type) %>%
    summarise(
      accuracy = mean(accuracy),
      norm_reward = mean(norm_reward),
      simulation_seed_count = n_distinct(simulation_model_seed),
      .groups = "drop"
    ) %>%
    tree_labels()
  if (
    any(!is.finite(participant_summary$accuracy)) ||
      any(!is.finite(participant_summary$norm_reward))
  ) {
    fail("At least one fitted participant simulation summary is undefined.")
  }
  cat(sprintf(
    paste0(
      "Read participant-fitted simulations: %d participants, %d tree types, ",
      "%d seeds, %d trials per participant/seed/tree.\n"
    ),
    n_distinct(simulation$subject),
    n_distinct(simulation$tree_type),
    n_distinct(simulation$simulation_model_seed),
    unique(trial_counts$trials)
  ))
  list(
    participant_summary = participant_summary,
    recall_rows = simulation
  )
}

prepare_behavior_summary <- function(path) {
  behavior <- readr::read_csv(
    path,
    show_col_types = FALSE,
    progress = FALSE
  )
  require_columns(
    behavior,
    c(
      "subject",
      "graph",
      "tree_type",
      "actual_reward",
      "estimated_reward",
      "V",
      "min_path_reward",
      "max_path_reward"
    ),
    "Processed Experiment 3 behavior"
  )
  behavior <- behavior %>%
    filter(tree_type %in% TREE_TYPES) %>%
    mutate(
      subject = as.character(subject),
      graph = as.character(graph),
      accuracy = as.numeric(
        round(as.numeric(estimated_reward)) ==
          round(as.numeric(actual_reward))
      )
    )
  trial_rewards <- behavior %>%
    distinct(
      subject,
      graph,
      tree_type,
      V,
      min_path_reward,
      max_path_reward
    ) %>%
    mutate(
      reward_range = as.numeric(max_path_reward) -
        as.numeric(min_path_reward),
      norm_reward = ifelse(
        reward_range > 0,
        (as.numeric(V) - as.numeric(min_path_reward)) / reward_range,
        NA_real_
      )
    )

  accuracy_summary <- behavior %>%
    group_by(subject, tree_type) %>%
    summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")
  reward_summary <- trial_rewards %>%
    group_by(subject, tree_type) %>%
    summarise(
      norm_reward = mean(norm_reward, na.rm = TRUE),
      .groups = "drop"
    )
  participant_summary <- accuracy_summary %>%
    inner_join(
      reward_summary,
      by = c("subject", "tree_type"),
      relationship = "one-to-one"
    ) %>%
    tree_labels()
  list(
    participant_summary = participant_summary,
    trial_rewards = trial_rewards,
    recall_rows = behavior
  )
}

plot_exp3_fitted_parameter_performance <- function(fits, behavior_outputs, output_directory) {
  require_columns(fits, c("subject", "best_lambda", "best_alpha"), "Exp3 best fits")
  parameters <- fits %>%
    transmute(subject = as.character(subject), fitted_lambda = as.numeric(best_lambda),
              fitted_alpha = as.numeric(best_alpha))
  if (anyDuplicated(parameters$subject) || anyNA(parameters$subject) || !nrow(parameters)) {
    fail("Exp3 fits must contain one row per participant.")
  }
  if (any(!is.finite(parameters$fitted_lambda) | parameters$fitted_lambda <= 0) ||
      any(!is.finite(parameters$fitted_alpha) | parameters$fitted_alpha < 0)) {
    fail("Exp3 fitted parameters are invalid.")
  }
  parameters$fitted_rho <- if ("best_rho" %in% names(fits)) {
    as.numeric(fits$best_rho)
  } else {
    signif(parameters$fitted_alpha / parameters$fitted_lambda, 12)
  }
  if (any(!is.finite(parameters$fitted_rho) | parameters$fitted_rho < 0) ||
      any(abs(parameters$fitted_alpha - parameters$fitted_lambda * parameters$fitted_rho) >
          1e-10 * pmax(abs(parameters$fitted_alpha), .Machine$double.eps))) {
    fail("Exp3 fitted rho is invalid or inconsistent with alpha / lambda.")
  }
  trials <- behavior_outputs$trial_rewards
  if (anyDuplicated(trials[c("subject", "graph", "tree_type")])) {
    fail("Exp3 behavioral trials have inconsistent normalized rewards.")
  }
  # Pool each participant's planning trials across all four conditions first;
  # then give participants equal weight within each selected parameter cell.
  recall <- behavior_outputs$recall_rows %>%
    filter(is.finite(accuracy)) %>%
    group_by(subject) %>%
    summarise(human_recall_accuracy = mean(accuracy), .groups = "drop")
  reward <- trials %>%
    filter(is.finite(norm_reward)) %>%
    group_by(subject) %>%
    summarise(human_norm_reward = mean(norm_reward), .groups = "drop")
  participants <- parameters %>%
    left_join(recall, by = "subject", relationship = "one-to-one") %>%
    left_join(reward, by = "subject", relationship = "one-to-one")
  if (any(!is.finite(participants$human_recall_accuracy)) ||
      any(!is.finite(participants$human_norm_reward))) {
    fail("Some fitted Exp3 participants have no valid behavioral performance.")
  }
  lambda_values <- sort(unique(parameters$fitted_lambda))
  rho_values <- sort(unique(parameters$fitted_rho))
  summary <- participants %>%
    group_by(fitted_lambda, fitted_rho) %>%
    summarise(mean_human_norm_reward = mean(human_norm_reward),
              mean_human_recall_accuracy = mean(human_recall_accuracy),
              participant_count = n(), .groups = "drop") %>%
    tidyr::complete(fitted_lambda = lambda_values, fitted_rho = rho_values)
  plot_data <- summary %>%
    mutate(
      fitted_lambda_factor = factor(fitted_lambda, levels = lambda_values,
                                    labels = fitted_reviewer_parameter_labels(lambda_values)),
      fitted_rho_factor = factor(fitted_rho, levels = rho_values,
                                 labels = fitted_reviewer_parameter_labels(rho_values))
    )
  panels <- list(
    fitted_reviewer_parameter_heatmap_panel(
      plot_data, "mean_human_norm_reward", "Participant\nnormalized\nreward"
    ) + labs(y = expression("Fitted " * alpha / lambda)),
    fitted_reviewer_parameter_heatmap_panel(
      plot_data, "mean_human_recall_accuracy", "Participant\nrecall\naccuracy"
    ) + labs(y = expression("Fitted " * alpha / lambda))
  )
  plot <- cowplot::plot_grid(plotlist = panels, nrow = 1, rel_widths = c(1, 1),
                            align = "hv", axis = "tblr")
  output_path <- file.path(output_directory, "fitted_parameters_human_performance.pdf")
  ensure_parent_directory(output_path)
  ggplot2::ggsave(output_path, plot = plot, width = 90, height = 40,
                  units = "mm", bg = "white", limitsize = FALSE)
  summary_path <- file.path(ANALYSIS_SUMMARY_DIR, "tests", basename(output_directory),
                            "fitted_parameters_human_performance.csv")
  ensure_parent_directory(summary_path)
  readr::write_csv(summary, summary_path)
  cat(sprintf("Saved Exp3 fitted-parameter performance heatmaps to: %s\n", output_path))
  cat(sprintf("Saved Exp3 fitted-parameter performance summary to: %s\n", summary_path))
  output_path
}

interaction_plot <- function(
    participant_data,
    metric,
    y_label,
    y_limits,
    y_breaks) {
  metric_symbol <- rlang::sym(metric)
  summary <- participant_data %>%
    filter(is.finite(!!metric_symbol)) %>%
    group_by(tree, order) %>%
    summarise(
      mean_value = mean(!!metric_symbol),
      participant_count = n_distinct(subject),
      se = ifelse(
        participant_count > 1L,
        sd(!!metric_symbol) / sqrt(participant_count),
        0
      ),
      .groups = "drop"
    ) %>%
    mutate(
      lower = mean_value - se,
      upper = mean_value + se
    )

  dodge <- position_dodge(width = 0.1)
  ggplot(
    summary,
    aes(x = order, y = mean_value, color = tree, group = tree)
  ) +
    geom_line(position = dodge, linewidth = 0.5) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      position = dodge,
      width = 0.1,
      linewidth = 0.4
    ) +
    geom_point(
      aes(fill = tree, color = paste0(tree, "_outline")),
      position = dodge, shape = 21, size = 0.25, stroke = 0.1
    ) +
    scale_color_manual(values = c(INTERACTION_COLORS, INTERACTION_OUTLINE_COLORS)) +
    scale_fill_manual(values = INTERACTION_COLORS) +
    scale_y_continuous(
      breaks = y_breaks,
      labels = scales::label_number(accuracy = 0.01)
    ) +
    coord_cartesian(ylim = y_limits) +
    labs(x = "Order", y = y_label, color = NULL) +
    theme_minimal(base_size = 7) +
    theme(
      panel.grid = element_blank(),
      panel.border = element_blank(),
      axis.line = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(1.5, "mm"),
      axis.title = element_text(size = 7, color = "black"),
      axis.text = element_text(size = 7, color = "black"),
      legend.position = "none",
      plot.background = element_rect(fill = "white", color = NA)
    )
}

save_interaction_plot <- function(plot, path) {
  ensure_parent_directory(path)
  output_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  ggsave(
    path,
    plot,
    width = 40,
    height = 30,
    units = "mm",
    device = output_device,
    bg = "white"
  )
}

save_combined_interaction_plot <- function(
    participant_plot,
    model_plot,
    path) {
  combined_data <- dplyr::bind_rows(
    participant_plot$data %>%
      dplyr::mutate(source = "Participants"),
    model_plot$data %>%
      dplyr::mutate(source = "Model")
  ) %>%
    dplyr::mutate(source = factor(source, levels = c("Participants", "Model")))

  # Build one faceted ggplot, as in the Exp1 fitted line plots. Combining two
  # complete ggplots duplicated the y axis, tick labels, and margins and left
  # very little of the 42.5-mm output width for the actual data panels.
  combined <- participant_plot %+% combined_data +
    facet_wrap(~source, nrow = 1) +
    theme(
      aspect.ratio = 1.2,
      strip.background = element_blank(),
      strip.text = element_text(size = 7, color = "black", face = "plain"),
      strip.clip = "off",
      panel.spacing = grid::unit(0.75, "lines"),
      plot.margin = margin(0, 1, 0, 1, unit = "mm")
    )
  ensure_parent_directory(path)
  output_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  ggsave(
    path,
    combined,
    width = 42.5,
    height = 36,
    units = "mm",
    device = output_device,
    bg = "white"
  )
}

summarize_exp3_rank_change_curves <- function(rank_rows) {
  rank_levels <- c("observation_greater", "observation_lower")
  rank_labels <- c(
    "Final rank better than\nencoding rank",
    "Final rank worse than\nencoding rank"
  )
  rank_rows %>%
    filter(
      is.finite(observation_rank),
      is.finite(final_rank),
      observation_rank != final_rank,
      is.finite(node_reward),
      is.finite(accuracy)
    ) %>%
    mutate(
      # Smaller numeric rank is better. Exclude ties before averaging, rather
      # than merely relabeling the previous same-or-better group.
      rank_change = ifelse(
        observation_rank > final_rank, "observation_greater", "observation_lower"
      )
    ) %>%
    group_by(
      source,
      participant_id,
      tree_type,
      tree_order,
      rank_change,
      node_reward
    ) %>%
    summarise(
      participant_accuracy = mean(accuracy),
      observation_count = n(),
      .groups = "drop"
    ) %>%
    group_by(source, tree_type, tree_order, rank_change, node_reward) %>%
    summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        n() > 1L,
        stats::sd(participant_accuracy) / sqrt(n()),
        NA_real_
      ),
      participant_count = n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    mutate(
      source = factor(
        source,
        levels = c("participants", "model"),
        labels = c("Participants", "Model")
      ),
      tree_order = exp3_tree_order_labels(tree_type),
      rank_change_condition = factor(
        rank_change,
        levels = rank_levels,
        labels = rank_labels
      ),
      lower = pmax(0, mean_accuracy - se),
      upper = pmin(1, mean_accuracy + se),
      included_in_plot = participant_count >= 5L
    )
}

plot_exp3_rank_change_curves <- function(rank_summary, path) {
  plotted <- rank_summary %>% filter(included_in_plot)
  if (nrow(plotted) == 0L) {
    fail("No Experiment 3 rank-change curve cells have five participants.")
  }
  rank_colors <- c(
    "Final rank better than\nencoding rank" = "#46585D",
    "Final rank worse than\nencoding rank" = "#6F3934"
  )
  tree_families <- list(
    Deep = c("deep_breadth", "deep_depth"),
    Wide = c("wide_breadth", "wide_depth")
  )
  source_labels <- c("Participants", "Model")

  make_source_group <- function(tree_family, current_source) {
    group_data <- plotted %>%
      filter(
        tree_type %in% tree_families[[tree_family]],
        as.character(source) == current_source
      ) %>%
      mutate(
        plot_order = factor(
          ifelse(endsWith(tree_type, "_breadth"), "BFS", "DFS"),
          levels = c("BFS", "DFS")
        )
      )
    ggplot(
      group_data,
      aes(
        x = node_reward,
        y = mean_accuracy,
        color = rank_change_condition,
        group = rank_change_condition
      )
    ) +
      geom_errorbar(
        aes(ymin = lower, ymax = upper),
        width = 0.35,
        linewidth = 0.25,
        na.rm = TRUE
      ) +
      geom_line(linewidth = 0.55) +
      geom_point(size = 0.45) +
      facet_grid(. ~ plot_order, drop = FALSE) +
      scale_color_manual(values = rank_colors, drop = FALSE) +
      scale_x_continuous(
        breaks = seq(-4, 4, by = 2),
        expand = expansion(add = 0.4)
      ) +
      coord_cartesian(ylim = c(0, 0.7)) +
      labs(
        x = "Probed Reward",
        y = "Recall accuracy",
        color = NULL
      ) +
      guides(color = guide_legend(nrow = 2, byrow = TRUE)) +
      theme_minimal(base_size = 7) +
      theme(
        text = element_text(size = 7, color = "black"),
        axis.title = element_text(size = 7, color = "black"),
        axis.text = element_text(size = 7, color = "black"),
        aspect.ratio = 1.2,
        panel.grid = element_blank(),
        panel.border = element_rect(
          color = "black",
          fill = NA,
          linewidth = 0.35
        ),
        strip.background = element_blank(),
        strip.text = element_text(size = 7, color = "black", face = "plain"),
        strip.clip = "off",
        legend.position = "none",
        panel.spacing = grid::unit(1.5, "mm"),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "mm")
      )
  }

  make_tree_family_row <- function(tree_family) {
    source_groups <- lapply(
      source_labels,
      function(current_source) make_source_group(tree_family, current_source)
    )
    group_row <- cowplot::plot_grid(
      plotlist = source_groups,
      nrow = 1,
      align = "h",
      axis = "tb"
    )
    row_label <- cowplot::ggdraw() +
      cowplot::draw_label(
        tree_family,
        size = 7,
        color = "black",
        angle = 90,
        fontface = "plain"
      )
    cowplot::plot_grid(
      row_label,
      group_row,
      nrow = 1,
      rel_widths = c(0.07, 1)
    )
  }

  legend <- cowplot::get_legend(
    make_source_group("Deep", "Participants") +
      theme(
        legend.position = "top",
        legend.justification = "center",
        legend.text = element_text(size = 7, color = "black"),
        legend.key.height = grid::unit(2.5, "mm"),
        legend.key.width = grid::unit(4, "mm"),
        legend.margin = margin(0, 0, 0, 0, unit = "mm"),
        legend.box.margin = margin(0, 0, 0, 0, unit = "mm")
      )
  )
  source_header <- cowplot::plot_grid(
    plotlist = lapply(
      source_labels,
      function(source_label) {
        cowplot::ggdraw() +
          cowplot::draw_label(
            source_label,
            size = 7,
            color = "black",
            fontface = "plain"
          )
      }
    ),
    nrow = 1
  )
  header_row <- cowplot::plot_grid(
    NULL,
    source_header,
    nrow = 1,
    rel_widths = c(0.07, 1)
  )
  plot <- cowplot::plot_grid(
    NULL,
    legend,
    header_row,
    make_tree_family_row("Deep"),
    make_tree_family_row("Wide"),
    ncol = 1,
    rel_heights = c(5, 10, 5, 30, 30)
  )
  ensure_parent_directory(path)
  output_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  ggsave(
    path,
    plot,
    width = 116,
    height = 80,
    units = "mm",
    device = output_device,
    bg = "white"
  )
}

summarize_exp3_rank_heatmaps <- function(rank_rows) {
  rank_rows %>%
    filter(
      is.finite(accuracy),
      is.finite(observation_rank),
      is.finite(final_rank)
    ) %>%
    group_by(
      source,
      participant_id,
      tree_type,
      tree_order,
      observation_rank,
      final_rank
    ) %>%
    summarise(
      participant_accuracy = mean(accuracy),
      observation_count = n(),
      .groups = "drop"
    ) %>%
    group_by(
      source,
      tree_type,
      tree_order,
      observation_rank,
      final_rank
    ) %>%
    summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        n() > 1L,
        stats::sd(participant_accuracy) / sqrt(n()),
        NA_real_
      ),
      participant_count = n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    mutate(
      source = factor(
        source,
        levels = c("participants", "model"),
        labels = c("Participants", "Model")
      ),
      tree_order = exp3_tree_order_labels(tree_type),
      included_in_plot = participant_count >= 5L,
      mean_accuracy = ifelse(included_in_plot, mean_accuracy, NA_real_),
      se = ifelse(included_in_plot, se, NA_real_)
    )
}

plot_exp3_rank_heatmaps <- function(heatmap_summary, path) {
  if (all(is.na(heatmap_summary$mean_accuracy))) {
    fail("No Experiment 3 rank heatmap cells have five participants.")
  }
  tree_families <- list(
    Deep = c("deep_breadth", "deep_depth"),
    Wide = c("wide_breadth", "wide_depth")
  )
  source_labels <- c("Participants", "Model")

  make_source_group <- function(tree_family, current_source) {
    current_tree_types <- tree_families[[tree_family]]
    path_counts <- vapply(
      current_tree_types,
      function(current_tree_type) length(exp3_tree_paths(current_tree_type)),
      integer(1)
    )
    if (length(unique(path_counts)) != 1L) {
      fail("BFS and DFS path counts differ for the %s tree.", tree_family)
    }
    path_count <- path_counts[[1]]
    group_data <- heatmap_summary %>%
      filter(
        tree_type %in% current_tree_types,
        as.character(source) == current_source
      ) %>%
      mutate(
        tree_order = factor(
          ifelse(endsWith(tree_type, "_breadth"), "BFS", "DFS"),
          levels = c("BFS", "DFS")
        )
      ) %>%
      select(
        tree_order,
        observation_rank,
        final_rank,
        mean_accuracy,
        participant_count,
        observation_count
      ) %>%
      tidyr::complete(
        tree_order,
        observation_rank = seq_len(path_count),
        final_rank = seq_len(path_count),
        fill = list(participant_count = 0L, observation_count = 0L)
      )
    ggplot(
      group_data,
      aes(x = observation_rank, y = final_rank, fill = mean_accuracy)
    ) +
      geom_tile(color = "white", linewidth = 0.08) +
      facet_grid(. ~ tree_order, drop = FALSE) +
      scale_x_continuous(
        breaks = unique(round(seq(1, path_count, length.out = 4)))
      ) +
      scale_y_continuous(
        breaks = unique(round(seq(1, path_count, length.out = 4)))
      ) +
      scale_fill_gradientn(
        colors = c("#F7F3F2", "#B77A73", "#532B27"),
        limits = c(0, 1),
        breaks = c(0, 0.25, 0.5, 0.75, 1),
        na.value = "#000000"
      ) +
      coord_fixed() +
      labs(
        x = "Encoding path rank",
        y = "Final path rank",
        fill = "Recall accuracy"
      ) +
      theme_minimal(base_size = 7) +
      theme(
        text = element_text(size = 7, color = "black"),
        axis.title = element_text(size = 7, color = "black"),
        axis.text = element_text(size = 7, color = "black"),
        panel.grid = element_blank(),
        panel.border = element_rect(
          color = "black",
          fill = NA,
          linewidth = 0.35
        ),
        strip.background = element_blank(),
        strip.text = element_text(size = 7, color = "black", face = "plain"),
        legend.position = "none",
        panel.spacing.x = grid::unit(2, "mm"),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "mm")
      )
  }

  make_tree_family_row <- function(tree_family) {
    source_groups <- lapply(
      source_labels,
      function(current_source) make_source_group(tree_family, current_source)
    )
    group_row <- cowplot::plot_grid(
      plotlist = source_groups,
      nrow = 1,
      align = "h",
      axis = "tb"
    )
    row_label <- cowplot::ggdraw() +
      cowplot::draw_label(
        tree_family,
        size = 7,
        color = "black",
        angle = 90,
        fontface = "plain"
      )
    cowplot::plot_grid(
      row_label,
      group_row,
      nrow = 1,
      rel_widths = c(0.07, 1)
    )
  }

  legend <- cowplot::get_legend(
    make_source_group("Deep", "Participants") +
      guides(
        fill = guide_colorbar(
          direction = "horizontal",
          title.position = "top",
          label.position = "bottom",
          barwidth = grid::unit(20, "mm"),
          barheight = grid::unit(2.5, "mm")
        )
      ) +
      theme(
        legend.position = "top",
        legend.title = element_text(size = 7, color = "black"),
        legend.text = element_text(size = 7, color = "black"),
        legend.margin = margin(0, 0, 0, 0, unit = "mm"),
        legend.box.margin = margin(0, 0, 0, 0, unit = "mm")
      )
  )
  source_header <- cowplot::plot_grid(
    plotlist = lapply(
      source_labels,
      function(source_label) {
        cowplot::ggdraw() +
          cowplot::draw_label(
            source_label,
            size = 7,
            color = "black",
            fontface = "plain"
          )
      }
    ),
    nrow = 1
  )
  header_row <- cowplot::plot_grid(
    NULL,
    source_header,
    nrow = 1,
    rel_widths = c(0.07, 1)
  )
  combined <- cowplot::plot_grid(
    legend,
    header_row,
    make_tree_family_row("Deep"),
    make_tree_family_row("Wide"),
    ncol = 1,
    rel_heights = c(9, 5, 30, 30)
  )
  ensure_parent_directory(path)
  output_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  ggsave(
    path,
    combined,
    width = 116,
    height = 74,
    units = "mm",
    device = output_device,
    bg = "white"
  )
}

summarize_exp3_reward_step_rank_difference_heatmaps <- function(rank_rows) {
  rank_direction_levels <- c("encoding_better", "encoding_worse")
  source_levels <- c("participants", "model")
  source_labels <- c("Participants", "Model")

  classified <- rank_rows %>%
    filter(
      source %in% source_levels,
      tree_type %in% TREE_TYPES,
      is.finite(node_reward),
      is.finite(node),
      is.finite(accuracy),
      is.finite(observation_rank),
      is.finite(final_rank)
    ) %>%
    mutate(
      rank_direction = case_when(
        observation_rank < final_rank ~ "encoding_better",
        observation_rank > final_rank ~ "encoding_worse",
        TRUE ~ NA_character_
      )
    ) %>%
    filter(!is.na(rank_direction))
  if (nrow(classified) == 0L) {
    fail(
      paste0(
        "No strict encoding-versus-final rank changes were available for ",
        "the Experiment 3 reward-by-step difference heatmaps."
      )
    )
  }

  summarized_cells <- classified %>%
    group_by(
      source,
      participant_id,
      tree_type,
      rank_direction,
      node_reward,
      observation_step = node
    ) %>%
    summarise(
      participant_accuracy = mean(accuracy),
      observation_count = n(),
      .groups = "drop"
    ) %>%
    group_by(
      source,
      tree_type,
      rank_direction,
      node_reward,
      observation_step
    ) %>%
    summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        n() > 1L,
        stats::sd(participant_accuracy) / sqrt(n()),
        NA_real_
      ),
      participant_count = n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(
      names_from = rank_direction,
      values_from = c(
        mean_accuracy,
        se,
        participant_count,
        observation_count
      ),
      names_sep = "_"
    ) %>%
    filter(
      is.finite(mean_accuracy_encoding_better),
      is.finite(mean_accuracy_encoding_worse)
    ) %>%
    mutate(
      mean_accuracy_difference = mean_accuracy_encoding_worse -
        mean_accuracy_encoding_better,
      se = sqrt(se_encoding_better^2 + se_encoding_worse^2),
      participant_count = pmin(
        participant_count_encoding_better,
        participant_count_encoding_worse
      ),
      observation_count = observation_count_encoding_better +
        observation_count_encoding_worse
    )
  if (nrow(summarized_cells) == 0L) {
    fail(
      paste0(
        "No Experiment 3 reward-by-observation-step cell contained both ",
        "rank directions."
      )
    )
  }

  reward_levels <- sort(unique(classified$node_reward))
  observation_step_levels <- seq_len(29L)
  tidyr::expand_grid(
    source = source_levels,
    tree_type = TREE_TYPES,
    node_reward = reward_levels,
    observation_step = observation_step_levels
  ) %>%
    left_join(
      summarized_cells,
      by = c(
        "source",
        "tree_type",
        "node_reward",
        "observation_step"
      )
    ) %>%
    mutate(
      participant_count = tidyr::replace_na(participant_count, 0L),
      observation_count = tidyr::replace_na(observation_count, 0L),
      mean_accuracy_difference = ifelse(
        participant_count >= 5L,
        mean_accuracy_difference,
        NA_real_
      ),
      se = ifelse(participant_count >= 5L, se, NA_real_),
      included_in_plot = participant_count >= 5L,
      source = factor(
        source,
        levels = source_levels,
        labels = source_labels
      ),
      tree_order = exp3_tree_order_labels(tree_type),
      probed_reward = factor(node_reward, levels = reward_levels),
      observation_step_factor = factor(
        observation_step,
        levels = observation_step_levels
      )
    )
}

plot_exp3_reward_step_rank_difference_heatmaps <- function(
    heatmap_summary,
    path,
    panel_size_mm = 21) {
  if (all(is.na(heatmap_summary$mean_accuracy_difference))) {
    fail(
      paste0(
        "No Experiment 3 paired reward-by-observation-step difference ",
        "cells contained five participants."
      )
    )
  }
  difference_limit <- max(
    abs(heatmap_summary$mean_accuracy_difference),
    na.rm = TRUE
  )
  difference_limit <- max(0.05, min(1, difference_limit))
  tree_families <- list(
    Deep = c("deep_breadth", "deep_depth"),
    Wide = c("wide_breadth", "wide_depth")
  )
  source_labels <- c("Participants", "Model")

  make_source_group <- function(tree_family, current_source) {
    group_data <- heatmap_summary %>%
      filter(
        tree_type %in% tree_families[[tree_family]],
        as.character(source) == current_source
      ) %>%
      mutate(
        plot_order = factor(
          ifelse(endsWith(tree_type, "_breadth"), "BFS", "DFS"),
          levels = c("BFS", "DFS")
        )
      )
    ggplot(
      group_data,
      aes(
        x = probed_reward,
        y = observation_step_factor,
        fill = mean_accuracy_difference
      )
    ) +
      geom_tile(color = "white", linewidth = 0.08) +
      facet_grid(. ~ plot_order, drop = FALSE) +
      scale_fill_gradient2(
        low = "#532B27",
        mid = "#F7F3F2",
        high = "#273F3F",
        midpoint = 0,
        limits = c(-difference_limit, difference_limit),
        breaks = c(-difference_limit, 0, difference_limit),
        labels = scales::label_number(accuracy = 0.01),
        na.value = "#000000"
      ) +
      scale_x_discrete(drop = FALSE) +
      scale_y_discrete(
        breaks = as.character(c(1L, 5L, 10L, 15L, 20L, 25L, 29L)),
        drop = FALSE
      ) +
      labs(
        x = "Probed reward",
        y = "Observation step",
        fill = "Accuracy difference\n(worse - better)"
      ) +
      theme_minimal(base_size = 7) +
      theme(
        text = element_text(size = 7, color = "black"),
        axis.title = element_text(size = 7, color = "black"),
        axis.text = element_text(size = 5.6, color = "black"),
        axis.ticks = element_line(color = "black", linewidth = 0.3),
        axis.ticks.length = grid::unit(0.8, "mm"),
        aspect.ratio = 1,
        panel.grid = element_blank(),
        panel.border = element_rect(
          color = "black",
          fill = NA,
          linewidth = 0.35
        ),
        panel.spacing = grid::unit(1.5, "mm"),
        strip.background = element_blank(),
        strip.text = element_text(
          size = 5.6,
          color = "black",
          face = "plain"
        ),
        strip.clip = "off",
        legend.position = "none",
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "mm")
      )
  }

  make_tree_family_row <- function(tree_family) {
    source_groups <- lapply(
      source_labels,
      function(current_source) make_source_group(tree_family, current_source)
    )
    group_row <- cowplot::plot_grid(
      plotlist = source_groups,
      nrow = 1,
      align = "h",
      axis = "tb"
    )
    row_label <- cowplot::ggdraw() +
      cowplot::draw_label(
        tree_family,
        size = 7,
        color = "black",
        angle = 90,
        fontface = "plain"
      )
    cowplot::plot_grid(
      row_label,
      group_row,
      nrow = 1,
      rel_widths = c(0.07, 1)
    )
  }

  legend <- cowplot::get_legend(
    make_source_group("Deep", "Participants") +
      guides(
        fill = guide_colorbar(
          direction = "horizontal",
          title.position = "top",
          title.hjust = 0.5,
          label.position = "bottom",
          barwidth = grid::unit(20, "mm"),
          barheight = grid::unit(2, "mm")
        )
      ) +
      theme(
        legend.position = "top",
        legend.justification = "center",
        legend.title = element_text(size = 5.6, color = "black"),
        legend.text = element_text(size = 5.6, color = "black"),
        legend.margin = margin(0, 0, 0, 0, unit = "mm"),
        legend.box.margin = margin(0, 0, 0, 0, unit = "mm"),
        legend.box.spacing = grid::unit(0, "mm")
      )
  )
  source_header <- cowplot::plot_grid(
    plotlist = lapply(
      source_labels,
      function(source_label) {
        cowplot::ggdraw() +
          cowplot::draw_label(
            source_label,
            size = 7,
            color = "black",
            fontface = "plain"
          )
      }
    ),
    nrow = 1
  )
  header_row <- cowplot::plot_grid(
    NULL,
    source_header,
    nrow = 1,
    rel_widths = c(0.07, 1)
  )
  combined <- cowplot::plot_grid(
    NULL,
    legend,
    header_row,
    make_tree_family_row("Deep"),
    make_tree_family_row("Wide"),
    ncol = 1,
    rel_heights = c(5, 10, 5, 30, 30)
  )

  ensure_parent_directory(path)
  output_device <- if (capabilities("cairo")) {
    grDevices::cairo_pdf
  } else {
    grDevices::pdf
  }
  ggsave(
    path,
    combined,
    width = 116,
    height = 80,
    units = "mm",
    device = output_device,
    bg = "white"
  )
}

exp3_output_label <- function(fitted_simulation_path) {
  file_name <- canonical_simulation_basename(fitted_simulation_path)
  configuration_hash <- sub(
    ".*_cfg_([[:xdigit:]]+)_.*",
    "\\1",
    file_name
  )
  run_identity <- regexec(
    "_n([0-9]+)_per_seed_tree_simseed([0-9]+)[.]csv$",
    file_name
  )
  identity_match <- regmatches(file_name, run_identity)[[1]]
  if (
    identical(configuration_hash, file_name) ||
      length(identity_match) != 3L
  ) {
    safe_stem <- gsub(
      "[^A-Za-z0-9_-]",
      "_",
      tools::file_path_sans_ext(file_name)
    )
    return(paste0("exp3_preregistered_fitted_", safe_stem))
  }
  paste0(
    "exp3_preregistered_fitted_cfg_",
    configuration_hash,
    "_n",
    identity_match[[2]],
    "_per_seed_tree_simseed",
    identity_match[[3]]
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0L || length(args) > 4L) {
  usage()
  quit(status = ifelse(length(args) == 0L, 0L, 2L))
}

fit_path <- .path_absolute(args[[1]], REPO_ROOT)
behavior_path <- if (length(args) >= 2L) {
  .path_absolute(args[[2]], REPO_ROOT)
} else {
  project_behavior_path("exp3")
}
simulation_seeds <- parse_seed_list(
  if (length(args) >= 3L) args[[3]] else "1,2,3,4,5"
)
fitted_simulation_path <- if (length(args) >= 4L) {
  .path_absolute(args[[4]], REPO_ROOT)
} else {
  NULL
}
require_existing_path(fit_path, "31-node participant-fit CSV")
require_existing_path(behavior_path, "Processed Experiment 3 behavior")
if (!is.null(fitted_simulation_path)) {
  require_existing_path(
    fitted_simulation_path,
    "31-node participant-fitted simulation CSV"
  )
}

fits <- readr::read_csv(
  fit_path,
  show_col_types = FALSE,
  progress = FALSE
)
if (!is.null(fitted_simulation_path)) {
  routing <- fits %>%
    transmute(subject = as.character(subject))
  fitted_outputs <- prepare_fitted_simulation_summary(
    fitted_simulation_path,
    fits,
    simulation_seeds
  )
  model_participants <- fitted_outputs$participant_summary
  model_rank_rows <- prepare_exp3_model_rank_rows(
    fitted_outputs$recall_rows
  )
} else {
  routing <- parse_fit_routing(fits, simulation_seeds)
  unique_simulation_files <- sort(unique(routing$simulation_file))
  cat(sprintf(
    "Summarizing %d legacy grid simulation files.\n",
    length(unique_simulation_files)
  ))
  simulation_summaries <- purrr::map_dfr(
    unique_simulation_files,
    function(path) {
      cat(sprintf("Reading grid simulation: %s\n", path))
      flush.console()
      summarize_simulation_file(path)
    }
  )
  model_participants <- routing %>%
    left_join(
      simulation_summaries,
      by = "simulation_file",
      relationship = "many-to-one"
    ) %>%
    group_by(subject, best_lambda, best_alpha, nll_model_seed, tree_type) %>%
    summarise(
      accuracy = mean(accuracy),
      norm_reward = mean(norm_reward),
      simulation_seed_count = n_distinct(simulation_model_seed),
      .groups = "drop"
    ) %>%
    tree_labels()
  if (
    any(!is.finite(model_participants$accuracy)) ||
      any(!is.finite(model_participants$norm_reward))
  ) {
    fail("At least one fitted participant simulation summary is missing.")
  }
  model_rank_rows <- NULL
}
behavior_outputs <- prepare_behavior_summary(behavior_path)
behavior_participants <- behavior_outputs$participant_summary
human_rank_rows <- prepare_exp3_human_rank_rows(behavior_outputs$recall_rows)

output_label <- if (!is.null(fitted_simulation_path)) {
  exp3_output_label(fitted_simulation_path)
} else {
  "exp3_preregistered_fitted_legacy_grid"
}
output_directory <- file.path(ANALYSIS_PLOT_DIR, "tests", output_label)
parameter_performance_path <- plot_exp3_fitted_parameter_performance(
  fits, behavior_outputs, output_directory
)
recovery_plot_path <- plot_fitted_matched_seed_recovery(
  tree_size = 31L,
  output_directory = output_directory
)
output_paths <- c(
  fitted_parameter_performance = parameter_performance_path,
  matched_seed_recovery = recovery_plot_path,
  accuracy_beh_sim = file.path(output_directory, "accuracy_beh_sim.pdf"),
  norm_reward_beh_sim = file.path(output_directory, "norm_reward_beh_sim.pdf")
)

accuracy_behavior_plot <- interaction_plot(
  behavior_participants,
  "accuracy",
  "Recall\naccuracy",
  c(0.1, 0.25),
  seq(0.1, 0.25, by = 0.05)
)
accuracy_model_plot <- interaction_plot(
  model_participants,
  "accuracy",
  "Recall\naccuracy",
  c(0.1, 0.25),
  seq(0.1, 0.25, by = 0.05)
)
reward_behavior_plot <- interaction_plot(
  behavior_participants,
  "norm_reward",
  "Normalized\nReward",
  c(0.8, 0.95),
  seq(0.8, 0.95, by = 0.05)
)
reward_model_plot <- interaction_plot(
  model_participants,
  "norm_reward",
  "Normalized\nReward",
  c(0.8, 0.95),
  seq(0.8, 0.95, by = 0.05)
)

save_combined_interaction_plot(
  accuracy_behavior_plot,
  accuracy_model_plot,
  output_paths[["accuracy_beh_sim"]]
)
save_combined_interaction_plot(
  reward_behavior_plot,
  reward_model_plot,
  output_paths[["norm_reward_beh_sim"]]
)

rank_summary_paths <- character()
if (!is.null(model_rank_rows)) {
  combined_rank_rows <- bind_rows(human_rank_rows, model_rank_rows)
  rank_curve_summary <- summarize_exp3_rank_change_curves(combined_rank_rows)
  rank_plot_paths <- c(node_reward_rank = file.path(
    output_directory, "node_reward_accuracy_by_observation_final_rank.pdf"))
  rank_summary_paths <- c(node_reward_rank = file.path(
    ANALYSIS_SUMMARY_DIR, "tests", output_label,
    "node_reward_accuracy_by_observation_final_rank.csv"))
  plot_exp3_rank_change_curves(rank_curve_summary, rank_plot_paths[["node_reward_rank"]])
  purrr::walk(rank_summary_paths, ensure_parent_directory)
  readr::write_csv(rank_curve_summary, rank_summary_paths[["node_reward_rank"]])
  output_paths <- c(output_paths, rank_plot_paths)
}

cat(sprintf("Participant-fit input: %s\n", fit_path))
cat(sprintf("Behavioral input: %s\n", behavior_path))
if (!is.null(fitted_simulation_path)) {
  cat(sprintf("Fitted simulation input: %s\n", fitted_simulation_path))
}
cat(sprintf("Fitted participants: %d\n", n_distinct(routing$subject)))
cat(sprintf(
  "Simulation model seeds: %s\n",
  paste(simulation_seeds, collapse = ",")
))
for (path in output_paths) {
  cat(sprintf("Saved Experiment 3 interaction plot to: %s\n", path))
}
for (path in rank_summary_paths) {
  cat(sprintf("Saved Experiment 3 rank summary to: %s\n", path))
}
print_local_scp_command(
  output_directory,
  file.path("analysis", "figures", "tests"),
  recursive = TRUE
)
