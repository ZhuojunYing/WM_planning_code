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

# Torch manuscript statistics. This file defines functions; sourcing it runs no fits.
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(here)
  library(lmerTest)
})

fail <- function(...) stop(sprintf(...), call. = FALSE)
require_columns <- function(data, columns, label) {
  absent <- setdiff(columns, names(data))
  if (length(absent)) fail("%s is missing: %s", label, paste(absent, collapse = ", "))
}

# Import only explicitly named definitions. The legacy figure/test entry points
# execute analyses at top level, so they must NOT be sourced wholesale.
import_definitions <- function(path, names, envir = parent.frame()) {
  expressions <- parse(path)
  found <- character()
  for (expr in expressions) {
    if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
        is.symbol(expr[[2]]) && as.character(expr[[2]]) %in% names) {
      eval(expr, envir = envir)
      found <- c(found, as.character(expr[[2]]))
    }
  }
  if (!setequal(found, names)) fail("Missing imported definitions in %s: %s",
    path, paste(setdiff(names, found), collapse = ", "))
}

import_definitions(project_path("analysis/scripts/figures/exp1.r"),
  c("default_main_simulation_tag", "default_exposure_simulation_tag",
    "default_independent_simulation_tag", "default_simultaneous_simulation_tag"))
import_definitions(project_path("analysis/scripts/figures/exp2.r"),
  c("default_exp2_main_simulation_tag", "default_exp2_exposure_simulation_tag",
    "default_exp2_independent_simulation_tag", "default_exp2_simultaneous_simulation_tag"))
import_definitions(project_path("analysis/scripts/figures/exp1.r"),
  c("children_to_edges", "experiment_edges", "tree_path_context", "experiment_path_context"))
import_definitions(project_path("analysis/scripts/figures/plot_utils_fitted_simulation.R"),
  c("fitted_competition_rank_matrix", "fitted_observation_final_rank_lookup"))

default_exp3_stats_simulation_tag <- paste0(
  "31_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_0.5_head_nonlin_cfg_a9fcce63bb09_",
  "radient_scale_1_final_lr_0.0003_lr_plateau_200_300_",
  "eb_0.05_rns_rho_grid_n25_per_seed_tree_simseed2027"
)

stats_inputs <- function(experiment, root) {
  tags <- switch(experiment,
    exp1 = c(sequential = default_main_simulation_tag,
      exposure = default_exposure_simulation_tag,
      simultaneous = default_simultaneous_simulation_tag,
      independent = default_independent_simulation_tag),
    exp2 = c(sequential = default_exp2_main_simulation_tag,
      exposure = default_exp2_exposure_simulation_tag,
      simultaneous = default_exp2_simultaneous_simulation_tag,
      independent = default_exp2_independent_simulation_tag),
    exp3 = c(sequential = default_exp3_stats_simulation_tag),
    fail("Unknown experiment: %s", experiment))
  c(participants = if (identical(normalizePath(root, mustWork = FALSE), STORAGE_ROOT)) project_behavior_path(experiment) else
      file.path(root, "data/analysis/processed_data", paste0(experiment, if (experiment == "exp3") "_beh.csv" else "_beh_new.csv")),
    setNames(vapply(paste0("participant_fitted_simulations_", tags, ".csv"),
      simulation_input_path, character(1), root = root), names(tags)))
}

parse_stats_options <- function(args, experiment) {
  options <- list(root = STORAGE_ROOT, output = NULL,
    sources = if (experiment == "exp3") c("participants", "sequential") else
      c("participants", "sequential", "simultaneous", "independent"),
    tests = c("descriptive", "parameter_performance",
      if (experiment == "exp3") c("rank_change", "tree_order") else
        c("context", "path_rank", "branch_timing", "observation_timestep", "nll"),
      if (experiment == "exp2") "best_branch_timing"),
    check = FALSE)
  while (length(args)) {
    flag <- args[[1]]
    if (flag == "--check-inputs") {
      options$check <- TRUE; args <- args[-1]; next
    }
    if (flag == "--help") {
      cat("Options: --source participants,sequential,simultaneous,independent\n",
        "--tests descriptive,context,path_rank,rank_change,branch_timing,best_branch_timing,tree_order,observation_timestep,memory_allocation,parameter_performance,nll\n",
        "--storage-root PATH --output-dir PATH --check-inputs\n")
      return(NULL)
    }
    if (!flag %in% c("--source", "--tests", "--storage-root", "--output-dir") || length(args) < 2)
      fail("Unknown option or missing value: %s", flag)
    value <- args[[2]]
    if (flag == "--source") options$sources <- strsplit(value, ",", fixed = TRUE)[[1]]
    if (flag == "--tests") options$tests <- strsplit(value, ",", fixed = TRUE)[[1]]
    if (flag == "--storage-root") options$root <- value
    if (flag == "--output-dir") options$output <- value
    args <- args[-c(1, 2)]
  }
  allowed <- if (experiment == "exp3") c("participants", "sequential") else
    c("participants", "sequential", "simultaneous", "independent")
  if (any(!options$sources %in% allowed)) fail("Invalid source for %s", experiment)
  if (any(!options$tests %in% c("descriptive", "context", "path_rank", "rank_change", "branch_timing", "best_branch_timing", "tree_order", "observation_timestep", "memory_allocation", "parameter_performance", "nll")))
    fail("Unknown test selection")
  if (experiment != "exp2" && "best_branch_timing" %in% options$tests)
    fail("best_branch_timing is defined only for Exp2")
  if (experiment == "exp3" && "observation_timestep" %in% options$tests)
    fail("The observation_timestep analysis is defined only for Exp1 and Exp2")
  if (experiment == "exp3" && "nll" %in% options$tests)
    fail("The NLL model comparisons are defined only for Exp1 and Exp2")
  if (is.null(options$output)) options$output <- STATS_ROOT
  options
}

# Construct rewards once per trial/layout, not once per recall probe. For
# simulations all seeds retain the fitted participant as the grouping factor.
sample_stats_trials <- function(raw, keys, trials_per_condition = 15L, seed = 2028L) {
  groups <- c("subject", "condition", "tree_type")
  trials <- raw %>% distinct(across(all_of(keys))) %>% arrange(across(all_of(keys)))
  counts <- trials %>% count(across(all_of(groups)))
  if (any(counts$n < trials_per_condition))
    fail("Cannot sample %d trials: some participant/tree/order conditions have fewer trials", trials_per_condition)
  # Preserve the caller's RNG state and make selection independent of CSV row order.
  had_rng <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_rng) old_rng <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_rng) assign(".Random.seed", old_rng, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  })
  set.seed(seed)
  selected <- trials %>% group_by(across(all_of(groups))) %>%
    slice_sample(n = trials_per_condition, replace = FALSE) %>% ungroup()
  list(data = semi_join(raw, selected, by = keys) %>% arrange(across(all_of(c(keys, "node")))), selected = selected)
}

sample_stats_nodes <- function(data, nodes_per_trial = 2L, seed = 2029L) {
  # Context and rank must already have been computed from all 30 node rewards.
  eligible <- data %>% filter(node < 30L) %>% arrange(participant, trial_id, node)
  counts <- eligible %>% count(participant, trial_id)
  if (any(counts$n < nodes_per_trial) || nrow(counts) != n_distinct(data$trial_id))
    fail("Too few eligible non-final nodes in a sampled simulation trial")
  had_rng <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_rng) old_rng <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (had_rng) assign(".Random.seed", old_rng, envir = .GlobalEnv)
    else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  })
  set.seed(seed)
  eligible %>% group_by(participant, trial_id) %>%
    slice_sample(n = nodes_per_trial, replace = FALSE) %>% ungroup()
}

read_stats_data <- function(path, experiment, source, condition = NULL,
                            sample_trials_per_condition = NULL, sample_seed = 2028L,
                            sampling_directory = NULL, sample_nodes_per_trial = NULL) {
  message("Reading ", source, ": ", path)
  n <- switch(experiment, exp1 = 6L, exp2 = 12L, exp3 = 30L)
  rewards <- paste0("n", seq_len(n), "_reward")
  needed <- c("subject", "graph", "condition", "tree_type", "node", "accuracy",
    "actual_reward", "estimated_reward", "simulation_model_seed", "simulation_trial",
    "norm_reward", "V", "chosen_path_reward", "min_path_reward", "max_path_reward", rewards)
  raw <- read_csv(path, col_select = any_of(needed), show_col_types = FALSE, progress = FALSE)
  require_columns(raw, c("subject", "graph", "node"), path)
  raw$subject <- as.character(raw$subject)
  raw$graph <- as.character(raw$graph)
  raw$node <- as.integer(raw$node)
  if (experiment != "exp3") raw$tree_type <- "standard"
  require_columns(raw, "tree_type", path)
  if (!is.null(condition)) raw$condition <- condition
  require_columns(raw, "condition", path)
  raw$condition <- tolower(raw$condition)
  raw <- filter(raw, condition %in% c("planning", "exposure"), node %in% seq_len(n))
  if (source != "participants") {
    require_columns(raw, c("simulation_model_seed", "actual_reward", "estimated_reward"), path)
    raw$accuracy <- as.numeric(raw$estimated_reward == raw$actual_reward)
  }
  require_columns(raw, "accuracy", path)
  if (any(!is.na(raw$accuracy) & !raw$accuracy %in% c(0, 1))) fail("Non-binary recall accuracy: %s", path)
  if (anyNA(raw$subject) || anyNA(raw$node)) fail("Missing participant/node identity: %s", path)
  keys <- c("subject", "condition", "tree_type", "graph")
  if (source != "participants") {
    keys <- c(keys, "simulation_model_seed", intersect("simulation_trial", names(raw)))
    if (anyDuplicated(raw[c(keys, "node")])) fail("Duplicate simulated nodes in %s", path)
    sizes <- raw %>% count(across(all_of(keys)))
    if (any(sizes$n != n)) fail("Incomplete simulated trial in %s", path)
    if (!is.null(sample_trials_per_condition)) {
      if (experiment != "exp3") fail("Trial subsampling is only enabled for Exp3 simulations")
      sampled <- sample_stats_trials(raw, keys, sample_trials_per_condition, sample_seed)
      message("Sampling Exp3 simulations: ", sample_trials_per_condition,
        " trials per participant/tree/order, pooled across model seeds; seed=", sample_seed,
        " | trials=", nrow(sampled$selected), " | nodes=", nrow(sampled$data))
      if (!is.null(sampling_directory)) {
        dir.create(sampling_directory, recursive = TRUE, showWarnings = FALSE)
        write_csv(sampled$selected, file.path(sampling_directory, "sampled_trials.csv"))
        writeLines(c(paste("Input:", path), paste("Sampling seed:", sample_seed),
          paste("Trials per participant/tree/order:", sample_trials_per_condition),
          "Uniform sampling without replacement from trials pooled across model seeds; all nodes retained.",
          paste("Selected trials:", nrow(sampled$selected)), paste("Selected nodes:", nrow(sampled$data))),
          file.path(sampling_directory, "sampling.txt"))
      }
      raw <- sampled$data
      rm(sampled)
    }
    wide <- raw %>% select(all_of(keys), node, actual_reward) %>%
      pivot_wider(names_from = node, values_from = actual_reward, names_prefix = "n")
    names(wide)[match(paste0("n", seq_len(n)), names(wide))] <- rewards
    raw <- select(raw, -any_of(rewards)) %>% left_join(wide, by = keys, relationship = "many-to-one")
  }
  require_columns(raw, rewards, path)
  if (any(!is.finite(as.matrix(raw[rewards])))) fail("Missing/non-finite trial rewards in %s", path)
  # Reward layouts also disambiguate reused human graph IDs across blocks.
  layouts <- raw %>% distinct(tree_type, across(all_of(rewards))) %>% mutate(.rank_trial_id = row_number())
  raw <- left_join(raw, layouts, by = c("tree_type", rewards), relationship = "many-to-one")
  trial_keys <- c(keys, ".rank_trial_id")
  trials <- raw %>% distinct(across(all_of(trial_keys))) %>% mutate(trial_id = row_number())
  raw <- left_join(raw, trials, by = trial_keys, relationship = "many-to-one")
  if (!"norm_reward" %in% names(raw)) {
    # Simulation CSVs use V, as in the plotting readers; also accept the
    # descriptive alias used by some exported datasets.
    chosen <- if (source == "participants" || !"chosen_path_reward" %in% names(raw)) "V" else "chosen_path_reward"
    require_columns(raw, c(chosen, "min_path_reward", "max_path_reward"), path)
    width <- raw$max_path_reward - raw$min_path_reward
    raw$norm_reward <- ifelse(width > 0, (raw[[chosen]] - raw$min_path_reward) / width, NA_real_)
  }
  observations <- raw %>% transmute(participant = factor(subject), trial_id,
    condition = factor(condition, levels = c("planning", "exposure")),
    tree_type, .rank_trial_id, node, accuracy, norm_reward)
  rm(raw) # large wide rewards are no longer needed
  if (exists("wide", inherits = FALSE)) rm(wide)
  contexts <- experiment_path_context(experiment_edges(experiment))
  enriched <- bind_rows(lapply(names(contexts), function(type) {
    context <- contexts[[type]]
    type_layouts <- filter(layouts, tree_type == type)
    if (!nrow(type_layouts)) return(NULL)
    lookup <- fitted_observation_final_rank_lookup(type_layouts, rewards,
      context$path_nodes, path_reward_summary = "max", include_context_rewards = TRUE)
    # For a parent, use the best complete path that contains it, matching plots.
    parent_nodes <- unique(unlist(lapply(context$path_nodes, function(p) head(p, -1))))
    lookup$is_leaf <- !lookup$node %in% parent_nodes
    matrix_rewards <- as.matrix(type_layouts[rewards])
    path_values <- matrix_rewards %*% t(context$path_incidence)
    branch <- bind_rows(lapply(seq_len(n), function(node) {
      own <- context$node_branch[[node]]
      other <- which(context$path_branch != own)
      values <- path_values[, other, drop = FALSE]
      best <- apply(values, 1, max)
      tied <- values == best
      before <- context$path_branch[other] < own
      earlier <- rowSums(tied[, before, drop = FALSE]) > 0
      later <- rowSums(tied[, !before, drop = FALSE]) > 0
      tibble(.rank_trial_id = type_layouts$.rank_trial_id, node,
        context_reward = best,
        relative_branch_timing = ifelse(earlier & later, NA_character_,
          ifelse(earlier, "before", "after")))
    }))
    lookup <- left_join(lookup, branch, by = c(".rank_trial_id", "node"), relationship = "one-to-one")
    lookup$node_reward <- matrix_rewards[cbind(match(lookup$.rank_trial_id, type_layouts$.rank_trial_id), lookup$node)]
    filter(observations, tree_type == type) %>%
      left_join(lookup, by = c(".rank_trial_id", "node"), relationship = "many-to-one")
  }))
  if (nrow(enriched) != nrow(observations) || anyNA(enriched$final_rank)) fail("Unmapped tree/rank rows in %s", path)
  if (!is.null(sample_nodes_per_trial)) {
    if (experiment != "exp3" || source == "participants") fail("Node subsampling is only enabled for Exp3 simulations")
    enriched <- sample_stats_nodes(enriched, sample_nodes_per_trial)
    message("Sampling Exp3 nodes: ", sample_nodes_per_trial,
      " non-final nodes per trial; seed=2029 | analysis rows=", nrow(enriched))
    if (!is.null(sampling_directory)) {
      dir.create(sampling_directory, recursive = TRUE, showWarnings = FALSE)
      audit <- enriched %>% select(trial_id, node) %>%
        left_join(trials, by = "trial_id", relationship = "many-to-one")
      write_csv(audit, file.path(sampling_directory, "sampled_nodes.csv"))
      writeLines(c("Node sampling seed: 2029", paste("Nodes per trial:", sample_nodes_per_trial),
        "Uniform sampling without replacement among observation steps 1--29; final observation excluded.",
        "All rank/context predictors were calculated from the full trial before node selection.",
        paste("Selected observations:", nrow(enriched))), file.path(sampling_directory, "node_sampling.txt"))
    }
  }
  if (experiment == "exp3" && source != "participants" && is.null(sample_nodes_per_trial) &&
      !is.null(sample_trials_per_condition)) {
    message("Retaining all 30 non-root nodes per sampled Exp3 trial; analysis rows=", nrow(enriched))
    if (!is.null(sampling_directory)) {
      # Refresh the old two-node audit files as well, so they cannot look current.
      dir.create(sampling_directory, recursive = TRUE, showWarnings = FALSE)
      write_csv(enriched %>% select(trial_id, node) %>% left_join(trials, by = "trial_id", relationship = "many-to-one"),
        file.path(sampling_directory, "sampled_nodes.csv"))
      writeLines(c("Node sampling: none; all 30 non-root nodes retained per selected trial.",
        "Rank-change analysis subsequently excludes equal ranks and the final observation.",
        paste("Selected observations:", nrow(enriched))), file.path(sampling_directory, "node_sampling.txt"))
    }
  }
  enriched %>% transmute(participant, trial_id, condition, tree_type, accuracy, norm_reward,
    observation_step = node, node_reward, encoding_rank = observation_rank, final_rank,
    path_reward = .rank_path_reward,
    current_branch_best_reward = .rank_current_branch_best_reward,
    current_branch_worst_reward = .rank_current_branch_worst_reward,
    max_other_branch_reward = .rank_max_other_branch_reward,
    min_other_branch_reward = .rank_min_other_branch_reward,
    other_branch_timing = .rank_other_branch_timing,
    worst_other_branch_timing = .rank_worst_other_branch_timing,
    sibling_reward = .rank_sibling_reward, parent_reward = .rank_parent_reward,
    aunt_reward = .rank_aunt_reward, max_child_reward = .rank_child_reward,
    avg_niece_reward = .rank_avg_niece_reward, is_leaf = factor(is_leaf, levels = c(FALSE, TRUE)),
    context_reward, relative_branch_timing = factor(relative_branch_timing, levels = c("after", "before")),
    tree = factor(ifelse(startsWith(tree_type, "wide"), "wide", "deep"), levels = c("deep", "wide")),
    order = factor(ifelse(endsWith(tree_type, "depth"), "DFS", "BFS"), levels = c("BFS", "DFS")))
}

continuous_covariates <- c("observation_step", "abs_node_reward", "node_reward",
  "sibling_reward", "parent_reward", "aunt_reward", "max_child_reward", "avg_niece_reward",
  "preceding_sum_reward")
context_predictors <- c("node_reward_z", "node_reward_squared_z", "sibling_reward_z",
  "sibling_reward_squared_z", "parent_reward_z", "aunt_reward_z")
rank_reward_predictors <- c("node_reward_z", "sibling_reward_z", "parent_reward_z",
  "aunt_reward_z", "max_child_reward_z", "avg_niece_reward_z")
# run_stats_model gives every expanded fixed-effect contrast its own independent
# participant slope, including timestep-by-reward and Exp3 tree/order interactions.
rank_predictors <- c("rank_condition", paste0("observation_step_z * (",
  paste(rank_reward_predictors, collapse = " + "), ")"))

standardize_stats <- function(data) {
  if ("node_reward" %in% names(data)) {
    data$abs_node_reward <- abs(data$node_reward)
    data$node_reward_squared <- data$node_reward^2
  }
  if ("sibling_reward" %in% names(data)) data$sibling_reward_squared <- data$sibling_reward^2
  vars <- intersect(c(continuous_covariates, "node_reward_squared", "sibling_reward_squared", "context_reward"), names(data))
  scaling <- bind_rows(lapply(vars, function(v) {
    x <- data[[v]]
    center <- mean(x, na.rm = TRUE); spread <- sd(x, na.rm = TRUE)
    # A structurally constant covariate is recorded, not used to drop all rows.
    data[[paste0(v, "_z")]] <<- if (is.finite(spread) && spread > 0) (x - center) / spread else rep(0, length(x))
    tibble(variable = v, mean = center, sd = spread)
  }))
  attr(data, "scaling") <- scaling
  data
}

rank_stats_data <- function(data, experiment) {
  n <- switch(experiment, exp1 = 6L, exp2 = 12L, exp3 = 30L)
  data <- filter(data, condition == "planning", observation_step < n,
    is.finite(encoding_rank), is.finite(final_rank), is.finite(accuracy),
    encoding_rank != final_rank)
  if (experiment == "exp3") {
    data$rank_condition <- factor(ifelse(data$final_rank > data$encoding_rank, "worse", "better"),
      levels = c("better", "worse"))
  } else {
    data$rank_condition <- factor(ifelse(data$final_rank < data$encoding_rank, "better", "worse"),
      levels = c("worse", "better"))
  }
  data
}

# Numeric columns for every random-design contrast avoid lme4's correlated
# factor-column expansion of ||. Every slope (including interactions) gets its
# own independent variance, exactly the requested uncorrelated structure.
independent_random_formula <- function(data, response, rhs) {
  fixed <- as.formula(paste(response, "~", rhs))
  design <- model.matrix(as.formula(paste("~", rhs)), data)
  slopes <- setdiff(colnames(design), "(Intercept)")
  random <- "(1 | participant)"
  mapping <- tibble(random_column = character(), fixed_contrast = character())
  for (i in seq_along(slopes)) {
    column <- paste0(".random_", i)
    data[[column]] <- design[, slopes[[i]]]
    random <- paste(random, sprintf("(0 + %s | participant)", column), sep = " + ")
    mapping <- bind_rows(mapping, tibble(random_column = column, fixed_contrast = slopes[[i]]))
  }
  list(data = data, formula = as.formula(paste(deparse1(fixed), random, sep = " + ")), mapping = mapping)
}

export_stats_terms <- function(table, directory) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(table, file.path(directory, "coefficients.csv"))
  for (i in seq_len(nrow(table))) {
    row <- table[i, ]
    term <- row$term
    p <- if (is.na(row$p)) "NA" else if (row$p < .001) "< .001" else sub("^0", "", sprintf("%.3f", row$p))
    values <- c(beta = sprintf("%.3f", row$beta), ci_low = sprintf("%.3f", row$ci_low),
      ci_high = sprintf("%.3f", row$ci_high), ci = sprintf("[%.3f, %.3f]", row$ci_low, row$ci_high),
      df = if (is.na(row$df)) "NA" else sprintf("%d", round(row$df)), p = p)
    for (stat in names(values)) writeLines(values[[stat]], file.path(directory, paste0(term, "_", stat, ".tex")))
  }
}

run_stats_model <- function(data, response, rhs, directory, focal = NULL) {
  if (response == "accuracy") return(fit_accuracy_glmer(data, response, rhs, directory, focal))
  fit_exp3_bam(data, response, rhs, directory, focal, threads = 1L,
    workflow_note = "Manuscript statistics: Gaussian BAM for continuous responses; binary accuracy is fitted separately with glmer.")
}

run_source_stats <- function(data, experiment, source, output, tests) {
  directory <- file.path(output, experiment, source)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  planning <- filter(data, condition == "planning")
  if (experiment == "exp2" && "best_branch_timing" %in% tests &&
      source %in% c("participants", "sequential", "simultaneous", "independent")) {
    run_best_branch_timing_stats(planning, file.path(directory, "best_branch_timing"))
  }
  if ("descriptive" %in% tests) {
    summary <- data %>% group_by(participant, condition) %>% summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")
    write_csv(summary, file.path(directory, "participant_accuracy.csv"))
    reward <- planning %>% distinct(participant, trial_id, norm_reward) %>%
      group_by(participant) %>% summarise(norm_reward = mean(norm_reward, na.rm = TRUE), .groups = "drop")
    write_csv(reward, file.path(directory, "participant_norm_reward.csv"))
    for (cond in unique(as.character(summary$condition))) {
      x <- summary$accuracy[summary$condition == cond]
      writeLines(sprintf("$M = %.3f$", mean(x)), file.path(directory, paste0("accuracy_", cond, "_mean.tex")))
      writeLines(sprintf("$SD = %.3f$", sd(x)), file.path(directory, paste0("accuracy_", cond, "_sd.tex")))
    }
    writeLines(sprintf("$M = %.3f$", mean(reward$norm_reward)), file.path(directory, "norm_reward_mean.tex"))
    writeLines(sprintf("$SD = %.3f$", sd(reward$norm_reward)), file.path(directory, "norm_reward_sd.tex"))
  }
  if (experiment != "exp3" && "context" %in% tests) {
    rhs <- paste(context_predictors, collapse = " + ")
    joint <- source %in% c("participants", "sequential")
    if (joint && !setequal(unique(as.character(data$condition)), c("planning", "exposure")))
      fail("Joint context model requires planning AND exposure for %s", source)
    run_stats_model(if (joint) data else planning, "accuracy",
      if (joint) paste0("(", rhs, ") * condition") else rhs,
      file.path(directory, if (joint) "context_joint_conditions" else "context_planning"))
  }
  if (experiment != "exp3" && "path_rank" %in% tests) {
    ranks <- seq_len(if (experiment == "exp1") 3L else 8L)
    anova <- adjacent_rank_anova(planning, ranks)
    anova_dir <- file.path(directory, "adjacent_path_ranks_anova")
    dir.create(anova_dir, recursive = TRUE, showWarnings = FALSE)
    write_csv(anova$results, file.path(anova_dir, "comparisons.csv"))
    write_csv(anova$means, file.path(anova_dir, "participant_rank_means.csv"))
    writeLines(c("Classical two-level repeated-measures ANOVA; F = paired t squared.",
      "Planning leaf and non-leaf probes; one equally weighted mean per participant and rank.",
      "Each node uses the best (lowest numerical) final rank among complete paths containing it.",
      "Each pair uses participants observed at both ranks; simulation means pool seeds/trials/nodes.",
      "Unadjusted F-test p and Holm p across adjacent pairs within each experiment/source."),
      file.path(anova_dir, "model.txt"))
  }
  if (experiment == "exp3" && "rank_change" %in% tests) {
    ranked <- rank_stats_data(planning, experiment)
    rhs <- paste(rank_predictors, collapse = " + ")
    if (experiment == "exp3") {
      rhs <- paste0("(", rhs, ") * tree + (", rhs, ") * order")
      focal <- c("rank_conditionworse:treewide", "rank_conditionworse:orderDFS")
    } else focal <- "rank_conditionbetter"
    run_stats_model(ranked, "accuracy", rhs, file.path(directory, "rank_change"), focal)
  }
  if (experiment != "exp3" && "branch_timing" %in% tests) {
    branch <- filter(planning, !is.na(relative_branch_timing))
    writeLines(sprintf("Mixed earlier/later maximizing-branch ties excluded: %d",
      sum(is.na(planning$relative_branch_timing))), file.path(directory, "branch_timing_exclusions.txt"))
    run_stats_model(branch, "accuracy", "context_reward_z * relative_branch_timing + observation_step_z",
      file.path(directory, "branch_timing"), "context_reward_z:relative_branch_timingbefore")
  }
  if (experiment != "exp3" && "observation_timestep" %in% tests) {
    # All planning probes, including equal ranks and the final observation.
    # Independent participant intercept, leaf-status slope, and timestep slope.
    run_stats_model(planning, "accuracy", "is_leaf + observation_step_z",
      file.path(directory, "observation_timestep"))
  }
  if (experiment == "exp3" && "tree_order" %in% tests) {
    # +/- 0.5 coding makes main effects equal-weight averages over the other
    # factor, and the interaction the wide-minus-deep DFS-minus-BFS difference.
    planning <- planning %>% mutate(tree_c = ifelse(tree == "wide", .5, -.5),
      order_c = ifelse(order == "DFS", .5, -.5))
    run_stats_model(planning, "accuracy", "tree_c * order_c", file.path(directory, "tree_order_accuracy"))
    trial_rewards <- planning %>% distinct(participant, trial_id, .keep_all = TRUE)
    run_stats_model(trial_rewards, "norm_reward", "tree_c * order_c", file.path(directory, "tree_order_norm_reward"))
  }
}

run_experiment_stats <- function(experiment, args = commandArgs(trailingOnly = TRUE)) {
  options <- parse_stats_options(args, experiment)
  if (is.null(options)) return(invisible(NULL))
  if ("memory_allocation" %in% options$tests) fail("Retired analysis: use the participant quadratic-coefficient analysis instead")
  if ("parameter_performance" %in% options$tests) run_parameter_performance_stats(experiment, options)
  if ("nll" %in% options$tests) run_nll_comparison_stats(experiment, options)
  options$tests <- setdiff(options$tests, c("memory_allocation", "parameter_performance", "nll"))
  if (!length(options$tests)) return(invisible(NULL))
  inputs <- stats_inputs(experiment, options$root)
  required <- options$sources
  needs_exposure <- experiment != "exp3" && "sequential" %in% required &&
    any(c("context", "descriptive") %in% options$tests)
  if (needs_exposure) required <- c(required, "exposure")
  manifest <- tibble(source = required, path = unname(inputs[required]), exists = file.exists(inputs[required]))
  print(manifest, width = Inf)
  if (any(!manifest$exists)) fail("Missing required input CSVs; see paths above. No fits were started.")
  if (options$check) return(invisible(manifest))
  experiment_dir <- file.path(options$output, experiment)
  dir.create(experiment_dir, recursive = TRUE, showWarnings = FALSE)
  manifest$bytes <- file.info(manifest$path)$size
  manifest$modified <- as.character(file.info(manifest$path)$mtime)
  write_csv(manifest, file.path(experiment_dir, "input_manifest.csv"))
  for (source in options$sources) {
    data <- read_stats_data(inputs[[source]], experiment, source,
      if (source == "participants") NULL else "planning",
      sample_trials_per_condition = NULL,
      sampling_directory = file.path(experiment_dir, source),
      sample_nodes_per_trial = NULL)
    if (source == "sequential" && needs_exposure) {
      exposure <- read_stats_data(inputs[["exposure"]], experiment, source, "exposure")
      data <- bind_rows(data, exposure)
      rm(exposure)
    }
    run_source_stats(data, experiment, source, options$output, options$tests)
    rm(data); invisible(gc())
  }
  writeLines(capture.output(sessionInfo()), file.path(experiment_dir, "sessionInfo.txt"))
  message("Saved statistics to: ", experiment_dir)
}

# ==============================================================================
# bam_common: consolidated shared definitions (function bodies unchanged)
# ==============================================================================
exp3_response_family <- function(data, response) {
  if (response == "accuracy") {
    if (any(!is.finite(data[[response]]) | !data[[response]] %in% c(0, 1)))
      fail("Individual accuracy requires binary 0/1 observations")
    return(binomial(link = "logit"))
  }
  gaussian()
}

add_exp3_odds_ratios <- function(table, family) {
  if (family$family == "binomial") table <- mutate(table,
    odds_ratio = exp(beta), odds_ratio_ci_low = exp(ci_low), odds_ratio_ci_high = exp(ci_high))
  table
}

bam_random_formula <- function(data, response, rhs, random_rhs = rhs) {
  spec <- independent_random_formula(data, response, random_rhs)
  spec$data$participant <- factor(spec$data$participant)
  random <- c('s(participant, bs = "re")',
    sprintf('s(participant, %s, bs = "re")', spec$mapping$random_column))
  s <- mgcv::s
  spec$formula <- as.formula(paste(response, "~", rhs, "+", paste(random, collapse = " + ")))
  spec
}

export_exp3_bam_fit <- function(fit, directory, focal = NULL) {
  # Keep convergence failures out of coefficient exports.
  if (!isTRUE(fit$converged) ||
      (!is.null(fit$outer.info$conv) && !identical(fit$outer.info$conv, "full convergence")))
    fail("BAM convergence check failed; inspect model.rds and model.txt: %s", directory)
  sm <- summary(fit)
  co <- sm$p.table
  if (any(!is.finite(co)) || !is.finite(sm$residual.df) || sm$residual.df <= 0)
    fail("Invalid BAM coefficient inference: %s", directory)
  logistic <- family(fit)$family == "binomial"
  critical <- if (logistic) qnorm(.975) else qt(.975, sm$residual.df)
  table <- tibble(term = rownames(co), beta = co[, 1], se = co[, 2],
    df = if (logistic) NA_real_ else sm$residual.df, p = co[, 4],
    ci_low = beta - critical * se, ci_high = beta + critical * se)
  table$estimator <- "bam"
  table$family <- family(fit)$family
  table$link <- family(fit)$link
  table[[if (logistic) "z" else "t"]] <- co[, 3]
  table <- add_exp3_odds_ratios(table, family(fit))
  if (!is.null(focal) && any(!focal %in% table$term)) fail("Missing focal BAM coefficients")
  write_csv(table, file.path(directory, "all_coefficients.csv"))
  # Small plotting snapshot: avoid loading the full observation-level BAM object.
  fixed_terms <- rownames(co)
  saveRDS(list(coefficients = co,
    covariance = stats::vcov(fit)[fixed_terms, fixed_terms, drop = FALSE],
    formula = deparse1(stats::formula(fit)), estimator = "bam",
    family = family(fit)$family, link = family(fit)$link),
    file.path(directory, "fixed_effects.rds"))
  selected <- if (is.null(focal)) filter(table, term != "(Intercept)") else filter(table, term %in% focal)
  export_stats_terms(selected, directory)
  if (logistic) for (i in seq_len(nrow(selected))) {
    writeLines(sprintf("%.3f", selected$z[i]), file.path(directory, paste0(selected$term[i], "_z.tex")))
    writeLines("---", file.path(directory, paste0(selected$term[i], "_df.tex")))
  }
  writeLines(capture.output(dput(list(smoothing_parameters = fit$sp, residual_variance = fit$sig2))),
    file.path(directory, "penalty_diagnostics.txt"))
  print(selected)
}

fit_exp3_bam <- function(data, response, rhs, directory, focal = NULL, threads = 1L,
                         sampling_note = "Sampling: none; all trials/nodes before analysis-specific exclusions.",
                         workflow_note = "Benchmark only; no automatic manuscript table integration.",
                         random_rhs = rhs) {
  if (!requireNamespace("mgcv", quietly = TRUE)) fail("Install mgcv in the statistics R environment")
  if ("node_reward" %in% names(data)) {
    data$abs_node_reward <- abs(data$node_reward)
    data$node_reward_squared <- data$node_reward^2
  }
  if ("sibling_reward" %in% names(data)) data$sibling_reward_squared <- data$sibling_reward^2
  unscaled <- unique(c(response, "participant", all.vars(as.formula(paste("~", gsub("_z\\b", "", rhs))))))
  require_columns(data, unscaled, directory)
  keep <- complete.cases(data[unscaled])
  data <- droplevels(data[keep, , drop = FALSE])
  if (!nrow(data)) fail("No complete rows: %s", directory)
  response_family <- exp3_response_family(data, response)
  data <- standardize_stats(data)
  spec <- bam_random_formula(data, response, rhs, random_rhs)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write_csv(attr(data, "scaling"), file.path(directory, "scaling.csv"))
  write_csv(spec$mapping, file.path(directory, "random_design.csv"))
  message("Fitting BAM: ", directory, " | rows=", nrow(data), " | participants=", n_distinct(data$participant))
  message(deparse1(spec$formula)); flush.console()
  warnings <- character()
  elapsed <- system.time(fit <- withCallingHandlers(
    mgcv::bam(spec$formula, data = spec$data, family = response_family, method = "fREML",
      discrete = TRUE, nthreads = threads, gc.level = 1, na.action = na.fail),
    warning = function(w) { warnings <<- c(warnings, conditionMessage(w)) }))[["elapsed"]]
  message("Fit finished; saving model and diagnostics.")
  saveRDS(fit, file.path(directory, "model.rds"), compress = FALSE)
  writeLines(c(paste("Formula:", deparse1(spec$formula)),
    "Estimator: mgcv::bam fREML, discrete=TRUE; NOT ML",
    paste("Family:", response_family$family, "Link:", response_family$link),
    "Inference: binomial accuracy uses approximate Wald z tests and normal 95% log-odds intervals, plus odds ratios; Gaussian normalized reward uses approximate t tests/residual effective df and t intervals. NOT Satterthwaite.",
    workflow_note,
    sampling_note,
    paste("Rows:", nrow(data)), paste("Excluded incomplete rows:", sum(!keep)),
    paste("Participants:", n_distinct(data$participant)), paste("Threads requested:", threads),
    paste("Elapsed fit seconds:", elapsed), paste("Converged:", fit$converged),
    paste("Outer convergence:", fit$outer.info$conv),
    paste("Warnings:", paste(unique(warnings), collapse = "; "))), file.path(directory, "model.txt"))
  export_exp3_bam_fit(fit, directory, focal)
  invisible(gc())
}
# ==============================================================================
# glmer_common: consolidated shared definitions (function bodies unchanged)
# ==============================================================================
glmer_convergence_diagnostics <- function(fit) {
  messages <- unlist(fit@optinfo$conv$lme4$messages)
  other <- messages[!grepl("^boundary \\(singular\\) fit", messages)]
  code <- fit@optinfo$conv$opt
  optimizer_ok <- length(code) > 0L && !anyNA(code) && all(code == 0)
  H <- fit@optinfo$derivs$Hessian
  g <- fit@optinfo$derivs$gradient
  relative <- min_eigen <- reciprocal <- NA_real_
  positive <- NA
  if (!is.null(H) && !is.null(g) && all(is.finite(H)) && all(is.finite(g))) {
    min_eigen <- min(eigen(H, symmetric = TRUE, only.values = TRUE)$values)
    positive <- min_eigen > 0
    reciprocal <- rcond(H)
    relative <- tryCatch(max(abs(solve(H, g))), error = function(e) NA_real_)
  }
  gradient_only <- length(other) > 0L && all(grepl(
    "^Model failed to converge with max\\|grad\\|", other))
  accepted_exception <- optimizer_ok && gradient_only && isTRUE(positive) &&
    is.finite(relative) && relative < .001
  passed <- optimizer_ok && (length(other) == 0L || accepted_exception) &&
    !identical(positive, FALSE)
  tibble(passed = passed, accepted_relative_gradient_exception = accepted_exception,
    max_absolute_relative_gradient = relative, relative_gradient_threshold = .001,
    hessian_positive_definite = positive, minimum_hessian_eigenvalue = min_eigen,
    hessian_reciprocal_condition_number = reciprocal,
    optimizer_code = paste(code, collapse = "; "),
    original_convergence_messages = paste(messages, collapse = "; "))
}

fit_accuracy_glmer <- function(data, response, rhs, directory, focal = NULL,
                              random_rhs = rhs, nAGQ = 1L) {
  stopifnot(response == "accuracy")
  stopifnot(length(nAGQ) == 1L, nAGQ %in% c(0L, 1L))
  if ("node_reward" %in% names(data)) {
    data$abs_node_reward <- abs(data$node_reward)
    data$node_reward_squared <- data$node_reward^2
  }
  if ("sibling_reward" %in% names(data)) data$sibling_reward_squared <- data$sibling_reward^2
  variables <- unique(c(response, "participant", all.vars(as.formula(paste("~",
    gsub("_z\\b", "", paste(rhs, random_rhs, sep = " + ")))))))
  require_columns(data, variables, directory)
  keep <- complete.cases(data[variables])
  data <- droplevels(data[keep, , drop = FALSE])
  if (!nrow(data)) fail("No complete rows: %s", directory)
  response_family <- exp3_response_family(data, response)
  data <- standardize_stats(data)
  spec <- independent_random_formula(data, response, random_rhs)
  spec$data$participant <- factor(spec$data$participant)
  # Preserve independent numeric contrasts, including factor interactions.
  random <- c("(1 | participant)", sprintf("(0 + %s | participant)", spec$mapping$random_column))
  spec$formula <- as.formula(paste(response, "~", rhs, "+", paste(random, collapse = " + ")))
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write_csv(attr(data, "scaling"), file.path(directory, "scaling.csv"))
  write_csv(spec$mapping, file.path(directory, "random_design.csv"))
  message("Fitting glmer: ", directory, " | rows=", nrow(data), " | participants=", n_distinct(data$participant))
  message(deparse1(spec$formula))
  warnings <- character()
  elapsed <- system.time(fit <- withCallingHandlers(
    lme4::glmer(spec$formula, data = spec$data, family = response_family, nAGQ = nAGQ,
      na.action = na.fail, control = lme4::glmerControl(optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5))),
    warning = function(w) warnings <<- c(warnings, conditionMessage(w))))[["elapsed"]]
  saveRDS(fit, file.path(directory, "model.rds"), compress = FALSE)
  messages <- fit@optinfo$conv$lme4$messages
  code <- fit@optinfo$conv$opt
  singular <- lme4::isSingular(fit)
  write_csv(as.data.frame(lme4::VarCorr(fit)), file.path(directory, "random_effect_variances.csv"))
  writeLines(c(paste("Formula:", deparse1(spec$formula)),
    if (nAGQ == 1L) "Estimator: lme4::glmer; binomial-logit GLMM; maximum likelihood, Laplace approximation (nAGQ=1)" else
      "Estimator: lme4::glmer; binomial-logit GLMM; faster approximate estimation (nAGQ=0; fixed and random effects optimized in PIRLS)",
    "Family: binomial Link: logit",
    "Inference: two-sided Wald z tests; normal-Wald 95% log-odds confidence intervals",
    "Sampling: none; all trials/nodes before analysis-specific exclusions.",
    paste("Rows:", nrow(data)), paste("Excluded incomplete rows:", sum(!keep)),
    paste("Participants:", n_distinct(data$participant)), paste("Singular:", singular),
    paste("Optimizer code:", paste(code, collapse = ",")),
    paste("Convergence:", paste(messages, collapse = "; ")),
    paste("Elapsed fit seconds:", elapsed), paste("Warnings:", paste(unique(warnings), collapse = "; "))),
    file.path(directory, "model.txt"))
  export_saved_accuracy_glmer(fit, directory, focal)
}

# Export a saved fit without repeating optimization. Original model.rds/model.txt
# are retained; the acceptance decision is recorded in convergence_diagnostics.csv.
export_saved_accuracy_glmer <- function(fit, directory, focal = NULL) {
  stopifnot(inherits(fit, "glmerMod"), family(fit)$family == "binomial",
    family(fit)$link == "logit")
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  diagnostics <- glmer_convergence_diagnostics(fit)
  write_csv(diagnostics, file.path(directory, "convergence_diagnostics.csv"))
  if (!diagnostics$passed)
    fail("glmer convergence check failed; inspect model.rds and model.txt: %s", directory)
  if (diagnostics$accepted_relative_gradient_exception)
    message("Accepting gradient-only warning under relative-gradient < 0.001 policy; ",
      "this is an explicit diagnostic exception, not independent convergence verification: ", directory)
  singular <- lme4::isSingular(fit)
  nAGQ <- unname(fit@devcomp$dims[["nAGQ"]])
  response_family <- family(fit)
  co <- coef(summary(fit))
  if (any(!is.finite(co))) fail("Non-finite glmer inference: %s", directory)
  if (!is.null(focal) && any(!focal %in% rownames(co))) fail("Missing focal glmer coefficients: %s", directory)
  table <- tibble(term = rownames(co), beta = co[, 1], se = co[, 2], df = NA_real_,
    z = co[, 3], p = co[, 4], ci_low = beta - qnorm(.975) * se,
    ci_high = beta + qnorm(.975) * se, estimator = "glmer", family = "binomial",
    link = "logit", singular_fit = singular, nAGQ = nAGQ,
    accepted_relative_gradient_exception = diagnostics$accepted_relative_gradient_exception,
    max_absolute_relative_gradient = diagnostics$max_absolute_relative_gradient)
  table <- add_exp3_odds_ratios(table, response_family)
  write_csv(table, file.path(directory, "all_coefficients.csv"))
  saveRDS(list(coefficients = co, covariance = as.matrix(vcov(fit)),
    formula = deparse1(formula(fit)), estimator = "glmer", family = "binomial", link = "logit", nAGQ = nAGQ,
    convergence_diagnostics = diagnostics),
    file.path(directory, "fixed_effects.rds"))
  selected <- if (is.null(focal)) filter(table, term != "(Intercept)") else filter(table, term %in% focal)
  export_stats_terms(selected, directory)
  for (i in seq_len(nrow(selected))) {
    writeLines(sprintf("%.3f", selected$z[i]), file.path(directory, paste0(selected$term[i], "_z.tex")))
    writeLines("---", file.path(directory, paste0(selected$term[i], "_df.tex")))
  }
  print(selected)
  invisible(fit)
}
# ==============================================================================
# performance: consolidated shared definitions (function bodies unchanged)
# ==============================================================================
source(project_path("analysis/scripts/figures/utils.R"))
performance_fit_path <- function(simulation, root) {
  analysis_selected_fit(simulation, root)
}

performance_plot_directory <- function(simulation, experiment, root) {
  name <- canonical_simulation_basename(simulation)
  hash <- regmatches(name, regexec("_cfg_([a-f0-9]+)_", name))[[1]]
  tail <- regmatches(name, regexec("(_n[0-9]+(_per_seed_tree)?_simseed[0-9]+(_new)?)\\.csv$", name))[[1]]
  if (length(hash) < 2 || length(tail) < 2) fail("Cannot derive plotting-summary folder: %s", simulation)
  file.path(root, "summaries/analysis/tests", paste0(experiment, "_preregistered_fitted_cfg_", hash[[2]], tail[[2]]))
}

performance_read <- function(path) {
  if (!file.exists(path)) fail("Required performance-analysis input is missing: %s", path)
  message("Reading performance input: ", path)
  read_csv(path, show_col_types = FALSE, progress = FALSE)
}

performance_manifest <- function(paths, directory) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write_csv(tibble(path = paths, bytes = file.info(paths)$size,
    modified = as.character(file.info(paths)$mtime)), file.path(directory, "input_manifest.csv"))
}

performance_z <- function(x, name) {
  if (any(!is.finite(x)) || !is.finite(sd(x)) || sd(x) == 0)
    fail("Cannot standardize non-finite or constant predictor: %s", name)
  as.numeric(scale(x))
}

performance_bam_f_test <- function(fit, y) {
  # No smooths or random effects in participant-level regressions: classical
  # Gaussian regression F test, calculated from the BAM residual sum of squares.
  df_num <- length(coef(fit)) - 1L
  df_den <- df.residual(fit)
  rss <- sum((y - fitted(fit))^2)
  tss <- sum((y - mean(y))^2)
  F <- max(0, (tss - rss) / df_num) / (rss / df_den)
  tibble(F = F, df_num = df_num, df_den = df_den,
    p = pf(F, df_num, df_den, lower.tail = FALSE))
}

export_performance_lm <- function(data, outcome, predictors, directory, interaction = FALSE, omit_constants = FALSE, standardize = TRUE) {
  require_columns(data, c("participant", outcome, predictors), directory)
  data <- as.data.frame(data)
  if (anyNA(data$participant) || anyDuplicated(data$participant)) fail("Expected one row per participant: %s", directory)
  original_n <- nrow(data)
  data <- data[apply(as.matrix(data[c(outcome, predictors)]), 1, function(x) all(is.finite(x))), ]
  if (nrow(data) < length(predictors) + 3) fail("Insufficient complete participants: %s", directory)
  scaling <- bind_rows(lapply(predictors, function(name) tibble(variable = name, mean = mean(data[[name]]), sd = sd(data[[name]]))))
  omitted <- character()
  if (omit_constants) {
    if (interaction) fail("Constant omission is only supported for additive memory-allocation regressions")
    omitted <- predictors[vapply(data[predictors], function(x) length(unique(x)) == 1L, logical(1))]
    predictors <- setdiff(predictors, omitted)
    if (!length(predictors)) fail("No varying memory-allocation predictors remain: %s", directory)
    if (length(omitted)) message("Omitting constant predictors (not estimable): ", paste(omitted, collapse = ", "))
  }
  if (standardize) for (name in predictors) data[[paste0(name, "_z")]] <- performance_z(data[[name]], name)
  rhs <- paste(if (standardize) paste0(predictors, "_z") else predictors, collapse = if (interaction) " * " else " + ")
  formula <- as.formula(paste(outcome, "~", rhs))
  fit <- mgcv::bam(formula, data = data, family = gaussian(), method = "fREML", nthreads = 1)
  if (fit$rank < ncol(model.matrix(fit))) fail("Association model is rank deficient: %s", directory)
  coefficients <- summary(fit)$p.table
  ci <- cbind(coefficients[, 1] - qt(.975, df.residual(fit)) * coefficients[, 2],
    coefficients[, 1] + qt(.975, df.residual(fit)) * coefficients[, 2])
  rownames(ci) <- rownames(coefficients)
  table <- tibble(term = rownames(coefficients), beta = coefficients[, 1], se = coefficients[, 2],
    df = df.residual(fit), p = coefficients[, 4], ci_low = ci[, 1], ci_high = ci[, 2])
  table <- filter(table, term != "(Intercept)") %>% mutate(status = "estimated", constant_value = NA_real_)
  for (name in omitted) table <- bind_rows(table, tibble(term = paste0(name, "_z"),
    beta = NA_real_, se = NA_real_, df = NA_real_, p = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
    status = "not_estimable_constant", constant_value = data[[name]][1]))
  export_stats_terms(table, directory)
  for (name in omitted) {
    term <- paste0(name, "_z")
    writeLines("Not estimable (constant predictor)", file.path(directory, paste0(term, "_beta.tex")))
    for (stat in c("ci", "ci_low", "ci_high", "df", "p"))
      writeLines("---", file.path(directory, paste0(term, "_", stat, ".tex")))
  }
  write_csv(data, file.path(directory, "participant_data.csv"))
  write_csv(scaling, file.path(directory, "scaling.csv"))
  omnibus <- performance_bam_f_test(fit, data[[outcome]])
  saveRDS(fit, file.path(directory, "model.rds"))
  export_performance_anova(omnibus, directory)
  writeLines(c(paste("Formula:", deparse1(formula)), paste("Participants:", nrow(data)),
    paste("Excluded incomplete participants:", original_n - nrow(data)),
    paste("Included predictors:", paste(predictors, collapse = ", ")),
    paste("Excluded constant predictors (not estimable):", paste(omitted, collapse = ", ")),
    "The overall F test applies only to included predictors.",
    "Estimator: Gaussian BAM without smooths or random effects (ordinary linear regression); one equally weighted row per participant.",
    "Inference: t tests, 95% t confidence intervals, overall F test; unadjusted p values.",
    if (standardize) "Predictors standardized on raw scales (no log transform); response unstandardized." else
      "Predictors and response unstandardized; no log transform.",
    "Exploratory association: fitted parameters and estimated recall slopes are not independent of performance; their estimation uncertainty is not propagated."),
    file.path(directory, "model.txt"))
  print(table)
  print(omnibus)
  invisible(fit)
}

export_performance_anova <- function(table, directory) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write_csv(table, file.path(directory, "anova.csv"))
  for (name in c("F", "df_num", "df_den", "p")) {
    value <- table[[name]][[1]]
    formatted <- if (name == "p" && value < .001) "< .001" else sprintf("%.3f", value)
    writeLines(formatted, file.path(directory, paste0("omnibus_", name, ".tex")))
  }
}

run_memory_allocation_stats <- function(experiment, options) {
  if (experiment == "exp3") fail("Memory-allocation performance tests are defined for Exp1/Exp2")
  simulation <- stats_inputs(experiment, options$root)[["sequential"]]
  plot_dir <- performance_plot_directory(simulation, experiment, options$root)
  predictors <- paste0(c("node", "sibling", "parent", "aunt"), "_reward_coefficient")
  for (source in intersect(options$sources, c("participants", "sequential"))) {
    path <- file.path(plot_dir, if (source == "participants") "reviewer_participant_metrics.csv" else "reviewer_sequential_model_metrics.csv")
    if (options$check) { if (!file.exists(path)) fail("Missing plotting metrics: %s", path); message(path); next }
    data <- performance_read(path)
    outcome <- if (source == "participants") "human_norm_reward" else "model_norm_reward"
    require_columns(data, c("subject", outcome, predictors), path)
    # Reject old mixed-model conditional slopes; plots and tests must use the same unpooled fits.
    accepted <- context_performance_sample(data, outcome)
    directory <- file.path(options$output, experiment, source, "memory_allocation_performance")
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    write_csv(data, file.path(directory, "coefficient_fit_diagnostics.csv"))
    excluded <- nrow(data) - nrow(accepted)
    data <- accepted
    data <- data %>% transmute(participant = as.character(subject), norm_reward = .data[[outcome]], across(all_of(predictors)))
    performance_manifest(path, directory)
    export_performance_lm(data, "norm_reward", predictors, directory, omit_constants = TRUE)
    coefficients <- read_csv(file.path(directory, "coefficients.csv"), show_col_types = FALSE)
    coefficients$context_coefficient_method <- "separate_binomial_glm_linear_v2"
    write_csv(coefficients, file.path(directory, "coefficients.csv"))
    cat(paste0("\nFirst stage: separate participant binomial-logit GLMs; no pooling or random effects.\n",
      "Predictors: standardized node, sibling, parent, and aunt rewards; linear terms only; planning only.\n",
      "First-stage standardization uses a common experiment/source scale; parent/aunt are zero for non-leaves.\n",
      "Excluded flagged/nonfinite first-stage fits: ", excluded, ". See coefficient_fit_diagnostics.csv.\n",
      "Four linear contextual coefficients enter the second-stage regression jointly.\n"),
      file = file.path(directory, "model.txt"), append = TRUE)
  }
}

read_planning_performance <- function(path, experiment, source) {
  # No rank/context preparation: retain the complete planning performance sample.
  data <- read_csv(path, col_select = any_of(c("subject", "condition", "node", "accuracy", "actual_reward",
    "estimated_reward", "graph", "tree_type", "norm_reward", "V", "min_path_reward", "max_path_reward")),
    show_col_types = FALSE, progress = FALSE)
  require_columns(data, c("subject", "node"), path)
  if (source == "participants") {
    require_columns(data, c("condition", "accuracy"), path)
    data <- filter(data, tolower(condition) == "planning")
  } else {
    require_columns(data, c("actual_reward", "estimated_reward"), path)
    data$accuracy <- as.numeric(data$actual_reward == data$estimated_reward)
  }
  data <- filter(data, node >= 1, node < switch(experiment, exp1 = 7, exp2 = 13, exp3 = 31))
  if (anyNA(data$subject) || any(!is.na(data$accuracy) & !data$accuracy %in% c(0, 1))) fail("Invalid participant/accuracy: %s", path)
  recall <- data %>% group_by(subject) %>% summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")
  # Parameter-performance tests use human reward data only.
  if (source == "participants") {
    require_columns(data, "graph", path)
    if (!"norm_reward" %in% names(data)) {
      require_columns(data, c("V", "min_path_reward", "max_path_reward"), path)
      width <- data$max_path_reward - data$min_path_reward
      data$norm_reward <- ifelse(width > 0, (data$V - data$min_path_reward) / width, NA_real_)
    }
    trials <- data %>% distinct(across(any_of(c("subject", "condition", "tree_type", "graph", "norm_reward"))))
    reward <- trials %>% group_by(subject) %>% summarise(norm_reward = mean(norm_reward, na.rm = TRUE), .groups = "drop")
    recall <- left_join(recall, reward, by = "subject", relationship = "one-to-one")
  }
  recall %>% rename(participant = subject) %>% mutate(participant = as.character(participant))
}

run_parameter_performance_stats <- function(experiment, options) {
  if (!"participants" %in% options$sources) return(invisible(NULL))
  paths <- stats_inputs(experiment, options$root)
  fit_path <- performance_fit_path(paths[["sequential"]], options$root)
  if (options$check) {
    for (path in c(paths[["participants"]], fit_path)) {
      if (!file.exists(path)) fail("Missing parameter-performance input: %s", path)
      message(path)
    }
    return(invisible(NULL))
  }
  fits <- performance_read(fit_path)
  require_columns(fits, c("subject", "best_lambda", "best_alpha"), fit_path)
  if ("best_rho" %in% names(fits) && any(!is.finite(fits$best_rho) | fits$best_rho < 0 |
      abs(fits$best_alpha - fits$best_lambda * fits$best_rho) > 1e-10 * pmax(1, abs(fits$best_alpha))))
    fail("Invalid or inconsistent best_rho: %s", fit_path)
  fits <- fits %>% transmute(participant = as.character(subject), fitted_lambda = best_lambda,
    fitted_alpha = best_alpha, fitted_rho = if ("best_rho" %in% names(fits)) fits$best_rho else best_alpha / best_lambda)
  if (anyNA(fits$participant) || anyDuplicated(fits$participant) || any(!is.finite(fits$fitted_lambda) | fits$fitted_lambda <= 0) ||
      any(!is.finite(fits$fitted_alpha) | fits$fitted_alpha < 0) ||
      any(!is.finite(fits$fitted_rho) | fits$fitted_rho < 0)) fail("Invalid fitted parameters: %s", fit_path)
  performance <- read_planning_performance(paths[["participants"]], experiment, "participants")
  if (nrow(anti_join(fits, performance, by = "participant"))) fail("Fitted participants missing behavioral performance")
  data <- left_join(fits, performance, by = "participant", relationship = "one-to-one")
  for (outcome in c("norm_reward", "accuracy")) {
    directory <- file.path(options$output, experiment, "participants", paste0("fitted_parameters_", outcome))
    performance_manifest(c(paths[["participants"]], fit_path), directory)
    export_performance_lm(data, outcome, c("fitted_lambda", "fitted_alpha"), directory, interaction = TRUE, standardize = FALSE)
  }
}

planning_accuracy_pairwise <- function(fit) {
  experiments <- c("exp1", "exp2", "exp3")
  design <- model.matrix(~ experiment, data.frame(experiment = factor(experiments, levels = experiments)),
    contrasts.arg = fit$contrasts)
  pairs <- combn(3L, 2L)
  result <- bind_rows(lapply(seq_len(ncol(pairs)), function(k) {
    i <- pairs[1, k]; j <- pairs[2, k]
    contrast <- design[j, ] - design[i, ]
    beta <- sum(contrast * coef(fit))
    se <- sqrt(drop(contrast %*% vcov(fit) %*% contrast))
    df <- df.residual(fit)
    tibble(term = paste0(experiments[j], "_minus_", experiments[i]), beta = beta, se = se,
      df = df, t = beta / se, p = 2 * pt(-abs(beta / se), df),
      ci_low = beta - qt(.975, df) * se, ci_high = beta + qt(.975, df) * se)
  }))
  result$p_holm <- p.adjust(result$p, method = "holm")
  result
}

run_cross_experiment_performance <- function(args = commandArgs(trailingOnly = TRUE)) {
  options <- parse_stats_options(args, "exp3")
  if (is.null(options)) return(invisible(NULL))
  for (source in intersect(options$sources, c("participants", "sequential"))) {
    paths <- vapply(c("exp1", "exp2", "exp3"), function(experiment) stats_inputs(experiment, options$root)[[source]], character(1))
    if (any(!file.exists(paths))) fail("Missing cross-experiment input: %s", paste(paths[!file.exists(paths)], collapse = "; "))
    if (options$check) { print(paths); next }
    data <- bind_rows(lapply(names(paths), function(experiment) {
      read_planning_performance(paths[[experiment]], experiment, source) %>%
        select(participant, accuracy) %>% mutate(experiment = experiment)
    }))
    data$experiment <- factor(data$experiment, levels = c("exp1", "exp2", "exp3"))
    if (any(!is.finite(data$accuracy))) fail("Missing participant-level planning accuracy")
    fit <- mgcv::bam(accuracy ~ experiment, data = data, family = gaussian(), method = "fREML", nthreads = 1)
    table <- performance_bam_f_test(fit, data$accuracy)
    directory <- file.path(options$output, "cross_experiment", source, "planning_accuracy_anova")
    performance_manifest(paths, directory)
    saveRDS(fit, file.path(directory, "model.rds"))
    write_csv(data, file.path(directory, "participant_data.csv"))
    export_performance_anova(table, directory)
    pairwise <- planning_accuracy_pairwise(fit)
    pair_directory <- file.path(dirname(directory), "planning_accuracy_pairwise")
    performance_manifest(paths, pair_directory)
    export_stats_terms(pairwise, pair_directory)
    write_csv(pairwise, file.path(pair_directory, "pairwise.csv"))
    for (k in seq_len(nrow(pairwise))) writeLines(if (pairwise$p_holm[k] < .001) "< .001" else sprintf("%.3f", pairwise$p_holm[k]),
      file.path(pair_directory, paste0(pairwise$term[k], "_p_holm.tex")))
    writeLines(c("Pairwise contrasts from the same one-way ANOVA, using pooled residual variance.",
      "Direction: larger tree minus smaller tree. Confidence intervals are unadjusted 95% t intervals.",
      "p_holm corrects all three pairwise tests within each source separately; p is unadjusted."),
      file.path(pair_directory, "model.txt"))
    writeLines(c("Formula: participant_mean_planning_accuracy ~ experiment",
      "One-way Gaussian BAM without smooths (ANOVA); independent experiment cohorts and common residual variance assumed.",
      "One equally weighted mean per participant per experiment; IDs are not matched across experiments.",
      "Uses all available planning nodes/trials for participants and sequential simulations; no Exp3 rank-test subsampling.",
      "Model comparisons describe the fitted participant cohorts, not independent replications of model training."), file.path(directory, "model.txt"))
    print(table)
  }
}
# ==============================================================================
# nll: consolidated shared definitions (function bodies unchanged)
# ==============================================================================
validate_nll_fits <- function(data, label) {
  columns <- c("subject", "path_nll", "reconstruction_nll", "scaled_reconstruction_nll",
    "total_nll", "reward_nll_scale", "n_path_choices", "n_reward_choices")
  require_columns(data, columns, label)
  data <- as.data.frame(data[columns])
  data$subject <- as.character(data$subject)
  if (nrow(data) < 2L || anyNA(data$subject) || any(!nzchar(data$subject)) || anyDuplicated(data$subject))
    fail("Expected unique, nonmissing participant IDs and at least two participants: %s", label)
  numeric <- columns[-1]
  if (!all(vapply(data[numeric], is.numeric, logical(1))) || any(!is.finite(as.matrix(data[numeric]))))
    fail("Non-finite or nonnumeric NLL inputs: %s", label)
  for (name in c("n_path_choices", "n_reward_choices"))
    if (any(data[[name]] <= 0 | data[[name]] != floor(data[[name]]))) fail("Invalid observation counts: %s", label)
  close <- function(x, y) all(abs(x - y) <= 1e-8 * pmax(1, abs(x), abs(y)))
  if (!close(data$reward_nll_scale, .5)) fail("Expected reward-NLL scale 0.5: %s", label)
  if (!close(data$scaled_reconstruction_nll, data$reward_nll_scale * data$reconstruction_nll) ||
      !close(data$total_nll, data$path_nll + data$scaled_reconstruction_nll))
    fail("Inconsistent total/scaled NLL: %s", label)
  data[order(data$subject), , drop = FALSE]
}

compare_nll_fits <- function(full, variant, name) {
  full <- validate_nll_fits(full, "sequential")
  variant <- validate_nll_fits(variant, name)
  if (!identical(full$subject, variant$subject)) fail("Participant sets differ for %s and sequential", name)
  for (column in c("n_path_choices", "n_reward_choices"))
    if (any(full[[column]] != variant[[column]])) fail("Observation counts differ for %s: %s", name, column)
  difference <- variant$total_nll - full$total_nll
  # t.test fails explicitly on essentially constant differences; do not invent p values.
  directional <- t.test(difference, mu = 0, alternative = "greater")
  two_sided <- t.test(difference, mu = 0, alternative = "two.sided")
  list(result = tibble(term = paste0(name, "_minus_sequential"),
    reference = "sequential", participants = length(difference),
    mean_full_nll = mean(full$total_nll), mean_variant_nll = mean(variant$total_nll),
    variant = name,
    beta = mean(difference), se = sd(difference) / sqrt(length(difference)),
    t = unname(directional$statistic), df = unname(directional$parameter),
    p = directional$p.value, p_two_sided = two_sided$p.value,
    ci_low = two_sided$conf.int[1], ci_high = two_sided$conf.int[2],
    one_sided_95_lower = directional$conf.int[1],
    alternative = "variant NLL > sequential NLL", reward_nll_scale = .5),
    pairs = tibble(participant = full$subject,
      full_nll = full$total_nll, variant_nll = variant$total_nll, difference = difference,
      n_path_choices = full$n_path_choices, n_reward_choices = full$n_reward_choices,
      variant = name))
}

run_nll_comparison_stats <- function(experiment, options) {
  if (!experiment %in% c("exp1", "exp2")) fail("NLL comparisons require Exp1 or Exp2")
  variants <- intersect(c("simultaneous", "independent"), options$sources)
  if (!length(variants)) { message("Skipping NLL comparisons: no variant sources selected."); return(invisible(NULL)) }
  # Do not mix an old independent-model cohort into the new-cohort analyses.
  # Retain the planned two-comparison family conservatively when one is pending.
  inputs <- stats_inputs(experiment, options$root)
  available_variants <- c("simultaneous", "independent")
  available_variants <- available_variants[vapply(inputs[available_variants], function(path)
    file.exists(performance_fit_path(path, options$root)), logical(1))]
  models <- c("sequential", union(variants, available_variants))
  paths <- setNames(vapply(inputs[models], performance_fit_path, character(1), root = options$root), models)
  fits <- lapply(paths, function(path) validate_nll_fits(performance_read(path), path))
  comparisons <- lapply(models[-1], function(name) compare_nll_fits(fits$sequential, fits[[name]], name))
  results <- bind_rows(lapply(comparisons, `[[`, "result"))
  results$p_holm <- p.adjust(results$p, method = "holm", n = 2L)
  results$p_two_sided_holm <- p.adjust(results$p_two_sided, method = "holm", n = 2L)
  if (options$check) { message("Validated selected-fit NLL inputs for ", experiment); return(invisible(paths)) }
  for (name in variants) {
    directory <- file.path(options$output, experiment, name, "nll_vs_sequential")
    row <- filter(results, variant == name)
    export_stats_terms(row, directory)
    write_csv(row, file.path(directory, "comparisons.csv"))
    write_csv(comparisons[[match(name, models[-1])]]$pairs, file.path(directory, "participant_differences.csv"))
    for (stat in c("t", "p_holm", "p_two_sided", "p_two_sided_holm", "one_sided_95_lower")) {
      value <- row[[stat]]
      formatted <- if (startsWith(stat, "p") && value < .001) "< .001" else sprintf("%.3f", value)
      writeLines(formatted, file.path(directory, paste0(row$term, "_", stat, ".tex")))
    }
    performance_manifest(unname(paths), directory)
    writeLines(c("Paired t test: one equally weighted selected-fit total NLL per participant and model.",
      "Difference = variant minus full sequential model; positive values favor the full model.",
      "Total NLL = path_nll + 0.5 * reconstruction_nll, matching the NLL bar plot.",
      "Primary p: one-sided greater-than-zero; p_holm adjusts BOTH variant comparisons within this experiment.",
      paste0("Computed variants: ", paste(models[-1], collapse = ", "),
        "; Holm n=2 retains the planned family, treating an uncomputed comparison conservatively."),
      "ci_low/ci_high: unadjusted TWO-SIDED 95% t interval for the mean difference (not a regression slope).",
      "one_sided_95_lower: unadjusted one-sided 95% lower bound; upper bound is infinity.",
      "Two-sided p values and their Holm correction are also exported as sensitivity results.",
      "CSV beta field is the raw mean NLL difference, retained for compatibility with statistic-fragment exports.",
      "Identical participant sets, per-participant observation counts and reward-NLL scales are required.",
      "These are in-sample selected-fit comparisons, not held-out prediction tests or likelihood-ratio tests.",
      "Inference is across participants conditional on the trained models; grid selection and model-training uncertainty are not propagated.",
      "The directional analysis is exploratory if chosen after inspecting the NLL results."), file.path(directory, "model.txt"))
    print(row)
  }
  invisible(results)
}
# ==============================================================================
# best_branch_timing: consolidated shared definitions (function bodies unchanged)
# ==============================================================================
best_branch_timing_stats_data <- function(data) {
  data %>% filter(condition == "planning", observation_step %in% 6:8,
    is.finite(current_branch_best_reward), is.finite(max_other_branch_reward),
    max_other_branch_reward > current_branch_best_reward,
    is.finite(current_branch_worst_reward), is.finite(min_other_branch_reward),
    min_other_branch_reward < current_branch_worst_reward,
    other_branch_timing %in% c("before", "after"),
    worst_other_branch_timing %in% c("before", "after"),
    other_branch_timing != worst_other_branch_timing) %>%
    mutate(best_branch_timing = factor(other_branch_timing, levels = c("after", "before")))
}

run_best_branch_timing_stats <- function(data, directory) {
  selected <- best_branch_timing_stats_data(data)
  if (!nrow(selected) || anyNA(selected$accuracy) ||
      any(!selected$accuracy %in% c(0, 1)) ||
      n_distinct(selected$best_branch_timing) != 2L)
    stop("Best-branch timing requires binary observations and both timing conditions: ", directory)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write_csv(selected %>% count(best_branch_timing, name = "observations"),
    file.path(directory, "counts.csv"))
  spec <- independent_random_formula(selected, "accuracy", "best_branch_timing")
  spec$data$participant <- factor(spec$data$participant)
  write_csv(spec$mapping, file.path(directory, "random_design.csv"))
  message("Fitting glmer: ", directory, " | rows=", nrow(selected),
    " | participants=", n_distinct(selected$participant))
  fit <- lme4::glmer(spec$formula, data = spec$data, family = binomial(link = "logit"),
    nAGQ = 1L, control = lme4::glmerControl(optimizer = "bobyqa",
      check.rankX = "message+drop.cols", optCtrl = list(maxfun = 2e5)))
  saveRDS(fit, file.path(directory, "model.rds"))
  writeLines(c(paste("Formula:", deparse1(spec$formula)),
    "Estimator: lme4::glmer; binomial-logit GLMM; maximum likelihood, Laplace approximation (nAGQ=1)",
    "Conceptual formula: accuracy ~ best_branch_timing + (1 + best_branch_timing || participant)",
    "Inference: two-sided Wald z tests; normal-Wald 95% confidence intervals",
    paste("Rows:", nrow(selected)), paste("Participants:", n_distinct(selected$participant)),
    paste("Singular:", lme4::isSingular(fit)),
    paste("Exp2 middle-branch leaves only (positions 6:8), global best/worst outside in distinct branches.",
      "Probe between best/worst; ambiguous ties excluded; after reference.",
      "All binary observations, no equal-rank/final-observation exclusion.")),
    file.path(directory, "model.txt"))
  export_best_branch_timing_fit(fit, directory)
  invisible(fit)
}

export_best_branch_timing_fit <- function(fit, directory) {
  if (inherits(fit, "bam")) {
    export_exp3_bam_fit(fit, directory, "best_branch_timingbefore")
    return(invisible(fit))
  }
  # A boundary variance is not an optimizer failure. Ignore only the exact
  # lme4 singularity notification; retain gradient/Hessian and other failures.
  messages <- fit@optinfo$conv$lme4$messages
  singular <- lme4::isSingular(fit)
  other_messages <- messages[!grepl(
    "^boundary \\(singular\\) fit: see help\\('isSingular'\\)$", messages)]
  optimizer_code <- fit@optinfo$conv$opt
  if (length(other_messages) || !length(optimizer_code) ||
      anyNA(optimizer_code) || any(optimizer_code != 0))
    stop("Best-branch timing convergence check failed; see ", directory)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  write_csv(as.data.frame(lme4::VarCorr(fit)), file.path(directory, "random_effect_variances.csv"))
  writeLines(c(paste("Optimizer code:", paste(optimizer_code, collapse = ", ")),
    paste("Singular:", singular), paste("Messages:", paste(messages, collapse = "; ")),
    "Export policy: singular-only fits with successful optimizer allowed; all other convergence messages rejected"),
    file.path(directory, "export_diagnostics.txt"))
  if (singular) message("Exporting singular fit; boundary variances recorded in ", directory)
  co <- coef(summary(fit))
  if (!"best_branch_timingbefore" %in% rownames(co) || any(!is.finite(co)))
    stop("Missing/non-finite best-branch timing estimates; see ", directory)
  table <- tibble(term = rownames(co), beta = co[, "Estimate"], se = co[, "Std. Error"],
    df = NA_real_, z = co[, "z value"], p = co[, "Pr(>|z|)"],
    ci_low = beta - qnorm(.975) * se, ci_high = beta + qnorm(.975) * se,
    odds_ratio = exp(beta), singular_fit = singular,
    estimator = "glmer", family = "binomial", link = "logit",
    convergence_messages = paste(messages, collapse = "; "))
  write_csv(table, file.path(directory, "all_coefficients.csv"))
  focal <- filter(table, term == "best_branch_timingbefore")
  export_stats_terms(focal, directory)
  writeLines(sprintf("%.3f", focal$z), file.path(directory, "best_branch_timingbefore_z.tex"))
  print(focal)
  invisible(fit)
}
# ==============================================================================
# adjacent_rank: consolidated shared definitions (function bodies unchanged)
# ==============================================================================
# With two levels, the within-participant F statistic is exactly paired t^2.
adjacent_rank_leaf_data <- function(data) {
  data[!is.na(data$is_leaf) & as.character(data$is_leaf) == "TRUE" &
    is.finite(data$final_rank) & is.finite(data$accuracy), , drop = FALSE]
}

adjacent_rank_anova <- function(data, ranks) {
  # final_rank is the minimum rank of all complete paths containing the node,
  # calculated upstream from the complete tree (also for non-leaf probes).
  data <- data[is.finite(data$final_rank) & is.finite(data$accuracy), , drop = FALSE]
  means <- aggregate(accuracy ~ participant + final_rank, data, mean)
  rows <- lapply(ranks, function(rank) {
    low <- means[means$final_rank == rank, c("participant", "accuracy")]
    high <- means[means$final_rank == rank + 1, c("participant", "accuracy")]
    paired <- merge(low, high, by = "participant", suffixes = c("_lower", "_higher"))
    n <- nrow(paired)
    difference <- paired$accuracy_higher - paired$accuracy_lower
    estimable <- n >= 2L && is.finite(var(difference)) && var(difference) > 0
    f <- if (estimable) n * mean(difference)^2 / var(difference) else NA_real_
    difference_se <- if (estimable) sd(difference) / sqrt(n) else NA_real_
    margin <- if (estimable) qt(.975, n - 1) * difference_se else NA_real_
    data.frame(rank_lower = rank, rank_higher = rank + 1, participants = n,
      mean_lower = if (n) mean(paired$accuracy_lower) else NA_real_,
      mean_higher = if (n) mean(paired$accuracy_higher) else NA_real_,
      difference = if (n) mean(difference) else NA_real_,
      difference_se = difference_se,
      difference_ci_low = if (estimable) mean(difference) - margin else NA_real_,
      difference_ci_high = if (estimable) mean(difference) + margin else NA_real_,
      F = f, df1 = 1, df2 = n - 1,
      partial_eta_squared = if (estimable) f / (f + n - 1) else NA_real_,
      p = if (estimable) pf(f, 1, n - 1, lower.tail = FALSE) else NA_real_,
      status = if (estimable) "ok" else "not_estimable",
      scope = "planning_all_probes_best_containing_path")
  })
  results <- do.call(rbind, rows)
  results$p_holm <- p.adjust(results$p, method = "holm", n = length(ranks))
  list(results = results, means = means)
}


# Optional cross-experiment entry point. Sourcing common.R never starts analyses.
if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) && identical(args[[1]], "--cross-experiment")) {
    run_cross_experiment_performance(args[-1])
  } else {
    cat("Shared statistics definitions. Run exp1.R, exp2.R, exp3.R, or exp2_40n.R.\n",
        "For set-size comparisons: common.R --cross-experiment [statistics options]\n")
  }
}

# Exposure-memory and quadratic-performance analyses used by the final tables.
require_fields <- function(data, fields, label) {
  missing <- setdiff(fields, names(data))
  if (length(missing)) stop(label, " missing: ", paste(missing, collapse = ", "))
}

read_planning_means <- function(path, experiment) {
  data <- read.csv(path, colClasses = c(subject = "character"))
  require_fields(data, c("subject", "condition", "node", "graph", "accuracy", "norm_reward"), path)
  data <- data[tolower(trimws(data$condition)) %in% "planning", ]
  limit <- switch(experiment, exp1 = 7, exp2 = 13, stop("Unknown experiment"))
  data <- data[!is.na(data$node) & data$node >= 1 & data$node < limit, ]
  if (!nrow(data) || anyNA(data$subject) || anyNA(data$graph)) stop("Invalid planning subject/trial IDs")
  if (any(!is.na(data$accuracy) & !data$accuracy %in% c(0, 1))) stop("Recall accuracy must be binary")
  if (any(is.finite(data$norm_reward) & (data$norm_reward < 0 | data$norm_reward > 1)))
    stop("Normalized reward outside [0,1]")
  rows <- lapply(split(data, data$subject), function(d) {
    # Every recalled node contributes once to accuracy; each trial contributes
    # only once to normalized reward, even when it has two recall observations.
    keys <- c("graph", intersect("tree_type", names(d)))
    trial_key <- do.call(paste, c(d[keys], sep = "\r"))
    rewards <- vapply(split(d$norm_reward, trial_key), function(x) {
      values <- unique(x[is.finite(x)])
      if (length(values) > 1) stop("Conflicting normalized rewards within a participant trial")
      if (length(values)) values else NA_real_
    }, numeric(1))
    recalls <- d$accuracy[is.finite(d$accuracy)]
    data.frame(subject = d$subject[1],
      accuracy = if (length(recalls)) mean(recalls) else NA_real_,
      norm_reward = if (any(is.finite(rewards))) mean(rewards, na.rm = TRUE) else NA_real_,
      recall_observations = length(recalls), planning_trials = sum(is.finite(rewards)))
  })
  do.call(rbind, rows)
}

parse_flag <- function(x) {
  value <- tolower(as.character(x))
  if (anyNA(value) || any(!value %in% c("true", "false", "1", "0"))) stop("Invalid fit flag")
  value %in% c("true", "1")
}

join_exposure_performance <- function(fits, performance) {
  require_fields(fits, c("subject", "status", "guess_rate", "sigma", "precision_inverse_sigma",
    "boundary_fit", "precision_unidentified_at_all_guess"), "Exposure fits")
  if (anyNA(fits$subject) || anyDuplicated(fits$subject)) stop("Expected one fit per subject")
  # Anchor the audit to planning participants; absent/failed exposure fits are
  # explicitly recorded rather than silently disappearing in an inner join.
  data <- merge(performance, fits, by = "subject", all.x = TRUE, sort = TRUE)
  data$p_miss <- data$guess_rate
  data$p_memory <- 1 - data$p_miss
  data$precision <- data$precision_inverse_sigma
  data$exclusion <- ""
  data$exclusion[is.na(data$status)] <- "missing_exposure_fit"
  failed <- !is.na(data$status) & data$status != "ok"
  data$exclusion[failed] <- "failed_exposure_fit"
  ok <- which(data$exclusion == "")
  if (length(ok)) {
    if (any(!is.finite(data$p_miss[ok]) | data$p_miss[ok] < 0 | data$p_miss[ok] > 1 |
            !is.finite(data$sigma[ok]) | data$sigma[ok] <= 0)) stop("Invalid fitted parameter")
    data$boundary_fit[ok] <- parse_flag(data$boundary_fit[ok])
    unidentified <- parse_flag(data$precision_unidentified_at_all_guess[ok]) | data$p_miss[ok] >= 1 - 1e-6
    data$exclusion[ok[unidentified]] <- "precision_unidentified_all_guess"
  }
  data
}

fit_associations <- function(data, directory) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  coefficients <- list(); correlations <- list(); model_summary <- list(); scaling <- list()
  for (outcome in c("norm_reward", "accuracy")) {
    d <- data[is.finite(data[[outcome]]), ]
    write.csv(d, file.path(directory, paste0(outcome, "_participant_data.csv")), row.names = FALSE)
    predictors <- "sigma"
    valid <- nrow(d) >= 5 && all(vapply(d[c(outcome, predictors)], function(x) sd(x) > 0, logical(1)))
    if (!valid) {
      writeLines("Not estimable: fewer than five participants or constant predictor/outcome",
        file.path(directory, paste0(outcome, "_model.txt")))
      for (predictor in predictors) coefficients[[length(coefficients) + 1L]] <- data.frame(
        outcome, term = paste0(predictor, "_z"), n = nrow(d), b = NA_real_, se = NA_real_,
        df = NA_real_, t = NA_real_, p = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
        status = "not_estimable")
      next
    }
    for (predictor in predictors) {
      scaling[[length(scaling) + 1L]] <- data.frame(outcome, predictor,
        mean = mean(d[[predictor]]), sd = sd(d[[predictor]]))
      d[[paste0(predictor, "_z")]] <- as.numeric(scale(d[[predictor]]))
      for (method in c("pearson", "spearman")) {
        test <- cor.test(d[[predictor]], d[[outcome]], method = method, exact = FALSE)
        correlations[[length(correlations) + 1L]] <- data.frame(outcome, predictor, method,
          n = nrow(d), estimate = unname(test$estimate), p = test$p.value,
          ci_low = if (method == "pearson") test$conf.int[1] else NA_real_,
          ci_high = if (method == "pearson") test$conf.int[2] else NA_real_)
      }
    }
    formula <- reformulate("sigma_z", response = outcome)
    fit <- lm(formula, data = d)
    if (fit$rank != 2L) {
      writeLines("Not estimable: collinear predictors", file.path(directory, paste0(outcome, "_model.txt")))
      for (predictor in predictors) coefficients[[length(coefficients) + 1L]] <- data.frame(
        outcome, term = paste0(predictor, "_z"), n = nrow(d), b = NA_real_, se = NA_real_,
        df = NA_real_, t = NA_real_, p = NA_real_, ci_low = NA_real_, ci_high = NA_real_, status = "rank_deficient")
      next
    }
    tab <- summary(fit)$coefficients
    ci <- confint(fit)
    for (term in rownames(tab)[-1]) coefficients[[length(coefficients) + 1L]] <- data.frame(
      outcome, term, n = nrow(d), b = tab[term, 1], se = tab[term, 2], df = df.residual(fit),
      t = tab[term, 3], p = tab[term, 4], ci_low = ci[term, 1], ci_high = ci[term, 2], status = "estimated")
    s <- summary(fit)
    f <- s$fstatistic
    model_summary[[length(model_summary) + 1L]] <- data.frame(outcome, n = nrow(d),
      r_squared = s$r.squared, adjusted_r_squared = s$adj.r.squared,
      F = unname(f[1]), df_num = unname(f[2]), df_den = unname(f[3]),
      p = pf(f[1], f[2], f[3], lower.tail = FALSE))
    saveRDS(fit, file.path(directory, paste0(outcome, "_model.rds")))
    write.csv(data.frame(subject = d$subject, fitted = fitted(fit), residual = residuals(fit),
      leverage = hatvalues(fit), cooks_distance = cooks.distance(fit)),
      file.path(directory, paste0(outcome, "_diagnostics.csv")), row.names = FALSE)
    writeLines(c(paste("Formula:", deparse(formula)),
      "OLS; equally weighted participant means; no participant random effects.",
      "Sigma z-scored within the outcome-specific analysis sample; response unstandardized.",
      "Two-sided t tests and 95% t intervals; fitted-parameter uncertainty not propagated."),
      file.path(directory, paste0(outcome, "_model.txt")))
  }
  tab <- do.call(rbind, coefficients)
  tab$p_holm <- p.adjust(tab$p, method = "holm", n = 2)
  write.csv(tab, file.path(directory, "coefficients.csv"), row.names = FALSE)
  if (length(correlations)) {
    tab <- do.call(rbind, correlations)
    tab$p_holm <- NA_real_
    for (method in unique(tab$method)) {
      i <- tab$method == method
      tab$p_holm[i] <- p.adjust(tab$p[i], method = "holm", n = 2)
    }
    write.csv(tab, file.path(directory, "correlations.csv"), row.names = FALSE)
  }
  if (length(model_summary)) write.csv(do.call(rbind, model_summary), file.path(directory, "model_summary.csv"), row.names = FALSE)
  if (length(scaling)) write.csv(do.call(rbind, scaling), file.path(directory, "scaling.csv"), row.names = FALSE)
}

run_exposure_associations <- function(experiment, behavior, fits_path, output) {
  if (!experiment %in% c("exp1", "exp2")) stop("Choose exp1 or exp2")
  performance <- read_planning_means(behavior, experiment)
  fits <- read.csv(fits_path, colClasses = c(subject = "character"))
  data <- join_exposure_performance(fits, performance)
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  write.csv(data, file.path(output, "participant_audit.csv"), row.names = FALSE)
  write.csv(fits[!fits$subject %in% performance$subject, ], file.path(output, "fits_without_planning.csv"), row.names = FALSE)
  write.csv(data.frame(path = c(normalizePath(behavior), normalizePath(fits_path)),
    md5 = unname(tools::md5sum(c(behavior, fits_path)))), file.path(output, "input_manifest.csv"), row.names = FALSE)
  eligible <- data[data$exclusion == "", ]
  if (!nrow(eligible)) stop("No identifiable exposure fits with planning performance")
  fit_associations(eligible, file.path(output, "all_identifiable_fits"))
  fit_associations(eligible[!parse_flag(eligible$boundary_fit), ], file.path(output, "nonboundary_sensitivity"))
  writeLines(c(
    "Exploratory: exposure-fitted sigma predicts planning-condition performance only.",
    "Sigma is the memory-distribution standard deviation, not inverse-sigma precision.",
    "All planning choices included; no correct-choice-only selection. Reward averaged once per graph/trial.",
    "Recall accuracy is a participant mean, not a binary dependent variable in these regressions.",
    "Sigma-only regressions; guessing probability is not a covariate.",
    "Pearson correlations are marginal associations; Spearman correlations are sensitivity analyses.",
    "Holm families: two slopes per experiment/sample; two correlations per method/experiment/sample.",
    "CIs are unadjusted. Primary sample includes identifiable boundary fits; sensitivity excludes all boundary fits.",
    "No first-stage uncertainty propagation or causal interpretation; inspect diagnostics and parameter recovery.",
    "These results are separate from manuscript statistics and are not automatically included in tables."),
    file.path(output, "README.txt"))
  message("Saved: ", output)
}


lambda_regression <- function(data, directory) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  variables <- c("sigma", "fitted_lambda")
  require_fields(data, variables, "Lambda regression")
  data <- data[Reduce(`&`, lapply(data[variables], is.finite)), ]
  result <- data.frame(term = "sigma_z", n = nrow(data),
    b = NA_real_, se = NA_real_, df = NA_real_, t = NA_real_,
    ci_low = NA_real_, ci_high = NA_real_, p = NA_real_, status = "not_estimable")
  model_note <- "Not estimable: insufficient observations or constant variable"
  if (nrow(data) >= 5 && all(vapply(data[variables], function(x) sd(x) > 0, logical(1)))) {
    scaling <- data.frame(variable = variables[1],
      mean = vapply(data[variables[1]], mean, numeric(1)),
      sd = vapply(data[variables[1]], sd, numeric(1)))
    for (name in variables[1]) data[[paste0(name, "_z")]] <- as.numeric(scale(data[[name]]))
    fit <- lm(fitted_lambda ~ sigma_z, data = data)
    if (fit$rank != 2L) {
      result$status <- "rank_deficient"
      model_note <- "Not estimable: collinear predictors"
    } else {
      summary <- summary(fit); tab <- summary$coefficients[result$term, , drop = FALSE]
      ci <- confint(fit)[result$term, , drop = FALSE]
      result$b <- tab[, 1]; result$se <- tab[, 2]; result$t <- tab[, 3]; result$p <- tab[, 4]
      result$df <- df.residual(fit); result$ci_low <- ci[, 1]; result$ci_high <- ci[, 2]
      result$status <- "estimated"
      saveRDS(fit, file.path(directory, "model.rds"))
      write.csv(scaling, file.path(directory, "scaling.csv"), row.names = FALSE)
      write.csv(data.frame(subject = data$subject, fitted = fitted(fit), residual = residuals(fit),
        leverage = hatvalues(fit), cooks_distance = cooks.distance(fit)),
        file.path(directory, "diagnostics.csv"), row.names = FALSE)
      f <- summary$fstatistic
      write.csv(data.frame(n = nrow(data), r_squared = summary$r.squared,
        adjusted_r_squared = summary$adj.r.squared, F = unname(f[1]),
        df_num = unname(f[2]), df_den = unname(f[3]), p = pf(f[1], f[2], f[3], lower.tail = FALSE)),
        file.path(directory, "model_summary.csv"), row.names = FALSE)
      model_note <- "Estimated by ordinary least squares; one equally weighted row per participant"
    }
  }
  write.csv(data, file.path(directory, "participant_data.csv"), row.names = FALSE)
  write.csv(result, file.path(directory, "coefficients.csv"), row.names = FALSE)
  writeLines(c("Formula: fitted_lambda ~ scale(sigma)", model_note,
    "Raw lambda outcome; predictors standardized; no interaction or log transform.",
    "Two-sided t tests and 95% t confidence intervals; unadjusted p values."), file.path(directory, "model.txt"))
  result
}

run_lambda_associations <- function(planning_path, exposure_path, output) {
  planning <- read.csv(planning_path, colClasses = c(subject = "character"))
  require_fields(planning, c("subject", "best_lambda"), planning_path)
  if (anyNA(planning$subject) || anyDuplicated(planning$subject) ||
      any(!is.finite(planning$best_lambda) | planning$best_lambda <= 0)) stop("Invalid planning fits")
  performance <- data.frame(subject = planning$subject, fitted_lambda = planning$best_lambda)
  exposure <- read.csv(exposure_path, colClasses = c(subject = "character"))
  audit <- join_exposure_performance(exposure, performance)
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  write.csv(audit, file.path(output, "participant_audit.csv"), row.names = FALSE)
  write.csv(exposure[!exposure$subject %in% planning$subject, ],
    file.path(output, "exposure_without_planning.csv"), row.names = FALSE)
  write.csv(data.frame(path = normalizePath(c(planning_path, exposure_path)),
    md5 = unname(tools::md5sum(c(planning_path, exposure_path)))),
    file.path(output, "input_manifest.csv"), row.names = FALSE)
  eligible <- audit[audit$exclusion == "", ]
  if (!nrow(eligible)) stop("No eligible matched participants")
  samples <- list(all_identifiable_fits = eligible,
    nonboundary_sensitivity = eligible[!parse_flag(eligible$boundary_fit), ])
  for (sample in names(samples)) {
    directory <- file.path(output, sample)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    result <- lambda_regression(samples[[sample]], directory)
    print(result)
  }
  writeLines(c("Planning lambda: full sequential model; one fitted value per participant.",
    "Exposure memory spread (sigma); raw lambda, no log transform.",
    "Sigma-only OLS: fitted_lambda ~ scale(sigma); no interaction or random effects.",
    "Two-sided t tests and 95% t confidence intervals; sigma is the sole predictor.",
    "P values are unadjusted; exploratory analyses, not independent confirmatory tests.",
    "Includes identifiable boundary fits; separate sensitivity excludes boundary exposure fits.",
    "Missing/failed exposure fits and all-guess unidentified precision are excluded.",
    "Neither exposure nor planning parameter-estimation uncertainty is propagated; inspect residuals and influence for skew/grid-bound effects.",
    "Legacy correlations.csv files, if present from older runs, are not used by this analysis or its table."),
    file.path(output, "README.txt"))
  message("Saved: ", output)
}


quadratic_node_performance <- function(input, output, data = NULL, data_source = "participants") {
  if (is.null(data)) data <- read.csv(input, stringsAsFactors = FALSE)
  required <- c("subject", "graph", "condition", "norm_reward", "accuracy",
    "actual_reward", "sibling_reward", "parent_reward", "aunt_reward", "is_leaf")
  if (!all(required %in% names(data))) stop("Missing columns: ", paste(setdiff(required, names(data)), collapse=", "))
  if (anyNA(data$subject) || anyNA(data$condition)) stop("Missing participant/condition")
  data <- data[tolower(trimws(data$condition)) == "planning", , drop = FALSE]
  if (!nrow(data)) stop("No planning observations")
  if (anyNA(data$graph)) stop("Missing trial identity")
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  write.csv(data.frame(input=normalizePath(input), md5=unname(tools::md5sum(input))),
    file.path(output,"input_provenance.csv"), row.names=FALSE)
  # Do not double-weight trials that contribute multiple recall probes.
  rewards <- unique(data[c("subject", "graph", "norm_reward")])
  if (anyDuplicated(rewards[c("subject", "graph")])) stop("Conflicting normalized rewards within participant/trial")
  finite <- is.finite(rewards$norm_reward)
  if (any(rewards$norm_reward[finite] < 0 | rewards$norm_reward[finite] > 1)) stop("Normalized reward outside [0,1]")
  means <- aggregate(norm_reward ~ subject, rewards[finite, , drop=FALSE], mean)
  counts <- aggregate(graph ~ subject, rewards[finite, , drop=FALSE], length)
  names(counts)[2] <- "reward_trials"
  means <- merge(means, counts, by="subject")
  first <- fit_individual_context_glms(data, include_node_quadratic=TRUE,
    include_sibling_quadratic=TRUE)
  write.csv(attr(first,"context_coefficient_scaling"), file.path(output,"first_stage_scaling.csv"), row.names=FALSE)
  sample <- merge(first, means, by="subject", all.x=TRUE)
  sample$included <- sample$context_fit_status == "ok" &
    is.finite(sample$node_reward_squared_coefficient) & is.finite(sample$norm_reward)
  write.csv(sample, file.path(output,"participant_coefficients_and_performance.csv"), row.names=FALSE)
  d <- sample[sample$included, , drop=FALSE]
  spread <- sd(d$node_reward_squared_coefficient)
  if (nrow(d) < 5L || !is.finite(spread) || spread <= 0) stop("Insufficient accepted participants or constant quadratic coefficient")
  center <- mean(d$node_reward_squared_coefficient)
  d$quadratic_coefficient_z <- (d$node_reward_squared_coefficient-center)/spread
  fit <- lm(norm_reward ~ quadratic_coefficient_z, data=d, na.action=na.fail)
  saveRDS(fit, file.path(output,"model.rds"))
  co <- coef(summary(fit)); ci <- confint(fit)
  if (any(!is.finite(co)) || any(!is.finite(ci))) stop("Nonfinite performance regression inference")
  result <- data.frame(term=rownames(co), b=co[,1], se=co[,2], t=co[,3],
    df=df.residual(fit), p=co[,4], ci_low=ci[,1], ci_high=ci[,2], n=nrow(d))
  write.csv(result, file.path(output,"coefficients.csv"), row.names=FALSE)
  write.csv(data.frame(term="node_reward_squared_coefficient", mean=center, sd=spread),
    file.path(output,"second_stage_scaling.csv"), row.names=FALSE)
  writeLines(c(
    paste("Source:", data_source, "; planning trials; Exp1/Exp2 and sources analyzed separately."),
    "First stage: separate unpooled binomial-logit GLM per participant:",
    "accuracy ~ node_z + sibling_z + parent_z + aunt_z + node_squared_z + sibling_squared_z",
    "Reward and raw reward squared standardized separately across complete observations within experiment and source.",
    "Parent/aunt set to zero for non-leaf observations before scaling.",
    "Flagged first-stage fits excluded, following the existing contextual-GLM policy; see participant audit.",
    "Second stage: OLS norm_reward ~ scale(node_reward_squared_coefficient).",
    "One equally weighted participant mean; normalized reward counted once per planning trial.",
    "For sequential simulations, trials and seeds are pooled within fitted participant, not treated as separate participants.",
    "Two-sided t tests, 95% t intervals, no multiplicity correction.",
    "Exploratory plug-in analysis; first-stage coefficient uncertainty is not propagated.",
    paste("Participants:", nrow(sample), "Included:", nrow(d), "Excluded:", sum(!sample$included))),
    file.path(output,"model.txt"))
  print(result, row.names=FALSE)
  invisible(result)
}
