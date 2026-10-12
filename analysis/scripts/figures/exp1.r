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
# Script: Experiment 1 preregistered plots using participant-fitted simulations
# Description: Reuses the preregistered variables and regression formulas
#              while overlaying multiple participant-fitted model simulations.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2); library(cowplot)
  library(grid); library(rlang); library(lmerTest); library(here)
})
source(project_path("analysis/scripts/figures/utils.R"))
source(project_path("analysis/scripts/figures/plot_utils.R"))
source(project_path(
  "analysis",
  "scripts",
  "figures",
  "plot_utils_fitted_simulation.R"
))

# Shared tree/path definitions for Experiments 1--3. The statistics workflow
# imports these definitions only (without sourcing or running this plot script).
children_to_edges <- function(children, tree_type) {
  dplyr::bind_rows(lapply(names(children), function(parent_text) {
    tibble::tibble(
      tree_type = tree_type,
      parent_node = as.integer(parent_text),
      node = as.integer(children[[parent_text]])
    )
  }))
}

experiment_edges <- function(experiment) {
  if (identical(experiment, "exp1")) {
    return(tibble::tribble(
      ~tree_type, ~parent_node, ~node,
      "standard", 0L, 1L,
      "standard", 0L, 4L,
      "standard", 1L, 2L,
      "standard", 1L, 3L,
      "standard", 4L, 5L,
      "standard", 4L, 6L
    ))
  }
  if (identical(experiment, "exp2")) {
    return(tibble::tribble(
      ~tree_type, ~parent_node, ~node,
      "standard", 0L, 1L,
      "standard", 0L, 5L,
      "standard", 0L, 9L,
      "standard", 1L, 2L,
      "standard", 1L, 3L,
      "standard", 1L, 4L,
      "standard", 5L, 6L,
      "standard", 5L, 7L,
      "standard", 5L, 8L,
      "standard", 9L, 10L,
      "standard", 9L, 11L,
      "standard", 9L, 12L
    ))
  }
  if (!identical(experiment, "exp3")) {
    fail("Unsupported experiment: %s", experiment)
  }

  tree_children <- list(
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
    )
  )
  dplyr::bind_rows(lapply(names(tree_children), function(tree_type) {
    children_to_edges(tree_children[[tree_type]], tree_type)
  }))
}

tree_path_context <- function(edges, tree_type) {
  tree_edges <- edges %>%
    dplyr::filter(.data$tree_type == .env$tree_type) %>%
    dplyr::arrange(.data$parent_node, .data$node)
  children <- split(tree_edges$node, tree_edges$parent_node)
  root_children <- as.integer(children[["0"]])
  if (length(root_children) < 2L) {
    fail("Tree type '%s' must contain at least two root branches.", tree_type)
  }

  walk_paths <- function(node, path) {
    next_nodes <- children[[as.character(node)]]
    if (is.null(next_nodes) || length(next_nodes) == 0L) {
      return(list(as.integer(path)))
    }
    unlist(
      lapply(next_nodes, function(next_node) {
        walk_paths(next_node, c(path, as.integer(next_node)))
      }),
      recursive = FALSE
    )
  }
  collect_subtree <- function(node) {
    next_nodes <- children[[as.character(node)]]
    c(
      as.integer(node),
      if (is.null(next_nodes)) integer() else {
        unlist(lapply(next_nodes, collect_subtree), use.names = FALSE)
      }
    )
  }

  paths <- unlist(lapply(root_children, function(root_node) {
    walk_paths(root_node, as.integer(root_node))
  }), recursive = FALSE)
  if (length(paths) < 2L || any(lengths(paths) == 0L)) {
    fail("Tree type '%s' does not contain valid terminal paths.", tree_type)
  }

  node_count <- max(tree_edges$node)
  path_incidence <- matrix(
    0L,
    nrow = length(paths),
    ncol = node_count
  )
  path_branch <- integer(length(paths))
  path_completion_position <- integer(length(paths))
  for (path_index in seq_along(paths)) {
    path <- as.integer(paths[[path_index]])
    path_incidence[path_index, path] <- 1L
    path_branch[[path_index]] <- path[[1L]]
    path_completion_position[[path_index]] <- max(path)
  }

  node_branch <- rep(NA_integer_, node_count)
  for (root_node in root_children) {
    branch_nodes <- as.integer(collect_subtree(root_node))
    if (any(!is.na(node_branch[branch_nodes]))) {
      fail("Root branches overlap in tree type '%s'.", tree_type)
    }
    node_branch[branch_nodes] <- root_node
  }
  if (any(is.na(node_branch))) {
    fail("Some nodes lack a root-branch assignment in '%s'.", tree_type)
  }

  list(
    path_nodes = paths,
    path_incidence = path_incidence,
    path_branch = path_branch,
    path_completion_position = path_completion_position,
    node_branch = node_branch
  )
}

experiment_path_context <- function(edges) {
  tree_types <- unique(edges$tree_type)
  result <- lapply(tree_types, function(tree_type) {
    tree_path_context(edges, tree_type)
  })
  names(result) <- tree_types
  result
}

# Reproduce the Exp1 node-reward reference layout used by the linear
# bias-performance panels. Values only establish its axis labels and limits;
# no fitted data are needed to measure the panel geometry.
quadratic_performance_reference <- function() {
  d <- expand.grid(x=c(-4,4),y=c(.25,.65),source=c("participants","model"))
  d$source <- factor(d$source,levels=c("participants","model"))
  ggplot2::ggplot(d,ggplot2::aes(x,y)) + ggplot2::geom_blank() +
    ggplot2::facet_wrap(~source,nrow=1) +
    ggplot2::scale_x_continuous(breaks=seq(-4,4,2),expand=ggplot2::expansion(add=.5)) +
    ggplot2::scale_y_continuous(breaks=seq(.25,.65,.2)) +
    ggplot2::coord_cartesian(ylim=c(.25,.65)) +
    ggplot2::labs(x="Probed Reward",y="Accuracy") +
    plot_theme_axes(7,aspect_ratio=1.2) +
    ggplot2::theme(strip.clip="off",panel.spacing=grid::unit(1.31,"mm"),
      plot.margin=ggplot2::margin(0,1,0,1,"mm"))
}

plot_quadratic_performance <- function(output_directory = NULL,
    storage = STORAGE_ROOT,
    required = TRUE) {
  if (is.null(output_directory)) output_directory <- file.path(ANALYSIS_PLOT_DIR,"tests/quadratic_node_reward_performance")
  root <- file.path(ANALYSIS_SUMMARY_DIR,"tests/quadratic_node_reward_performance")
  paths <- file.path(root,c("exp1","exp2"),"participants","participant_coefficients_and_performance.csv")
  if (!all(file.exists(paths))) {
    if (required) stop("Run submit_quadratic_node_reward_performance.sh first; missing: ",paste(paths[!file.exists(paths)],collapse=", "))
    message("Skipping combined quadratic performance plot: participant analysis missing")
    return(invisible(NULL))
  }
  rows <- lapply(seq_along(paths),function(i) {
    d <- read.csv(paths[i])
    if (!all(c("sibling_reward_squared_coefficient","node_reward_squared_coefficient","included","norm_reward") %in% names(d)))
      stop("Outdated quadratic analysis; rerun the six-predictor test: ",paths[i])
    d <- d[!is.na(d$included) & d$included & is.finite(d$node_reward_squared_coefficient) & is.finite(d$norm_reward),]
    data.frame(subject=d$subject, coefficient=d$node_reward_squared_coefficient,
      norm_reward=d$norm_reward, experiment=c("Experiment 1","Experiment 2")[i])
  })
  d <- do.call(rbind,rows)
  d$experiment <- factor(d$experiment,levels=c("Experiment 1","Experiment 2"))
  p <- ggplot2::ggplot(d,ggplot2::aes(coefficient,norm_reward)) +
    ggplot2::geom_smooth(method="lm",formula=y~x,se=TRUE,linewidth=.45,
      color="#6e3934",fill="#6e3934",alpha=.12) +
    ggplot2::geom_point(size=.7,alpha=.55,color="#6e3934") +
    ggplot2::facet_wrap(~experiment,nrow=1,scales="fixed") +
    ggplot2::scale_x_continuous(breaks=scales::pretty_breaks(n=3)) +
    ggplot2::coord_cartesian(ylim=c(0,1)) +
    ggplot2::labs(x="Quadratic probed-reward\ncoefficient",y="Normalized reward") +
    plot_theme_axes(7,aspect_ratio=1.2) +
    ggplot2::theme(strip.clip="off",panel.spacing=grid::unit(1.31,"mm"),
      plot.margin=ggplot2::margin(0,1,0,1,"mm"))
  dir.create(output_directory,recursive=TRUE,showWarnings=FALSE)
  path <- file.path(output_directory,"node_squared_memory_allocation_planning_performance_participants.pdf")
  save_fitted_paired_reference_plot(p,path,quadratic_performance_reference(),
    width=40,height=36,side_padding_mm=2)
  message("Saved: ",path)
  invisible(p)
}

default_main_simulation_tag <- paste0(
  "7_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_0.5_head_nonlin_cfg_6101c496d7ee_",
  "epochs_320_gradient_scale_1_final_lr_0.0003_",
  "lr_plateau_200_300_rho_grid_n100_simseed2027_new"
)
default_exposure_simulation_tag <- paste0(
  "7_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_1_exposure_head_cfg_594ee6ad2a83_",
  "ochs_320_critic_off_gradient_scale_1_final_lr_0.0003_",
  "lr_plateau_200_300_n100_simseed2027"
)
default_independent_simulation_tag <- paste0(
  "independent_channels_7_seeds_1-2-3-4-5_sel_2026_mc1_",
  "eval2026_reward_nll_scale_0.5_head_nonlin_cfg_e4e00107acfb_",
  "_1_final_lr_0.0003_rnn_units_8_latent_dim_1_",
  "lr_plateau_200_300_rho_grid_n100_simseed2027_new"
)
default_simultaneous_simulation_tag <- paste0(
  "simultaneous_channel_7_seeds_1-2-3-4-5_sel_2026_mc1_",
  "eval2026_reward_nll_scale_0.5_head_nonlin_cfg_40adf29adafa_",
  "epochs_320_gradient_scale_1_final_lr_0.0003_",
  "lr_plateau_200_300_rho_grid_n100_simseed2027_new"
)
default_tree_size_exp2_simulation_tag <- paste0(
  "13_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_0.5_head_nonlin_cfg_8fbf3736337f_",
  "e_1_final_lr_0.0003_lr_plateau_200_300_",
  "pm_1_critic_coef_1_eb_0_rho_grid_n100_simseed2027_new"
)
default_tree_size_exp3_simulation_tag <- paste0(
  "31_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_0.5_head_nonlin_cfg_a9fcce63bb09_",
  "radient_scale_1_final_lr_0.0003_lr_plateau_200_300_",
  "eb_0.05_rns_rho_grid_n25_per_seed_tree_simseed2027"
)
default_temporal_simulation_tag <- paste0(
  "later_presentation_resource_7_seeds_1-2-3-4-5_",
  "sel_2026_mc1_eval2026_reward_nll_scale_0.5_",
  "later_order_inverse_rank_precision_power_law_n100_simseed2027"
)

fail <- function(...) {
  stop(sprintf(...), call. = FALSE)
}

resolve_fitted_simulation_path <- function(
    argument = NULL,
    default_tag = NULL,
    description = "Participant-fitted simulation CSV",
    optional = FALSE) {
  if (
    is.null(argument) ||
      !nzchar(trimws(argument)) ||
      identical(tolower(trimws(argument)), "default")
  ) {
    argument <- default_tag
  }
  if (
    is.null(argument) ||
      !nzchar(trimws(argument)) ||
      tolower(trimws(argument)) %in% c("none", "skip")
  ) {
    if (optional) {
      return(NULL)
    }
    fail("No %s was configured.", description)
  }
  argument <- trimws(argument)

  expanded_argument <- .path_absolute(argument, REPO_ROOT)
  if (file.exists(expanded_argument)) {
    return(normalizePath(expanded_argument, mustWork = TRUE))
  }
  if (grepl("[/\\\\]", argument)) {
    require_existing_path(expanded_argument, description)
  }

  if (basename(argument) %in% .simulation_inputs$public_file) {
    resolved_path <- simulation_input_path(basename(argument))
    require_existing_path(resolved_path, description)
    return(resolved_path)
  }
  file_name <- argument
  if (!grepl("[.]csv$", file_name, ignore.case = TRUE)) {
    file_name <- paste0(file_name, ".csv")
  }
  if (!startsWith(file_name, "participant_fitted_simulations_")) {
    file_name <- paste0("participant_fitted_simulations_", file_name)
  }

  resolved_path <- simulation_input_path(file_name)
  require_existing_path(resolved_path, description)
  resolved_path
}

require_columns <- function(data, required_columns, description) {
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0L) {
    fail(
      "%s is missing required columns: %s",
      description,
      paste(missing_columns, collapse = ", ")
    )
  }
}

summarize_related_reward <- function(data, links, output_name) {
  related_rewards <- data %>%
    dplyr::select(
      subject,
      simulation_model_seed,
      graph,
      related_node = node,
      related_reward = actual_reward
    )

  summary <- data %>%
    dplyr::distinct(subject, simulation_model_seed, graph, node) %>%
    dplyr::inner_join(
      links,
      by = "node",
      relationship = "many-to-many"
    ) %>%
    dplyr::left_join(
      related_rewards,
      by = c(
        "subject",
        "simulation_model_seed",
        "graph",
        "related_node"
      )
    ) %>%
    dplyr::group_by(subject, simulation_model_seed, graph, node) %>%
    dplyr::summarise(
      value = mean(related_reward, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(value = ifelse(is.nan(value), NA_real_, value))

  names(summary)[names(summary) == "value"] <- output_name
  summary
}

prepare_fitted_exp1_model_data <- function(
    simulation_path,
    model_name,
    model_key,
    condition_name = "planning") {
  condition_name <- match.arg(
    condition_name,
    c("planning", "exposure")
  )
  simulation <- readr::read_csv(
    simulation_path,
    show_col_types = FALSE,
    progress = FALSE
  )
  require_columns(
    simulation,
    c(
      "subject",
      "simulation_model_seed",
      "graph",
      "node",
      "actual_reward",
      "estimated_reward"
    ),
    "Fitted-simulation CSV"
  )

  simulation <- simulation %>%
    dplyr::mutate(
      model_name = as.character(model_name),
      model_key = as.character(model_key),
      subject = as.character(subject),
      simulation_model_seed = as.integer(simulation_model_seed),
      graph = as.integer(graph),
      node = as.integer(node),
      actual_reward = as.numeric(actual_reward),
      estimated_reward = as.numeric(estimated_reward)
    )
  if (any(!simulation$node %in% 1:6)) {
    fail("The fitted simulation contains nodes outside the 7-node task.")
  }
  if (
    any(!is.finite(simulation$actual_reward)) ||
      any(!is.finite(simulation$estimated_reward))
  ) {
    fail("The fitted simulation contains non-finite reward values.")
  }
  if (
    anyDuplicated(
      simulation[c("subject", "simulation_model_seed", "graph", "node")]
    )
  ) {
    fail("The fitted simulation contains duplicate subject/seed/graph/node rows.")
  }

  graph_sizes <- simulation %>%
    dplyr::count(subject, simulation_model_seed, graph, name = "node_count")
  if (any(graph_sizes$node_count != 6L)) {
    fail("Every fitted simulation graph must contain exactly six reward nodes.")
  }

  edges <- tibble::tribble(
    ~parent_node, ~node,
    0L, 1L,
    0L, 4L,
    1L, 2L,
    1L, 3L,
    4L, 5L,
    4L, 6L
  )
  node_relations <- edges %>%
    dplyr::transmute(node, parent_node)
  sibling_links <- edges %>%
    dplyr::inner_join(
      edges,
      by = "parent_node",
      suffix = c("", "_related"),
      relationship = "many-to-many"
    ) %>%
    dplyr::filter(node != node_related) %>%
    dplyr::transmute(node, related_node = node_related)
  aunt_links <- node_relations %>%
    dplyr::inner_join(
      sibling_links,
      by = c("parent_node" = "node"),
      relationship = "many-to-many"
    ) %>%
    dplyr::transmute(node, related_node)
  cousin_links <- fitted_cousin_links(6L)

  reward_lookup <- simulation %>%
    dplyr::select(
      subject,
      simulation_model_seed,
      graph,
      parent_node = node,
      parent_reward = actual_reward
    )
  sibling_summary <- summarize_related_reward(
    simulation,
    sibling_links,
    "sibling_reward"
  )
  aunt_summary <- summarize_related_reward(
    simulation,
    aunt_links,
    "aunt_reward"
  )
  cousin_summary <- summarize_related_reward(
    simulation,
    cousin_links,
    "cousin_reward"
  )

  simulation %>%
    dplyr::left_join(node_relations, by = "node") %>%
    dplyr::left_join(
      reward_lookup,
      by = c(
        "subject",
        "simulation_model_seed",
        "graph",
        "parent_node"
      )
    ) %>%
    dplyr::left_join(
      sibling_summary,
      by = c("subject", "simulation_model_seed", "graph", "node")
    ) %>%
    dplyr::left_join(
      aunt_summary,
      by = c("subject", "simulation_model_seed", "graph", "node")
    ) %>%
    dplyr::left_join(
      cousin_summary,
      by = c("subject", "simulation_model_seed", "graph", "node")
    ) %>%
    dplyr::mutate(
      condition = condition_name,
      parent_reward = ifelse(parent_node == 0L, 0, parent_reward),
      is_leaf = !node %in% unique(edges$parent_node),
      accuracy = as.numeric(estimated_reward == actual_reward),
      fitted_subject = subject,
      simulation_graph = graph,
      subject = paste0(
        model_key,
        "_subject_",
        fitted_subject,
        "_seed_",
        simulation_model_seed
      ),
      graph = paste0(
        model_key,
        "_subject_",
        fitted_subject,
        "_seed_",
        simulation_model_seed,
        "_graph_",
        simulation_graph
      )
    )
}

derive_save_directory <- function(simulation, simulation_path, override = NULL) {
  if (!is.null(override) && nzchar(trimws(override))) {
    safe_override <- gsub("[^A-Za-z0-9_-]", "_", trimws(override))
    return(safe_override)
  }

  configuration_hash <- character()
  if ("participant_fit_configuration_hash" %in% names(simulation)) {
    configuration_hash <- unique(
      stats::na.omit(simulation$participant_fit_configuration_hash)
    )
  }
  if (length(configuration_hash) != 1L) {
    fail(
      paste0(
        "Expected one participant_fit_configuration_hash; pass an explicit ",
        "output label as the second argument."
      )
    )
  }

  simulation_stem <- tools::file_path_sans_ext(canonical_simulation_basename(simulation_path))
  simulation_tail <- sub(
    ".*(_n[0-9]+_simseed[0-9]+(?:_new)?)$",
    "\\1",
    simulation_stem
  )
  if (identical(simulation_tail, simulation_stem)) {
    simulation_tail <- ""
  }
  paste0(
    "exp1_preregistered_fitted_cfg_",
    configuration_hash,
    simulation_tail
  )
}

process_fitted_simulation <- function(
    main_simulation_argument = NULL,
    output_label = NULL,
    independent_simulation_argument = NULL,
    simultaneous_simulation_argument = NULL,
    temporal_simulation_argument = NULL,
    exposure_simulation_argument = NULL,
    tree_size_exp2_simulation_argument = NULL,
    tree_size_exp3_simulation_argument = NULL,
    skip_coef_plot = FALSE) {
  main_simulation_path <- resolve_fitted_simulation_path(
    main_simulation_argument,
    default_main_simulation_tag,
    "Main VAE fitted-simulation CSV"
  )
  exposure_simulation_path <- resolve_fitted_simulation_path(
    exposure_simulation_argument,
    default_exposure_simulation_tag,
    "Exposure Main VAE fitted-simulation CSV",
    optional = TRUE
  )
  independent_simulation_path <- resolve_fitted_simulation_path(
    independent_simulation_argument,
    default_independent_simulation_tag,
    "Independent-channel fitted-simulation CSV",
    optional = TRUE
  )
  simultaneous_simulation_path <- resolve_fitted_simulation_path(
    simultaneous_simulation_argument,
    default_simultaneous_simulation_tag,
    "Simultaneous-encoding fitted-simulation CSV",
    optional = TRUE
  )
  tree_size_exp2_simulation_path <- NULL
  if (
    is.null(tree_size_exp2_simulation_argument) ||
      !nzchar(trimws(tree_size_exp2_simulation_argument)) ||
      identical(
        tolower(trimws(tree_size_exp2_simulation_argument)),
        "default"
      )
  ) {
    candidate_path <- simulation_input_path(
      paste0(
        "participant_fitted_simulations_",
        default_tree_size_exp2_simulation_tag,
        ".csv"
      )
    )
    if (file.exists(candidate_path)) {
      tree_size_exp2_simulation_path <- normalizePath(
        candidate_path,
        mustWork = TRUE
      )
    }
  } else if (
    !tolower(trimws(tree_size_exp2_simulation_argument)) %in%
      c("none", "skip")
  ) {
    tree_size_exp2_simulation_path <- resolve_fitted_simulation_path(
      tree_size_exp2_simulation_argument,
      description = "Experiment 2 tree-size fitted-simulation CSV"
    )
  }
  tree_size_exp3_simulation_path <- NULL
  if (
    is.null(tree_size_exp3_simulation_argument) ||
      !nzchar(trimws(tree_size_exp3_simulation_argument)) ||
      identical(
        tolower(trimws(tree_size_exp3_simulation_argument)),
        "default"
      )
  ) {
    candidate_path <- simulation_input_path(
      paste0(
        "participant_fitted_simulations_",
        default_tree_size_exp3_simulation_tag,
        ".csv"
      )
    )
    if (file.exists(candidate_path)) {
      tree_size_exp3_simulation_path <- normalizePath(
        candidate_path,
        mustWork = TRUE
      )
    }
  } else if (
    !tolower(trimws(tree_size_exp3_simulation_argument)) %in%
      c("none", "skip")
  ) {
    tree_size_exp3_simulation_path <- resolve_fitted_simulation_path(
      tree_size_exp3_simulation_argument,
      description = "Experiment 3 tree-size fitted-simulation CSV"
    )
  }
  behavior_path <- analysis_path("data", "processed_data", "exp1_beh_new.csv")
  require_existing_path(behavior_path, "Experiment 1 behavioral CSV")

  df_beh <- read.csv(behavior_path)
  df_beh$subject <- as.character(df_beh$subject)
  df_beh$graph <- as.character(df_beh$graph)
  df_beh <- add_fitted_human_cousin_reward(df_beh, 6L)
  model_inputs <- list(
    list(
      name = "Main VAE",
      display_name = "Sequential\nshared-representation\nmodel",
      key = "main_vae",
      path = main_simulation_path
    ),
    list(
      name = "Independent\nencoding\nmodel",
      key = "independent_channels",
      path = independent_simulation_path
    ),
    list(
      name = "Simultaneous\nencoding\nmodel",
      key = "simultaneous_channel",
      path = simultaneous_simulation_path
    )
  )
  model_inputs <- Filter(
    function(model_input) !is.null(model_input$path),
    model_inputs
  )
  model_frames <- lapply(model_inputs, function(model_input) {
    prepare_fitted_exp1_model_data(
      model_input$path,
      model_input$name,
      model_input$key
    )
  })
  exposure_model_frame <- if (is.null(exposure_simulation_path)) {
    NULL
  } else {
    prepare_fitted_exp1_model_data(
      exposure_simulation_path,
      "Main VAE",
      "main_vae_exposure",
      condition_name = "exposure"
    )
  }
  df_model <- dplyr::bind_rows(model_frames) %>%
    dplyr::mutate(
      model_name = factor(
        model_name,
        levels = vapply(model_inputs, `[[`, character(1), "name")
      )
    )
  df_main_model <- df_model %>%
    dplyr::filter(model_name == "Main VAE")
  df_context_model <- if (is.null(exposure_model_frame)) {
    df_model
  } else {
    dplyr::bind_rows(
      df_model %>% dplyr::mutate(model_name = as.character(model_name)),
      exposure_model_frame %>%
        dplyr::mutate(model_name = as.character(model_name))
    )
  }
  main_model_data <- model_frames[[1]]
  save_dir <- derive_save_directory(
    main_model_data,
    main_simulation_path,
    output_label
  )

  y_min <- 0.2
  y_max <- 0.8
  context_y_min <- 0.25
  context_y_max <- 0.65
  context_y_step <- 0.2
  line_plot_width <- 42.5
  paired_line_plot_width <- 40
  reviewer_participant_data <- prepare_fitted_reviewer_participant_data(
    df_beh,
    main_simulation_path, include_context_coefficients = FALSE
  )

  if (isTRUE(skip_coef_plot)) {
    cat(
      paste0(
        "Skipping coef_coef_by_source_accuracy.pdf from saved context statistics ",
        "(--skip-coef-plot).\n"
      )
    )
  } else {
    plot_fitted_context_stats_coefs(
      experiment = "exp1",
      show_exposure = !is.null(exposure_model_frame),
      figure_name = "coef_by_source",
      save_dir = save_dir
    )
  }
  node_reward_plot <- plot_fitted_binned_data_shade(
    df_beh,
    df_context_model %>% dplyr::filter(model_name %in% c("Main VAE", "Exposure")),
    dependent_var = "actual_reward",
    x_lab = "Probed Reward",
    y_lab = "Accuracy",
    figure_name = "node_reward",
    x_size = paired_line_plot_width,
    y_min = context_y_min,
    y_max = context_y_max,
    y_step = context_y_step,
    x_step = 2,
    y_var = "accuracy",
    show_exposure = FALSE,
    append_exposure_suffix = FALSE,
    save_dir = save_dir
  )
  plot_fitted_reviewer_participant_associations(
    reviewer_participant_data,
    save_dir = save_dir,
    reference_plot = node_reward_plot,
    reference_width = paired_line_plot_width,
    df_main_model = df_main_model,
    heatmaps_only = TRUE
  )
  plot_fitted_matched_seed_recovery(
    tree_size = 7L,
    output_directory = analysis_path("figures", save_dir)
  )
  aunt_reward_plot <- plot_fitted_binned_data_shade(
    df_beh %>% dplyr::filter(is_leaf == TRUE),
    df_context_model %>% dplyr::filter(is_leaf == TRUE, model_name %in% c("Main VAE", "Exposure")),
    dependent_var = "aunt_reward",
    x_lab = "Aunt Reward",
    y_lab = "Accuracy",
    figure_name = "aunt_reward",
    x_size = paired_line_plot_width,
    y_min = context_y_min,
    y_max = context_y_max,
    y_step = context_y_step,
    x_step = 2,
    y_var = "accuracy",
    save_dir = save_dir,
    show_model_comparison_legend = FALSE,
    show_exposure = FALSE,
    append_exposure_suffix = FALSE
  )
  plot_fitted_binned_data_shade(
    df_beh,
    df_context_model %>% dplyr::filter(model_name %in% c("Main VAE", "Exposure")),
    dependent_var = "sibling_reward",
    x_lab = "Sibling Reward",
    y_lab = "Accuracy",
    figure_name = "sibling_reward",
    x_size = paired_line_plot_width,
    y_min = context_y_min,
    y_max = context_y_max,
    y_step = context_y_step,
    x_step = 2,
    y_var = "accuracy",
    show_exposure = FALSE,
    append_exposure_suffix = FALSE,
    save_dir = save_dir
  )
  plot_fitted_binned_data_shade(
    df_beh %>% dplyr::filter(is_leaf == TRUE),
    df_context_model %>% dplyr::filter(is_leaf == TRUE, model_name %in% c("Main VAE", "Exposure")),
    dependent_var = "parent_reward",
    x_lab = "Parent Reward",
    y_lab = "Accuracy",
    figure_name = "parent_reward",
    x_size = paired_line_plot_width,
    y_min = context_y_min,
    y_max = context_y_max,
    y_step = context_y_step,
    x_step = 2,
    y_var = "accuracy",
    show_exposure = FALSE,
    append_exposure_suffix = FALSE,
    save_dir = save_dir
  )
  plot_fitted_reward_accuracy_source_panels(
    df_beh, df_model, save_dir, y_min = 0, y_max = 0.6,
    average_reward_labels = FALSE
  )

  exp1_rank_path_nodes <- list(
    c(1L, 2L),
    c(1L, 3L),
    c(4L, 5L),
    c(4L, 6L)
  )
  plot_fitted_max_other_path_reward_accuracy(
    df_beh = df_beh,
    df_model = df_model,
    node_count = 6L,
    path_nodes = exp1_rank_path_nodes,
    save_dir = save_dir,
    reward_variable = "max_other_branch_path_reward",
    timing_variable = "relative_other_branch_timing",
    x_label = "Maximum path reward in other branch",
    timing_labels = c(
      "Other branch appeared\nbefore probed branch",
      "Other branch appeared\nafter probed branch"
    ),
    output_stem = "max_other_branch_path_reward",
    legend_position = "bottom",
    equal_count_bins = 10L,
    y_min = 0.15,
    y_max = 0.75,
    x_size = 64,
    panel_spacing_mm = 5,
    aligned_panel_headers = TRUE,
    participant_model_gap_mm = 6.5 * 2 / 3,
    reference_plot = aunt_reward_plot,
    reference_width = paired_line_plot_width,
    reference_height = 36,
    match_panel_width = TRUE
  )


  simultaneous_rank_data <- df_model %>%
    dplyr::filter(as.character(model_name) == "Simultaneous\nencoding\nmodel")
  if (nrow(simultaneous_rank_data) > 0L) {
  }

  exp2_behavior_path <- analysis_path(
    "data",
    "processed_data",
    "exp2_beh_new.csv"
  )
  exp3_behavior_path <- analysis_path(
    "data",
    "processed_data",
    "exp3_beh.csv"
  )
  if (
    !is.null(tree_size_exp2_simulation_path) &&
      !is.null(tree_size_exp3_simulation_path) &&
      file.exists(exp2_behavior_path) &&
      file.exists(exp3_behavior_path)
  ) {
    exp2_behavior <- readr::read_csv(
      exp2_behavior_path,
      col_select = dplyr::all_of(c(
        "subject",
        "graph",
        "condition",
        "accuracy",
        "norm_reward"
      )),
      show_col_types = FALSE,
      progress = FALSE
    )
    require_columns(
      exp2_behavior,
      c("subject", "graph", "condition", "accuracy", "norm_reward"),
      "Experiment 2 tree-size behavioral data"
    )
    exp3_behavior <- readr::read_csv(
      exp3_behavior_path,
      col_select = dplyr::all_of(c(
        "subject",
        "graph",
        "condition",
        "tree_type",
        "accuracy",
        "norm_reward"
      )),
      show_col_types = FALSE,
      progress = FALSE
    )
    require_columns(
      exp3_behavior,
      c(
        "subject",
        "graph",
        "condition",
        "tree_type",
        "accuracy",
        "norm_reward"
      ),
      "Experiment 3 tree-size behavioral data"
    )
    exp2_simulation <- readr::read_csv(
      tree_size_exp2_simulation_path,
      col_select = dplyr::all_of(c(
        "subject",
        "simulation_model_seed",
        "graph",
        "actual_reward",
        "estimated_reward",
        "V",
        "min_path_reward",
        "max_path_reward"
      )),
      show_col_types = FALSE,
      progress = FALSE
    )
    require_columns(
      exp2_simulation,
      c(
        "subject",
        "simulation_model_seed",
        "graph",
        "actual_reward",
        "estimated_reward",
        "V",
        "min_path_reward",
        "max_path_reward"
      ),
      "Experiment 2 tree-size fitted simulation"
    )
    exp3_simulation <- readr::read_csv(
      tree_size_exp3_simulation_path,
      col_select = dplyr::all_of(c(
        "subject",
        "simulation_model_seed",
        "graph",
        "condition",
        "tree_type",
        "actual_reward",
        "estimated_reward",
        "V",
        "min_path_reward",
        "max_path_reward"
      )),
      show_col_types = FALSE,
      progress = FALSE
    )
    require_columns(
      exp3_simulation,
      c(
        "subject",
        "simulation_model_seed",
        "graph",
        "condition",
        "tree_type",
        "actual_reward",
        "estimated_reward",
        "V",
        "min_path_reward",
        "max_path_reward"
      ),
      "Experiment 3 tree-size fitted simulation"
    )
    tree_size_behavior <- dplyr::bind_rows(
      df_beh %>%
        dplyr::transmute(
          subject = as.character(subject),
          condition = as.character(condition),
          tree_size = 7L,
          trial_id = paste("exp1", subject, graph, sep = "|"),
          accuracy = as.numeric(accuracy),
          norm_reward = as.numeric(norm_reward)
        ),
      exp2_behavior %>%
        dplyr::transmute(
          subject = as.character(subject),
          condition = as.character(condition),
          tree_size = 13L,
          trial_id = paste("exp2", subject, graph, sep = "|"),
          accuracy = as.numeric(accuracy),
          norm_reward = as.numeric(norm_reward)
        ),
      exp3_behavior %>%
        dplyr::transmute(
          subject = as.character(subject),
          condition = as.character(condition),
          tree_size = 31L,
          trial_id = paste(
            "exp3",
            subject,
            tree_type,
            graph,
            sep = "|"
          ),
          accuracy = as.numeric(accuracy),
          norm_reward = as.numeric(norm_reward)
        )
    )
    tree_size_model <- dplyr::bind_rows(
      df_main_model %>%
        dplyr::transmute(
          fitted_subject = as.character(fitted_subject),
          condition = as.character(condition),
          tree_size = 7L,
          trial_id = paste(
            "exp1",
            fitted_subject,
            simulation_model_seed,
            simulation_graph,
            sep = "|"
          ),
          accuracy = as.numeric(accuracy),
          norm_reward = dplyr::if_else(
            as.numeric(max_path_reward) > as.numeric(min_path_reward),
            (as.numeric(V) - as.numeric(min_path_reward)) /
              (as.numeric(max_path_reward) - as.numeric(min_path_reward)),
            NA_real_
          )
        ),
      exp2_simulation %>%
        dplyr::transmute(
          fitted_subject = as.character(subject),
          condition = "planning",
          tree_size = 13L,
          trial_id = paste(
            "exp2",
            subject,
            simulation_model_seed,
            graph,
            sep = "|"
          ),
          accuracy = as.numeric(
            round(as.numeric(estimated_reward)) ==
              round(as.numeric(actual_reward))
          ),
          norm_reward = dplyr::if_else(
            as.numeric(max_path_reward) > as.numeric(min_path_reward),
            (as.numeric(V) - as.numeric(min_path_reward)) /
              (as.numeric(max_path_reward) - as.numeric(min_path_reward)),
            NA_real_
          )
        ),
      exp3_simulation %>%
        dplyr::transmute(
          fitted_subject = as.character(subject),
          condition = as.character(condition),
          tree_size = 31L,
          trial_id = paste(
            "exp3",
            subject,
            tree_type,
            simulation_model_seed,
            graph,
            sep = "|"
          ),
          accuracy = as.numeric(
            round(as.numeric(estimated_reward)) ==
              round(as.numeric(actual_reward))
          ),
          norm_reward = dplyr::if_else(
            as.numeric(max_path_reward) > as.numeric(min_path_reward),
            (as.numeric(V) - as.numeric(min_path_reward)) /
              (as.numeric(max_path_reward) - as.numeric(min_path_reward)),
            NA_real_
          )
        )
    )
    plot_fitted_tree_size_accuracy(
      df_beh = tree_size_behavior,
      df_model = tree_size_model,
      save_dir = save_dir,
      x_size = paired_line_plot_width,
      y_size = 36,
      reference_plot = node_reward_plot
    )
  } else {
    cat(paste0(
      "Skipping tree_size_accuracy.pdf because an Experiment 2 or 3 ",
      "behavior/fitted-simulation CSV is unavailable.\n"
    ))
  }

  # Include all probes; parents use the best rank among paths containing them.
  df_beh_rank <- calculate_path_rank_exp1_beh(
    df_beh %>% dplyr::filter(condition == "planning")
  )
  model_rank_participant_lookup <- df_main_model %>%
    dplyr::distinct(subject, graph, fitted_subject, model_name)
  df_model_rank <- calculate_path_rank_exp1_model(
    df_main_model %>% dplyr::filter(condition == "planning")
  ) %>%
    dplyr::left_join(
      model_rank_participant_lookup,
      by = c("subject", "graph"),
      relationship = "many-to-one"
    )
  if (any(is.na(df_model_rank$fitted_subject))) {
    fail(
      "Could not recover original participant identifiers after path ranking."
    )
  }
  df_beh_rank$path_rank <- df_beh_rank$p_rank
  df_model_rank$path_rank <- df_model_rank$p_rank
  # Pool individual probes directly within participant/rank, as in the ANOVA.
  # The plotting helper performs that aggregation before computing group SEMs.
  df_beh_rank_grouped <- df_beh_rank
  df_model_rank_grouped <- df_model_rank
  path_rank_plot <- plot_fitted_binned_data_bar(
    df_beh_rank_grouped,
    df_model_rank_grouped,
    dependent_var = "path_rank",
    x_lab = "Path Rank",
    y_lab = "Accuracy",
    figure_name = "path_rank",
    y_min = y_min,
    y_max = y_max,
    y_var = "accuracy",
    x_min = 1,
    x_max = 4.5,
    x_step = NULL,
    save_dir = save_dir
  )

  plot_fitted_model_nll(
    model_inputs = model_inputs, save_dir = save_dir,
    reference_plot = node_reward_plot, reference_width = paired_line_plot_width,
    bar_width_mm = 3.69
  )
  output_directory <- analysis_path("figures", save_dir)
  plot_quadratic_performance(required = TRUE)
  cat("Fitted simulation inputs:\n")
  for (model_input in model_inputs) {
    cat(sprintf(
      "  %s: %s\n",
      gsub("\n", " ", model_input$name, fixed = TRUE),
      model_input$path
    ))
  }
  if (!is.null(exposure_simulation_path)) {
    cat(sprintf(
      "  Exposure Main VAE (joint-condition coefficient plot): %s\n",
      exposure_simulation_path
    ))
  }
  if (is.null(independent_simulation_path)) {
    cat(
      paste0(
        "  Independent channels: omitted; pass ",
        "--independent-simulation when its fitted CSV is ready.\n"
      )
    )
  }
  print_local_scp_command(
    output_directory,
    file.path("analysis", "figures", "tests"),
    recursive = TRUE
  )
  print_local_scp_command(
    analysis_path("stats", "tests", save_dir),
    file.path("analysis", "summaries", "tests"),
    recursive = TRUE
  )
}

parse_figure_arguments <- function(arguments) {
  parsed <- list(
    main_simulation = NULL,
    output_label = NULL,
    independent_simulation = NULL,
    simultaneous_simulation = NULL,
    temporal_simulation = NULL,
    exposure_simulation = NULL,
    tree_size_exp2_simulation = NULL,
    tree_size_exp3_simulation = NULL,
    skip_coef_plot = FALSE
  )
  value_options <- c(
    "--main-simulation" = "main_simulation",
    "--output-label" = "output_label",
    "--independent-simulation" = "independent_simulation",
    "--simultaneous-simulation" = "simultaneous_simulation",
    "--temporal-simulation" = "temporal_simulation",
    "--exposure-simulation" = "exposure_simulation",
    "--tree-size-exp2-simulation" = "tree_size_exp2_simulation",
    "--tree-size-exp3-simulation" = "tree_size_exp3_simulation",
    # Backward-compatible alias for the former earlier-presentation default.
    "--earlier-simulation" = "temporal_simulation"
  )
  disable_options <- c(
    "--no-independent" = "independent_simulation",
    "--no-simultaneous" = "simultaneous_simulation",
    "--no-temporal" = "temporal_simulation",
    "--no-exposure" = "exposure_simulation",
    "--no-earlier" = "temporal_simulation"
  )
  positional_names <- c(
    "main_simulation",
    "output_label",
    "independent_simulation",
    "simultaneous_simulation",
    "temporal_simulation",
    "exposure_simulation"
  )
  positional_index <- 1L
  argument_index <- 1L
  while (argument_index <= length(arguments)) {
    argument <- arguments[[argument_index]]
    if (argument %in% names(value_options)) {
      if (argument_index == length(arguments)) {
        fail("%s requires a value.", argument)
      }
      parsed[[value_options[[argument]]]] <- arguments[[argument_index + 1L]]
      argument_index <- argument_index + 2L
    } else if (argument %in% names(disable_options)) {
      parsed[[disable_options[[argument]]]] <- "none"
      argument_index <- argument_index + 1L
    } else if (identical(argument, "--skip-coef-plot")) {
      parsed$skip_coef_plot <- TRUE
      argument_index <- argument_index + 1L
    } else if (startsWith(argument, "--")) {
      fail("Unknown option: %s", argument)
    } else {
      if (positional_index > length(positional_names)) {
        fail("Too many positional arguments.")
      }
      parsed[[positional_names[[positional_index]]]] <- argument
      positional_index <- positional_index + 1L
      argument_index <- argument_index + 1L
    }
  }
  if (
    !is.null(parsed$output_label) &&
      tolower(trimws(parsed$output_label)) == "default"
  ) {
    parsed$output_label <- NULL
  }
  parsed
}

if (sys.nframe() == 0L) {
  arguments <- parse_figure_arguments(commandArgs(trailingOnly = TRUE))
  process_fitted_simulation(
    main_simulation_argument = arguments$main_simulation,
    output_label = arguments$output_label,
    independent_simulation_argument = arguments$independent_simulation,
    simultaneous_simulation_argument = arguments$simultaneous_simulation,
    temporal_simulation_argument = arguments$temporal_simulation,
    exposure_simulation_argument = arguments$exposure_simulation,
    tree_size_exp2_simulation_argument =
      arguments$tree_size_exp2_simulation,
    tree_size_exp3_simulation_argument =
      arguments$tree_size_exp3_simulation,
    skip_coef_plot = arguments$skip_coef_plot
  )
}
