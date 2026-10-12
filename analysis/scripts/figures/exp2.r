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
# Experiment 2 preregistered-style plots using participant-fitted simulations
# ==============================================================================

# Reuse the established participant-level averaging, visual styling, model
# labels, path resolution, and command-line parser. The Exp1 entry point is
# guarded by sys.nframe(), so sourcing it defines helpers without making plots.
source(project_path(
  "analysis",
  "scripts",
  "figures",
  "exp1.r"
))

default_exp2_main_simulation_tag <- paste0(
  "13_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_0.5_head_nonlin_cfg_8fbf3736337f_",
  "e_1_final_lr_0.0003_lr_plateau_200_300_",
  "pm_1_critic_coef_1_eb_0_rho_grid_n100_simseed2027_new"
)
default_exp2_exposure_simulation_tag <- paste0(
  "13_seeds_1-2-3-4-5_sel_2026_mc1_eval2026_",
  "reward_nll_scale_1_exposure_head_cfg_768442bbc34a_",
  "ritic_off_gradient_scale_1_final_lr_0.0003_",
  "lr_plateau_200_300_pm_1_eb_0_n100_simseed2027"
)
default_exp2_independent_simulation_tag <- paste0(
  "independent_channels_13_seeds_1-2-3-4-5_sel_2026_mc1_",
  "eval2026_reward_nll_scale_0.5_head_nonlin_cfg_e4e00107acfb_",
  "_1_final_lr_0.0003_rnn_units_8_latent_dim_1_",
  "lr_plateau_200_300_rho_grid_n100_simseed2027_new"
)
default_exp2_simultaneous_simulation_tag <- paste0(
  "simultaneous_channel_13_seeds_1-2-3-4-5_sel_2026_mc1_",
  "eval2026_reward_nll_scale_0.5_head_nonlin_cfg_522e1194ca56_",
  "dient_scale_1_final_lr_0.0003_lr_plateau_200_300_",
  "critic_coef_1_rho_grid_n100_simseed2027_new"
)
default_exp2_temporal_simulation_tag <- paste0(
  "later_presentation_resource_13_seeds_1-2-3-4-5_",
  "sel_2026_mc1_eval2026_reward_nll_scale_0.5_",
  "later_order_inverse_rank_precision_power_law_",
  "n100_simseed2027"
)

prepare_fitted_exp2_model_data <- function(
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
    "Experiment 2 fitted-simulation CSV"
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
  if (any(!simulation$node %in% 1:12)) {
    fail("The fitted simulation contains nodes outside the 13-node task.")
  }
  if (
    any(!is.finite(simulation$actual_reward)) ||
      any(!is.finite(simulation$estimated_reward))
  ) {
    fail("The fitted simulation contains non-finite reward values.")
  }
  identity_columns <- c(
    "subject",
    "simulation_model_seed",
    "graph",
    "node"
  )
  if (anyDuplicated(simulation[identity_columns])) {
    fail("The fitted simulation contains duplicate subject/seed/graph/node rows.")
  }
  graph_sizes <- simulation %>%
    dplyr::count(subject, simulation_model_seed, graph, name = "node_count")
  if (any(graph_sizes$node_count != 12L)) {
    fail("Every Experiment 2 fitted simulation graph must contain 12 nodes.")
  }

  edges <- tibble::tribble(
    ~parent_node, ~node,
    0L, 1L,
    0L, 5L,
    0L, 9L,
    1L, 2L,
    1L, 3L,
    1L, 4L,
    5L, 6L,
    5L, 7L,
    5L, 8L,
    9L, 10L,
    9L, 11L,
    9L, 12L
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
  cousin_links <- fitted_cousin_links(12L)

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

derive_exp2_save_directory <- function(
    simulation,
    simulation_path,
    override = NULL) {
  if (!is.null(override) && nzchar(trimws(override))) {
    return(gsub("[^A-Za-z0-9_-]", "_", trimws(override)))
  }
  configuration_hash <- character()
  if ("participant_fit_configuration_hash" %in% names(simulation)) {
    configuration_hash <- unique(stats::na.omit(
      simulation$participant_fit_configuration_hash
    ))
  }
  if (length(configuration_hash) != 1L) {
    fail(
      paste0(
        "Expected one participant_fit_configuration_hash; pass ",
        "--output-label LABEL explicitly."
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
    "exp2_preregistered_fitted_cfg_",
    configuration_hash,
    simulation_tail
  )
}

process_exp2_fitted_simulation <- function(
    main_simulation_argument = NULL,
    output_label = NULL,
    independent_simulation_argument = NULL,
    simultaneous_simulation_argument = NULL,
    temporal_simulation_argument = NULL,
    exposure_simulation_argument = NULL,
    skip_coef_plot = FALSE) {
  main_simulation_path <- resolve_fitted_simulation_path(
    main_simulation_argument,
    default_exp2_main_simulation_tag,
    "Main VAE Experiment 2 fitted-simulation CSV"
  )
  exposure_simulation_path <- resolve_fitted_simulation_path(
    exposure_simulation_argument,
    default_exp2_exposure_simulation_tag,
    "Exposure Main VAE Experiment 2 fitted-simulation CSV",
    optional = TRUE
  )
  independent_simulation_path <- resolve_fitted_simulation_path(
    independent_simulation_argument,
    default_exp2_independent_simulation_tag,
    "Independent-channel Experiment 2 fitted-simulation CSV",
    optional = TRUE
  )
  simultaneous_simulation_path <- resolve_fitted_simulation_path(
    simultaneous_simulation_argument,
    default_exp2_simultaneous_simulation_tag,
    "Simultaneous-encoding Experiment 2 fitted-simulation CSV",
    optional = TRUE
  )
  behavior_path <- analysis_path("data", "processed_data", "exp2_beh_new.csv")
  require_existing_path(behavior_path, "Experiment 2 behavioral CSV")
  df_beh <- readr::read_csv(
    behavior_path,
    show_col_types = FALSE,
    progress = FALSE
  ) %>%
    dplyr::mutate(
      subject = as.character(subject),
      graph = as.character(graph)
    )
  df_beh <- add_fitted_human_cousin_reward(df_beh, 12L)

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
    prepare_fitted_exp2_model_data(
      model_input$path,
      model_input$name,
      model_input$key
    )
  })
  exposure_model_frame <- if (is.null(exposure_simulation_path)) {
    NULL
  } else {
    prepare_fitted_exp2_model_data(
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
  save_dir <- derive_exp2_save_directory(
    model_frames[[1]],
    main_simulation_path,
    output_label
  )
  y_min <- 0
  y_max <- 0.6
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
      experiment = "exp2",
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
    y_min = y_min,
    y_max = y_max,
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
    tree_size = 13L,
    output_directory = analysis_path("figures", save_dir)
  )
  aunt_reward_plot <- plot_fitted_binned_data_shade(
    df_beh %>% dplyr::filter(is_leaf == TRUE),
    df_context_model %>% dplyr::filter(is_leaf == TRUE, model_name %in% c("Main VAE", "Exposure")),
    dependent_var = "aunt_reward",
    x_lab = "Average Aunt Reward",
    y_lab = "Accuracy",
    figure_name = "aunt_reward",
    x_size = paired_line_plot_width,
    y_min = y_min,
    y_max = y_max,
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
    x_lab = "Average Sibling Reward",
    y_lab = "Accuracy",
    figure_name = "sibling_reward",
    x_size = paired_line_plot_width,
    y_min = y_min,
    y_max = y_max,
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
    y_min = y_min,
    y_max = y_max,
    x_step = 2,
    y_var = "accuracy",
    show_exposure = FALSE,
    append_exposure_suffix = FALSE,
    save_dir = save_dir
  )

  exp2_rank_path_nodes <- list(
    c(1L, 2L),
    c(1L, 3L),
    c(1L, 4L),
    c(5L, 6L),
    c(5L, 7L),
    c(5L, 8L),
    c(9L, 10L),
    c(9L, 11L),
    c(9L, 12L)
  )
  # Middle branch: parent node 5 and leaf nodes 6-8. Preserve full-tree ranks.
  # Middle branch has neither global extremum; the best/worst branches are opposite sides.
  plot_fitted_observation_final_rank_node_reward(
    df_beh = df_beh,
    df_model = df_model,
    node_count = 12L,
    path_nodes = exp2_rank_path_nodes,
    save_dir = save_dir,
    y_min = y_min,
    y_max = y_max,
    x_size = line_plot_width,
    final_better_color = "#46585D",
    output_suffix = "_middle_branch",
    probe_nodes = 6:8,
    compare_best_worst_branch_timing = TRUE,
    collapsed_only = TRUE,
    collapsed_reference_plot = aunt_reward_plot
  )
  plot_fitted_max_other_path_reward_accuracy(
    df_beh = df_beh,
    df_model = df_model,
    node_count = 12L,
    path_nodes = exp2_rank_path_nodes,
    save_dir = save_dir,
    reward_variable = "max_other_branch_path_reward",
    timing_variable = "relative_other_branch_timing",
    x_label = "Maximum path reward in other branches",
    timing_labels = c(
      "Other branches appeared\nbefore probed branch",
      "Other branches appeared\nafter probed branch"
    ),
    output_stem = "max_other_branch_path_reward",
    legend_position = "bottom",
    equal_count_bins = 10L,
    y_min = y_min,
    y_max = y_max,
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

  # Include all probes; parents use the best rank among paths containing them.
  df_beh_rank <- calculate_path_rank_exp2_beh(
    df_beh %>% dplyr::filter(condition == "planning")
  )
  model_rank_participant_lookup <- df_main_model %>%
    dplyr::distinct(subject, graph, fitted_subject, model_name)
  df_model_rank <- calculate_path_rank_exp2_model(
    df_main_model %>% dplyr::filter(condition == "planning")
  ) %>%
    dplyr::left_join(
      model_rank_participant_lookup,
      by = c("subject", "graph"),
      relationship = "many-to-one"
    )
  if (any(is.na(df_model_rank$fitted_subject))) {
    fail("Could not recover fitted participant identifiers after path ranking.")
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
    y_max = 0.8,
    y_var = "accuracy",
    x_min = 1,
    x_max = 9.5,
    x_step = 1,
    save_dir = save_dir
  )

  plot_fitted_model_nll(
    model_inputs = model_inputs, save_dir = save_dir,
    reference_plot = node_reward_plot, reference_width = paired_line_plot_width,
    bar_width_mm = 3.69
  )
  output_directory <- analysis_path("figures", save_dir)
  # Shared quadratic plotting functions are defined in the sourced Exp1 script.
  cat("Experiment 2 fitted simulation inputs:\n")
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

parse_exp2_figure_arguments <- function(arguments) {
  parsed <- list(
    main_simulation = NULL,
    output_label = NULL,
    independent_simulation = NULL,
    simultaneous_simulation = NULL,
    temporal_simulation = NULL,
    exposure_simulation = NULL,
    skip_coef_plot = FALSE
  )
  value_options <- c(
    "--main-simulation" = "main_simulation",
    "--output-label" = "output_label",
    "--independent-simulation" = "independent_simulation",
    "--simultaneous-simulation" = "simultaneous_simulation",
    "--temporal-simulation" = "temporal_simulation",
    "--exposure-simulation" = "exposure_simulation",
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

arguments <- parse_exp2_figure_arguments(commandArgs(trailingOnly = TRUE))
process_exp2_fitted_simulation(
  main_simulation_argument = arguments$main_simulation,
  output_label = arguments$output_label,
  independent_simulation_argument = arguments$independent_simulation,
  simultaneous_simulation_argument = arguments$simultaneous_simulation,
  temporal_simulation_argument = arguments$temporal_simulation,
  exposure_simulation_argument = arguments$exposure_simulation,
  skip_coef_plot = arguments$skip_coef_plot
)
