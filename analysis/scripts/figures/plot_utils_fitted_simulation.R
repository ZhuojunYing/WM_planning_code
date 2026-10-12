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
# Participant-fitted simulation plotting utilities
#
# These helpers intentionally average every fitted model observation for a
# participant (all model seeds and simulated trials) before averaging across
# participants. They are kept separate from plot_utils.R so that the original
# preregistered plotting behavior remains unchanged.
# ==============================================================================

source(project_path("analysis/scripts/figures/utils.R"))
source(project_path("analysis/scripts/figures/plot_utils.R"))

fitted_participant_color <- "#6e3934"
fitted_model_color <- "#899E8B"

fitted_timing_model_levels <- function(model_names) {
  preferred_levels <- c(
    "Main VAE",
    "Independent\nencoding\nmodel",
    "Simultaneous\nencoding\nmodel"
  )
  preferred_levels[preferred_levels %in% unique(as.character(model_names))]
}

fitted_timing_facet_labels <- function() {
  c(
    "participants" = " ",
    "Participants" = " ",
    "Main VAE" = "Sequential\nshared-representation\nmodel",
    "Independent\nencoding\nmodel" = "Independent\nencoding model",
    "Simultaneous\nencoding\nmodel" = "Simultaneous\nencoding model"
  )
}

fitted_model_colors <- function(model_names) {
  preferred_colors <- c(
    "Main VAE" = fitted_participant_color,
    "Sequential\nshared-representation\nmodel" = fitted_participant_color,
    "Exposure" = "#c2beb5",
    "Independent\nencoding\nmodel" = "#5C9999",
    "Node reward\nmodel" = "#4D7F80",
    "Simultaneous\nencoding\nmodel" = "#4D7F80",
    "Temporal\nmodel" = "#365959"
  )
  model_names <- unique(as.character(model_names))
  colors <- preferred_colors[model_names]
  colors[is.na(colors)] <- fitted_model_color
  colors
}

fitted_model_shapes <- function(model_names) {
  preferred_shapes <- c(
    "Main VAE" = 16,
    "Sequential\nshared-representation\nmodel" = 16,
    "Exposure" = 16,
    "Independent\nencoding\nmodel" = 16,
    "Node reward\nmodel" = 15,
    "Simultaneous\nencoding\nmodel" = 15,
    "Temporal\nmodel" = 17
  )
  model_names <- unique(as.character(model_names))
  shapes <- preferred_shapes[model_names]
  missing_names <- model_names[is.na(shapes)]
  if (length(missing_names) > 0L) {
    replacement <- rep(c(18, 3, 4, 8), length.out = length(missing_names))
    names(replacement) <- missing_names
    shapes[missing_names] <- replacement
  }
  shapes
}

fitted_model_linetypes <- function(model_names) {
  preferred_linetypes <- c(
    "Main VAE" = "solid",
    "Sequential\nshared-representation\nmodel" = "solid",
    "Exposure" = "solid",
    "Independent\nencoding\nmodel" = "longdash",
    "Node reward\nmodel" = "dotdash",
    "Simultaneous\nencoding\nmodel" = "dotdash",
    "Temporal\nmodel" = "dotted"
  )
  fallback_linetypes <- c(
    "twodash",
    "dashed",
    "1234",
    "F282"
  )
  model_names <- unique(as.character(model_names))
  linetypes <- preferred_linetypes[model_names]
  missing_names <- model_names[is.na(linetypes)]
  if (length(missing_names) > 0L) {
    replacement <- rep(fallback_linetypes, length.out = length(missing_names))
    names(replacement) <- missing_names
    linetypes[missing_names] <- replacement
  }
  linetypes
}

fitted_leaf_branch_groups <- function(node_count) {
  node_count <- as.integer(node_count)
  if (identical(node_count, 6L)) {
    return(list(2:3, 5:6))
  }
  if (identical(node_count, 12L)) {
    return(list(2:4, 6:8, 10:12))
  }
  stop(
    "Cousin-reward plots require the 7-node or 13-node tree.",
    call. = FALSE
  )
}

fitted_cousin_links <- function(node_count) {
  branch_groups <- fitted_leaf_branch_groups(node_count)
  dplyr::bind_rows(lapply(seq_along(branch_groups), function(branch_index) {
    tidyr::expand_grid(
      node = as.integer(branch_groups[[branch_index]]),
      related_node = as.integer(unlist(
        branch_groups[-branch_index],
        use.names = FALSE
      ))
    )
  }))
}

add_fitted_human_cousin_reward <- function(data, node_count) {
  require_columns(
    data,
    c("node", paste0("n", seq_len(node_count), "_reward")),
    "Human cousin-reward data"
  )
  branch_groups <- fitted_leaf_branch_groups(node_count)
  cousin_reward <- rep(NA_real_, nrow(data))
  node <- suppressWarnings(as.integer(data$node))
  for (branch_index in seq_along(branch_groups)) {
    target_rows <- which(node %in% branch_groups[[branch_index]])
    if (length(target_rows) == 0L) next
    cousin_nodes <- as.integer(unlist(
      branch_groups[-branch_index],
      use.names = FALSE
    ))
    reward_columns <- paste0("n", cousin_nodes, "_reward")
    cousin_reward[target_rows] <- rowMeans(
      as.matrix(data[target_rows, reward_columns, drop = FALSE]),
      na.rm = FALSE
    )
  }
  data$cousin_reward <- cousin_reward
  data
}

fitted_model_legend_theme <- function(
    base_font_size = 7,
    legend_position = "bottom") {
  ggplot2::theme(
    legend.position = legend_position,
    legend.direction = if (identical(legend_position, "right")) {
      "vertical"
    } else {
      "horizontal"
    },
    legend.justification = "center",
    legend.title = ggplot2::element_blank(),
    legend.text = ggplot2::element_text(
      size = base_font_size,
      color = "black",
      face = "plain"
    ),
    legend.key.height = grid::unit(2.5, "mm"),
    legend.key.width = grid::unit(7, "mm"),
    legend.spacing.x = grid::unit(1, "mm"),
    legend.margin = ggplot2::margin(0, 0, 0, 0, "mm"),
    legend.box.margin = ggplot2::margin(0, 0, 0, 0, "mm")
  )
}

save_fitted_model_comparison_legend <- function(
    model_names,
    save_dir,
    figure_name = "model_comparison_legend",
    x_size = 85,
    y_size = 24,
    base_font_size = 7) {
  model_names <- setdiff(
    unique(as.character(model_names)),
    "Temporal\nmodel"
  )
  if (length(model_names) == 0L) {
    stop("No fitted models were supplied for the comparison legend.", call. = FALSE)
  }
  legend_data <- expand.grid(
    model_name = model_names,
    legend_x = c(0, 1),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  legend_data$model_name <- factor(
    legend_data$model_name,
    levels = model_names
  )
  legend_data$legend_y <- as.integer(legend_data$model_name)
  model_colors <- fitted_model_colors(model_names)
  model_linetypes <- fitted_model_linetypes(model_names)
  model_shapes <- fitted_model_shapes(model_names)
  legend_plot <- ggplot2::ggplot(
    legend_data,
    ggplot2::aes(
      x = legend_x,
      y = legend_y,
      color = model_name,
      linetype = model_name,
      shape = model_name,
      group = model_name
    )
  ) +
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::geom_point(size = 1.2, stroke = 0.3) +
    ggplot2::scale_color_manual(
      name = NULL,
      values = model_colors,
      breaks = model_names,
      drop = FALSE
    ) +
    ggplot2::scale_linetype_manual(
      name = NULL,
      values = model_linetypes,
      breaks = model_names,
      drop = FALSE,
      guide = "none"
    ) +
    ggplot2::scale_shape_manual(
      name = NULL,
      values = model_shapes,
      breaks = model_names,
      drop = FALSE,
      guide = "none"
    ) +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        nrow = 2,
        byrow = TRUE,
        override.aes = list(
          linetype = unname(model_linetypes),
          shape = unname(model_shapes),
          linewidth = 0.7,
          size = 1.2
        )
      )
    ) +
    ggplot2::theme_void() +
    fitted_model_legend_theme(
      base_font_size = base_font_size,
      legend_position = "top"
    )
  legend_grob <- cowplot::get_legend(
    legend_plot
  )
  if (is.null(legend_grob)) {
    stop("Could not construct the fitted-model comparison legend.", call. = FALSE)
  }
  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0(figure_name, ".pdf")
  )
  ensure_parent_directory(output_path)
  legend_canvas <- cowplot::ggdraw() +
    cowplot::draw_grob(
      legend_grob,
      x = 0.03,
      y = 0.08,
      width = 0.94,
      height = 0.84
    )
  ggsave_pdf(
    output_path,
    plot = legend_canvas,
    width = x_size,
    height = y_size
  )
  cat(sprintf("Saved fitted-model legend to: %s\n", output_path))
  output_path
}

prepare_fitted_binned_plot_data <- function(
    df_beh,
    df_model = NULL,
    dependent_var,
    y_var = "accuracy",
    show_exposure = FALSE) {
  participant_data <- df_beh %>%
    dplyr::mutate(
      source = "participants",
      model_name = "Participants",
      fitted_plot_participant = as.character(subject)
    )

  if (is.null(df_model)) {
    df <- participant_data
  } else {
    required_model_columns <- c("fitted_subject", "model_name")
    missing_model_columns <- setdiff(required_model_columns, names(df_model))
    if (length(missing_model_columns) > 0L) {
      stop(
        paste0(
          "Fitted model data is missing: ",
          paste(missing_model_columns, collapse = ", "),
          "."
        ),
        call. = FALSE
      )
    }

    model_levels <- unique(as.character(df_model$model_name))

    model_data <- df_model %>%
      dplyr::mutate(
        source = "model",
        model_name = factor(model_name, levels = model_levels),
        fitted_plot_participant = as.character(fitted_subject)
      )
    df <- dplyr::bind_rows(participant_data, model_data)
  }

  df <- df %>%
    dplyr::mutate(
      source = factor(source, levels = c("participants", "model"))
    )

  if (!show_exposure) {
    df <- df %>% dplyr::filter(condition == "planning")
  }

  df %>%
    dplyr::group_by(
      source,
      model_name,
      condition,
      !!rlang::sym(dependent_var),
      fitted_plot_participant
    ) %>%
    dplyr::summarise(
      avg_error = mean(.data[[y_var]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::group_by(
      source,
      model_name,
      condition,
      !!rlang::sym(dependent_var)
    ) %>%
    dplyr::summarise(
      mean_error = mean(avg_error, na.rm = TRUE),
      se_error = stats::sd(avg_error, na.rm = TRUE) / sqrt(dplyr::n()),
      count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::filter(count >= 5 | source == "model") %>%
    dplyr::rename(x_value = !!rlang::sym(dependent_var))
}

prepare_fitted_equal_count_binned_plot_data <- function(
    df_beh,
    df_model = NULL,
    dependent_var,
    y_var = "accuracy",
    show_exposure = FALSE,
    equal_count_bins = 10L) {
  equal_count_bins <- as.integer(equal_count_bins)
  if (
    length(equal_count_bins) != 1L ||
      is.na(equal_count_bins) ||
      equal_count_bins < 2L
  ) {
    stop("equal_count_bins must be an integer of at least 2.", call. = FALSE)
  }

  participant_data <- df_beh %>%
    dplyr::mutate(
      source = "participants",
      model_name = "Participants",
      fitted_plot_participant = as.character(subject)
    )
  if (is.null(df_model)) {
    observations <- participant_data
  } else {
    required_model_columns <- c("fitted_subject", "model_name")
    missing_model_columns <- setdiff(required_model_columns, names(df_model))
    if (length(missing_model_columns) > 0L) {
      stop(
        paste0(
          "Fitted model data is missing: ",
          paste(missing_model_columns, collapse = ", "),
          "."
        ),
        call. = FALSE
      )
    }
    model_levels <- unique(as.character(df_model$model_name))
    model_data <- df_model %>%
      dplyr::mutate(
        source = "model",
        model_name = factor(model_name, levels = model_levels),
        fitted_plot_participant = as.character(fitted_subject)
      )
    observations <- dplyr::bind_rows(participant_data, model_data)
  }

  observations <- observations %>%
    dplyr::mutate(
      source = factor(source, levels = c("participants", "model")),
      .equal_count_x = as.numeric(.data[[dependent_var]]),
      .equal_count_y = as.numeric(.data[[y_var]]),
      .equal_count_row = dplyr::row_number()
    )
  if (!show_exposure) {
    observations <- observations %>% dplyr::filter(condition == "planning")
  }
  observations <- observations %>%
    dplyr::filter(
      is.finite(.equal_count_x),
      is.finite(.equal_count_y),
      !is.na(fitted_plot_participant)
    ) %>%
    dplyr::group_by(source, model_name, condition) %>%
    dplyr::arrange(
      .equal_count_x,
      fitted_plot_participant,
      .equal_count_row,
      .by_group = TRUE
    ) %>%
    dplyr::mutate(
      equal_count_bin = dplyr::ntile(dplyr::row_number(), equal_count_bins)
    ) %>%
    dplyr::ungroup()
  if (nrow(observations) == 0L) {
    stop("No finite observations were available for equal-count binning.", call. = FALSE)
  }

  bin_metadata <- observations %>%
    dplyr::group_by(source, model_name, condition, equal_count_bin) %>%
    dplyr::summarise(
      x_value = mean(.equal_count_x),
      x_min = min(.equal_count_x),
      x_max = max(.equal_count_x),
      observation_count = dplyr::n(),
      .groups = "drop"
    )
  participant_bins <- observations %>%
    dplyr::group_by(
      source,
      model_name,
      condition,
      equal_count_bin,
      fitted_plot_participant
    ) %>%
    dplyr::summarise(
      avg_error = mean(.equal_count_y),
      participant_observation_count = dplyr::n(),
      .groups = "drop"
    )
  participant_bins %>%
    dplyr::group_by(source, model_name, condition, equal_count_bin) %>%
    dplyr::summarise(
      mean_error = mean(avg_error),
      se_error = ifelse(
        dplyr::n() > 1L,
        stats::sd(avg_error) / sqrt(dplyr::n()),
        NA_real_
      ),
      count = dplyr::n_distinct(fitted_plot_participant),
      .groups = "drop"
    ) %>%
    dplyr::left_join(
      bin_metadata,
      by = c("source", "model_name", "condition", "equal_count_bin"),
      relationship = "one-to-one"
    ) %>%
    dplyr::filter(count >= 5L | source == "model") %>%
    dplyr::arrange(source, model_name, condition, equal_count_bin)
}

fitted_complete_recall_predictors <- function() {
  paste(
    "scale(actual_reward) + scale(actual_reward^2) +",
    "scale(sibling_reward) + scale(sibling_reward^2) +",
    "scale(aunt_reward) + scale(parent_reward)"
  )
}

prepare_fitted_complete_recall_data <- function(
    df_beh,
    condition_name = "planning") {
  condition_name <- match.arg(condition_name, c("planning", "exposure"))
  require_columns(
    df_beh,
    c(
      "subject",
      "condition",
      "accuracy",
      "actual_reward",
      "sibling_reward",
      "aunt_reward",
      "parent_reward",
      "is_leaf"
    ),
    "Behavioral complete-recall-model data"
  )
  df_beh %>%
    dplyr::filter(
      tolower(as.character(.data$condition)) == condition_name
    ) %>%
    dplyr::mutate(
      subject = as.character(subject),
      parent_reward = ifelse(!as.logical(is_leaf), 0, parent_reward),
      aunt_reward = ifelse(!as.logical(is_leaf), 0, aunt_reward)
    )
}

save_fitted_participant_recall_audit <- function(fit, save_dir) {
  directory <- analysis_path("stats", "tests", save_dir)
  ensure_dir(directory)
  prefix <- file.path(directory, "participant_recall_model")
  saveRDS(fit, paste0(prefix, ".rds"))
  write.csv(as.data.frame(lme4::VarCorr(fit)), paste0(prefix, "_variance.csv"), row.names = FALSE)
  deviations <- as.data.frame(lme4::ranef(fit, condVar = FALSE)$subject)
  conditional <- as.data.frame(stats::coef(fit)$subject)
  write.csv(data.frame(subject = rownames(deviations), deviations, check.names = FALSE),
    paste0(prefix, "_participant_deviations.csv"), row.names = FALSE)
  write.csv(data.frame(subject = rownames(conditional), conditional, check.names = FALSE),
    paste0(prefix, "_participant_coefficients.csv"), row.names = FALSE)
  fixed <- lme4::fixef(fit)
  write.csv(data.frame(term = names(fixed), estimate = unname(fixed)),
    paste0(prefix, "_fixed_effects.csv"), row.names = FALSE)
  writeLines(c(paste("Formula:", paste(deparse(stats::formula(fit)), collapse = " ")),
    "Data: planning condition only; this is the model used for reviewer participant coefficients.",
    paste("Singular:", lme4::isSingular(fit)),
    capture.output(summary(fit)), capture.output(sessionInfo())), paste0(prefix, "_diagnostics.txt"))
  message("Saved participant recall model and random-slope diagnostics: ", prefix)
}

fit_fitted_participant_complete_recall_model <- function(
    df_beh,
    y_var = "accuracy",
    condition_name = "planning") {
  regression_data <- prepare_fitted_complete_recall_data(
    df_beh,
    condition_name = condition_name
  )
  base_formula <- fitted_complete_recall_predictors()
  participant_formula <- stats::as.formula(paste0(
    y_var,
    " ~ ",
    base_formula,
    " + (",
    base_formula,
    " || subject)"
  ))

  if (identical(y_var, "accuracy")) {
    lme4::glmer(
      participant_formula,
      family = stats::binomial,
      data = regression_data,
      control = lme4::glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
      )
    )
  } else {
    lmerTest::lmer(
      participant_formula,
      data = regression_data,
      REML = FALSE
    )
  }
}

plot_fitted_exp1_preregistered_planning_coefs <- function(
    df_beh,
    df_model,
    df_exposure_model = NULL,
    y_var = "accuracy",
    x_lab = "Regression Weight",
    y_lab = "",
    figure_name = "coef_by_source",
    save_dir = "plots",
    x_size = 80,
    y_size = 36,
    base_font_size = 7,
    participant_fit = NULL) {
  required_model_columns <- c("model_name", "fitted_subject")
  missing_model_columns <- setdiff(required_model_columns, names(df_model))
  if (length(missing_model_columns) > 0L) {
    stop(
      paste0(
        "Fitted model coefficient data is missing: ",
        paste(missing_model_columns, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }
  if (!is.null(df_exposure_model)) {
    missing_exposure_columns <- setdiff(
      required_model_columns,
      names(df_exposure_model)
    )
    if (length(missing_exposure_columns) > 0L) {
      stop(
        paste0(
          "Exposure fitted-model coefficient data is missing: ",
          paste(missing_exposure_columns, collapse = ", "),
          "."
        ),
        call. = FALSE
      )
    }
  }
  text_size_mm <- base_font_size * 25.4 / 72.27

  planning_behavior <- prepare_fitted_complete_recall_data(
    df_beh,
    condition_name = "planning"
  )
  df_model <- df_model %>%
    dplyr::filter(condition == "planning") %>%
    dplyr::mutate(
      parent_reward = ifelse(!is_leaf, 0, parent_reward),
      aunt_reward = ifelse(!is_leaf, 0, aunt_reward)
    )
  if (!is.null(df_exposure_model)) {
    df_exposure_model <- df_exposure_model %>%
      dplyr::filter(tolower(as.character(condition)) == "exposure") %>%
      dplyr::mutate(
        parent_reward = ifelse(!is_leaf, 0, parent_reward),
        aunt_reward = ifelse(!is_leaf, 0, aunt_reward)
      )
    if (nrow(df_exposure_model) == 0L) {
      stop(
        "The exposure fitted-simulation CSV contains no exposure rows.",
        call. = FALSE
      )
    }
  }

  terms_model <- c(
    "scale(actual_reward)" = "Probed",
    "scale(sibling_reward)" = "Sibling",
    "scale(aunt_reward)" = "Aunt",
    "scale(parent_reward)" = "Parent"
  )
  predictor_levels <- c("Aunt", "Parent", "Sibling", "Probed")
  base_fml <- fitted_complete_recall_predictors()
  model_formula <- stats::as.formula(paste0(y_var, " ~ ", base_fml))
  if (is.null(participant_fit)) {
    participant_fit <- fit_fitted_participant_complete_recall_model(
      planning_behavior,
      y_var = y_var,
      condition_name = "planning"
    )
  }
  participant_coefs <- extract_planning_condition_coefs(
    summary(participant_fit)$coefficients,
    terms_model,
    "participants"
  ) %>%
    dplyr::mutate(model_name = "Participants")
  if (!is.null(df_exposure_model)) {
    exposure_participant_fit <-
      fit_fitted_participant_complete_recall_model(
        df_beh,
        y_var = y_var,
        condition_name = "exposure"
      )
    exposure_participant_coefs <- extract_planning_condition_coefs(
      summary(exposure_participant_fit)$coefficients,
      terms_model,
      "participants"
    ) %>%
      dplyr::mutate(
        condition = "exposure",
        model_name = "Participants"
      )
    participant_coefs <- dplyr::bind_rows(
      participant_coefs,
      exposure_participant_coefs
    )
  }

  model_names <- unique(as.character(df_model$model_name))
  model_coefs <- lapply(model_names, function(current_model_name) {
    current_data <- df_model %>%
      dplyr::filter(model_name == current_model_name)
    current_fit <- if (y_var == "accuracy") {
      stats::glm(
        model_formula,
        family = stats::binomial,
        data = current_data
      )
    } else {
      stats::lm(model_formula, data = current_data)
    }
    extract_planning_condition_coefs(
      summary(current_fit)$coefficients,
      terms_model,
      "model"
    ) %>%
      dplyr::mutate(model_name = current_model_name)
  }) %>%
    dplyr::bind_rows()
  if (!is.null(df_exposure_model)) {
    exposure_model_names <- unique(
      as.character(df_exposure_model$model_name)
    )
    if (length(exposure_model_names) != 1L) {
      stop(
        "Expected exactly one exposure fitted-model name.",
        call. = FALSE
      )
    }
    exposure_model_fit <- if (y_var == "accuracy") {
      stats::glm(
        model_formula,
        family = stats::binomial,
        data = df_exposure_model
      )
    } else {
      stats::lm(model_formula, data = df_exposure_model)
    }
    exposure_model_coefs <- extract_planning_condition_coefs(
      summary(exposure_model_fit)$coefficients,
      terms_model,
      "model"
    ) %>%
      dplyr::mutate(
        condition = "exposure",
        model_name = exposure_model_names[[1]]
      )
    model_coefs <- dplyr::bind_rows(
      model_coefs,
      exposure_model_coefs
    )
  }

  coefficient_data <- dplyr::bind_rows(
    participant_coefs,
    model_coefs
  ) %>%
    dplyr::mutate(
      # Match the shared coefficient plot: error bars show +/-1 SE.
      ci_low = estimate - se,
      ci_high = estimate + se,
      source = factor(source, levels = c("participants", "model")),
      condition = factor(condition, levels = c("planning", "exposure")),
      predictor_label = factor(
        terms_model[predictor],
        levels = predictor_levels
      ),
      predictor_position = as.numeric(predictor_label),
      model_name = factor(
        model_name,
        levels = c("Participants", model_names)
      ),
      series_name = dplyr::if_else(
        condition == "exposure",
        paste0(as.character(model_name), " exposure"),
        as.character(model_name)
      ),
      color_key = dplyr::case_when(
        condition == "exposure" ~ "Exposure",
        source == "participants" ~ "Participants",
        TRUE ~ as.character(model_name)
      ),
      linetype_key = dplyr::if_else(
        condition == "exposure",
        "Exposure",
        as.character(model_name)
      )
    )
  participant_series <- unique(
    coefficient_data$series_name[
      coefficient_data$source == "participants"
    ]
  )
  participant_offsets <- if (length(participant_series) <= 1L) {
    stats::setNames(0, participant_series)
  } else {
    stats::setNames(
      seq(-0.08, 0.08, length.out = length(participant_series)),
      participant_series
    )
  }
  model_series <- unique(
    coefficient_data$series_name[coefficient_data$source == "model"]
  )
  model_offsets <- if (length(model_series) <= 1L) {
    stats::setNames(0, model_series)
  } else {
    stats::setNames(
      seq(-0.24, 0.24, length.out = length(model_series)),
      model_series
    )
  }
  coefficient_data <- coefficient_data %>%
    dplyr::mutate(
      plot_position = dplyr::if_else(
        source == "model",
        predictor_position + unname(model_offsets[series_name]),
        predictor_position + unname(participant_offsets[series_name])
      )
    )

  color_values <- c(
    "Participants" = fitted_participant_color,
    fitted_model_colors(model_names),
    "Exposure" = "#c2beb5"
  )
  linetype_values <- c(
    fitted_model_linetypes(model_names),
    "Exposure" = "solid"
  )

  finite_limits <- coefficient_data %>%
    dplyr::filter(is.finite(ci_low), is.finite(ci_high))
  if (nrow(finite_limits) == 0L) {
    stop("No finite coefficient confidence intervals were available.", call. = FALSE)
  }
  x_min <- min(finite_limits$ci_low)
  x_max <- max(finite_limits$ci_high)
  x_pad <- 0.1 * (x_max - x_min + 1e-8)
  participant_plot_data <- coefficient_data %>%
    dplyr::filter(source == "participants")
  model_plot_data <- coefficient_data %>%
    dplyr::filter(source == "model")

  plot <- ggplot2::ggplot() +
    ggplot2::geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.3,
      color = "grey60"
    ) +
    ggplot2::geom_errorbarh(
      data = participant_plot_data,
      ggplot2::aes(
        y = plot_position,
        xmin = ci_low,
        xmax = ci_high,
        color = color_key
      ),
      height = 0,
      linewidth = 0.4
    ) +
    ggplot2::geom_point(
      data = participant_plot_data,
      ggplot2::aes(
        x = estimate,
        y = plot_position,
        color = color_key
      ),
      size = 0.5
    ) +
    ggplot2::geom_errorbarh(
      data = model_plot_data,
      ggplot2::aes(
        y = plot_position,
        xmin = ci_low,
        xmax = ci_high,
        color = color_key,
        linetype = linetype_key
      ),
      height = 0,
      linewidth = 0.4
    ) +
    ggplot2::geom_point(
      data = model_plot_data,
      ggplot2::aes(
        x = estimate,
        y = plot_position,
        color = color_key
      ),
      size = 0.5
    ) +
    ggplot2::geom_text(
      data = coefficient_data %>% dplyr::filter(star != ""),
      ggplot2::aes(x = estimate, y = plot_position, label = star),
      vjust = -0.4,
      color = "black",
      size = text_size_mm,
      show.legend = FALSE
    ) +
    ggplot2::scale_linetype_manual(
      name = "Model",
      values = linetype_values
    ) +
    ggplot2::scale_color_manual(
      name = "Model",
      values = color_values
    ) +
    ggplot2::scale_x_continuous(
      name = x_lab,
      breaks = scales::pretty_breaks(n = 3),
      limits = c(x_min - x_pad, x_max + x_pad)
    ) +
    ggplot2::scale_y_continuous(
      name = y_lab,
      breaks = seq_along(predictor_levels),
      labels = predictor_levels,
      # Match the padding that ggplot supplied when this was a discrete axis.
      # Without extra room above the top ("Probed") row, its significance
      # stars are shifted outside the panel by geom_text(vjust = -0.4) and
      # clipped even though the underlying p-values are unchanged.
      expand = ggplot2::expansion(add = c(0.2, 0.6))
    ) +
    ggplot2::facet_wrap(~source, nrow = 1) +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      strip.text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(color = "black", linewidth = 0.4),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.15, "cm"),
      aspect.ratio = 0.8,
      panel.spacing = grid::unit(1, "lines"),
      plot.margin = ggplot2::margin(2, 2, 2, 2, "mm")
    ) +
    ggplot2::theme(legend.position = "none")

  output_directory <- analysis_path("figures", save_dir)
  ensure_dir(output_directory)
  ggsave_pdf(
    file.path(
      output_directory,
      sprintf("coef_%s_%s.pdf", figure_name, y_var)
    ),
    plot = plot,
    width = x_size,
    height = y_size,
    limitsize = FALSE
  )
  plot
}

plot_fitted_context_stats_coefs <- function(
    experiment, save_dir, show_exposure = TRUE,
    stats_root = STATS_ROOT,
    x_lab = "Coefficient (log odds)", y_lab = "", figure_name = "coef_by_source",
    x_size = 80, y_size = 42, base_font_size = 7) {
  stopifnot(experiment %in% c("exp1", "exp2"))
  terms <- c(node_reward_z = "Probed", sibling_reward_z = "Sibling",
    aunt_reward_z = "Aunt", parent_reward_z = "Parent")
  coefs <- lapply(c("participants", "sequential"), function(source) {
    directory <- file.path(stats_root, experiment, source, "context_joint_conditions")
    path <- file.path(directory, "fixed_effects.rds")
    if (!file.exists(path)) stop("Run fitted statistics --tests context --source participants,sequential before plotting; missing: ", path)
    snapshot <- readRDS(path)
    tab <- readr::read_csv(file.path(directory, "all_coefficients.csv"), show_col_types = FALSE)
    needed <- c(names(terms), paste0(names(terms), ":conditionexposure"),
      "node_reward_squared_z", "sibling_reward_squared_z")
    if (!identical(snapshot$estimator, "glmer") || !identical(snapshot$family, "binomial") ||
        !identical(snapshot$link, "logit") || !all(needed %in% rownames(snapshot$coefficients)) ||
        !all(tab$estimator == "glmer")) stop("Invalid joint logistic glmer context export: ", directory)
    saved <- snapshot$coefficients[tab$term, , drop = FALSE]
    if (!isTRUE(all.equal(unname(saved[, "Estimate"]), tab$beta, tolerance = 1e-10)) ||
        !isTRUE(all.equal(unname(saved[, "Std. Error"]), tab$se, tolerance = 1e-10)))
      stop("Coefficient snapshot and table exports differ; rerun context statistics: ", directory)
    extract_condition_coefs(snapshot$coefficients, terms,
      if (source == "participants") "participants" else "model",
      vcov_matrix = snapshot$covariance)
  })
  coefs <- dplyr::bind_rows(coefs)
  if (!show_exposure) coefs <- dplyr::filter(coefs, condition == "planning")
  # Keep a compact audit of the exact estimates used in the figure.
  audit <- analysis_path("stats", "tests", save_dir, "context_coefficient_plot.csv")
  ensure_parent_directory(audit)
  readr::write_csv(coefs, audit)
  plot_coef_df(coef_df = coefs, terms_model = terms, x_lab = x_lab, y_lab = y_lab,
    figure_name = figure_name, save_dir = save_dir, x_size = x_size, y_size = y_size,
    base_font_size = base_font_size, facet_source = TRUE,
    predictor_levels = c("Aunt", "Parent", "Sibling", "Probed"),
    show_condition_legend = show_exposure)
}

plot_fitted_joint_main_vae_condition_coefs <- function(
    df_beh,
    df_planning_model,
    df_exposure_model = NULL,
    y_var = "accuracy",
    x_lab = "Regression Weight",
    y_lab = "",
    figure_name = "coef_by_source",
    save_dir = "plots",
    x_size = 80,
    y_size = 36,
    base_font_size = 7) {
  required_model_columns <- c(
    "condition",
    "model_name",
    "fitted_subject",
    "accuracy",
    "actual_reward",
    "sibling_reward",
    "aunt_reward",
    "parent_reward",
    "is_leaf"
  )
  missing_planning_columns <- setdiff(
    required_model_columns,
    names(df_planning_model)
  )
  if (length(missing_planning_columns) > 0L) {
    stop(
      paste0(
        "Planning Main-VAE coefficient data is missing: ",
        paste(missing_planning_columns, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }

  planning_model <- df_planning_model %>%
    dplyr::filter(
      tolower(as.character(condition)) == "planning",
      as.character(model_name) == "Main VAE"
    )
  if (nrow(planning_model) == 0L) {
    stop("No planning Main-VAE rows were available for the coefficient plot.", call. = FALSE)
  }

  if (is.null(df_exposure_model)) {
    return(plot_fitted_exp1_preregistered_planning_coefs(
      df_beh = df_beh,
      df_model = planning_model,
      df_exposure_model = NULL,
      y_var = y_var,
      x_lab = x_lab,
      y_lab = y_lab,
      figure_name = figure_name,
      save_dir = save_dir,
      x_size = x_size,
      y_size = y_size,
      base_font_size = base_font_size
    ))
  }

  missing_exposure_columns <- setdiff(
    required_model_columns,
    names(df_exposure_model)
  )
  if (length(missing_exposure_columns) > 0L) {
    stop(
      paste0(
        "Exposure Main-VAE coefficient data is missing: ",
        paste(missing_exposure_columns, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }
  exposure_model <- df_exposure_model %>%
    dplyr::filter(
      tolower(as.character(condition)) == "exposure",
      as.character(model_name) == "Main VAE"
    )
  if (nrow(exposure_model) == 0L) {
    stop("No exposure Main-VAE rows were available for the coefficient plot.", call. = FALSE)
  }

  joint_model <- dplyr::bind_rows(planning_model, exposure_model) %>%
    dplyr::mutate(
      condition = factor(
        tolower(as.character(condition)),
        levels = c("planning", "exposure")
      )
    )

  # plot_combined_coefs fits one condition-interaction model per source:
  #   outcome ~ contextual predictors * condition
  # Participant coefficients come from one mixed model containing both
  # conditions; Main-VAE coefficients come from one GLM/LM containing both.
  # The resulting conditional planning and exposure effects are therefore
  # derived from the same fitted model rather than separate regressions.
  plot_combined_coefs(
    df_beh = df_beh,
    df_model = joint_model,
    y_var = y_var,
    x_lab = x_lab,
    y_lab = y_lab,
    figure_name = figure_name,
    save_dir = save_dir,
    x_size = x_size,
    y_size = y_size + 6,
    base_font_size = base_font_size,
    show_condition_legend = TRUE
  )
}

resolve_fitted_parameter_fit_path <- function(simulation_path) {
  fit_path <- analysis_selected_fit(simulation_path)
  require_existing_path(fit_path, "Selected participant-fit CSV")
  fit_path
}

fitted_reviewer_theme <- function(base_font_size = 7, aspect_ratio = 1) {
  plot_theme_axes(
    base_font_size = base_font_size,
    aspect_ratio = aspect_ratio
  ) +
    ggplot2::theme(
      text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      strip.text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      plot.margin = ggplot2::margin(2, 2, 2, 2, "mm")
    )
}

fitted_reviewer_parameter_transform <- function(values) {
  values <- as.numeric(values)
  positive_values <- values[is.finite(values) & values > 0]
  sigma <- if (length(positive_values) > 0L) {
    min(positive_values) / 2
  } else {
    1
  }
  scales::pseudo_log_trans(sigma = sigma, base = 10)$transform
}

fitted_reviewer_parameter_labels <- function(values) {
  vapply(
    as.numeric(values),
    function(value) {
      if (!is.finite(value)) {
        return(NA_character_)
      }
      if (value == 0) {
        return("0")
      }
      if (abs(value) < 0.01 || abs(value) >= 1000) {
        label <- format(value, scientific = TRUE, digits = 2, trim = TRUE)
        label <- sub("e\\+", "e", label)
        label <- sub("e(-?)0+", "e\\1", label)
        return(label)
      }
      format(signif(value, 4), scientific = FALSE, trim = TRUE)
    },
    character(1)
  )
}

fitted_reviewer_scatter_panel <- function(
    data,
    x_var,
    y_var,
    x_lab,
    y_lab,
    x_transform = function(x) x,
    x_break_values = NULL,
    x_limits = NULL,
    y_limits = c(0, 1),
    rotate_x_labels = FALSE,
    compact_x_axis = FALSE,
    base_font_size = 7) {
  plot_data <- data %>%
    dplyr::transmute(
      reviewer_x = x_transform(as.numeric(.data[[x_var]])),
      reviewer_y = as.numeric(.data[[y_var]])
    ) %>%
    dplyr::filter(is.finite(reviewer_x), is.finite(reviewer_y))
  if (nrow(plot_data) == 0L) {
    stop(
      sprintf("No finite observations were available for %s versus %s.", y_var, x_var),
      call. = FALSE
    )
  }

  plot <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = reviewer_x, y = reviewer_y)
  ) +
    ggplot2::geom_smooth(
      method = "lm",
      formula = y ~ x,
      se = TRUE,
      linewidth = 0.45,
      color = fitted_participant_color,
      fill = fitted_participant_color,
      alpha = 0.12
    ) +
    ggplot2::geom_point(
      size = 0.7,
      alpha = 0.55,
      color = fitted_participant_color
    ) +
    ggplot2::labs(x = x_lab, y = y_lab) +
    ggplot2::coord_cartesian(
      xlim = if (is.null(x_limits)) NULL else x_transform(x_limits),
      ylim = y_limits
    ) +
    ggplot2::scale_y_continuous(
      breaks = seq(y_limits[[1]], y_limits[[2]], length.out = 5),
      expand = ggplot2::expansion(mult = c(0.02, 0.04))
    ) +
    fitted_reviewer_theme(
      base_font_size = base_font_size,
      aspect_ratio = 1
    )

  if (!is.null(x_break_values)) {
    x_break_values <- sort(unique(as.numeric(x_break_values)))
    plot <- plot + ggplot2::scale_x_continuous(
      breaks = x_transform(x_break_values),
      labels = fitted_reviewer_parameter_labels(x_break_values),
      guide = ggplot2::guide_axis(check.overlap = TRUE)
    )
  } else {
    if (isTRUE(compact_x_axis)) {
      plot <- plot + ggplot2::scale_x_continuous(
        breaks = scales::pretty_breaks(n = 3),
        labels = fitted_reviewer_parameter_labels,
        guide = ggplot2::guide_axis(check.overlap = TRUE)
      )
    } else {
      plot <- plot + ggplot2::scale_x_continuous(
        breaks = scales::pretty_breaks(n = 4)
      )
    }
  }
  if (rotate_x_labels) {
    plot <- plot + ggplot2::theme(
      axis.text.x = ggplot2::element_text(
        angle = 45,
        hjust = 1,
        vjust = 1,
        size = base_font_size,
        color = "black"
      )
    )
  }
  plot
}

fitted_reviewer_parameter_heatmap_panel <- function(
    heatmap_data,
    fill_var,
    fill_label,
    base_font_size = 7,
    fill_colors = c("#F7F3F2", "#B77A73", "#532B27")) {
  ggplot2::ggplot(
    heatmap_data,
    ggplot2::aes(
      x = fitted_lambda_factor,
      y = fitted_rho_factor,
      fill = .data[[fill_var]]
    )
  ) +
    ggplot2::geom_tile(
      color = "white",
      linewidth = 0.2
    ) +
    ggplot2::scale_fill_gradientn(
      colors = fill_colors,
      limits = c(0, 1),
      breaks = c(0, 0.25, 0.5, 0.75, 1),
      na.value = "#000000"
    ) +
    # Let the panel fill the available page height rather than constrain it
    # by its width; axes and legends retain their own space.
    ggplot2::coord_cartesian() +
    ggplot2::labs(
      x = expression("Fitted " * lambda),
      y = expression("Fitted " * alpha / lambda),
      fill = fill_label
    ) +
    ggplot2::guides(
      fill = ggplot2::guide_colorbar(
        direction = "vertical",
        title.position = "top",
        title.hjust = 0,
        label.position = "right",
        barwidth = grid::unit(1.5, "mm"),
        barheight = grid::unit(15, "mm")
      )
    ) +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      aspect.ratio = NULL,
      text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      axis.text.x = ggplot2::element_text(
        angle = 45,
        size = base_font_size * 0.8,
        hjust = 1,
        vjust = 1
      ),
      axis.text.y = ggplot2::element_text(size = base_font_size * 0.8),
      axis.ticks = ggplot2::element_line(
        color = "black",
        linewidth = 0.3
      ),
      axis.ticks.length = grid::unit(0.8, "mm"),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.4
      ),
      legend.position = "right",
      legend.direction = "vertical",
      legend.justification = "center",
      legend.box = "vertical",
      legend.title = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain",
        lineheight = 0.9,
        margin = ggplot2::margin(0, 0, 2, 0, "mm")
      ),
      legend.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain",
        margin = ggplot2::margin(0, 0, 0, 1, "mm")
      ),
      legend.margin = ggplot2::margin(0, 0, 0, 0, "mm"),
      legend.box.margin = ggplot2::margin(0, 0, 0, 0, "mm"),
      legend.spacing = grid::unit(0.8, "mm"),
      plot.margin = ggplot2::margin(1, 1, 1, 1, "mm")
    )
}

prepare_fitted_matched_seed_recovery <- function(fits, second_parameter) {
  parameters <- c("lambda", second_parameter)
  required <- c("subject", "recovery_mode", paste0("true_", parameters),
                paste0("recovered_", parameters))
  missing <- setdiff(required, names(fits))
  if (length(missing)) stop(
    paste("Recovery fits are missing columns:", paste(missing, collapse = ", ")),
    call. = FALSE
  )
  fits <- fits[fits$recovery_mode %in% "matched_seed", , drop = FALSE]
  if (!nrow(fits)) stop("No matched-seed recovery fits were found.", call. = FALSE)
  if (anyDuplicated(fits$subject)) stop(
    "Matched-seed recovery fits contain duplicate synthetic participants.", call. = FALSE
  )
  if (all(c("generating_seed", "fitting_seed") %in% names(fits)) &&
      any(is.na(fits$fitting_seed) | is.na(fits$generating_seed) |
          fits$fitting_seed != fits$generating_seed)) {
    stop("Matched-seed recovery contains nonmatching training seeds.", call. = FALSE)
  }
  summaries <- lapply(parameters, function(parameter) {
    rows <- data.frame(
      true_value = signif(as.numeric(fits[[paste0("true_", parameter)]]), 10),
      recovered_value = signif(as.numeric(fits[[paste0("recovered_", parameter)]]), 10)
    )
    if (any(!is.finite(as.matrix(rows))) || any(as.matrix(rows) < 0) ||
        (parameter == "lambda" && any(as.matrix(rows) <= 0))) {
      stop("Recovery parameter values are invalid.", call. = FALSE)
    }
    values <- sort(unique(c(rows$true_value, rows$recovered_value)))
    counts <- dplyr::count(rows, true_value, recovered_value, name = "recovery_count")
    counts <- tidyr::complete(
      counts, true_value = values, recovered_value = values,
      fill = list(recovery_count = 0L)
    )
    counts <- dplyr::group_by(counts, true_value)
    counts <- dplyr::mutate(
      counts,
      generating_count = sum(recovery_count),
      recovery_proportion = ifelse(generating_count > 0, recovery_count / generating_count, NA_real_)
    )
    dplyr::mutate(dplyr::ungroup(counts), parameter = parameter)
  })
  dplyr::bind_rows(summaries)
}

resolve_current_exp3_recovery <- function(directories) {
  # Never fall back to the obsolete lambda/alpha recovery merely because it exists.
  expected <- list(tree = "31", parameter_grid_mode = "rho", rnn_units = 128,
    recovery_workflow_version = 3, recovery_modes = "matched_seed", execution_layout = "sharded",
    lambda_values = c(1e-6, 1e-5, 1e-4, .001, .01, .1, 1, 10),
    rho_values = c(0, .01, .02, .1, .2, .5, 1, 2, 5, 10),
    model_seeds = 1:5, replicates_per_cell = 10, mc_samples = 1,
    generation_mc_samples = 1, batch_size = 128, condition = "planning",
    reward_nll_scale = .5, recovery_seed = 2028, evaluation_seed = 2026,
    model_epochs = 320, action_entropy_bonus = .05, nonlinear_scale = TRUE,
    no_critic_loss = FALSE, reconstruction_hidden_units = 128,
    reconstruction_loss_scope = "final_timestep", max_action_gradient_scale = 1,
    final_learning_rate = .0003, late_lr_plateau_start_epoch = 200,
    late_lr_plateau_end_epoch = 300,
    model_configuration_artifact_suffix = paste0(
      "_head_nonlinear_location_scale_recon_range_minus5_plus5_recon_units_128_",
      "recon_scope_final_timestep_rnn_units_128_epochs_320_gradient_scale_1_",
      "final_lr_0.0003_lr_plateau_200_300_eb_0.05_rns"))
  paths <- unlist(lapply(directories, list.files,
    pattern = "^exp3_full_recovery[.]json$", full.names = TRUE))
  candidates <- character()
  for (path in paths) {
    metadata <- tryCatch(jsonlite::fromJSON(path), error = function(e) NULL)
    if (is.null(metadata)) next
    matches <- vapply(names(expected), function(key)
      isTRUE(all.equal(unname(metadata[[key]]), expected[[key]], check.attributes = FALSE)), logical(1))
    if (!all(matches) || !is.null(metadata$parameter_multiplier) ||
        !is.null(metadata$initial_critic_coefficient)) next
    # The default behavior file is the same exp3_beh.csv used by manuscript stats.
    if (!is.null(metadata$behavior_csv) && basename(metadata$behavior_csv) != "exp3_beh.csv") next
    fit <- sub("[.]json$", ".csv", path)
    if (file.exists(fit)) candidates <- c(candidates, fit)
  }
  if (!length(candidates)) stop(
    "No completed current Exp3 lambda/rho recovery found in the configured RecoveryInputs folder; ",
    "download both the fits CSV and config JSON when plotting locally.", call. = FALSE)
  candidates[[which.max(file.info(candidates)$mtime)]]
}

plot_fitted_matched_seed_recovery <- function(
    tree_size, output_directory, fits_csv = NULL, base_font_size = 7) {
  tags <- c(
    `7` = "exp1_full_recovery",
    `13` = "exp2_full_recovery",
    `31` = "exp3_full_recovery"
  )
  tag <- tags[[as.character(tree_size)]]
  if (is.null(tag)) stop("Unsupported recovery tree size.", call. = FALSE)
  if (tree_size == 31 && is.null(fits_csv)) {
    fits_csv <- resolve_current_exp3_recovery(RECOVERY_INPUT_DIR)
  }
  if (is.null(fits_csv)) {
    filename <- paste0(tag, ".csv")
    candidates <- file.path(RECOVERY_INPUT_DIR, filename)
    available <- candidates[file.exists(candidates)]
    if (!length(available)) stop(
      paste0("Matched-seed recovery fits were not found. Expected: ",
             paste(candidates, collapse = " or ")), call. = FALSE
    )
    fits_csv <- available[[1L]]
  }
  cat(sprintf("Matched-seed recovery input: %s\n", fits_csv))
  fits <- readr::read_csv(fits_csv, show_col_types = FALSE, progress = FALSE)
  second_parameter <- "rho"
  summary <- prepare_fitted_matched_seed_recovery(fits, second_parameter)
  panels <- lapply(c("lambda", second_parameter), function(parameter) {
    data <- summary[summary$parameter == parameter, , drop = FALSE]
    values <- sort(unique(c(data$true_value, data$recovered_value)))
    data$fitted_lambda_factor <- factor(
      data$recovered_value, levels = values,
      labels = fitted_reviewer_parameter_labels(values)
    )
    # Match the original recovery matrix: smallest true value is at the top.
    data$fitted_rho_factor <- factor(
      data$true_value, levels = rev(values),
      labels = fitted_reviewer_parameter_labels(rev(values))
    )
    fitted_reviewer_parameter_heatmap_panel(
      data, "recovery_proportion", "Recovery\nproportion", base_font_size,
      fill_colors = c("#F7FBFF", "#6BAED6", "#08306B")
    ) + ggplot2::labs(
      x = if (parameter == "rho") expression("Recovered " * alpha / lambda) else expression("Recovered " * lambda),
      y = if (parameter == "rho") expression("True " * alpha / lambda) else expression("True " * lambda)
    ) + ggplot2::geom_text(
      ggplot2::aes(
        label = ifelse(is.finite(recovery_proportion) & recovery_proportion >= 0.01,
                       sprintf("%.2f", recovery_proportion), ""),
        color = recovery_proportion > 0.55
      ), size = 2.2, show.legend = FALSE, na.rm = TRUE
    ) + ggplot2::scale_color_manual(values = c(`FALSE` = "black", `TRUE` = "white"))
  })
  plot <- cowplot::plot_grid(plotlist = panels, nrow = 1, rel_widths = c(1, 1),
                            align = "hv", axis = "tblr")
  output_path <- file.path(output_directory, "matched_seed_parameter_recovery_heatmaps.pdf")
  ensure_parent_directory(output_path)
  # Recovery matrices contain numeric cell annotations, so give them more
  # space than the unannotated fitted-parameter performance heatmaps.
  ggplot2::ggsave(output_path, plot = plot, width = 150, height = 65,
                  units = "mm", bg = "white", limitsize = FALSE)
  summary_path <- file.path(ANALYSIS_SUMMARY_DIR, "tests", basename(output_directory),
                            "matched_seed_parameter_recovery_heatmaps.csv")
  ensure_parent_directory(summary_path)
  summary$input_fits_csv <- normalizePath(fits_csv, mustWork = TRUE)
  summary$recovery_mode <- "matched_seed"
  readr::write_csv(summary, summary_path)
  cat(sprintf("Saved matched-seed recovery heatmaps to: %s\n", output_path))
  invisible(output_path)
}

fitted_reviewer_association_summary <- function(
    data,
    analysis,
    predictor,
    outcome,
    predictor_transform = function(x) x,
    predictor_scale = "identity") {
  analysis_data <- data %>%
    dplyr::transmute(
      raw_predictor = as.numeric(.data[[predictor]]),
      transformed_predictor = predictor_transform(raw_predictor),
      outcome_value = as.numeric(.data[[outcome]])
    ) %>%
    dplyr::filter(
      is.finite(raw_predictor),
      is.finite(transformed_predictor),
      is.finite(outcome_value)
    )
  if (
    nrow(analysis_data) < 3L ||
      dplyr::n_distinct(analysis_data$raw_predictor) < 2L
  ) {
    return(tibble::tibble(
      analysis = analysis,
      predictor = predictor,
      outcome = outcome,
      predictor_scale = predictor_scale,
      participant_count = nrow(analysis_data),
      spearman_rho = NA_real_,
      spearman_p_value = NA_real_,
      linear_slope = NA_real_,
      linear_slope_se = NA_real_,
      linear_slope_p_value = NA_real_,
      linear_r_squared = NA_real_
    ))
  }

  correlation <- suppressWarnings(stats::cor.test(
    analysis_data$raw_predictor,
    analysis_data$outcome_value,
    method = "spearman",
    exact = FALSE
  ))
  linear_fit <- stats::lm(
    outcome_value ~ transformed_predictor,
    data = analysis_data
  )
  linear_summary <- summary(linear_fit)
  linear_coefficients <- linear_summary$coefficients
  predictor_row <- match(
    "transformed_predictor",
    rownames(linear_coefficients)
  )
  if (is.na(predictor_row)) {
    warning(
      sprintf(
        paste0(
          "Linear slope is not estimable for %s (%s predicting %s); ",
          "writing NA statistics for that slope."
        ),
        analysis,
        predictor,
        outcome
      ),
      call. = FALSE,
      immediate. = TRUE
    )
    linear_slope <- NA_real_
    linear_slope_se <- NA_real_
    linear_slope_p_value <- NA_real_
  } else {
    linear_slope <- linear_coefficients[predictor_row, "Estimate"]
    linear_slope_se <- linear_coefficients[predictor_row, "Std. Error"]
    linear_slope_p_value <- linear_coefficients[predictor_row, "Pr(>|t|)"]
  }
  linear_r_squared <- linear_summary$r.squared
  if (!is.finite(linear_r_squared)) {
    linear_r_squared <- NA_real_
  }
  tibble::tibble(
    analysis = analysis,
    predictor = predictor,
    outcome = outcome,
    predictor_scale = predictor_scale,
    participant_count = nrow(analysis_data),
    spearman_rho = unname(correlation$estimate),
    spearman_p_value = correlation$p.value,
    linear_slope = linear_slope,
    linear_slope_se = linear_slope_se,
    linear_slope_p_value = linear_slope_p_value,
    linear_r_squared = linear_r_squared
  )
}

prepare_fitted_reviewer_participant_data <- function(
    df_beh,
    main_simulation_path, include_context_coefficients = TRUE) {
  require_columns(
    df_beh,
    c("subject", "graph", "condition", "accuracy", "norm_reward"),
    "Reviewer behavioral data"
  )
  planning <- df_beh %>%
    dplyr::mutate(
      subject = as.character(subject),
      graph = as.character(graph),
      condition = tolower(as.character(condition)),
      accuracy = as.numeric(accuracy),
      norm_reward = as.numeric(norm_reward)
    ) %>%
    dplyr::filter(condition == "planning")

  recall_metrics <- planning %>%
    dplyr::filter(is.finite(accuracy)) %>%
    dplyr::group_by(subject) %>%
    dplyr::summarise(
      human_recall_accuracy = mean(accuracy),
      recall_observation_count = dplyr::n(),
      .groups = "drop"
    )
  planning_trials <- planning %>%
    dplyr::filter(is.finite(norm_reward)) %>%
    dplyr::distinct(subject, graph, norm_reward)
  if (anyDuplicated(planning_trials[c("subject", "graph")])) {
    stop(
      "A human planning trial contains inconsistent normalized rewards.",
      call. = FALSE
    )
  }
  planning_metrics <- planning_trials %>%
    dplyr::group_by(subject) %>%
    dplyr::summarise(
      human_norm_reward = mean(norm_reward),
      planning_trial_count = dplyr::n(),
      .groups = "drop"
    )

  fit_path <- resolve_fitted_parameter_fit_path(main_simulation_path)
  fitted_parameters <- readr::read_csv(
    fit_path,
    col_select = dplyr::any_of(c("subject", "best_lambda", "best_alpha")),
    show_col_types = FALSE,
    progress = FALSE
  )
  require_columns(
    fitted_parameters,
    c("subject", "best_lambda", "best_alpha"),
    "Main-VAE selected participant-fit CSV"
  )
  fitted_parameters <- fitted_parameters %>%
    dplyr::transmute(
      subject = as.character(subject),
      fitted_lambda = as.numeric(best_lambda),
      fitted_alpha = as.numeric(best_alpha)
    )
  if (anyDuplicated(fitted_parameters$subject)) {
    stop(
      "The Main-VAE selected-fit CSV contains duplicate participants.",
      call. = FALSE
    )
  }
  if (
    any(!is.finite(fitted_parameters$fitted_lambda)) ||
      any(fitted_parameters$fitted_lambda <= 0) ||
      any(!is.finite(fitted_parameters$fitted_alpha)) ||
      any(fitted_parameters$fitted_alpha < 0)
  ) {
    stop(
      "Fitted lambda must be positive and fitted alpha must be non-negative.",
      call. = FALSE
    )
  }

  participant_coefficients <- if (include_context_coefficients) fit_individual_context_glms(planning) else
    dplyr::distinct(planning, subject)

  participant_data <- fitted_parameters %>%
    dplyr::left_join(recall_metrics, by = "subject", relationship = "one-to-one") %>%
    dplyr::left_join(planning_metrics, by = "subject", relationship = "one-to-one") %>%
    dplyr::left_join(
      participant_coefficients,
      by = "subject",
      relationship = "one-to-one"
    )
  required_numeric <- c(
    "human_recall_accuracy",
    "human_norm_reward"
  )
  incomplete <- participant_data %>%
    dplyr::filter(dplyr::if_any(dplyr::all_of(required_numeric), ~ !is.finite(.x)))
  if (nrow(incomplete) > 0L) {
    stop(
      paste0(
        "Reviewer analyses are missing human metrics or recall coefficients for ",
        nrow(incomplete),
        " fitted participant(s)."
      ),
      call. = FALSE
    )
  }
  attr(participant_data, "selected_fit_path") <- fit_path
  attr(participant_data, "context_coefficient_scaling") <- attr(participant_coefficients, "context_coefficient_scaling")
  participant_data
}

prepare_fitted_reviewer_model_data <- function(df_model) {
  required <- c("fitted_subject", "model_name", "condition", "graph", "accuracy",
                "actual_reward", "sibling_reward", "parent_reward", "aunt_reward",
                "is_leaf", "V", "min_path_reward", "max_path_reward")
  require_columns(df_model, required, "Reviewer sequential-model simulations")
  rows <- df_model %>%
    dplyr::select(dplyr::all_of(required)) %>%
    dplyr::filter(model_name == "Main VAE", condition == "planning") %>%
    dplyr::mutate(subject = as.character(fitted_subject),
                  parent_reward = ifelse(is_leaf, parent_reward, 0),
                  aunt_reward = ifelse(is_leaf, aunt_reward, 0))
  if (!nrow(rows)) stop("No planning sequential-model rows for reviewer plots.", call. = FALSE)
  trials <- rows %>%
    dplyr::distinct(subject, graph, V, min_path_reward, max_path_reward)
  if (anyDuplicated(trials[c("subject", "graph")])) {
    stop("A model trial contains inconsistent path rewards.", call. = FALSE)
  }
  performance <- trials %>%
    dplyr::mutate(norm_reward = ifelse(max_path_reward > min_path_reward,
                  (V - min_path_reward) / (max_path_reward - min_path_reward), NA_real_)) %>%
    dplyr::filter(is.finite(norm_reward)) %>%
    dplyr::group_by(subject) %>%
    dplyr::summarise(model_norm_reward = mean(norm_reward), .groups = "drop")
  coefficients <- fit_individual_context_glms(rows)
  result <- dplyr::left_join(coefficients, performance, by = "subject", relationship = "one-to-one")
  if (any(!is.finite(result$model_norm_reward))) stop("Missing model planning performance.", call. = FALSE)
  attr(result, "context_coefficient_scaling") <- attr(coefficients, "context_coefficient_scaling")
  result
}

save_fitted_paired_reference_plot <- function(
    plot, path, reference_plot, width = 40, height = 36, side_padding_mm = 0) {
  grob <- ggplot2::ggplotGrob(plot)
  panels <- grob$layout[grepl("^panel", grob$layout$name), ]
  panel_height <- fitted_panel_height_mm(reference_plot, width, height)
  grob$heights[unique(panels$t)] <- grid::unit(panel_height, "mm")
  grob$widths[unique(panels$l)] <- grid::unit(panel_height / reference_plot$theme$aspect.ratio, "mm")
  grob$respect <- FALSE
  if (side_padding_mm > 0) {
    grob <- gtable::gtable_add_cols(grob, grid::unit(side_padding_mm, "mm"), pos = 0)
    grob <- gtable::gtable_add_cols(grob, grid::unit(side_padding_mm, "mm"), pos = -1)
  }
  ensure_parent_directory(path)
  ggsave_pdf(path, plot = grob, width = width + 2 * side_padding_mm, height = height)
}

plot_fitted_context_performance_pairs <- function(
    participant_data, model_data, save_dir, reference_plot, width = 40, height = 36,
    base_font_size = 7) {
  columns <- paste0(c("node", "parent", "sibling", "aunt"), "_reward_coefficient")
  # Export all fits, including flagged ones, before selecting the common joint-analysis sample.
  audit_directory <- analysis_path("stats", "tests", save_dir)
  save_individual_context_audit(model_data, audit_directory, "sequential")
  metrics_path <- file.path(audit_directory, "reviewer_sequential_model_metrics.csv")
  readr::write_csv(model_data, metrics_path)
  participant_data <- context_performance_sample(participant_data, "human_norm_reward")
  model_data <- context_performance_sample(model_data, "model_norm_reward")
  data <- dplyr::bind_rows(
    participant_data %>% dplyr::select(subject, dplyr::all_of(columns), human_norm_reward) %>%
      dplyr::rename(norm_reward = human_norm_reward) %>% dplyr::mutate(source = "participants"),
    model_data %>% dplyr::select(subject, dplyr::all_of(columns), model_norm_reward) %>%
      dplyr::rename(norm_reward = model_norm_reward) %>% dplyr::mutate(source = "model")
  ) %>% dplyr::mutate(source = factor(source, levels = c("participants", "model")))
  paths <- character()
  for (context in c("node", "parent", "sibling", "aunt")) {
    column <- paste0(context, "_reward_coefficient")
    plot_data <- data %>% dplyr::transmute(subject, source, coefficient = .data[[column]], norm_reward)
    plot <- ggplot2::ggplot(plot_data, ggplot2::aes(coefficient, norm_reward)) +
      ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = TRUE, linewidth = 0.45,
                          color = fitted_participant_color, fill = fitted_participant_color, alpha = 0.12) +
      ggplot2::geom_point(size = 0.7, alpha = 0.55, color = fitted_participant_color) +
      ggplot2::facet_wrap(~source, nrow = 1, scales = "fixed") +
      ggplot2::scale_x_continuous(breaks = scales::pretty_breaks(n = 3),
        labels = fitted_reviewer_parameter_labels,
        expand = ggplot2::expansion(mult = 0.08),
        guide = ggplot2::guide_axis(check.overlap = TRUE)) +
      ggplot2::coord_cartesian(ylim = c(0, 1)) +
      ggplot2::labs(x = paste0(tools::toTitleCase(context), "-reward\ncoefficient"), y = "Normalized reward") +
      plot_theme_axes(base_font_size, aspect_ratio = 1.2) +
      ggplot2::theme(strip.clip = "off", panel.spacing = grid::unit(1.31, "mm"),
                     plot.margin = ggplot2::margin(0, 1, 0, 1, "mm"))
    filename <- paste0(context, "_memory_allocation_planning_performance")
    path <- analysis_path("figures", save_dir, paste0(filename, ".pdf"))
    save_fitted_paired_reference_plot(plot, path, reference_plot, width, height, side_padding_mm = 2)
    summary_path <- analysis_path("stats", "tests", save_dir, paste0(filename, ".csv"))
    ensure_parent_directory(summary_path)
    readr::write_csv(plot_data, summary_path)
    paths <- c(paths, path)
    cat(sprintf("Saved participant/model context-performance plot to: %s\n", path))
  }
  paths
}

plot_fitted_reviewer_participant_associations <- function(
    participant_data,
    save_dir,
    base_font_size = 7,
    reference_plot = NULL,
    reference_width = 40,
    reference_height = 36,
    df_main_model = NULL, heatmaps_only = FALSE) {
  output_directory <- analysis_path("figures", save_dir)
  summary_directory <- analysis_path("stats", "tests", save_dir)
  ensure_dir(output_directory)
  ensure_dir(summary_directory)
  if (!heatmaps_only) {
    save_individual_context_audit(participant_data, summary_directory, "participants")
    context_participants <- context_performance_sample(participant_data, "human_norm_reward")
    if (nrow(context_participants) < 3L) stop("Too few estimable individual contextual GLMs; inspect participants_context_glm_diagnostics.csv.")
  }

  reviewer_panel_width_mm <- 42.5
  reviewer_panel_height_mm <- 36

  # Training uses alpha = lambda * rho. Round numerical division noise so
  # the same rho does not become several separate rows across lambda values.
  participant_data <- participant_data %>%
    dplyr::mutate(fitted_rho = signif(fitted_alpha / fitted_lambda, 10))
  if (any(!is.finite(participant_data$fitted_rho) | participant_data$fitted_rho < 0)) {
    stop("Fitted rho must be finite and non-negative.", call. = FALSE)
  }
  alpha_transform <- fitted_reviewer_parameter_transform(
    participant_data$fitted_alpha
  )
  lambda_transform <- fitted_reviewer_parameter_transform(
    participant_data$fitted_lambda
  )
  lambda_values <- sort(unique(as.numeric(participant_data$fitted_lambda)))
  # Retain all nine tested rho levels even if no participant selected a level.
  rho_values <- sort(unique(c(
    0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5,
    participant_data$fitted_rho
  )))
  parameter_heatmap_data <- participant_data %>%
    dplyr::group_by(fitted_lambda, fitted_rho) %>%
    dplyr::summarise(
      mean_human_norm_reward = mean(human_norm_reward),
      mean_human_recall_accuracy = mean(human_recall_accuracy),
      participant_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    tidyr::complete(
      fitted_lambda = lambda_values,
      fitted_rho = rho_values
    ) %>%
    dplyr::mutate(
      fitted_lambda_factor = factor(
        fitted_lambda,
        levels = lambda_values,
        labels = fitted_reviewer_parameter_labels(lambda_values)
      ),
      fitted_rho_factor = factor(
        fitted_rho,
        levels = rho_values,
        labels = fitted_reviewer_parameter_labels(rho_values)
      )
    )
  parameter_panels <- list(
    fitted_reviewer_parameter_heatmap_panel(
      parameter_heatmap_data,
      "mean_human_norm_reward",
      "Participant\nnormalized\nreward",
      base_font_size = base_font_size
    ),
    fitted_reviewer_parameter_heatmap_panel(
      parameter_heatmap_data,
      "mean_human_recall_accuracy",
      "Participant\nrecall\naccuracy",
      base_font_size = base_font_size
    )
  )
  parameter_plot <- cowplot::plot_grid(
    plotlist = parameter_panels,
    ncol = 2,
    nrow = 1,
    rel_widths = c(1, 1),
    align = "hv",
    axis = "tblr"
  )
  parameter_plot_path <- file.path(
    output_directory,
    "fitted_parameters_human_performance.pdf"
  )
  ggsave_pdf(
    parameter_plot_path,
    plot = parameter_plot,
    # Give each side-by-side panel its own axes and external color bar, plus
    # room for angled tick labels below the compact nine-row rho heatmaps.
    width = 90,
    height = 40,
    limitsize = FALSE
  )

  if (heatmaps_only) {
    readr::write_csv(
      dplyr::select(parameter_heatmap_data, -dplyr::ends_with("_factor")),
      file.path(summary_directory, "fitted_parameter_human_performance_heatmaps.csv"))
    return(invisible(parameter_plot_path))
  }

  coefficient_specs <- tibble::tribble(
    ~predictor, ~label,
    "node_reward_coefficient", "Node-reward coefficient",
    "parent_reward_coefficient", "Parent-reward coefficient",
    "sibling_reward_coefficient", "Sibling-reward coefficient",
    "aunt_reward_coefficient", "Aunt-reward coefficient"
  )
  context_panels <- lapply(seq_len(nrow(coefficient_specs)), function(index) {
    fitted_reviewer_scatter_panel(
      context_participants,
      coefficient_specs$predictor[[index]],
      "human_norm_reward",
      coefficient_specs$label[[index]],
      "Normalized reward",
      x_break_values = NULL,
      x_limits = NULL,
      base_font_size = base_font_size
    )
  })
  context_plot <- cowplot::plot_grid(
    plotlist = context_panels,
    ncol = 2,
    align = "hv",
    axis = "tblr"
  )
  context_plot_path <- file.path(
    output_directory,
    "context_memory_allocation_planning_performance.pdf"
  )
  context_width <- 2 * reviewer_panel_width_mm
  context_height <- 2 * reviewer_panel_height_mm
  if (!is.null(reference_plot) && is.null(df_main_model)) {
    matched_context <- fitted_reviewer_matched_context_grid(
      context_panels, reference_plot, reference_width, reference_height,
      base_font_size
    )
    context_plot <- matched_context$plot
    context_width <- matched_context$width
    context_height <- matched_context$height
  }
  if (is.null(df_main_model)) {
    ggsave_pdf(context_plot_path, plot = context_plot, width = context_width,
               height = context_height, limitsize = FALSE)
  } else {
    model_reviewer_data <- prepare_fitted_reviewer_model_data(df_main_model)
    context_plot_path <- plot_fitted_context_performance_pairs(
      participant_data, model_reviewer_data, save_dir, reference_plot,
      reference_width, reference_height, base_font_size
    )
  }

  positive_bias_plot <- context_panels[[1L]]
  positive_bias_plot_path <- file.path(
    output_directory,
    "positive_reward_memory_bias_planning_performance.pdf"
  )
  ggsave_pdf(
    positive_bias_plot_path,
    plot = positive_bias_plot,
    width = 42.5,
    height = 36,
    limitsize = FALSE
  )

  parameter_associations <- dplyr::bind_rows(
    fitted_reviewer_association_summary(
      participant_data,
      "fitted_alpha_predicts_planning",
      "fitted_alpha",
      "human_norm_reward",
      alpha_transform,
      "pseudo_log"
    ),
    fitted_reviewer_association_summary(
      participant_data,
      "fitted_alpha_predicts_recall",
      "fitted_alpha",
      "human_recall_accuracy",
      alpha_transform,
      "pseudo_log"
    ),
    fitted_reviewer_association_summary(
      participant_data,
      "fitted_lambda_predicts_planning",
      "fitted_lambda",
      "human_norm_reward",
      lambda_transform,
      "pseudo_log"
    ),
    fitted_reviewer_association_summary(
      participant_data,
      "fitted_lambda_predicts_recall",
      "fitted_lambda",
      "human_recall_accuracy",
      lambda_transform,
      "pseudo_log"
    )
  )
  context_associations <- dplyr::bind_rows(lapply(
    seq_len(nrow(coefficient_specs)),
    function(index) {
      fitted_reviewer_association_summary(
        context_participants,
        paste0(
          coefficient_specs$predictor[[index]],
          "_predicts_planning"
        ),
        coefficient_specs$predictor[[index]],
        "human_norm_reward"
      )
    }
  ))

  participant_output <- file.path(
    summary_directory,
    "reviewer_participant_metrics.csv"
  )
  parameter_summary_output <- file.path(
    summary_directory,
    "fitted_parameter_performance_associations.csv"
  )
  parameter_heatmap_output <- file.path(
    summary_directory,
    "fitted_parameter_human_performance_heatmaps.csv"
  )
  context_summary_output <- file.path(
    summary_directory,
    "context_memory_performance_associations.csv"
  )
  positive_bias_summary_output <- file.path(
    summary_directory,
    "positive_reward_memory_bias_planning_association.csv"
  )
  readr::write_csv(participant_data, participant_output)
  readr::write_csv(parameter_associations, parameter_summary_output)
  readr::write_csv(
    parameter_heatmap_data %>%
      dplyr::select(
        fitted_lambda,
        fitted_rho,
        participant_count,
        mean_human_norm_reward,
        mean_human_recall_accuracy
      ),
    parameter_heatmap_output
  )
  readr::write_csv(context_associations, context_summary_output)
  readr::write_csv(context_associations[1L, , drop = FALSE], positive_bias_summary_output)
  cat(sprintf("Saved fitted-parameter reviewer heatmaps to: %s\n", parameter_plot_path))
  cat(sprintf("Saved fitted-parameter heatmap summary to: %s\n", parameter_heatmap_output))
  cat(sprintf("Saved positive-reward-bias reviewer plot to: %s\n", positive_bias_plot_path))
  cat(sprintf("Saved context-allocation reviewer plot to: %s\n", context_plot_path))
  cat(sprintf("Saved reviewer participant metrics to: %s\n", participant_output))
  cat(sprintf("Saved fitted-parameter association tests to: %s\n", parameter_summary_output))
  cat(sprintf("Saved context-allocation association tests to: %s\n", context_summary_output))
  cat(sprintf("Saved positive-reward-bias association test to: %s\n", positive_bias_summary_output))

  invisible(list(
    parameter_plot = parameter_plot,
    positive_bias_plot = positive_bias_plot,
    context_plot = context_plot,
    parameter_associations = parameter_associations,
    context_associations = context_associations
  ))
}

plot_fitted_observation_timestep_accuracy <- function(
    df_beh,
    df_model,
    node_count,
    save_dir,
    y_min = 0,
    y_max = 0.8,
    x_size = 42.5,
    y_size = 36,
    base_font_size = 7,
    show_exposure = FALSE) {
  require_columns(
    df_beh,
    c("subject", "condition", "node", "accuracy"),
    "Human observation-timestep data"
  )
  require_columns(
    df_model,
    c("fitted_subject", "model_name", "condition", "node", "accuracy"),
    "Model observation-timestep data"
  )

  participant_rows <- df_beh %>%
    dplyr::mutate(
      subject = as.character(subject),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      accuracy = as.numeric(accuracy)
    ) %>%
    dplyr::filter(
      condition %in% if (show_exposure) c("planning", "exposure") else "planning",
      node %in% seq_len(node_count),
      is.finite(accuracy)
    )
  participant_summary <- participant_rows %>%
    dplyr::group_by(subject, condition, node) %>%
    dplyr::summarise(participant_accuracy = mean(accuracy), .groups = "drop") %>%
    dplyr::group_by(condition, node) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se_accuracy = stats::sd(participant_accuracy) / sqrt(dplyr::n()),
      participant_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      source = factor("participants", levels = c("participants", "models")),
      model_name = "Participants",
      plot_series = dplyr::if_else(
        condition == "exposure",
        "Exposure",
        "Participants"
      )
    )

  model_rows <- df_model %>%
    dplyr::mutate(
      fitted_subject = as.character(fitted_subject),
      model_name = as.character(model_name),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      accuracy = as.numeric(accuracy)
    ) %>%
    dplyr::filter(
      condition %in% if (show_exposure) c("planning", "exposure") else "planning",
      node %in% seq_len(node_count),
      is.finite(accuracy)
    )
  model_summary <- model_rows %>%
    dplyr::group_by(model_name, condition, fitted_subject, node) %>%
    dplyr::summarise(participant_accuracy = mean(accuracy), .groups = "drop") %>%
    dplyr::group_by(model_name, condition, node) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se_accuracy = stats::sd(participant_accuracy) / sqrt(dplyr::n()),
      participant_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      source = factor("models", levels = c("participants", "models")),
      plot_series = dplyr::if_else(
        condition == "exposure",
        "Exposure",
        model_name
      )
    )
  model_names <- unique(as.character(model_summary$plot_series))
  model_summary <- model_summary %>%
    dplyr::mutate(
      model_name = factor(model_name, levels = unique(model_rows$model_name)),
      plot_series = factor(plot_series, levels = model_names),
      plot_linewidth = dplyr::if_else(
        as.character(plot_series) %in% c("Main VAE", "Exposure"),
        0.8,
        0.45
      )
    )
  if (nrow(participant_summary) == 0L || nrow(model_summary) == 0L) {
    stop("No observation-timestep accuracy data were available.", call. = FALSE)
  }

  plot <- ggplot2::ggplot() +
    ggplot2::geom_line(
      data = participant_summary,
      ggplot2::aes(
        x = node,
        y = mean_accuracy,
        group = condition,
        color = plot_series
      ),
      linewidth = 0.8,
      show.legend = FALSE
    ) +
    ggplot2::geom_errorbar(
      data = participant_summary,
      ggplot2::aes(
        x = node,
        ymin = mean_accuracy - se_accuracy,
        ymax = mean_accuracy + se_accuracy,
        color = plot_series
      ),
      width = 0.2,
      linewidth = 0.3,
      show.legend = FALSE
    ) +
    ggplot2::geom_point(
      data = participant_summary,
      ggplot2::aes(x = node, y = mean_accuracy, color = plot_series),
      size = 0.5,
      show.legend = FALSE
    ) +
    ggplot2::geom_line(
      data = model_summary,
      ggplot2::aes(
        x = node,
        y = mean_accuracy,
        group = interaction(condition, model_name),
        color = plot_series,
        linetype = plot_series,
        linewidth = plot_linewidth
      )
    ) +
    ggplot2::geom_errorbar(
      data = model_summary,
      ggplot2::aes(
        x = node,
        ymin = mean_accuracy - se_accuracy,
        ymax = mean_accuracy + se_accuracy,
        color = plot_series
      ),
      width = 0.2,
      linewidth = 0.25
    ) +
    ggplot2::geom_point(
      data = model_summary,
      ggplot2::aes(
        x = node,
        y = mean_accuracy,
        color = plot_series,
        shape = plot_series
      ),
      size = 0.6,
      stroke = 0.3
    ) +
    ggplot2::facet_wrap(~source, nrow = 1) +
    ggplot2::scale_color_manual(
      name = "Model",
      values = c(
        "Participants" = fitted_participant_color,
        fitted_model_colors(model_names)
      )
    ) +
    ggplot2::scale_linetype_manual(
      name = "Model",
      values = fitted_model_linetypes(model_names)
    ) +
    ggplot2::scale_shape_manual(
      name = "Model",
      values = fitted_model_shapes(model_names)
    ) +
    ggplot2::scale_linewidth_identity(guide = "none") +
    ggplot2::scale_x_continuous(
      breaks = if (node_count <= 6L) {
        seq_len(node_count)
      } else {
        sort(unique(c(seq(1L, node_count, by = 2L), node_count)))
      },
      limits = c(1, node_count),
      expand = ggplot2::expansion(add = 0.25)
    ) +
    ggplot2::scale_y_continuous(
      breaks = seq(y_min, y_max, by = 0.2),
      expand = ggplot2::expansion(mult = c(0.01, 0.03))
    ) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(
      x = "Observation timestep",
      y = "Recall accuracy"
    ) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black"
      ),
      axis.text = ggplot2::element_text(
        # Match plot_fitted_binned_data_shade(), where theme_minimal uses
        # its default 0.8 relative size for axis tick labels.
        size = base_font_size * 0.8,
        color = "black"
      ),
      panel.spacing = grid::unit(1.31, "mm"),
      strip.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      strip.clip = "off",
      legend.position = "none",
      plot.margin = ggplot2::margin(0, 1, 0, 1, unit = "mm")
    )
  output_path <- analysis_path(
    "figures",
    save_dir,
    "observation_timestep_accuracy.pdf"
  )
  summary_path <- analysis_path(
    "stats",
    "tests",
    save_dir,
    "observation_timestep_accuracy.csv"
  )
  ensure_parent_directory(output_path)
  ensure_parent_directory(summary_path)
  ggsave_pdf(
    output_path,
    plot = plot,
    width = x_size,
    height = y_size,
    limitsize = FALSE
  )
  readr::write_csv(
    dplyr::bind_rows(
      participant_summary,
      model_summary %>% dplyr::mutate(model_name = as.character(model_name))
    ),
    summary_path
  )
  cat(sprintf("Saved observation-timestep accuracy plot to: %s\n", output_path))
  cat(sprintf("Saved observation-timestep accuracy summary to: %s\n", summary_path))
  invisible(list(plot = plot, summary_path = summary_path))
}

fitted_reviewer_matched_context_grid <- function(
    panels, reference_plot, reference_width, reference_height,
    base_font_size = 7) {
  stopifnot(length(panels) == 4L)
  panel_height <- fitted_panel_height_mm(reference_plot, reference_width, reference_height)
  panel_width <- panel_height / reference_plot$theme$aspect.ratio
  panel_gap <- reference_plot$theme$panel.spacing
  grDevices::pdf(NULL, width = reference_width / 25.4, height = reference_height / 25.4)
  on.exit(grDevices::dev.off(), add = TRUE)
  grobs <- lapply(seq_along(panels), function(i) {
    p <- panels[[i]] + ggplot2::labs(
      x = sub(" coefficient$", "\ncoefficient", panels[[i]]$labels$x)
    ) + ggplot2::theme(
      aspect.ratio = NULL,
      axis.text = ggplot2::element_text(size = base_font_size * 0.8),
      plot.margin = ggplot2::margin(0, 0, 0, 0, "mm")
    )
    # Share the y axis within each row, as in the reference's two facets.
    if (i %% 2L == 0L) p <- p + ggplot2::theme(
      axis.title.y = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank(),
      axis.ticks.y = ggplot2::element_blank()
    )
    g <- ggplot2::ggplotGrob(p)
    panel <- g$layout[g$layout$name == "panel", ]
    g$widths[panel$l] <- grid::unit(panel_width, "mm")
    g$heights[panel$t] <- grid::unit(panel_height, "mm")
    g
  })
  # Align axes across rows without stretching the fixed-size drawing panels.
  for (indices in list(c(1L, 3L), c(2L, 4L))) {
    widths <- grid::unit.pmax(grobs[[indices[1L]]]$widths, grobs[[indices[2L]]]$widths)
    for (i in indices) grobs[[i]]$widths <- widths
  }
  for (indices in list(c(1L, 2L), c(3L, 4L))) {
    heights <- grid::unit.pmax(grobs[[indices[1L]]]$heights, grobs[[indices[2L]]]$heights)
    for (i in indices) grobs[[i]]$heights <- heights
  }
  result <- gtable::gtable(
    widths = grid::unit.c(grid::unit(1, "mm"), sum(grobs[[1]]$widths),
                          panel_gap, sum(grobs[[2]]$widths), grid::unit(1, "mm")),
    # Reserve headroom for the rotated y title, which can exceed panel height.
    heights = grid::unit.c(grid::unit(6, "mm"), sum(grobs[[1]]$heights),
                           grid::unit(3, "mm"), sum(grobs[[3]]$heights), grid::unit(1, "mm"))
  )
  for (i in seq_along(grobs)) result <- gtable::gtable_add_grob(
    result, grobs[[i]], t = if (i <= 2L) 2L else 4L,
    l = if (i %% 2L == 1L) 2L else 4L, clip = "off"
  )
  list(plot = result,
       width = grid::convertWidth(sum(result$widths), "mm", valueOnly = TRUE),
       height = grid::convertHeight(sum(result$heights), "mm", valueOnly = TRUE))
}

fitted_panel_height_mm <- function(plot, width_mm, height_mm) {
  fitted_panel_size_mm(plot, width_mm, height_mm)[["height"]]
}

fitted_panel_size_mm <- function(plot, width_mm, height_mm) {
  # Measure the actual panel on a PDF device of the reference figure's size.
  grDevices::pdf(NULL, width = width_mm / 25.4, height = height_mm / 25.4)
  on.exit(grDevices::dev.off(), add = TRUE)
  grid::grid.newpage()
  grid::grid.draw(ggplot2::ggplotGrob(plot))
  grid::grid.force()
  viewport_names <- grid::grid.ls(
    viewports = TRUE, grobs = FALSE, print = FALSE
  )$name
  panel_viewport <- grep("^panel[.-]", viewport_names, value = TRUE)[1L]
  if (is.na(panel_viewport)) {
    stop("Could not measure the reference plot panel.", call. = FALSE)
  }
  grid::seekViewport(panel_viewport)
  c(width = grid::convertWidth(grid::unit(1, "npc"), "mm", valueOnly = TRUE),
    height = grid::convertHeight(grid::unit(1, "npc"), "mm", valueOnly = TRUE))
}

fitted_timing_matched_panel_layout <- function(
    plot, legend, reference_plot, reference_width, reference_height,
    legend_height = 10, legend_position = "bottom", match_panel_width = FALSE) {
  target <- fitted_panel_size_mm(reference_plot, reference_width, reference_height)
  grob <- if (inherits(plot, "gtable")) plot else ggplot2::ggplotGrob(plot)
  rows <- unique(grob$layout$t[grepl("^panel", grob$layout$name)])
  stopifnot(length(rows) > 0L)
  grob$heights[rows] <- grid::unit(target[["height"]], "mm")
  if (match_panel_width) {
    columns <- unique(grob$layout$l[grepl("^panel", grob$layout$name)])
    grob$widths[columns] <- grid::unit(target[["width"]], "mm")
  }
  # Retain the existing horizontal spacing, but do not let aspect-ratio
  # constraints shrink the explicitly sized panels to accommodate the legend.
  grob$respect <- FALSE
  heights <- grid::unit.c(sum(grob$heights), grid::unit(legend_height, "mm"))
  if (legend_position == "top") heights <- heights[2:1]
  result <- gtable::gtable(widths = grid::unit(1, "null"), heights = heights)
  panel_row <- if (legend_position == "bottom") 1L else 2L
  result <- gtable::gtable_add_grob(result, grob, t = panel_row, l = 1, clip = "off")
  result <- gtable::gtable_add_grob(result, legend, t = 3L - panel_row, l = 1, clip = "off")
  list(plot = result, height = grid::convertHeight(sum(heights), "mm", valueOnly = TRUE),
       width = if (match_panel_width) max(
         grid::convertWidth(sum(grob$widths), "mm", valueOnly = TRUE),
         grid::convertWidth(sum(legend$widths), "mm", valueOnly = TRUE)) else NULL)
}

plot_fitted_model_nll <- function(
    model_inputs,
    save_dir,
    x_size = 65,
    y_size = 36,
    base_font_size = 7,
    reference_plot = NULL,
    reference_width = 40,
    bar_reference_plot = NULL,
    bar_reference_width = 60,
    bar_width_mm = NULL) {
  if (!is.null(bar_width_mm) && (length(bar_width_mm) != 1L ||
      !is.numeric(bar_width_mm) || !is.finite(bar_width_mm) || bar_width_mm <= 0)) {
    stop("bar_width_mm must be one positive finite number.", call. = FALSE)
  }
  if (length(model_inputs) == 0L) {
    stop("At least one fitted model is required for the NLL plot.", call. = FALSE)
  }
  required_columns <- c(
    "subject",
    "path_nll",
    "n_path_choices",
    "reconstruction_nll",
    "n_reward_choices",
    "reward_nll_scale",
    "scaled_reconstruction_nll",
    "total_nll"
  )
  fit_frames <- lapply(model_inputs, function(model_input) {
    display_name <- if (!is.null(model_input$display_name)) {
      as.character(model_input$display_name)
    } else {
      as.character(model_input$name)
    }
    fit_path <- resolve_fitted_parameter_fit_path(model_input$path)
    fit <- readr::read_csv(
      fit_path,
      col_select = dplyr::any_of(required_columns),
      show_col_types = FALSE,
      progress = FALSE
    )
    require_columns(
      fit,
      required_columns,
      sprintf("%s selected-fit CSV", gsub("\n", " ", model_input$name))
    )
    fit <- fit %>%
      dplyr::mutate(
        subject = as.character(subject),
        dplyr::across(
          dplyr::all_of(setdiff(required_columns, "subject")),
          as.numeric
        )
      )
    if (anyDuplicated(fit$subject)) {
      stop(
        sprintf(
          "%s selected-fit CSV contains duplicate participants.",
          gsub("\n", " ", model_input$name)
        ),
        call. = FALSE
      )
    }
    numeric_columns <- setdiff(required_columns, "subject")
    if (any(!is.finite(as.matrix(fit[numeric_columns])))) {
      stop(
        sprintf(
          "%s selected-fit CSV contains non-finite NLL values.",
          gsub("\n", " ", model_input$name)
        ),
        call. = FALSE
      )
    }
    tolerance <- 1e-8 * pmax(1, abs(fit$total_nll))
    if (any(abs(
      fit$total_nll -
        (fit$path_nll + fit$scaled_reconstruction_nll)
    ) > tolerance)) {
      stop(
        sprintf(
          "%s selected-fit CSV has inconsistent total NLL values.",
          gsub("\n", " ", model_input$name)
        ),
        call. = FALSE
      )
    }
    fit %>%
      dplyr::mutate(
        model_name = display_name,
        selected_fit_path = fit_path
      )
  })

  reference_subjects <- sort(unique(fit_frames[[1L]]$subject))
  if (any(vapply(
    fit_frames,
    function(fit) !identical(sort(unique(fit$subject)), reference_subjects),
    logical(1)
  ))) {
    stop(
      "The fitted models do not contain identical participant sets.",
      call. = FALSE
    )
  }
  reference_counts <- fit_frames[[1L]] %>%
    dplyr::arrange(subject) %>%
    dplyr::select(subject, n_path_choices, n_reward_choices)
  if (any(vapply(
    fit_frames,
    function(fit) {
      comparison_counts <- fit %>%
        dplyr::arrange(subject) %>%
        dplyr::select(subject, n_path_choices, n_reward_choices)
      !identical(comparison_counts, reference_counts)
    },
    logical(1)
  ))) {
    stop(
      "The fitted models use different path or recall observation counts.",
      call. = FALSE
    )
  }
  reward_nll_scales <- unique(unlist(lapply(
    fit_frames,
    function(fit) unique(fit$reward_nll_scale)
  )))
  if (length(reward_nll_scales) != 1L) {
    stop(
      "The fitted models use different reward-NLL scales.",
      call. = FALSE
    )
  }

  model_names <- vapply(
    model_inputs,
    function(model_input) {
      if (!is.null(model_input$display_name)) {
        as.character(model_input$display_name)
      } else {
        as.character(model_input$name)
      }
    },
    character(1)
  )
  nll_summary <- dplyr::bind_rows(fit_frames) %>%
    dplyr::mutate(
      model_name = factor(model_name, levels = model_names)
    ) %>%
    dplyr::group_by(model_name) %>%
    dplyr::summarise(
      participant_count = dplyr::n_distinct(subject),
      path_choice_count = sum(n_path_choices),
      reward_choice_count = sum(n_reward_choices),
      path_nll = sum(path_nll),
      reconstruction_nll = sum(reconstruction_nll),
      reward_nll_scale = dplyr::first(reward_nll_scale),
      scaled_reconstruction_nll = sum(scaled_reconstruction_nll),
      mean_participant_path_nll = mean(path_nll),
      mean_participant_reconstruction_nll = mean(reconstruction_nll),
      mean_participant_scaled_reconstruction_nll = mean(
        scaled_reconstruction_nll
      ),
      mean_participant_total_nll = mean(total_nll),
      se_participant_total_nll = stats::sd(total_nll) / sqrt(dplyr::n()),
      total_nll = sum(total_nll),
      selected_fit_path = dplyr::first(selected_fit_path),
      .groups = "drop"
    )

  nll_plot <- ggplot2::ggplot(
    nll_summary,
    ggplot2::aes(
      x = model_name,
      y = mean_participant_total_nll,
      fill = model_name
    )
  ) +
    ggplot2::geom_col(width = 0.4) +
    ggplot2::geom_errorbar(
      ggplot2::aes(
        ymin = mean_participant_total_nll - se_participant_total_nll,
        ymax = mean_participant_total_nll + se_participant_total_nll
      ),
      width = 0,
      linewidth = 0.3,
      color = "black"
    ) +
    ggplot2::scale_fill_manual(
      values = fitted_model_colors(model_names),
      guide = "none"
    ) +
    ggplot2::scale_y_continuous(
      labels = scales::label_comma(),
      expand = ggplot2::expansion(mult = c(0, 0.05))
    ) +
    ggplot2::scale_x_discrete(
      labels = function(labels) gsub(
        "shared-representation", "shared-\nrepresentation", labels, fixed = TRUE
      )
    ) +
    ggplot2::labs(x = NULL, y = "Average NLL") +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(color = "black", linewidth = 0.4),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.15, "cm"),
      legend.position = "none",
      plot.margin = ggplot2::margin(1, 3, 1, 3, "mm")
    )

  if (!is.null(bar_reference_plot) || !is.null(bar_width_mm)) {
    # Match physical bar widths, not just geom widths on different x scales.
    panel_width <- function(plot, width) {
      grDevices::pdf(NULL, width = width / 25.4, height = y_size / 25.4)
      on.exit(grDevices::dev.off())
      grid::grid.newpage(); grid::grid.draw(ggplot2::ggplotGrob(plot)); grid::grid.force()
      names <- grid::grid.ls(viewports = TRUE, grobs = FALSE, print = FALSE)$name
      grid::seekViewport(grep("^panel[.-]", names, value = TRUE)[1L])
      grid::convertWidth(grid::unit(1, "npc"), "mm", valueOnly = TRUE)
    }
    target_mm <- bar_width_mm
    if (is.null(target_mm)) {
      ref <- ggplot2::ggplot_build(bar_reference_plot)
      ref_bar_width <- median(ref$data[[1]]$xmax - ref$data[[1]]$xmin)
      target_mm <- ref_bar_width / diff(ref$layout$panel_params[[1]]$x.range) *
        panel_width(bar_reference_plot, bar_reference_width)
    }
    own <- ggplot2::ggplot_build(nll_plot)
    nll_plot$layers[[1]]$geom_params$width <- target_mm /
      panel_width(nll_plot, x_size) * diff(own$layout$panel_params[[1]]$x.range)
  }
  plot_to_save <- nll_plot
  if (!is.null(reference_plot)) {
    panel_height_mm <- fitted_panel_height_mm(reference_plot, reference_width, y_size)
    plot_to_save <- ggplot2::ggplotGrob(nll_plot)
    panel_rows <- unique(plot_to_save$layout$t[plot_to_save$layout$name == "panel"])
    plot_to_save$heights[panel_rows] <- grid::unit(panel_height_mm, "mm")
  }

  plot_output_path <- analysis_path(
    "figures",
    save_dir,
    "bar_model_nll.pdf"
  )
  summary_output_path <- analysis_path(
    "stats",
    "tests",
    paste0("fitted_model_nll_", save_dir, ".csv")
  )
  ensure_parent_directory(plot_output_path)
  ensure_parent_directory(summary_output_path)
  ggsave_pdf(
    plot_output_path,
    plot = plot_to_save,
    width = x_size,
    height = y_size,
    limitsize = FALSE
  )
  readr::write_csv(
    nll_summary %>%
      dplyr::mutate(model_name = as.character(model_name)),
    summary_output_path
  )
  cat(sprintf("Saved fitted-model NLL plot to: %s\n", plot_output_path))
  cat(sprintf("Saved fitted-model NLL summary to: %s\n", summary_output_path))
  list(
    plot = nll_plot,
    summary = nll_summary,
    plot_path = plot_output_path,
    summary_path = summary_output_path
  )
}

plot_fitted_binned_data_bar <- function(
    df_beh,
    df_model = NULL,
    dependent_var,
    x_lab = "Value",
    y_lab = "Average Accuracy",
    title = NULL,
    figure_name,
    x_size = ifelse(is.null(df_model), 40, 60),
    y_size = 36,
    base_font_size = 7,
    y_min = NULL,
    y_max = NULL,
    x_step = NULL,
    x_label_step = NULL,
    y_step = NULL,
    combined_plot = FALSE,
    return_data = FALSE,
    show_shading = FALSE,
    x_min = -5,
    x_max = 5,
    y_var = c("accuracy", "estimated_reward"),
    show_exposure = FALSE,
    save_dir = "plots") {
  y_var <- match.arg(y_var)

  discrete_data <- prepare_fitted_binned_plot_data(
    df_beh = df_beh,
    df_model = df_model,
    dependent_var = dependent_var,
    y_var = y_var,
    show_exposure = show_exposure
  )

  if (return_data) {
    return(discrete_data)
  }

  dodge_w <- if (!is.null(x_step)) x_step * 0.8 else 0.8
  bar_w <- if (!is.null(x_step)) x_step * 0.7 else 0.7
  label_step <- if (!is.null(x_label_step)) {
    x_label_step
  } else if (!is.null(x_step)) {
    x_step
  } else {
    1
  }
  x_breaks <- seq(from = x_min, to = x_max, by = label_step)

  p <- ggplot2::ggplot(
    discrete_data,
    ggplot2::aes(x = x_value, y = mean_error, fill = condition)
  ) +
    ggplot2::geom_bar(
      stat = "identity",
      position = ggplot2::position_dodge(width = dodge_w),
      width = bar_w,
      alpha = 0.9
    ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(
        ymin = mean_error - se_error,
        ymax = mean_error + se_error,
        group = condition
      ),
      position = ggplot2::position_dodge(width = dodge_w),
      width = bar_w * 0.3,
      linewidth = 0.3,
      color = "black",
      na.rm = TRUE
    ) +
    ggplot2::scale_fill_manual(
      values = c("planning" = "#6e3934", "exposure" = "#c2beb5")
    ) +
    ggplot2::scale_x_continuous(breaks = x_breaks, expand = c(0, 0)) +
    ggplot2::coord_cartesian(
      xlim = c(x_min - (dodge_w / 2), x_max + (dodge_w / 2)),
      ylim = c(y_min, y_max)
    ) +
    ggplot2::labs(x = x_lab, y = y_lab, title = title) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.0)

  if (!is.null(df_model)) {
    p <- p + ggplot2::facet_wrap(~source, nrow = 1)
  }

  if (!is.null(y_step)) {
    p <- p + ggplot2::scale_y_continuous(
      breaks = seq(y_min, y_max, by = y_step)
    )
  }

  if (!combined_plot) {
    save_custom_plot(
      p,
      figure_name,
      y_var,
      save_dir,
      x_size,
      y_size,
      show_exposure,
      "bar_"
    )
  }

  p
}

plot_fitted_binned_data_shade <- function(
    df_beh,
    df_model = NULL,
    dependent_var,
    x_lab = "Value",
    y_lab = "Average Accuracy",
    title = NULL,
    figure_name,
    x_size = 42.5,
    y_size = 36,
    base_font_size = 7,
    y_min = NULL,
    y_max = NULL,
    x_step = NULL,
    y_step = NULL,
    combined_plot = FALSE,
    return_data = FALSE,
    show_shading = FALSE,
    x_min = -4,
    x_max = 4,
    y_var = c("accuracy", "estimated_reward"),
    show_exposure = FALSE,
    save_dir = "plots",
    show_model_comparison_legend = FALSE,
    model_legend_position = c("none", "right", "top"),
    model_legend_width = 0,
    model_legend_height = 0,
    model_legend_side_margin = 0,
    equal_count_bins = NULL,
    errorbar_cap_width = 0.5,
    append_exposure_suffix = show_exposure) {
  y_var <- match.arg(y_var)
  model_legend_position <- match.arg(model_legend_position)
  if (!is.numeric(model_legend_width) || model_legend_width < 0) {
    stop("model_legend_width must be a non-negative number.", call. = FALSE)
  }
  if (!is.numeric(model_legend_height) || model_legend_height < 0) {
    stop("model_legend_height must be a non-negative number.", call. = FALSE)
  }
  if (!is.numeric(model_legend_side_margin) || model_legend_side_margin < 0) {
    stop("model_legend_side_margin must be a non-negative number.", call. = FALSE)
  }
  if (
    length(errorbar_cap_width) != 1L ||
      !is.numeric(errorbar_cap_width) ||
      !is.finite(errorbar_cap_width) ||
      errorbar_cap_width < 0
  ) {
    stop("errorbar_cap_width must be one non-negative number.", call. = FALSE)
  }

  if (y_var == "estimated_reward") {
    y_min <- -4
    y_max <- 4
    show_exposure <- TRUE
  }

  if (is.null(df_model) && is.null(y_step)) {
    y_step <- 0.1
  }

  discrete_data <- if (is.null(equal_count_bins)) {
    prepare_fitted_binned_plot_data(
      df_beh = df_beh,
      df_model = df_model,
      dependent_var = dependent_var,
      y_var = y_var,
      show_exposure = show_exposure
    )
  } else {
    prepare_fitted_equal_count_binned_plot_data(
      df_beh = df_beh,
      df_model = df_model,
      dependent_var = dependent_var,
      y_var = y_var,
      show_exposure = show_exposure,
      equal_count_bins = equal_count_bins
    )
  }

  if (return_data) {
    return(discrete_data)
  }

  x_breaks <- if (!is.null(x_step)) {
    seq(from = x_min, to = x_max, by = x_step)
  } else {
    -4:4
  }

  participant_data <- discrete_data %>%
    dplyr::filter(source == "participants") %>%
    dplyr::mutate(
      plot_series = dplyr::if_else(
        tolower(as.character(condition)) == "exposure",
        "Exposure",
        "Participants"
      )
    )
  model_data <- discrete_data %>%
    dplyr::filter(source == "model") %>%
    dplyr::mutate(
      plot_series = dplyr::if_else(
        tolower(as.character(condition)) == "exposure",
        "Exposure",
        as.character(model_name)
      )
    )
  model_names <- unique(as.character(model_data$plot_series))
  model_linetypes <- fitted_model_linetypes(model_names)
  model_linewidths <- stats::setNames(
    rep(0.4, length(model_names)),
    model_names
  )
  model_linewidths[model_names %in% c("Main VAE", "Exposure")] <- 0.8
  model_point_sizes <- stats::setNames(
    rep(0.6, length(model_names)),
    model_names
  )
  model_point_sizes[model_names %in% c("Main VAE", "Exposure")] <- 0.5
  model_position <- ggplot2::position_dodge(width = 0.8)

  p <- ggplot2::ggplot() +
    ggplot2::geom_line(
      data = participant_data,
      ggplot2::aes(
        x = x_value,
        y = mean_error,
        group = condition,
        color = plot_series
      ),
      linewidth = 0.8,
      alpha = 0.8,
      show.legend = FALSE
    ) +
    ggplot2::geom_errorbar(
      data = participant_data,
      ggplot2::aes(
        x = x_value,
        ymin = mean_error - se_error,
        ymax = mean_error + se_error,
        color = plot_series
      ),
      width = errorbar_cap_width,
      linewidth = 0.3,
      alpha = 0.95,
      na.rm = TRUE,
      show.legend = FALSE
    ) +
    ggplot2::geom_point(
      data = participant_data,
      ggplot2::aes(x = x_value, y = mean_error, color = plot_series),
      size = 0.5,
      show.legend = FALSE
    ) +
    ggplot2::geom_line(
      data = model_data,
      ggplot2::aes(
        x = x_value,
        y = mean_error,
        group = interaction(condition, model_name),
        color = plot_series,
        linetype = plot_series,
        linewidth = plot_series
      ),
      position = model_position,
      alpha = 0.9
    ) +
    ggplot2::geom_errorbar(
      data = model_data,
      ggplot2::aes(
        x = x_value,
        ymin = mean_error - se_error,
        ymax = mean_error + se_error,
        group = interaction(condition, model_name),
        color = plot_series
      ),
      position = model_position,
      width = errorbar_cap_width,
      linewidth = 0.3,
      alpha = 0.9,
      na.rm = TRUE
    ) +
    ggplot2::geom_point(
      data = model_data,
      ggplot2::aes(
        x = x_value,
        y = mean_error,
        group = interaction(condition, model_name),
        color = plot_series,
        shape = plot_series,
        size = plot_series
      ),
      position = model_position,
      stroke = 0.3
    ) +
    ggplot2::scale_color_manual(
      name = "Model",
      values = c(
        "Participants" = fitted_participant_color,
        fitted_model_colors(model_names)
      )
    ) +
    ggplot2::scale_linetype_manual(
      name = "Model",
      values = model_linetypes
    ) +
    ggplot2::scale_shape_manual(
      name = "Model",
      values = fitted_model_shapes(model_names)
    ) +
    ggplot2::scale_linewidth_manual(
      name = "Model",
      values = model_linewidths,
      guide = "none"
    ) +
    ggplot2::scale_size_manual(
      name = "Model",
      values = model_point_sizes,
      guide = "none"
    ) +
    ggplot2::scale_x_continuous(
      breaks = x_breaks,
      expand = ggplot2::expansion(add = 0.5)
    ) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(
      x = x_lab,
      y = y_lab,
      title = title,
      linetype = "Model"
    ) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      strip.clip = "off",
      panel.spacing = grid::unit(1.31, "mm"),
      plot.margin = ggplot2::margin(
        t = 0,
        r = 1,
        b = 0,
        l = 1,
        unit = "mm"
      )
    )

  if (!is.null(df_model)) {
    p <- p + ggplot2::facet_wrap(~source, nrow = 1)
  }

  if (show_model_comparison_legend) {
    p <- p +
      fitted_model_legend_theme(
        base_font_size,
        legend_position = model_legend_position
      )
    if (identical(model_legend_position, "top")) {
      p <- p + ggplot2::guides(
        color = ggplot2::guide_legend(nrow = 2, byrow = TRUE),
        linetype = ggplot2::guide_legend(nrow = 2, byrow = TRUE),
        shape = ggplot2::guide_legend(nrow = 2, byrow = TRUE),
        linewidth = "none",
        size = "none"
      )
    } else {
      p <- p + ggplot2::guides(
        color = ggplot2::guide_legend(ncol = 1, byrow = TRUE),
        linetype = ggplot2::guide_legend(ncol = 1, byrow = TRUE),
        shape = ggplot2::guide_legend(ncol = 1, byrow = TRUE),
        linewidth = "none",
        size = "none"
      )
    }
  } else {
    p <- p + ggplot2::theme(legend.position = "none")
  }

  if (!is.null(y_step)) {
    p <- p + ggplot2::scale_y_continuous(
      breaks = seq(y_min, y_max, by = y_step)
    )
  }

  if (!combined_plot) {
    plot_to_save <- p
    output_width <- x_size + ifelse(
      show_model_comparison_legend &&
        identical(model_legend_position, "right"),
      model_legend_width,
      0
    )
    output_height <- y_size + ifelse(
      show_model_comparison_legend &&
        identical(model_legend_position, "top"),
      model_legend_height,
      0
    )

    if (
      show_model_comparison_legend &&
        identical(model_legend_position, "top") &&
        model_legend_side_margin > 0
    ) {
      legend_grob <- cowplot::get_legend(p)
      panel_plot <- p + ggplot2::theme(legend.position = "none")
      panel_row <- cowplot::plot_grid(
        NULL,
        panel_plot,
        NULL,
        nrow = 1,
        rel_widths = c(
          model_legend_side_margin,
          x_size,
          model_legend_side_margin
        )
      )
      plot_to_save <- cowplot::plot_grid(
        legend_grob,
        panel_row,
        ncol = 1,
        rel_heights = c(model_legend_height, y_size),
        align = "v",
        axis = "lr"
      )
      output_width <- x_size + (2 * model_legend_side_margin)
    }

    save_custom_plot(
      plot_to_save,
      figure_name,
      y_var,
      save_dir,
      output_width,
      output_height,
      show_exposure && append_exposure_suffix
    )
  }

  p
}

plot_fitted_directional_context_reward_accuracy <- function(
    df_beh,
    df_model,
    node_count,
    save_dir,
    reward_variable,
    reward_label,
    output_stem,
    y_min = 0,
    y_max = 0.8,
    x_size = 42.5,
    panel_height = 36,
    legend_height = 7,
    group_header_height = 4,
    base_font_size = 7,
    equal_count_bins = 10L) {
  reward_variable <- as.character(reward_variable)
  reward_label <- as.character(reward_label)
  output_stem <- as.character(output_stem)
  if (
    length(reward_variable) != 1L ||
      !nzchar(reward_variable) ||
      length(reward_label) != 1L ||
      !nzchar(reward_label) ||
      length(output_stem) != 1L ||
      !nzchar(output_stem)
  ) {
    stop(
      "reward_variable, reward_label, and output_stem must be nonempty strings.",
      call. = FALSE
    )
  }
  require_columns(
    df_beh,
    c("subject", "condition", "node", reward_variable, "accuracy"),
    sprintf("Human directional %s data", reward_variable)
  )
  require_columns(
    df_model,
    c(
      "fitted_subject",
      "model_name",
      "condition",
      "node",
      reward_variable,
      "accuracy"
    ),
    sprintf("Fitted-model directional %s data", reward_variable)
  )

  branch_groups <- fitted_leaf_branch_groups(node_count)
  early_nodes <- branch_groups[[1L]]
  late_nodes <- branch_groups[[length(branch_groups)]]
  timing_levels <- c("earlier", "later")
  timing_labels <- c(
    "Probed node presented earlier",
    "Probed node presented later"
  )

  human_data <- df_beh %>%
    dplyr::transmute(
      source = "Participants",
      participant_id = as.character(subject),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      reward_value = as.numeric(.data[[reward_variable]]),
      accuracy = as.numeric(accuracy)
    )
  model_data <- df_model %>%
    dplyr::transmute(
      source = as.character(model_name),
      participant_id = as.character(fitted_subject),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      reward_value = as.numeric(.data[[reward_variable]]),
      accuracy = as.numeric(accuracy)
    )
  available_model_sources <- fitted_timing_model_levels(model_data$source)
  source_levels <- c("Participants", available_model_sources)
  facet_labels <- fitted_timing_facet_labels()
  classified <- dplyr::bind_rows(human_data, model_data) %>%
    dplyr::filter(
      condition == "planning",
      node %in% c(early_nodes, late_nodes),
      is.finite(reward_value),
      is.finite(accuracy)
    ) %>%
    dplyr::mutate(
      source = factor(source, levels = source_levels),
      facet_label = factor(
        unname(facet_labels[as.character(source)]),
        levels = unname(facet_labels[source_levels])
      ),
      probe_timing = factor(
        dplyr::if_else(node %in% early_nodes, "earlier", "later"),
        levels = timing_levels,
        labels = timing_labels
      ),
      .equal_count_row = dplyr::row_number()
    )

  equal_count_bins <- as.integer(equal_count_bins)
  if (
    length(equal_count_bins) != 1L ||
      is.na(equal_count_bins) ||
      equal_count_bins < 2L
  ) {
    stop("equal_count_bins must be an integer of at least 2.", call. = FALSE)
  }
  binned <- classified %>%
    dplyr::group_by(source, facet_label, probe_timing) %>%
    dplyr::arrange(
      reward_value,
      participant_id,
      .equal_count_row,
      .by_group = TRUE
    ) %>%
    dplyr::mutate(
      equal_count_bin = dplyr::ntile(dplyr::row_number(), equal_count_bins)
    ) %>%
    dplyr::ungroup()
  bin_metadata <- binned %>%
    dplyr::group_by(source, facet_label, probe_timing, equal_count_bin) %>%
    dplyr::summarise(
      reward_value = mean(reward_value),
      reward_value_min = min(reward_value),
      reward_value_max = max(reward_value),
      bin_observation_count = dplyr::n(),
      .groups = "drop"
    )
  summary <- binned %>%
    dplyr::group_by(
      source,
      facet_label,
      participant_id,
      probe_timing,
      equal_count_bin
    ) %>%
    dplyr::summarise(
      participant_accuracy = mean(accuracy),
      observation_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::group_by(source, facet_label, probe_timing, equal_count_bin) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_accuracy) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    dplyr::left_join(
      bin_metadata,
      by = c("source", "facet_label", "probe_timing", "equal_count_bin"),
      relationship = "one-to-one"
    ) %>%
    dplyr::filter(participant_count >= 5L) %>%
    dplyr::mutate(
      lower = pmax(0, mean_accuracy - se),
      upper = pmin(1, mean_accuracy + se)
    )
  if (nrow(summary) == 0L) {
    stop(
      sprintf(
        "No directional %s cells contained five participants.",
        reward_variable
      ),
      call. = FALSE
    )
  }
  observed_conditions <- unique(as.character(summary$probe_timing))
  if (!all(timing_labels %in% observed_conditions)) {
    stop(
      sprintf(
        "Directional %s data do not contain both probe timings.",
        reward_variable
      ),
      call. = FALSE
    )
  }

  timing_colors <- stats::setNames(
    c("#8B4741", "#532B27"),
    timing_labels
  )
  accuracy_plot <- ggplot2::ggplot(
    summary,
    ggplot2::aes(
      x = reward_value,
      y = mean_accuracy,
      color = probe_timing,
      group = probe_timing
    )
  ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower, ymax = upper),
      width = 0.25,
      linewidth = 0.3,
      alpha = 0.95,
      na.rm = TRUE
    ) +
    ggplot2::geom_line(linewidth = 0.8, alpha = 0.8) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::facet_wrap(~facet_label, nrow = 1, drop = FALSE) +
    ggplot2::scale_color_manual(values = timing_colors, drop = FALSE) +
    ggplot2::guides(
      color = ggplot2::guide_legend(nrow = 2, byrow = TRUE)
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq(-4, 4, by = 2),
      expand = ggplot2::expansion(add = 0.5)
    ) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(
      x = reward_label,
      y = "Recall accuracy",
      color = NULL
    ) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size,
        color = "black"
      ),
      legend.position = "none",
      strip.text = ggplot2::element_text(
        size = base_font_size,
        color = "black",
        face = "plain"
      ),
      strip.clip = "off",
      panel.spacing = grid::unit(0.75, "lines"),
      plot.margin = ggplot2::margin(0, 1, 0, 1, unit = "mm")
    )

  compact_legend <- cowplot::get_legend(
    accuracy_plot +
      ggplot2::theme(
        legend.position = "top",
        legend.justification = "left",
        legend.box.just = "left",
        legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
        legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
        legend.box.spacing = grid::unit(0, "mm"),
        legend.spacing = grid::unit(0, "mm"),
        legend.spacing.y = grid::unit(0, "mm"),
        legend.key.height = grid::unit(2.5, "mm"),
        legend.key.width = grid::unit(4, "mm"),
        legend.text = ggplot2::element_text(
          size = base_font_size,
          color = "black",
          margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
        ),
        plot.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
      )
  )
  if (is.null(compact_legend)) {
    stop("Could not construct the probe-timing legend.", call. = FALSE)
  }
  participant_header <- cowplot::ggdraw() +
    cowplot::draw_label(
      "participants",
      size = base_font_size * 0.8,
      color = "black",
      fontface = "plain"
    )
  model_header <- cowplot::ggdraw() +
    cowplot::draw_label(
      "models",
      size = base_font_size * 0.8,
      color = "black",
      fontface = "plain"
    )
  group_header <- cowplot::plot_grid(
    participant_header,
    model_header,
    nrow = 1,
    rel_widths = c(1, length(available_model_sources))
  )
  combined_plot <- cowplot::plot_grid(
    compact_legend,
    group_header,
    accuracy_plot,
    ncol = 1,
    rel_heights = c(legend_height, group_header_height, panel_height),
    align = "v",
    axis = "lr"
  )
  output_width <- x_size * length(source_levels) / 2
  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0(output_stem, "_accuracy_by_probe_timing.pdf")
  )
  summary_path <- analysis_path(
    "stats",
    "tests",
    save_dir,
    paste0(output_stem, "_accuracy_by_probe_timing.csv")
  )
  ensure_parent_directory(output_path)
  ensure_parent_directory(summary_path)
  ggsave_pdf(
    output_path,
    plot = combined_plot,
    width = output_width,
    height = panel_height + legend_height + group_header_height
  )
  output_summary <- summary %>%
    dplyr::mutate(reward_variable = reward_variable) %>%
    dplyr::rename(
      !!reward_variable := reward_value,
      !!paste0(reward_variable, "_min") := reward_value_min,
      !!paste0(reward_variable, "_max") := reward_value_max
    )
  readr::write_csv(output_summary, summary_path)
  cat(sprintf("Saved directional %s plot to: %s\n", reward_variable, output_path))
  cat(sprintf(
    "Saved directional %s summary to: %s\n",
    reward_variable,
    summary_path
  ))
  invisible(list(plot = combined_plot, summary = output_summary))
}

plot_fitted_directional_cousin_reward_accuracy <- function(
    df_beh,
    df_model,
    node_count,
    save_dir,
    ...) {
  plot_fitted_directional_context_reward_accuracy(
    df_beh = df_beh,
    df_model = df_model,
    node_count = node_count,
    save_dir = save_dir,
    reward_variable = "cousin_reward",
    reward_label = "Average Cousin Reward",
    output_stem = "cousin_reward",
    ...
  )
}

fitted_max_other_path_lookup <- function(
    trials,
    reward_columns,
    path_nodes,
    trial_id_column = ".path_timing_trial_id") {
  require_columns(
    trials,
    c(trial_id_column, reward_columns),
    "Maximum-other-path trial table"
  )
  reward_values <- as.matrix(trials[, reward_columns, drop = FALSE])
  storage.mode(reward_values) <- "double"
  if (any(!is.finite(reward_values))) {
    stop("Maximum-other-path rewards must all be finite.", call. = FALSE)
  }

  node_count <- length(reward_columns)
  path_count <- length(path_nodes)
  path_incidence <- matrix(0L, nrow = path_count, ncol = node_count)
  path_completion_position <- integer(path_count)
  path_root_branch <- integer(path_count)
  for (path_index in seq_along(path_nodes)) {
    nodes <- as.integer(path_nodes[[path_index]])
    if (
      length(nodes) == 0L ||
        any(!nodes %in% seq_len(node_count))
    ) {
      stop("A maximum-other-path definition contains an invalid node.", call. = FALSE)
    }
    path_incidence[path_index, nodes] <- 1L
    path_completion_position[[path_index]] <- max(nodes)
    path_root_branch[[path_index]] <- nodes[[1L]]
  }
  node_root_branch <- vapply(seq_len(node_count), function(node) {
    containing_branches <- unique(
      path_root_branch[path_incidence[, node] == 1L]
    )
    if (length(containing_branches) != 1L) {
      stop(
        sprintf(
          "Node %d does not have exactly one root-branch assignment.",
          node
        ),
        call. = FALSE
      )
    }
    containing_branches[[1L]]
  }, integer(1))
  path_rewards <- reward_values %*% t(path_incidence)
  trial_ids <- trials[[trial_id_column]]

  dplyr::bind_rows(lapply(seq_len(node_count), function(probe_node) {
    # "Other" means a complete terminal path that does not contain the
    # probed node. A path is theoretically before the probe only if its final
    # node has already appeared according to the task's node-index order.
    eligible_paths <- which(path_incidence[, probe_node] == 0L)
    if (length(eligible_paths) == 0L) {
      return(NULL)
    }
    eligible_rewards <- path_rewards[, eligible_paths, drop = FALSE]
    maximum_reward <- apply(eligible_rewards, 1L, max)
    is_maximum <- sweep(eligible_rewards, 1L, maximum_reward, `==`)
    path_before_probe <-
      path_completion_position[eligible_paths] < probe_node
    maximum_occurs_before <- if (any(path_before_probe)) {
      rowSums(is_maximum[, path_before_probe, drop = FALSE]) > 0L
    } else {
      rep(FALSE, nrow(eligible_rewards))
    }
    maximum_occurs_after <- if (any(!path_before_probe)) {
      rowSums(is_maximum[, !path_before_probe, drop = FALSE]) > 0L
    } else {
      rep(FALSE, nrow(eligible_rewards))
    }

    probe_root_branch <- node_root_branch[[probe_node]]
    other_branch_paths <- which(path_root_branch != probe_root_branch)
    if (length(other_branch_paths) == 0L) {
      return(NULL)
    }
    other_branch_rewards <- path_rewards[
      ,
      other_branch_paths,
      drop = FALSE
    ]
    other_root_branches <- unique(path_root_branch[other_branch_paths])
    other_branches_before <- other_root_branches < probe_root_branch
    other_branch_timing <- if (all(other_branches_before)) {
      "before_probe"
    } else if (all(!other_branches_before)) {
      "after_probe"
    } else {
      NA_character_
    }

    tibble::tibble(
      .path_timing_trial_id = trial_ids,
      node = probe_node,
      max_other_path_reward = as.numeric(maximum_reward),
      max_path_timing_ambiguous =
        maximum_occurs_before & maximum_occurs_after,
      relative_path_timing = dplyr::case_when(
        maximum_occurs_before & !maximum_occurs_after ~ "before_probe",
        maximum_occurs_after & !maximum_occurs_before ~ "after_probe",
        TRUE ~ NA_character_
      ),
      max_other_branch_path_reward = as.numeric(
        apply(other_branch_rewards, 1L, max)
      ),
      mean_other_branch_path_reward = as.numeric(
        rowMeans(other_branch_rewards)
      ),
      relative_other_branch_timing = other_branch_timing,
      other_branch_timing_ambiguous = is.na(other_branch_timing)
    )
  }))
}

prepare_fitted_max_other_path_human <- function(
    df_beh,
    node_count,
    path_nodes) {
  reward_columns <- paste0("n", seq_len(node_count), "_reward")
  require_columns(
    df_beh,
    c(
      "subject", "graph", "condition", "node", "accuracy",
      reward_columns
    ),
    "Human maximum-other-path data"
  )
  behavior <- df_beh %>%
    dplyr::mutate(
      subject = as.character(subject),
      graph = as.character(graph),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      accuracy = as.numeric(accuracy),
      dplyr::across(dplyr::all_of(reward_columns), as.numeric)
    ) %>%
    dplyr::filter(
      condition == "planning",
      !is.na(subject),
      !is.na(graph),
      node %in% seq_len(node_count),
      is.finite(accuracy),
      dplyr::if_all(dplyr::all_of(reward_columns), is.finite)
    )
  trial_key_columns <- c("subject", "condition", "graph", reward_columns)
  if (anyDuplicated(behavior[c(trial_key_columns, "node")])) {
    stop("Human maximum-other-path data contain duplicate trial/node rows.", call. = FALSE)
  }
  trials <- behavior %>%
    dplyr::distinct(dplyr::across(dplyr::all_of(trial_key_columns))) %>%
    dplyr::mutate(.path_timing_trial_id = dplyr::row_number())
  path_lookup <- fitted_max_other_path_lookup(
    trials,
    reward_columns,
    path_nodes
  )
  behavior %>%
    dplyr::left_join(
      trials,
      by = trial_key_columns,
      relationship = "many-to-one"
    ) %>%
    dplyr::left_join(
      path_lookup,
      by = c(".path_timing_trial_id", "node"),
      relationship = "many-to-one"
    ) %>%
    dplyr::transmute(
      source = "Participants",
      participant_id = subject,
      max_other_path_reward,
      relative_path_timing,
      max_other_branch_path_reward,
      mean_other_branch_path_reward,
      relative_other_branch_timing,
      accuracy
    )
}

prepare_fitted_max_other_path_model <- function(
    df_model,
    node_count,
    path_nodes) {
  require_columns(
    df_model,
    c(
      "subject", "graph", "condition", "node", "actual_reward",
      "estimated_reward", "model_name"
    ),
    "Fitted maximum-other-path data"
  )
  participant_column <- if ("fitted_subject" %in% names(df_model)) {
    "fitted_subject"
  } else {
    "subject"
  }
  model <- df_model %>%
    dplyr::mutate(
      model_name = as.character(model_name),
      subject = as.character(subject),
      graph = as.character(graph),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      actual_reward = as.numeric(actual_reward),
      estimated_reward = as.numeric(estimated_reward)
    ) %>%
    dplyr::filter(
      condition == "planning",
      !is.na(subject),
      !is.na(graph),
      node %in% seq_len(node_count),
      is.finite(actual_reward),
      is.finite(estimated_reward)
    )
  trial_key_columns <- c("model_name", "subject", "condition", "graph")
  if (anyDuplicated(model[c(trial_key_columns, "node")])) {
    stop("Fitted maximum-other-path data contain duplicate trial/node rows.", call. = FALSE)
  }
  graph_sizes <- model %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(trial_key_columns))) %>%
    dplyr::summarise(node_count = dplyr::n(), .groups = "drop")
  if (any(graph_sizes$node_count != node_count)) {
    stop(
      "Every fitted maximum-other-path trial must contain every reward node.",
      call. = FALSE
    )
  }
  trials <- model %>%
    dplyr::select(
      dplyr::all_of(trial_key_columns),
      node,
      actual_reward
    ) %>%
    tidyr::pivot_wider(
      names_from = node,
      values_from = actual_reward,
      names_prefix = "reward_"
    ) %>%
    dplyr::mutate(.path_timing_trial_id = dplyr::row_number())
  reward_columns <- paste0("reward_", seq_len(node_count))
  path_lookup <- fitted_max_other_path_lookup(
    trials,
    reward_columns,
    path_nodes
  )
  model %>%
    dplyr::left_join(
      trials %>%
        dplyr::select(
          dplyr::all_of(trial_key_columns),
          .path_timing_trial_id
        ),
      by = trial_key_columns,
      relationship = "many-to-one"
    ) %>%
    dplyr::left_join(
      path_lookup,
      by = c(".path_timing_trial_id", "node"),
      relationship = "many-to-one"
    ) %>%
    dplyr::transmute(
      source = model_name,
      participant_id = as.character(.data[[participant_column]]),
      max_other_path_reward,
      relative_path_timing,
      max_other_branch_path_reward,
      mean_other_branch_path_reward,
      relative_other_branch_timing,
      accuracy = as.numeric(estimated_reward == actual_reward)
    )
}

plot_fitted_reward_accuracy_source_panels <- function(df_beh, df_model, save_dir,
    y_min = 0, y_max = 1, panel_width = 21.25, panel_height = 36,
    average_reward_labels = TRUE) {
  variables <- c(actual_reward = "Probed reward", parent_reward = "Parent reward",
    sibling_reward = "Average sibling reward", aunt_reward = "Average aunt reward")
  if (!average_reward_labels) {
    variables[c("sibling_reward", "aunt_reward")] <- c("Sibling reward", "Aunt reward")
  }
  for (variable in names(variables)) {
    leaf_only <- variable %in% c("parent_reward", "aunt_reward")
    humans <- if (leaf_only) dplyr::filter(df_beh, is_leaf %in% TRUE) else df_beh
    models <- if (leaf_only) dplyr::filter(df_model, is_leaf %in% TRUE) else df_model
    summary <- prepare_fitted_binned_plot_data(humans, models, variable, show_exposure = FALSE)
    levels <- c("Participants", unique(as.character(df_model$model_name)))
    summary$model_name <- factor(summary$model_name, levels = levels)
    p <- ggplot2::ggplot(summary, ggplot2::aes(x_value, mean_error)) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
        width = 0, linewidth = .3, na.rm = TRUE) +
      ggplot2::geom_line(linewidth = .6, color = fitted_participant_color) +
      ggplot2::geom_point(size = .6, color = fitted_participant_color) +
      ggplot2::facet_wrap(~model_name, nrow = 1, drop = FALSE,
        labeller = ggplot2::label_wrap_gen(width = 16)) +
      ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
      ggplot2::labs(x = variables[[variable]], y = "Accuracy") +
      plot_theme_axes(base_font_size = 7, aspect_ratio = 1.2) +
      ggplot2::theme(legend.position = "none", strip.clip = "off",
        strip.text = ggplot2::element_text(size = 5.6),
        panel.spacing = grid::unit(.75, "lines"),
        plot.margin = ggplot2::margin(3, 2, 2, 2, unit = "mm"))
    stem <- paste0(if (variable == "actual_reward") "node_reward" else variable, "_accuracy_by_source")
    path <- analysis_path("figures", save_dir, paste0(stem, ".pdf"))
    csv <- analysis_path("stats", "tests", save_dir, paste0(stem, ".csv"))
    ensure_parent_directory(path); ensure_parent_directory(csv)
    ggsave_pdf(path, plot = p, width = panel_width * length(levels), height = panel_height + 10)
    readr::write_csv(summary, csv)
    cat(sprintf("Saved source-panel reward plot to: %s\n", path))
  }
}

plot_fitted_max_other_path_reward_accuracy <- function(
    df_beh,
    df_model,
    node_count,
    path_nodes,
    save_dir,
    reward_variable = "max_other_path_reward",
    timing_variable = "relative_path_timing",
    x_label = "Maximum reward of any other path",
    timing_labels = c(
      "Max-reward path appeared\nbefore probed node",
      "Max-reward path appeared\nafter probed node"
    ),
    output_stem = "max_other_path_reward",
    y_min = 0,
    y_max = 0.8,
    x_size = 42.5,
    panel_height = 36,
    legend_height = 7,
    group_header_height = 4,
    minimum_observation_count = 10L,
    minimum_participant_count = 5L,
    base_font_size = 7,
    equal_count_bins = NULL,
    output_suffix = "",
    simple_panel_headers = FALSE,
    match_rank_layout = FALSE,
    legend_position = "top",
    panel_spacing_mm = NULL,
    aligned_panel_headers = FALSE,
    participant_model_gap_mm = NULL,
    reference_plot = NULL,
    reference_width = 40,
    reference_height = 36,
    match_panel_width = FALSE) {
  legend_position <- match.arg(legend_position, c("top", "bottom"))
  top_padding_height <- if (isTRUE(match_rank_layout)) 3 else 0
  # Simplified facet labels replace the group header; keep only a small spacer.
  if (isTRUE(simple_panel_headers) && identical(output_suffix, "_participants_sequential")) {
    group_header_height <- max(0, group_header_height - 2)
  }
  human_data <- prepare_fitted_max_other_path_human(
    df_beh,
    node_count,
    path_nodes
  )
  model_data <- prepare_fitted_max_other_path_model(
    df_model,
    node_count,
    path_nodes
  )
  reward_variable <- as.character(reward_variable)
  timing_variable <- as.character(timing_variable)
  x_label <- as.character(x_label)
  output_stem <- as.character(output_stem)
  if (
    length(reward_variable) != 1L || !nzchar(reward_variable) ||
      length(timing_variable) != 1L || !nzchar(timing_variable) ||
      length(x_label) != 1L || !nzchar(x_label) ||
      length(output_stem) != 1L || !nzchar(output_stem) ||
      length(timing_labels) != 2L
  ) {
    stop("Invalid path-context plotting configuration.", call. = FALSE)
  }
  if (
    length(minimum_observation_count) != 1L ||
      !is.finite(minimum_observation_count) ||
      minimum_observation_count < 1 ||
      length(minimum_participant_count) != 1L ||
      !is.finite(minimum_participant_count) ||
      minimum_participant_count < 1
  ) {
    stop(
      "Path-context sample thresholds must be positive finite values.",
      call. = FALSE
    )
  }
  minimum_observation_count <- as.integer(minimum_observation_count)
  minimum_participant_count <- as.integer(minimum_participant_count)
  require_columns(
    human_data,
    c(reward_variable, timing_variable),
    "Human path-context plotting data"
  )
  require_columns(
    model_data,
    c(reward_variable, timing_variable),
    "Fitted path-context plotting data"
  )
  available_model_sources <- fitted_timing_model_levels(model_data$source)
  source_levels <- c("Participants", available_model_sources)
  facet_labels <- fitted_timing_facet_labels()
  if (isTRUE(aligned_panel_headers)) {
    facet_labels["Participants"] <- "participants"
    group_header_height <- 0
  }
  if (isTRUE(simple_panel_headers)) {
    stopifnot(length(available_model_sources) == 1L)
    facet_labels[source_levels] <- c("participants", "model")
  }
  timing_levels <- c("before_probe", "after_probe")

  classified <- dplyr::bind_rows(human_data, model_data) %>%
    dplyr::mutate(
      reward_value = as.numeric(.data[[reward_variable]]),
      relative_timing = as.character(.data[[timing_variable]]),
      source = factor(source, levels = source_levels),
      facet_label = factor(
        unname(facet_labels[as.character(source)]),
        levels = unname(facet_labels[source_levels])
      ),
      relative_timing = factor(
        relative_timing,
        levels = timing_levels,
        labels = timing_labels
      )
    ) %>%
    dplyr::filter(
      !is.na(source),
      !is.na(relative_timing),
      is.finite(reward_value),
      is.finite(accuracy)
    )
  # Non-overlapping equal-count bins, separately for each model and timing line.
  # Tied reward values may straddle bins; retain bin identity in aggregation.
  if (!is.null(equal_count_bins)) {
    if (length(equal_count_bins) != 1L || !is.finite(equal_count_bins) ||
        equal_count_bins < 2L || equal_count_bins != as.integer(equal_count_bins)) {
      stop("equal_count_bins must be an integer of at least 2.", call. = FALSE)
    }
    classified <- classified %>%
      dplyr::group_by(source, facet_label, relative_timing) %>%
      dplyr::arrange(reward_value, participant_id, .by_group = TRUE) %>%
      dplyr::mutate(reward_bin = dplyr::ntile(dplyr::row_number(), equal_count_bins)) %>%
      dplyr::ungroup()
  } else {
    classified <- classified %>% dplyr::mutate(reward_bin = reward_value)
  }
  bin_metadata <- classified %>%
    dplyr::group_by(source, facet_label, relative_timing, reward_bin) %>%
    dplyr::summarise(reward_value = mean(reward_value), .groups = "drop")
  summary <- classified %>%
    dplyr::group_by(
      source,
      facet_label,
      participant_id,
      relative_timing,
      reward_bin
    ) %>%
    dplyr::summarise(
      participant_accuracy = mean(accuracy),
      observation_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::group_by(
      source,
      facet_label,
      relative_timing,
      reward_bin
    ) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_accuracy) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    dplyr::left_join(bin_metadata,
      by = c("source", "facet_label", "relative_timing", "reward_bin"),
      relationship = "many-to-one") %>%
    dplyr::filter(
      participant_count >= .env$minimum_participant_count,
      observation_count >= .env$minimum_observation_count
    ) %>%
    dplyr::mutate(
      lower = pmax(0, mean_accuracy - se),
      upper = pmin(1, mean_accuracy + se)
    )
  if (nrow(summary) == 0L) {
    stop(
      sprintf(
        paste0(
          "No %s cells contained at least %d observations from at least ",
          "%d participants."
        ),
        reward_variable,
        minimum_observation_count,
        minimum_participant_count
      ),
      call. = FALSE
    )
  }

  timing_colors <- stats::setNames(
    c("#8B4741", "#532B27"),
    timing_labels
  )
  accuracy_plot <- ggplot2::ggplot(
    summary,
    ggplot2::aes(
      x = reward_value,
      y = mean_accuracy,
      color = relative_timing,
      group = relative_timing
    )
  ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower, ymax = upper),
      width = 0,
      linewidth = 0.3,
      alpha = 0.95,
      na.rm = TRUE
    ) +
    ggplot2::geom_line(linewidth = 0.8, alpha = 0.8) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::facet_wrap(~facet_label, nrow = 1, drop = FALSE) +
    ggplot2::scale_color_manual(values = timing_colors, drop = FALSE) +
    ggplot2::guides(
      color = ggplot2::guide_legend(nrow = 2, byrow = TRUE)
    ) +
    ggplot2::scale_x_continuous(
      breaks = scales::breaks_pretty(n = 5),
      expand = ggplot2::expansion(add = 0.5)
    ) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(
      x = x_label,
      y = "Recall accuracy",
      color = NULL
    ) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      axis.text.x = ggplot2::element_text(size = base_font_size * 0.8),
      axis.text.y = ggplot2::element_text(size = base_font_size * 0.8),
      legend.position = "none",
      strip.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      strip.clip = "off",
      panel.spacing = if (!is.null(panel_spacing_mm)) grid::unit(panel_spacing_mm, "mm") else
        if (isTRUE(match_rank_layout)) grid::unit(1.31, "mm") else grid::unit(0.75, "lines"),
      plot.margin = ggplot2::margin(0, 1, 0, 1, unit = "mm")
    )
  # Keep timing labels side by side, with room for two-line text.
  accuracy_plot <- accuracy_plot + ggplot2::guides(color = ggplot2::guide_legend(nrow = 1, byrow = TRUE))
  legend_height <- max(legend_height, 10)
  legend_alignment <- if (identical(output_suffix, "_participants_sequential")) "right" else "center"
  compact_legend <- cowplot::get_legend(
    accuracy_plot +
      ggplot2::theme(
        legend.position = "top",
        legend.justification = legend_alignment,
        legend.box.just = legend_alignment,
        legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
        legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
        legend.box.spacing = grid::unit(0, "mm"),
        legend.spacing = grid::unit(0, "mm"),
        legend.spacing.y = grid::unit(0, "mm"),
        legend.key.height = grid::unit(2.5, "mm"),
        legend.key.width = grid::unit(4, "mm"),
        legend.text = ggplot2::element_text(
          size = base_font_size * 0.8,
          color = "black",
          margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
        ),
        plot.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
      )
  )
  if (is.null(compact_legend)) {
    stop("Could not construct the path-context legend.", call. = FALSE)
  }
  if (!is.null(participant_model_gap_mm)) {
    accuracy_plot <- ggplot2::ggplotGrob(accuracy_plot)
    panels <- accuracy_plot$layout[grepl("^panel", accuracy_plot$layout$name), ]
    columns <- sort(unique(panels$l))
    if (length(columns) > 1L) {
      spacer <- columns[[2]] - 1L
      accuracy_plot$widths[spacer] <- grid::unit(participant_model_gap_mm, "mm")
    }
  }
  participant_header <- cowplot::ggdraw() +
    cowplot::draw_label(
      "participants",
      size = base_font_size * 0.8,
      color = "black",
      fontface = "plain"
    )
  model_header <- cowplot::ggdraw() +
    cowplot::draw_label(
      "models",
      size = base_font_size * 0.8,
      color = "black",
      fontface = "plain"
    )
  group_header <- cowplot::plot_grid(
    participant_header,
    model_header,
    nrow = 1,
    rel_widths = c(1, length(available_model_sources))
  )
  combined_plot <- cowplot::plot_grid(
    compact_legend,
    if (isTRUE(simple_panel_headers) || isTRUE(aligned_panel_headers)) NULL else group_header,
    accuracy_plot,
    ncol = 1,
    rel_heights = c(legend_height, group_header_height, panel_height),
    align = "v",
    axis = "lr"
  )
  if (isTRUE(match_rank_layout)) {
    combined_plot <- cowplot::plot_grid(
      NULL, compact_legend,
      if (isTRUE(simple_panel_headers) || isTRUE(aligned_panel_headers)) NULL else group_header,
      accuracy_plot, ncol = 1,
      rel_heights = c(top_padding_height, legend_height, group_header_height, panel_height),
      align = "v", axis = "lr"
    )
  }
  if (identical(legend_position, "bottom")) {
    combined_plot <- cowplot::plot_grid(
      if (isTRUE(simple_panel_headers) || isTRUE(aligned_panel_headers)) NULL else group_header,
      accuracy_plot, compact_legend, ncol = 1,
      rel_heights = c(group_header_height, panel_height, legend_height),
      align = "v", axis = "lr"
    )
  }
  output_height <- panel_height + legend_height + group_header_height + top_padding_height
  output_width <- x_size * length(source_levels) / 2 -
    if (isTRUE(match_rank_layout)) 2.5 * (length(source_levels) - 1L) else 0
  if (!is.null(reference_plot)) {
    if (!isTRUE(aligned_panel_headers) && !isTRUE(simple_panel_headers)) {
      stop("Matched timing-panel heights require facet-level headers.", call. = FALSE)
    }
    matched <- fitted_timing_matched_panel_layout(
      accuracy_plot, compact_legend, reference_plot, reference_width,
      reference_height, legend_height, legend_position, match_panel_width
    )
    combined_plot <- matched$plot
    output_height <- matched$height
    if (!is.null(matched$width)) output_width <- matched$width
  }
  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0(output_stem, "_accuracy_by_relative_timing", output_suffix, ".pdf")
  )
  summary_path <- analysis_path(
    "stats",
    "tests",
    save_dir,
    paste0(output_stem, "_accuracy_by_relative_timing", output_suffix, ".csv")
  )
  ensure_parent_directory(output_path)
  ensure_parent_directory(summary_path)
  ggsave_pdf(
    output_path,
    plot = combined_plot,
    width = output_width,
    height = output_height
  )
  output_summary <- summary %>%
    dplyr::rename(
      !!reward_variable := reward_value,
      !!timing_variable := relative_timing
    ) %>%
    dplyr::mutate(
      reward_variable = reward_variable,
      timing_variable = timing_variable,
      minimum_observation_count = minimum_observation_count,
      minimum_participant_count = minimum_participant_count
    )
  readr::write_csv(output_summary, summary_path)
  cat(sprintf("Saved %s timing plot to: %s\n", reward_variable, output_path))
  cat(sprintf(
    "Saved %s timing summary to: %s\n",
    reward_variable,
    summary_path
  ))
  invisible(list(plot = combined_plot, summary = output_summary))
}

fitted_competition_rank_matrix <- function(path_rewards) {
  path_rewards <- as.matrix(path_rewards)
  ranks <- matrix(
    1L,
    nrow = nrow(path_rewards),
    ncol = ncol(path_rewards)
  )
  for (path_index in seq_len(ncol(path_rewards))) {
    ranks[, path_index] <- 1L + rowSums(
      path_rewards > path_rewards[, path_index]
    )
  }
  ranks
}

fitted_observation_final_rank_lookup <- function(
    trials,
    reward_columns,
    path_nodes,
    path_reward_summary = c("mean", "max"),
    include_context_rewards = FALSE) {
  path_reward_summary <- match.arg(path_reward_summary)
  require_columns(
    trials,
    c(".rank_trial_id", reward_columns),
    "Observation/final-rank trial table"
  )
  reward_values <- as.matrix(trials[, reward_columns, drop = FALSE])
  storage.mode(reward_values) <- "double"
  if (any(!is.finite(reward_values))) {
    stop("Observation/final-rank rewards must all be finite.", call. = FALSE)
  }

  node_count <- length(reward_columns)
  path_count <- length(path_nodes)
  path_incidence <- matrix(0, nrow = path_count, ncol = node_count)
  for (path_index in seq_along(path_nodes)) {
    nodes <- as.integer(path_nodes[[path_index]])
    if (
      length(nodes) == 0L ||
        any(!nodes %in% seq_len(node_count))
    ) {
      stop("A fitted rank path contains an invalid node.", call. = FALSE)
    }
    path_incidence[path_index, nodes] <- 1
  }
  if (any(colSums(path_incidence) == 0)) {
    stop("Every reward node must belong to at least one path.", call. = FALSE)
  }

  minimum_eligible_rank <- function(path_ranks, eligible_paths) {
    result <- path_ranks[, eligible_paths[[1L]]]
    if (length(eligible_paths) > 1L) {
      for (path_index in eligible_paths[-1L]) {
        result <- pmin(result, path_ranks[, path_index])
      }
    }
    as.integer(result)
  }

  final_path_rewards <- reward_values %*% t(path_incidence)
  final_path_ranks <- fitted_competition_rank_matrix(final_path_rewards)
  final_node_ranks <- matrix(
    NA_integer_,
    nrow = nrow(reward_values),
    ncol = node_count
  )
  final_node_path_rewards <- matrix(
    NA_real_,
    nrow = nrow(reward_values),
    ncol = node_count
  )
  observation_node_ranks <- final_node_ranks
  partial_path_rewards <- matrix(
    0,
    nrow = nrow(reward_values),
    ncol = path_count
  )

  for (node_index in seq_len(node_count)) {
    eligible_paths <- which(path_incidence[, node_index] == 1)
    partial_path_rewards[, eligible_paths] <- sweep(
      partial_path_rewards[, eligible_paths, drop = FALSE],
      1L,
      reward_values[, node_index],
      `+`
    )
    observation_path_ranks <- fitted_competition_rank_matrix(
      partial_path_rewards
    )
    observation_node_ranks[, node_index] <- minimum_eligible_rank(
      observation_path_ranks,
      eligible_paths
    )
    final_node_ranks[, node_index] <- minimum_eligible_rank(
      final_path_ranks,
      eligible_paths
    )
    # Reuse path totals across all trials; no per-trial path search is needed.
    if (path_reward_summary == "max") {
      best_reward <- final_path_rewards[, eligible_paths[[1L]]]
      for (path_index in eligible_paths[-1L]) {
        best_reward <- pmax(best_reward, final_path_rewards[, path_index])
      }
      final_node_path_rewards[, node_index] <- best_reward
    } else {
      final_node_path_rewards[, node_index] <- rowMeans(
        final_path_rewards[, eligible_paths, drop = FALSE]
      )
    }
  }

  lookup <- tibble::tibble(
    .rank_trial_id = rep(trials$.rank_trial_id, each = node_count),
    node = rep(seq_len(node_count), times = nrow(trials)),
    observation_rank = as.vector(t(observation_node_ranks)),
    final_rank = as.vector(t(final_node_ranks)),
    .rank_path_reward = as.vector(t(final_node_path_rewards)),
    rank_change = dplyr::if_else(
      observation_rank >= final_rank,
      "observation_greater_equal",
      "observation_lower"
    )
  )
  if (isTRUE(include_context_rewards)) {
    max_current <- min_current <- max_other <- matrix(NA_real_, nrow(reward_values), node_count)
    for (node in seq_len(node_count)) {
      current <- which(path_incidence[, node] == 1)
      other <- which(path_incidence[, node] == 0)
      for (p in current) {
        max_current[, node] <- if (p == current[[1L]]) final_path_rewards[, p] else
          pmax(max_current[, node], final_path_rewards[, p])
        min_current[, node] <- if (p == current[[1L]]) final_path_rewards[, p] else
          pmin(min_current[, node], final_path_rewards[, p])
      }
      for (p in other) {
        max_other[, node] <- if (p == other[[1L]]) final_path_rewards[, p] else
          pmax(max_other[, node], final_path_rewards[, p])
      }
    }
    lookup$.rank_max_current_path_reward <- as.vector(t(max_current))
    lookup$.rank_min_current_path_reward <- as.vector(t(min_current))
    lookup$.rank_max_other_path_reward <- as.vector(t(max_other))
    # Derive immediate relatives from the complete tree, not the filtered probes.
    parents <- rep(NA_integer_, node_count)
    for (path in path_nodes) {
      for (i in seq_along(path)) {
        parent <- if (i == 1L) 0L else path[[i - 1L]]
        if (!is.na(parents[path[[i]]]) && parents[path[[i]]] != parent) {
          stop("Inconsistent parent mapping in rank paths.", call. = FALSE)
        }
        parents[path[[i]]] <- parent
      }
    }
    leaves <- !seq_len(node_count) %in% parents
    sibling <- min_sibling <- max_sibling <- parent_reward <- aunt <- max_aunt <- cousin <- max_cousin <- child <- avg_child <- niece <- min_niece <- avg_niece <- matrix(0, nrow(reward_values), node_count)
    branch_reward <- matrix(NA_real_, nrow(reward_values), node_count)
    branch_timing <- matrix(NA_character_, nrow(reward_values), node_count)
    path_branches <- vapply(path_nodes, function(p) as.integer(p[[1L]]), integer(1))
    on_best_branch_path <- matrix(FALSE, nrow(reward_values), node_count)
    current_branch_best_reward <- matrix(NA_real_, nrow(reward_values), node_count)
    current_branch_worst_reward <- matrix(NA_real_, nrow(reward_values), node_count)
    worst_branch_reward <- matrix(NA_real_, nrow(reward_values), node_count)
    worst_branch_timing <- matrix(NA_character_, nrow(reward_values), node_count)
    unselected_branch_leaf_1 <- unselected_branch_leaf_2 <- matrix(NA_real_, nrow(reward_values), node_count)
    for (branch in unique(path_branches)) {
      candidates <- which(path_branches == branch)
      # Stable tie-break: first path in presentation order; never duplicate parents.
      best <- candidates[max.col(final_path_rewards[, candidates, drop = FALSE], ties.method = "first")]
      best_totals <- final_path_rewards[cbind(seq_len(nrow(reward_values)), best)]
      for (node in unique(unlist(path_nodes[candidates]))) current_branch_best_reward[, node] <- best_totals
      worst_totals <- apply(final_path_rewards[, candidates, drop = FALSE], 1L, min)
      for (node in unique(unlist(path_nodes[candidates]))) current_branch_worst_reward[, node] <- worst_totals
      for (p in candidates) {
        on_best_branch_path[best == p, path_nodes[[p]]] <- TRUE
        # Rewards of the unselected terminal leaves, ordered by presentation.
        remaining_leaves <- sort(setdiff(
          vapply(path_nodes[candidates], function(x) as.integer(tail(x, 1L)), integer(1)),
          tail(path_nodes[[p]], 1L)))
        selected_rows <- which(best == p)
        for (node in path_nodes[[p]]) {
          if (length(remaining_leaves) >= 1L) unselected_branch_leaf_1[selected_rows, node] <- reward_values[selected_rows, remaining_leaves[[1L]]]
          if (length(remaining_leaves) >= 2L) unselected_branch_leaf_2[selected_rows, node] <- reward_values[selected_rows, remaining_leaves[[2L]]]
        }
      }
    }
    lookup$.rank_on_best_branch_path <- as.vector(t(on_best_branch_path))
    lookup$.rank_current_branch_best_reward <- as.vector(t(current_branch_best_reward))
    lookup$.rank_current_branch_worst_reward <- as.vector(t(current_branch_worst_reward))
    lookup$.rank_unselected_branch_leaf_1 <- as.vector(t(unselected_branch_leaf_1))
    lookup$.rank_unselected_branch_leaf_2 <- as.vector(t(unselected_branch_leaf_2))
    for (node in seq_len(node_count)) {
      current_branch <- unique(path_branches[path_incidence[, node] == 1])
      if (length(current_branch) != 1L) stop("Probe belongs to multiple root branches.")
      other_paths <- which(path_branches != current_branch)
      if (!length(other_paths)) next
      other_rewards <- final_path_rewards[, other_paths, drop = FALSE]
      best <- apply(other_rewards, 1L, max)
      winners <- sweep(other_rewards, 1L, best, `==`)
      before <- path_branches[other_paths] < current_branch
      has_before <- rowSums(winners[, before, drop = FALSE]) > 0
      has_after <- rowSums(winners[, !before, drop = FALSE]) > 0
      branch_reward[, node] <- best
      branch_timing[, node] <- ifelse(has_before & !has_after, "before", ifelse(has_after & !has_before, "after", NA_character_))
      worst <- apply(other_rewards, 1L, min)
      losers <- sweep(other_rewards, 1L, worst, `==`)
      worst_before <- rowSums(losers[, before, drop = FALSE]) > 0
      worst_after <- rowSums(losers[, !before, drop = FALSE]) > 0
      worst_branch_reward[, node] <- worst
      worst_branch_timing[, node] <- ifelse(worst_before & !worst_after, "before",
        ifelse(worst_after & !worst_before, "after", NA_character_))
    }
    lookup$.rank_max_other_branch_reward <- as.vector(t(branch_reward))
    lookup$.rank_other_branch_timing <- as.vector(t(branch_timing))
    lookup$.rank_min_other_branch_reward <- as.vector(t(worst_branch_reward))
    lookup$.rank_worst_other_branch_timing <- as.vector(t(worst_branch_timing))
    for (node in seq_len(node_count)) {
      children <- which(parents == node)
      if (length(children)) {
        avg_child[, node] <- rowMeans(reward_values[, children, drop = FALSE])
        child[, node] <- reward_values[, children[[1L]]]
        for (child_node in children[-1L]) {
          child[, node] <- pmax(child[, node], reward_values[, child_node])
        }
      }
      siblings <- which(parents == parents[[node]] & seq_len(node_count) != node)
      nieces <- which(parents %in% siblings)
      if (!leaves[[node]] && length(nieces)) {
        avg_niece[, node] <- rowMeans(reward_values[, nieces, drop = FALSE])
        niece[, node] <- reward_values[, nieces[[1L]]]
        min_niece[, node] <- reward_values[, nieces[[1L]]]
        for (niece_node in nieces[-1L]) {
          niece[, node] <- pmax(niece[, node], reward_values[, niece_node])
          min_niece[, node] <- pmin(min_niece[, node], reward_values[, niece_node])
        }
      }
      if (length(siblings)) sibling[, node] <- rowMeans(reward_values[, siblings, drop = FALSE])
      if (length(siblings)) {
        max_sibling[, node] <- reward_values[, siblings[[1L]]]
        min_sibling[, node] <- reward_values[, siblings[[1L]]]
        for (sibling_node in siblings[-1L]) {
          max_sibling[, node] <- pmax(max_sibling[, node], reward_values[, sibling_node])
          min_sibling[, node] <- pmin(min_sibling[, node], reward_values[, sibling_node])
        }
      }
      # Parent and aunt terms are defined only for leaves in this analysis.
      if (leaves[[node]] && parents[[node]] > 0L) {
        p <- parents[[node]]
        parent_reward[, node] <- reward_values[, p]
        aunts <- which(parents == parents[[p]] & seq_len(node_count) != p)
        if (length(aunts)) aunt[, node] <- rowMeans(reward_values[, aunts, drop = FALSE])
        if (length(aunts)) {
          max_aunt[, node] <- reward_values[, aunts[[1L]]]
          for (aunt_node in aunts[-1L]) {
            max_aunt[, node] <- pmax(max_aunt[, node], reward_values[, aunt_node])
          }
        }
        cousins <- which(leaves & parents %in% aunts)
        if (length(cousins)) cousin[, node] <- rowMeans(reward_values[, cousins, drop = FALSE])
        if (length(cousins)) {
          max_cousin[, node] <- reward_values[, cousins[[1L]]]
          for (cousin_node in cousins[-1L]) {
            max_cousin[, node] <- pmax(max_cousin[, node], reward_values[, cousin_node])
          }
        }
      }
    }
    lookup$.rank_parent_reward <- as.vector(t(parent_reward))
    lookup$.rank_sibling_reward <- as.vector(t(sibling))
    lookup$.rank_max_sibling_reward <- as.vector(t(max_sibling))
    lookup$.rank_min_sibling_reward <- as.vector(t(min_sibling))
    lookup$.rank_aunt_reward <- as.vector(t(aunt))
    lookup$.rank_max_aunt_reward <- as.vector(t(max_aunt))
    lookup$.rank_cousin_reward <- as.vector(t(cousin))
    lookup$.rank_max_cousin_reward <- as.vector(t(max_cousin))
    lookup$.rank_child_reward <- as.vector(t(child))
    lookup$.rank_avg_child_reward <- as.vector(t(avg_child))
    lookup$.rank_max_niece_reward <- as.vector(t(niece))
    lookup$.rank_min_niece_reward <- as.vector(t(min_niece))
    lookup$.rank_avg_niece_reward <- as.vector(t(avg_niece))
  }
  lookup
}

prepare_fitted_observation_final_rank_human <- function(
    df_beh,
    node_count,
    path_nodes,
    include_context_rewards = FALSE,
    exclude_final_observation = TRUE) {
  reward_columns <- paste0("n", seq_len(node_count), "_reward")
  required_columns <- c(
    "subject",
    "graph",
    "condition",
    "node",
    "actual_reward",
    "estimated_reward",
    "accuracy",
    reward_columns
  )
  require_columns(df_beh, required_columns, "Behavioral rank-comparison data")
  behavior <- df_beh %>%
    dplyr::mutate(
      subject = as.character(subject),
      graph = as.character(graph),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      dplyr::across(
        dplyr::all_of(c(
          "actual_reward",
          "estimated_reward",
          "accuracy",
          reward_columns
        )),
        as.numeric
      )
    ) %>%
    dplyr::filter(
      condition == "planning",
      !is.na(subject),
      !is.na(graph),
      node %in% seq_len(node_count),
      is.finite(actual_reward),
      is.finite(estimated_reward),
      is.finite(accuracy),
      dplyr::if_all(dplyr::all_of(reward_columns), is.finite)
    )
  if (nrow(behavior) == 0L) {
    stop("No valid planning-condition behavioral rows were found.", call. = FALSE)
  }

  trial_key_columns <- c("subject", "condition", "graph", reward_columns)
  if (anyDuplicated(behavior[c(trial_key_columns, "node")])) {
    stop(
      "Behavioral rank-comparison data contain duplicate trial/node rows.",
      call. = FALSE
    )
  }
  trials <- behavior %>%
    dplyr::distinct(dplyr::across(dplyr::all_of(trial_key_columns))) %>%
    dplyr::mutate(.rank_trial_id = dplyr::row_number())
  rank_lookup <- fitted_observation_final_rank_lookup(
    trials,
    reward_columns,
    path_nodes,
    include_context_rewards = include_context_rewards
  )
  behavior %>%
    dplyr::left_join(
      trials,
      by = trial_key_columns,
      relationship = "many-to-one"
    ) %>%
    dplyr::left_join(
      rank_lookup,
      by = c(".rank_trial_id", "node"),
      relationship = "many-to-one"
    ) %>%
    # Compute final rank from the complete trial above, then exclude the last
    # presentation because its encoding-time rank cannot subsequently change.
    dplyr::filter(!exclude_final_observation | node < node_count) %>%
    dplyr::transmute(
      source = "participants",
      participant_id = subject,
      trial_id = paste0("human_", .rank_trial_id),
      observation_step = node,
      node_reward = actual_reward,
      path_reward = .rank_path_reward,
      parent_reward = if (include_context_rewards) .rank_parent_reward else 0,
      sibling_reward = if (include_context_rewards) .rank_sibling_reward else 0,
      max_sibling_reward = if (include_context_rewards) .rank_max_sibling_reward else 0,
      min_sibling_reward = if (include_context_rewards) .rank_min_sibling_reward else 0,
      max_other_branch_reward = if (include_context_rewards) .rank_max_other_branch_reward else NA_real_,
      other_branch_timing = if (include_context_rewards) .rank_other_branch_timing else NA_character_,
      aunt_reward = if (include_context_rewards) .rank_aunt_reward else 0,
      max_aunt_reward = if (include_context_rewards) .rank_max_aunt_reward else 0,
      cousin_reward = if (include_context_rewards) .rank_cousin_reward else 0,
      max_cousin_reward = if (include_context_rewards) .rank_max_cousin_reward else 0,
      child_reward = if (include_context_rewards) .rank_child_reward else 0,
      avg_child_reward = if (include_context_rewards) .rank_avg_child_reward else 0,
      max_current_path_reward = if (include_context_rewards) .rank_max_current_path_reward else NA_real_,
      min_current_path_reward = if (include_context_rewards) .rank_min_current_path_reward else NA_real_,
      on_best_branch_path = if (include_context_rewards) .rank_on_best_branch_path else NA,
      current_branch_best_reward = if (include_context_rewards) .rank_current_branch_best_reward else NA_real_,
      current_branch_worst_reward = if (include_context_rewards) .rank_current_branch_worst_reward else NA_real_,
      min_other_branch_reward = if (include_context_rewards) .rank_min_other_branch_reward else NA_real_,
      worst_other_branch_timing = if (include_context_rewards) .rank_worst_other_branch_timing else NA_character_,
      other_branch_node_1 = if (include_context_rewards) .rank_unselected_branch_leaf_1 else NA_real_,
      other_branch_node_2 = if (include_context_rewards) .rank_unselected_branch_leaf_2 else NA_real_,
      max_other_path_reward = if (include_context_rewards) .rank_max_other_path_reward else NA_real_,
      max_niece_reward = if (include_context_rewards) .rank_max_niece_reward else 0,
      min_niece_reward = if (include_context_rewards) .rank_min_niece_reward else 0,
      avg_niece_reward = if (include_context_rewards) .rank_avg_niece_reward else 0,
      accuracy,
      observation_rank,
      final_rank,
      rank_change
    )
}

prepare_fitted_observation_final_rank_model <- function(
    df_model,
    node_count,
    path_nodes,
    use_model_name = FALSE,
    include_context_rewards = FALSE,
    exclude_final_observation = TRUE) {
  required_columns <- c(
    "subject",
    "graph",
    "condition",
    "node",
    "actual_reward",
    "estimated_reward"
  )
  require_columns(df_model, required_columns, "Fitted rank-comparison data")
  participant_column <- if ("fitted_subject" %in% names(df_model)) {
    "fitted_subject"
  } else {
    "subject"
  }
  if (isTRUE(use_model_name) && !"model_name" %in% names(df_model)) {
    stop(
      "Fitted rank-comparison data require model_name for model facets.",
      call. = FALSE
    )
  }
  model <- df_model %>%
    dplyr::mutate(
      subject = as.character(subject),
      graph = as.character(graph),
      condition = tolower(as.character(condition)),
      node = as.integer(node),
      actual_reward = as.numeric(actual_reward),
      estimated_reward = as.numeric(estimated_reward)
    ) %>%
    dplyr::filter(
      condition == "planning",
      !is.na(subject),
      !is.na(graph),
      node %in% seq_len(node_count),
      is.finite(actual_reward),
      is.finite(estimated_reward)
    )
  trial_key_columns <- c("subject", "condition", "graph")
  if (anyDuplicated(model[c(trial_key_columns, "node")])) {
    stop(
      "Fitted rank-comparison data contain duplicate trial/node rows.",
      call. = FALSE
    )
  }
  graph_sizes <- model %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(trial_key_columns))) %>%
    dplyr::summarise(n = dplyr::n(), .groups = "drop")
  if (any(graph_sizes$n != node_count)) {
    stop(
      "Every fitted rank-comparison trial must contain every reward node.",
      call. = FALSE
    )
  }
  trials <- model %>%
    dplyr::select(dplyr::all_of(trial_key_columns), node, actual_reward) %>%
    tidyr::pivot_wider(
      names_from = node,
      values_from = actual_reward,
      names_prefix = "reward_"
    ) %>%
    dplyr::mutate(.rank_trial_id = dplyr::row_number())
  reward_columns <- paste0("reward_", seq_len(node_count))
  rank_lookup <- fitted_observation_final_rank_lookup(
    trials,
    reward_columns,
    path_nodes,
    include_context_rewards = include_context_rewards
  )
  model %>%
    dplyr::left_join(
      trials %>% dplyr::select(
        dplyr::all_of(trial_key_columns),
        .rank_trial_id
      ),
      by = trial_key_columns,
      relationship = "many-to-one"
    ) %>%
    dplyr::left_join(
      rank_lookup,
      by = c(".rank_trial_id", "node"),
      relationship = "many-to-one"
    ) %>%
    # Compute final rank from the complete trial above, then exclude the last
    # presentation because its encoding-time rank cannot subsequently change.
    dplyr::filter(!exclude_final_observation | node < node_count) %>%
    dplyr::transmute(
      source = if (isTRUE(use_model_name)) {
        as.character(model_name)
      } else {
        "model"
      },
      participant_id = as.character(.data[[participant_column]]),
      trial_id = paste0("model_", .rank_trial_id),
      observation_step = node,
      node_reward = actual_reward,
      path_reward = .rank_path_reward,
      parent_reward = if (include_context_rewards) .rank_parent_reward else 0,
      sibling_reward = if (include_context_rewards) .rank_sibling_reward else 0,
      max_sibling_reward = if (include_context_rewards) .rank_max_sibling_reward else 0,
      min_sibling_reward = if (include_context_rewards) .rank_min_sibling_reward else 0,
      max_other_branch_reward = if (include_context_rewards) .rank_max_other_branch_reward else NA_real_,
      other_branch_timing = if (include_context_rewards) .rank_other_branch_timing else NA_character_,
      aunt_reward = if (include_context_rewards) .rank_aunt_reward else 0,
      max_aunt_reward = if (include_context_rewards) .rank_max_aunt_reward else 0,
      cousin_reward = if (include_context_rewards) .rank_cousin_reward else 0,
      max_cousin_reward = if (include_context_rewards) .rank_max_cousin_reward else 0,
      child_reward = if (include_context_rewards) .rank_child_reward else 0,
      avg_child_reward = if (include_context_rewards) .rank_avg_child_reward else 0,
      max_current_path_reward = if (include_context_rewards) .rank_max_current_path_reward else NA_real_,
      min_current_path_reward = if (include_context_rewards) .rank_min_current_path_reward else NA_real_,
      on_best_branch_path = if (include_context_rewards) .rank_on_best_branch_path else NA,
      current_branch_best_reward = if (include_context_rewards) .rank_current_branch_best_reward else NA_real_,
      current_branch_worst_reward = if (include_context_rewards) .rank_current_branch_worst_reward else NA_real_,
      min_other_branch_reward = if (include_context_rewards) .rank_min_other_branch_reward else NA_real_,
      worst_other_branch_timing = if (include_context_rewards) .rank_worst_other_branch_timing else NA_character_,
      other_branch_node_1 = if (include_context_rewards) .rank_unselected_branch_leaf_1 else NA_real_,
      other_branch_node_2 = if (include_context_rewards) .rank_unselected_branch_leaf_2 else NA_real_,
      max_other_path_reward = if (include_context_rewards) .rank_max_other_path_reward else NA_real_,
      max_niece_reward = if (include_context_rewards) .rank_max_niece_reward else 0,
      min_niece_reward = if (include_context_rewards) .rank_min_niece_reward else 0,
      avg_niece_reward = if (include_context_rewards) .rank_avg_niece_reward else 0,
      accuracy = as.numeric(estimated_reward == actual_reward),
      observation_rank,
      final_rank,
      rank_change
    )
}

plot_fitted_best_worst_timing_collapsed <- function(classified, save_dir,
    y_min=0.1, y_max=0.4, base_font_size=7, reference_plot=NULL) {
  # Pool eligible probes within participant/timing BEFORE averaging participants.
  # Do not average the existing reward-bin means, which would reweight rewards.
  individual <- classified %>% dplyr::filter(is.finite(accuracy)) %>%
    dplyr::group_by(source,participant_id,rank_change) %>%
    dplyr::summarise(participant_accuracy=mean(accuracy),observations=dplyr::n(),.groups="drop")
  summary <- individual %>% dplyr::group_by(source,rank_change) %>%
    dplyr::summarise(mean_accuracy=mean(participant_accuracy),
      se=if (dplyr::n()>1) stats::sd(participant_accuracy)/sqrt(dplyr::n()) else NA_real_,
      participants=dplyr::n(),observations=sum(observations),.groups="drop")
  labels <- fitted_timing_facet_labels()
  labels["participants"] <- "participants"
  source_levels <- levels(classified$source)
  summary$facet_label <- factor(unname(labels[as.character(summary$source)]),
    levels=unname(labels[source_levels]))
  summary$rank_change <- factor(summary$rank_change,levels=c("best_first","worst_first"))
  p <- ggplot2::ggplot(summary,ggplot2::aes(rank_change,mean_accuracy,color=rank_change)) +
    ggplot2::geom_point(size=1) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin=mean_accuracy-se,ymax=mean_accuracy+se),
      width=.12,linewidth=.3,na.rm=TRUE) +
    ggplot2::facet_wrap(~facet_label,nrow=1,drop=FALSE) +
    ggplot2::scale_x_discrete(labels=c(best_first="Best\nbranch\nfirst",worst_first="Worst\nbranch\nfirst"),drop=FALSE) +
    ggplot2::scale_color_manual(values=c(best_first="#46585D",worst_first="#6F3934"),
      breaks=c("best_first","worst_first"),labels=c("Best branch first","Worst branch first"),drop=FALSE) +
    ggplot2::guides(color=ggplot2::guide_legend(nrow=1)) +
    ggplot2::coord_cartesian(ylim=c(y_min,y_max)) +
    ggplot2::labs(x=NULL,y="Recall accuracy",color=NULL) +
    plot_theme_axes(base_font_size,aspect_ratio=1.2) +
    ggplot2::theme(strip.clip="off",strip.text=ggplot2::element_text(size=base_font_size*.8),
      axis.text=ggplot2::element_text(size=base_font_size*.8,color="black"),
      panel.spacing=grid::unit(5,"mm"),plot.margin=ggplot2::margin(0,1,0,1,"mm"))
  legend <- cowplot::get_legend(p + ggplot2::theme(legend.position="bottom",
    legend.text=ggplot2::element_text(size=base_font_size*.8),
    legend.key.width=grid::unit(4,"mm"),legend.key.height=grid::unit(2.5,"mm")))
  if (is.null(reference_plot)) {
    # Same axis/strip layout as the Exp2 aunt-reward reference, for plot-only calls.
    ref <- expand.grid(x=c(-4,4),y=c(0,.6),source=c("participants","model"))
    ref$source <- factor(ref$source,levels=c("participants","model"))
    reference_plot <- ggplot2::ggplot(ref,ggplot2::aes(x,y)) + ggplot2::geom_blank() +
      ggplot2::facet_wrap(~source,nrow=1) +
      ggplot2::scale_x_continuous(breaks=seq(-4,4,2),expand=ggplot2::expansion(add=.5)) +
      ggplot2::coord_cartesian(ylim=c(0,.6)) +
      ggplot2::labs(x="Average Aunt Reward",y="Accuracy") +
      plot_theme_axes(base_font_size,aspect_ratio=1.2) +
      ggplot2::theme(strip.clip="off",panel.spacing=grid::unit(1.31,"mm"),
        plot.margin=ggplot2::margin(0,1,0,1,"mm"))
  }
  grob <- ggplot2::ggplotGrob(p)
  columns <- sort(unique(grob$layout$l[grepl("^panel",grob$layout$name)]))
  if (length(columns)>1) grob$widths[columns[2]-1] <- grid::unit(6.5*2/3,"mm")
  stem <- "accuracy_by_best_worst_branch_timing_middle_branch"
  path <- analysis_path("figures",save_dir,paste0(stem,".pdf"))
  ensure_parent_directory(path)
  matched <- fitted_timing_matched_panel_layout(grob,legend,reference_plot,40,36,
    legend_height=10,legend_position="bottom",match_panel_width=TRUE)
  ggsave_pdf(path,plot=matched$plot,width=matched$width,height=matched$height)
  summary_path <- analysis_path("stats","tests",save_dir,paste0(stem,".csv"))
  ensure_parent_directory(summary_path)
  readr::write_csv(summary,summary_path)
  readr::write_csv(individual,sub("[.]csv$","_participant_means.csv",summary_path))
  message("Saved collapsed best/worst timing plot: ",path)
  invisible(summary)
}

plot_fitted_observation_final_rank_node_reward <- function(
    df_beh,
    df_model,
    node_count,
    path_nodes,
    save_dir,
    y_min = 0,
    y_max = 0.75,
    x_size = 42.5,
    panel_height = 36,
    legend_height = 7,
    group_header_height = 4,
    top_padding_height = 3,
    base_font_size = 7,
    final_better_color = "#365959",
    output_suffix = "",
    simple_panel_headers = FALSE,
    probe_nodes = NULL,
    compare_best_worst_branch_timing = FALSE, collapsed_reference_plot = NULL,
    collapsed_only = FALSE) {
  if (collapsed_only && !compare_best_worst_branch_timing) stop("collapsed_only requires best/worst timing")
  if (isTRUE(compare_best_worst_branch_timing)) {
    stopifnot(node_count == 12L, identical(probe_nodes, 6:8))
  }
  # Match the compact legend-to-facet gap in the paired path-context plot.
  if (isTRUE(simple_panel_headers) && identical(output_suffix, "_participants_sequential")) {
    group_header_height <- max(0, group_header_height - 2)
  }
  human_data <- prepare_fitted_observation_final_rank_human(
    df_beh,
    node_count,
    path_nodes,
    include_context_rewards = compare_best_worst_branch_timing,
    exclude_final_observation = !compare_best_worst_branch_timing
  )
  model_data <- prepare_fitted_observation_final_rank_model(
    df_model,
    node_count,
    path_nodes,
    use_model_name = TRUE,
    include_context_rewards = TRUE,
    exclude_final_observation = !compare_best_worst_branch_timing
  )
  # Restrict probes only after full-tree rewards/ranks have been prepared.
  # Filtering raw input would remove paths needed for the rank calculation.
  if (!is.null(probe_nodes)) {
    if (!length(probe_nodes) || anyNA(probe_nodes) ||
        any(!probe_nodes %in% seq_len(node_count))) {
      stop("probe_nodes must identify non-root nodes in this tree.", call. = FALSE)
    }
    if (!nzchar(output_suffix)) stop("A restricted-node plot requires a separate output_suffix.", call. = FALSE)
    human_data <- dplyr::filter(human_data, observation_step %in% probe_nodes)
    model_data <- dplyr::filter(model_data, observation_step %in% probe_nodes)
  }
  available_model_sources <- fitted_timing_model_levels(model_data$source)
  source_levels <- c("participants", available_model_sources)
  facet_labels <- fitted_timing_facet_labels()
  if (isTRUE(simple_panel_headers)) {
    stopifnot(length(available_model_sources) == 1L)
    facet_labels[source_levels] <- c("participants", "model")
  }
  rank_levels <- c("final_better", "final_worse")
  rank_labels <- c(
    "Final rank better than\nencoding rank",
    "Final rank worse than\nencoding rank"
  )
  if (isTRUE(compare_best_worst_branch_timing)) {
    rank_levels <- c("best_first", "worst_first")
    rank_labels <- c("Best branch first", "Worst branch first")
  }
  classified <- dplyr::bind_rows(human_data, model_data) %>%
    dplyr::mutate(
      source = factor(source, levels = source_levels),
      facet_label = factor(
        unname(facet_labels[as.character(source)]),
        levels = unname(facet_labels[source_levels])
      ),
      rank_change = if (isTRUE(compare_best_worst_branch_timing)) {
        eligible <- is.finite(current_branch_best_reward) & is.finite(max_other_branch_reward) &
          max_other_branch_reward > current_branch_best_reward &
          is.finite(current_branch_worst_reward) & is.finite(min_other_branch_reward) &
          min_other_branch_reward < current_branch_worst_reward &
          other_branch_timing %in% c("before", "after") &
          worst_other_branch_timing %in% c("before", "after") &
          other_branch_timing != worst_other_branch_timing
        dplyr::case_when(
          eligible & other_branch_timing == "before" ~ "best_first",
          eligible & other_branch_timing == "after" ~ "worst_first",
          TRUE ~ NA_character_)
      } else dplyr::case_when(
        final_rank < observation_rank ~ "final_better",
        final_rank > observation_rank ~ "final_worse",
        TRUE ~ NA_character_
      ),
      rank_change = factor(rank_change, levels = rank_levels)
    ) %>%
    dplyr::filter(!is.na(rank_change))
  if (!setequal(as.character(unique(classified$rank_change)), rank_levels)) {
    stop("The combined data do not contain both rank conditions.", call. = FALSE)
  }

  if (isTRUE(compare_best_worst_branch_timing)) {
    plot_fitted_best_worst_timing_collapsed(classified,save_dir,
      y_min=0.1,y_max=0.4,base_font_size=base_font_size,reference_plot=collapsed_reference_plot)
    if (collapsed_only) return(invisible(NULL))
  }

  plot_summary <- classified %>%
    dplyr::filter(is.finite(node_reward), is.finite(accuracy)) %>%
    dplyr::group_by(
      source,
      facet_label,
      participant_id,
      rank_change,
      node_reward
    ) %>%
    dplyr::summarise(
      participant_accuracy = mean(accuracy),
      observation_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::group_by(source, facet_label, rank_change, node_reward) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_accuracy) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      lower = pmax(0, mean_accuracy - se),
      upper = pmin(1, mean_accuracy + se),
      rank_change_condition = factor(
        as.character(rank_change),
        levels = rank_levels,
        labels = rank_labels
      )
    ) %>%
    dplyr::filter(participant_count >= 5L)
  if (nrow(plot_summary) == 0L) {
    stop("No rank-comparison cells contained five participants.", call. = FALSE)
  }

  rank_colors <- stats::setNames(c(final_better_color, "#6F3934"), rank_labels)
  accuracy_plot <- ggplot2::ggplot(
    plot_summary,
    ggplot2::aes(
      x = node_reward,
      y = mean_accuracy,
      color = rank_change_condition,
      group = rank_change_condition
    )
  ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower, ymax = upper),
      width = 0.5,
      linewidth = 0.3,
      alpha = 0.95,
      na.rm = TRUE
    ) +
    ggplot2::geom_line(linewidth = 0.8, alpha = 0.8) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::facet_wrap(~facet_label, nrow = 1, drop = FALSE) +
    ggplot2::scale_color_manual(values = rank_colors, drop = FALSE) +
    ggplot2::guides(
      color = ggplot2::guide_legend(
        nrow = if (isTRUE(compare_best_worst_branch_timing)) 1 else 2,
        byrow = TRUE)
    ) +
    ggplot2::scale_x_continuous(
      breaks = seq(-4, 4, by = 2),
      expand = ggplot2::expansion(add = 0.5)
    ) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(
      x = "Probed reward",
      y = "Recall accuracy",
      color = NULL
    ) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.position = "none",
      strip.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      strip.clip = "off",
      panel.spacing = grid::unit(1.31, "mm"),
      plot.margin = ggplot2::margin(0, 1, 0, 1, unit = "mm")
    )
  legend_alignment <- if (identical(output_suffix, "_participants_sequential")) "right" else "left"
  if (isTRUE(compare_best_worst_branch_timing)) legend_alignment <- "center"
  compact_legend <- cowplot::get_legend(
    accuracy_plot +
      ggplot2::theme(
        legend.position = "top",
        legend.justification = legend_alignment,
        legend.box.just = legend_alignment,
        legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
        legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
        legend.box.spacing = grid::unit(0, "mm"),
        legend.spacing = grid::unit(0, "mm"),
        legend.spacing.y = grid::unit(0, "mm"),
        legend.key.height = grid::unit(2.5, "mm"),
        legend.key.width = grid::unit(4, "mm"),
        legend.text = ggplot2::element_text(
          size = base_font_size * 0.8,
          color = "black",
          margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
        ),
        plot.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm")
      )
  )
  if (is.null(compact_legend)) {
    stop("Could not construct the rank-condition legend.", call. = FALSE)
  }
  participant_header <- cowplot::ggdraw() +
    cowplot::draw_label(
      "participants",
      size = base_font_size * 0.8,
      color = "black",
      fontface = "plain"
    )
  model_header <- cowplot::ggdraw() +
    cowplot::draw_label(
      "models",
      size = base_font_size * 0.8,
      color = "black",
      fontface = "plain"
    )
  group_header <- cowplot::plot_grid(
    participant_header,
    model_header,
    nrow = 1,
    rel_widths = c(1, length(available_model_sources))
  )
  combined_plot <- cowplot::plot_grid(
    NULL,
    compact_legend,
    if (isTRUE(simple_panel_headers)) NULL else group_header,
    accuracy_plot,
    ncol = 1,
    rel_heights = c(
      top_padding_height,
      legend_height,
      group_header_height,
      panel_height
    ),
    align = "v",
    axis = "lr"
  )
  if (isTRUE(compare_best_worst_branch_timing)) {
    combined_plot <- cowplot::plot_grid(
      NULL,
      if (isTRUE(simple_panel_headers)) NULL else group_header,
      accuracy_plot,
      compact_legend,
      ncol = 1,
      rel_heights = c(top_padding_height, group_header_height, panel_height, legend_height),
      align = "v", axis = "lr"
    )
  }
  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0(if (isTRUE(compare_best_worst_branch_timing))
      "node_reward_accuracy_by_best_worst_branch_timing" else
      "node_reward_accuracy_by_observation_final_rank", output_suffix, ".pdf")
  )
  ensure_parent_directory(output_path)
  if (isTRUE(compare_best_worst_branch_timing)) {
    summary_path <- analysis_path("stats", "tests", save_dir,
      paste0("node_reward_accuracy_by_best_worst_branch_timing", output_suffix, ".csv"))
    ensure_parent_directory(summary_path)
    readr::write_csv(plot_summary, summary_path)
  }
  # Remove the same 2.5 mm per gap as in the paired contextual plots,
  # preserving panel space when alternative models add extra facets.
  output_width <- x_size * length(source_levels) / 2 -
    2.5 * (length(source_levels) - 1L)
  ggsave_pdf(
    output_path,
    plot = combined_plot,
    width = output_width,
    height = panel_height + legend_height + group_header_height +
      top_padding_height
  )
  cat(sprintf("Saved observation/final-rank plot to: %s\n", output_path))
  output_path
}

plot_fitted_encoding_final_rank_accuracy_heatmaps <- function(
    df_beh,
    df_model,
    node_count,
    path_nodes,
    save_dir,
    panel_size_mm = 21,
    base_font_size = 7,
    output_suffix = "") {
  human_data <- prepare_fitted_observation_final_rank_human(
    df_beh,
    node_count,
    path_nodes
  )
  model_data <- prepare_fitted_observation_final_rank_model(
    df_model,
    node_count,
    path_nodes
  )
  source_levels <- c("participants", "model")
  source_labels <- c("participants", "model")
  path_count <- length(path_nodes)

  participant_cells <- dplyr::bind_rows(human_data, model_data) %>%
    dplyr::filter(
      source %in% source_levels,
      is.finite(accuracy),
      observation_rank %in% seq_len(path_count),
      final_rank %in% seq_len(path_count)
    ) %>%
    dplyr::group_by(
      source,
      participant_id,
      observation_rank,
      final_rank
    ) %>%
    dplyr::summarise(
      participant_accuracy = mean(accuracy),
      observation_count = dplyr::n(),
      .groups = "drop"
    )
  if (nrow(participant_cells) == 0L) {
    stop("No valid encoding/final-rank observations were found.", call. = FALSE)
  }

  heatmap_summary <- participant_cells %>%
    dplyr::group_by(source, observation_rank, final_rank) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_accuracy) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      source = factor(source, levels = source_levels, labels = source_labels)
    ) %>%
    tidyr::complete(
      source,
      observation_rank = seq_len(path_count),
      final_rank = seq_len(path_count),
      fill = list(participant_count = 0L, observation_count = 0L)
    ) %>%
    dplyr::mutate(
      mean_accuracy = dplyr::if_else(
        participant_count >= 5L,
        mean_accuracy,
        NA_real_
      ),
      se = dplyr::if_else(participant_count >= 5L, se, NA_real_),
      encoding_path_rank = factor(
        observation_rank,
        levels = seq_len(path_count)
      ),
      final_path_rank = factor(final_rank, levels = seq_len(path_count))
    )
  if (all(is.na(heatmap_summary$mean_accuracy))) {
    stop(
      "No encoding/final-rank cells contained five participants.",
      call. = FALSE
    )
  }

  heatmap_plot <- ggplot2::ggplot(
    heatmap_summary,
    ggplot2::aes(
      x = encoding_path_rank,
      y = final_path_rank,
      fill = mean_accuracy
    )
  ) +
    ggplot2::geom_tile(
      color = "white",
      linewidth = 0.2,
      na.rm = FALSE
    ) +
    ggplot2::facet_grid(. ~ source, drop = FALSE) +
    ggplot2::scale_fill_gradientn(
      colors = c("#F7F3F2", "#B77A73", "#532B27"),
      limits = c(0, 1),
      breaks = c(0, 0.5, 1),
      na.value = "#000000"
    ) +
    ggplot2::guides(
      fill = ggplot2::guide_colorbar(
        direction = "horizontal",
        title.position = "top",
        title.hjust = 0.5,
        label.position = "bottom",
        barwidth = grid::unit(12, "mm"),
        barheight = grid::unit(1.5, "mm")
      )
    ) +
    ggplot2::coord_fixed() +
    ggplot2::labs(
      x = "Encoding path rank",
      y = "Final path rank",
      fill = "Recall accuracy"
    ) +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.8, "mm"),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.4
      ),
      panel.spacing.x = grid::unit(3, "mm"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.title = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.spacing = grid::unit(0, "mm"),
      legend.spacing.y = grid::unit(0, "mm"),
      plot.margin = ggplot2::margin(1, 1, 1, 1, unit = "mm")
    )

  # Fix the data region of every facet at exactly 21 x 21 mm. Axis text,
  # facet strips, spacing, and the shared legend remain outside those panels.
  heatmap_grob <- ggplot2::ggplotGrob(heatmap_plot)
  panel_indices <- grep("^panel", heatmap_grob$layout$name)
  if (length(panel_indices) != length(source_levels)) {
    stop(
      "Could not identify both heatmap panels for fixed-size export.",
      call. = FALSE
    )
  }
  panel_columns <- unique(heatmap_grob$layout$l[panel_indices])
  panel_rows <- unique(heatmap_grob$layout$t[panel_indices])
  heatmap_grob$widths[panel_columns] <- grid::unit(panel_size_mm, "mm")
  heatmap_grob$heights[panel_rows] <- grid::unit(panel_size_mm, "mm")
  output_width_mm <- grid::convertWidth(
    sum(heatmap_grob$widths),
    unitTo = "mm",
    valueOnly = TRUE
  )
  output_height_mm <- grid::convertHeight(
    sum(heatmap_grob$heights),
    unitTo = "mm",
    valueOnly = TRUE
  )

  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0("encoding_final_path_rank_accuracy_heatmaps", output_suffix, ".pdf")
  )
  summary_path <- analysis_path(
    "stats",
    "tests",
    save_dir,
    paste0("encoding_final_path_rank_accuracy_heatmaps", output_suffix, ".csv")
  )
  ensure_parent_directory(output_path)
  ensure_parent_directory(summary_path)
  ggsave_pdf(
    output_path,
    plot = heatmap_grob,
    width = output_width_mm,
    height = output_height_mm
  )
  readr::write_csv(
    heatmap_summary %>%
      dplyr::transmute(
        source = as.character(source),
        encoding_path_rank = observation_rank,
        final_path_rank = final_rank,
        mean_accuracy,
        se,
        participant_count,
        observation_count
      ),
    summary_path
  )
  cat(sprintf(
    "Saved encoding/final path-rank accuracy heatmaps to: %s\n",
    output_path
  ))
  cat(sprintf(
    "Saved encoding/final path-rank accuracy summary to: %s\n",
    summary_path
  ))
  invisible(list(
    plot = heatmap_grob,
    summary = heatmap_summary,
    output_path = output_path,
    summary_path = summary_path
  ))
}

plot_fitted_reward_timestep_rank_direction_heatmaps <- function(
    df_beh,
    df_model,
    node_count,
    path_nodes,
    save_dir,
    panel_size_mm = 21,
    base_font_size = 7,
    output_suffix = "") {
  human_data <- prepare_fitted_observation_final_rank_human(
    df_beh,
    node_count,
    path_nodes
  )
  model_data <- prepare_fitted_observation_final_rank_model(
    df_model,
    node_count,
    path_nodes
  )
  source_levels <- c("participants", "model")
  rank_direction_levels <- c("encoding_better", "encoding_worse")
  rank_direction_labels <- c(
    "Final rank worse than\nencoding rank",
    "Final rank better than\nencoding rank"
  )

  classified <- dplyr::bind_rows(human_data, model_data) %>%
    dplyr::filter(
      source %in% source_levels,
      is.finite(final_rank),
      is.finite(observation_step),
      is.finite(accuracy),
      is.finite(observation_rank),
      is.finite(final_rank)
    ) %>%
    dplyr::mutate(
      rank_direction = dplyr::case_when(
        observation_rank < final_rank ~ "encoding_better",
        observation_rank > final_rank ~ "encoding_worse",
        TRUE ~ NA_character_
      )
    ) %>%
    dplyr::filter(!is.na(rank_direction))
  if (nrow(classified) == 0L) {
    stop(
      "No strict encoding-versus-final rank changes were available.",
      call. = FALSE
    )
  }

  participant_cells <- classified %>%
    dplyr::group_by(
      source,
      participant_id,
      rank_direction,
      final_rank,
      observation_step
    ) %>%
    dplyr::summarise(
      participant_accuracy = mean(accuracy),
      observation_count = dplyr::n(),
      .groups = "drop"
    )
  summarized_cells <- participant_cells %>%
    dplyr::group_by(
      source,
      rank_direction,
      final_rank,
      observation_step
    ) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_accuracy) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
      observation_count = sum(observation_count),
      .groups = "drop"
    )
  final_rank_levels <- seq_along(path_nodes)
  observation_step_levels <- seq_len(node_count - 1L)
  heatmap_summary <- tidyr::expand_grid(
    source = source_levels,
    rank_direction = rank_direction_levels,
    final_rank = final_rank_levels,
    observation_step = observation_step_levels
  ) %>%
    dplyr::left_join(
      summarized_cells,
      by = c(
        "source",
        "rank_direction",
        "final_rank",
        "observation_step"
      )
    ) %>%
    dplyr::mutate(
      participant_count = tidyr::replace_na(participant_count, 0L),
      observation_count = tidyr::replace_na(observation_count, 0L),
      included_in_plot = is.finite(mean_accuracy),
      mean_accuracy = dplyr::if_else(
        included_in_plot,
        mean_accuracy,
        NA_real_
      ),
      se = dplyr::if_else(included_in_plot, se, NA_real_),
      source = factor(source, levels = source_levels),
      rank_direction = factor(
        rank_direction,
        levels = rank_direction_levels,
        labels = rank_direction_labels
      ),
      final_rank_factor = factor(final_rank, levels = final_rank_levels),
      observation_step_factor = factor(
        observation_step,
        levels = observation_step_levels
      )
    )
  if (all(is.na(heatmap_summary$mean_accuracy))) {
    stop(
      paste0(
        "No final-rank-by-observation-step rank-condition cell contained ",
        "an observed accuracy."
      ),
      call. = FALSE
    )
  }

  heatmap_plot <- ggplot2::ggplot(
    heatmap_summary,
    ggplot2::aes(
      x = final_rank_factor,
      y = observation_step_factor,
      fill = mean_accuracy
    )
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 0.2) +
    ggplot2::facet_grid(source ~ rank_direction, drop = FALSE) +
    ggplot2::scale_fill_gradient(
      low = "#FFFFFF",
      high = "#6F3934",
      limits = c(0, 1),
      breaks = c(0, 0.5, 1),
      na.value = "#000000"
    ) +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::guides(
      fill = ggplot2::guide_colorbar(
        direction = "horizontal",
        title.position = "top",
        title.hjust = 0.5,
        label.position = "bottom",
        barwidth = grid::unit(12, "mm"),
        barheight = grid::unit(1.5, "mm")
      )
    ) +
    ggplot2::labs(
      x = "Final path rank",
      y = "Observation step",
      fill = "Recall accuracy"
    ) +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.8, "mm"),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.4
      ),
      panel.spacing = grid::unit(3, "mm"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.title = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.spacing = grid::unit(0, "mm"),
      legend.spacing.y = grid::unit(0, "mm"),
      plot.margin = ggplot2::margin(1, 1, 1, 1, unit = "mm")
    )

  heatmap_grob <- ggplot2::ggplotGrob(heatmap_plot)
  panel_indices <- grep("^panel", heatmap_grob$layout$name)
  if (length(panel_indices) != 4L) {
    stop(
      "Could not identify all four reward-by-step heatmap panels.",
      call. = FALSE
    )
  }
  panel_columns <- unique(heatmap_grob$layout$l[panel_indices])
  panel_rows <- unique(heatmap_grob$layout$t[panel_indices])
  heatmap_grob$widths[panel_columns] <- grid::unit(panel_size_mm, "mm")
  heatmap_grob$heights[panel_rows] <- grid::unit(panel_size_mm, "mm")
  output_width_mm <- grid::convertWidth(
    sum(heatmap_grob$widths),
    unitTo = "mm",
    valueOnly = TRUE
  )
  output_height_mm <- grid::convertHeight(
    sum(heatmap_grob$heights),
    unitTo = "mm",
    valueOnly = TRUE
  )

  output_stem <- paste0(
    "node_reward_observation_step_accuracy_by_encoding_final_rank_",
    "heatmaps", output_suffix
  )
  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0(output_stem, ".pdf")
  )
  summary_path <- analysis_path(
    "stats",
    "tests",
    save_dir,
    paste0(output_stem, ".csv")
  )
  ensure_parent_directory(output_path)
  ensure_parent_directory(summary_path)
  ggsave_pdf(
    output_path,
    plot = heatmap_grob,
    width = output_width_mm,
    height = output_height_mm
  )
  readr::write_csv(
    heatmap_summary %>%
      dplyr::transmute(
        source = as.character(source),
        rank_direction = as.character(rank_direction),
        final_rank = final_rank,
        observation_step,
        mean_accuracy,
        se,
        participant_count,
        observation_count,
        included_in_plot
      ),
    summary_path
  )
  cat(sprintf(
    "Saved final-rank-by-step rank-direction heatmaps to: %s\n",
    output_path
  ))
  cat(sprintf(
    "Saved final-rank-by-step rank-direction summary to: %s\n",
    summary_path
  ))
  invisible(list(
    plot = heatmap_grob,
    summary = heatmap_summary,
    output_path = output_path,
    summary_path = summary_path
  ))
}

plot_fitted_reward_timestep_rank_difference_heatmaps <- function(
    df_beh,
    df_model,
    node_count,
    path_nodes,
    save_dir,
    panel_size_mm = 21,
    base_font_size = 7,
    output_suffix = "") {
  human_data <- prepare_fitted_observation_final_rank_human(
    df_beh,
    node_count,
    path_nodes
  )
  model_data <- prepare_fitted_observation_final_rank_model(
    df_model,
    node_count,
    path_nodes
  )
  source_levels <- c("participants", "model")
  rank_direction_levels <- c("encoding_better", "encoding_worse")

  classified <- dplyr::bind_rows(human_data, model_data) %>%
    dplyr::filter(
      source %in% source_levels,
      is.finite(final_rank),
      is.finite(observation_step),
      is.finite(accuracy),
      is.finite(observation_rank),
      is.finite(final_rank)
    ) %>%
    dplyr::mutate(
      rank_direction = dplyr::case_when(
        observation_rank < final_rank ~ "encoding_better",
        observation_rank > final_rank ~ "encoding_worse",
        TRUE ~ NA_character_
      )
    ) %>%
    dplyr::filter(!is.na(rank_direction))
  if (nrow(classified) == 0L) {
    stop(
      "No strict encoding-versus-final rank changes were available.",
      call. = FALSE
    )
  }

  # Average within participant first, then contrast the two independently
  # summarized rank directions. A strict within-participant pairing would
  # erase many structurally valid reward-by-step cells in the human data.
  summarized_cells <- classified %>%
    dplyr::group_by(
      source,
      participant_id,
      rank_direction,
      final_rank,
      observation_step
    ) %>%
    dplyr::summarise(
      participant_accuracy = mean(accuracy),
      observation_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::group_by(source, rank_direction, final_rank, observation_step) %>%
    dplyr::summarise(
      mean_accuracy = mean(participant_accuracy),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_accuracy) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
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
    dplyr::filter(
      is.finite(mean_accuracy_encoding_better),
      is.finite(mean_accuracy_encoding_worse)
    ) %>%
    dplyr::mutate(
      mean_accuracy_difference = mean_accuracy_encoding_worse -
        mean_accuracy_encoding_better,
      se = sqrt(
        se_encoding_better^2 + se_encoding_worse^2
      ),
      participant_count = pmin(
        participant_count_encoding_better,
        participant_count_encoding_worse
      ),
      minimum_condition_observation_count = pmin(
        observation_count_encoding_better,
        observation_count_encoding_worse
      ),
      observation_count = observation_count_encoding_better +
        observation_count_encoding_worse
    )
  if (nrow(summarized_cells) == 0L) {
    stop(
      paste0(
        "No reward-by-observation-step cell contained both rank ",
        "directions."
      ),
      call. = FALSE
    )
  }
  final_rank_levels <- seq_along(path_nodes)
  observation_step_levels <- seq_len(node_count - 1L)
  heatmap_summary <- tidyr::expand_grid(
    source = source_levels,
    final_rank = final_rank_levels,
    observation_step = observation_step_levels
  ) %>%
    dplyr::left_join(
      summarized_cells,
      by = c("source", "final_rank", "observation_step")
    ) %>%
    dplyr::mutate(
      participant_count = tidyr::replace_na(participant_count, 0L),
      minimum_condition_observation_count = tidyr::replace_na(
        minimum_condition_observation_count,
        0L
      ),
      observation_count = tidyr::replace_na(observation_count, 0L),
      included_in_plot = is.finite(mean_accuracy_difference),
      mean_accuracy_difference = dplyr::if_else(
        included_in_plot,
        mean_accuracy_difference,
        NA_real_
      ),
      se = dplyr::if_else(included_in_plot, se, NA_real_),
      source = factor(source, levels = source_levels),
      final_rank_factor = factor(final_rank, levels = final_rank_levels),
      observation_step_factor = factor(
        observation_step,
        levels = observation_step_levels
      )
    )
  if (all(is.na(heatmap_summary$mean_accuracy_difference))) {
    stop(
      paste0(
        "No final-rank-by-observation-step difference cells contained ",
        "observations in both rank conditions."
      ),
      call. = FALSE
    )
  }
  difference_limit <- max(
    abs(heatmap_summary$mean_accuracy_difference),
    na.rm = TRUE
  )
  difference_limit <- max(0.05, min(1, difference_limit))

  heatmap_plot <- ggplot2::ggplot(
    heatmap_summary,
    ggplot2::aes(
      x = final_rank_factor,
      y = observation_step_factor,
      fill = mean_accuracy_difference
    )
  ) +
    ggplot2::geom_tile(color = "white", linewidth = 0.2) +
    ggplot2::facet_grid(. ~ source, drop = FALSE) +
    ggplot2::scale_fill_gradient2(
      low = "#532B27",
      mid = "#F7F3F2",
      high = "#273F3F",
      midpoint = 0,
      limits = c(-difference_limit, difference_limit),
      breaks = c(-difference_limit, 0, difference_limit),
      labels = scales::label_number(accuracy = 0.01),
      na.value = "#000000"
    ) +
    ggplot2::scale_x_discrete(drop = FALSE) +
    ggplot2::scale_y_discrete(drop = FALSE) +
    ggplot2::guides(
      fill = ggplot2::guide_colorbar(
        direction = "horizontal",
        title.position = "top",
        title.hjust = 0.5,
        label.position = "bottom",
        barwidth = grid::unit(16, "mm"),
        barheight = grid::unit(1.5, "mm")
      )
    ) +
    ggplot2::labs(
      x = "Final path rank",
      y = "Observation step",
      fill = paste0(
        "Accuracy difference\n",
        "(final rank better - final rank worse)"
      )
    ) +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      text = ggplot2::element_text(size = base_font_size, color = "black"),
      axis.title = ggplot2::element_text(
        size = base_font_size,
        color = "black"
      ),
      axis.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.8, "mm"),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(
        color = "black",
        fill = NA,
        linewidth = 0.4
      ),
      panel.spacing = grid::unit(3, "mm"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black",
        face = "plain"
      ),
      legend.position = "top",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.title = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.text = ggplot2::element_text(
        size = base_font_size * 0.8,
        color = "black"
      ),
      legend.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.margin = ggplot2::margin(0, 0, 0, 0, unit = "mm"),
      legend.box.spacing = grid::unit(0, "mm"),
      legend.spacing.y = grid::unit(0, "mm"),
      plot.margin = ggplot2::margin(2, 1, 1, 1, unit = "mm")
    )

  heatmap_grob <- ggplot2::ggplotGrob(heatmap_plot)
  panel_indices <- grep("^panel", heatmap_grob$layout$name)
  if (length(panel_indices) != 2L) {
    stop(
      "Could not identify both reward-by-step difference panels.",
      call. = FALSE
    )
  }
  panel_columns <- unique(heatmap_grob$layout$l[panel_indices])
  panel_rows <- unique(heatmap_grob$layout$t[panel_indices])
  heatmap_grob$widths[panel_columns] <- grid::unit(panel_size_mm, "mm")
  heatmap_grob$heights[panel_rows] <- grid::unit(panel_size_mm, "mm")
  output_width_mm <- grid::convertWidth(
    sum(heatmap_grob$widths),
    unitTo = "mm",
    valueOnly = TRUE
  )
  output_height_mm <- grid::convertHeight(
    sum(heatmap_grob$heights),
    unitTo = "mm",
    valueOnly = TRUE
  )

  output_stem <- paste0(
    "node_reward_observation_step_accuracy_difference_by_encoding_",
    "final_rank_heatmaps", output_suffix
  )
  output_path <- analysis_path(
    "figures",
    save_dir,
    paste0(output_stem, ".pdf")
  )
  summary_path <- analysis_path(
    "stats",
    "tests",
    save_dir,
    paste0(output_stem, ".csv")
  )
  ensure_parent_directory(output_path)
  ensure_parent_directory(summary_path)
  ggsave_pdf(
    output_path,
    plot = heatmap_grob,
    width = output_width_mm,
    height = output_height_mm
  )
  readr::write_csv(
    heatmap_summary %>%
      dplyr::transmute(
        source = as.character(source),
        final_rank = final_rank,
        observation_step,
        mean_accuracy_difference,
        se,
        participant_count,
        participant_count_final_rank_worse =
          participant_count_encoding_better,
        participant_count_final_rank_better =
          participant_count_encoding_worse,
        observation_count_final_rank_worse =
          observation_count_encoding_better,
        observation_count_final_rank_better =
          observation_count_encoding_worse,
        minimum_condition_observation_count,
        observation_count,
        included_in_plot
      ),
    summary_path
  )
  cat(sprintf(
    "Saved final-rank-by-step rank-direction difference heatmaps to: %s\n",
    output_path
  ))
  cat(sprintf(
    "Saved final-rank-by-step rank-direction difference summary to: %s\n",
    summary_path
  ))
  invisible(list(
    plot = heatmap_grob,
    summary = heatmap_summary,
    output_path = output_path,
    summary_path = summary_path
  ))
}

plot_fitted_tree_size_accuracy <- function(
    df_beh,
    df_model,
    save_dir,
    x_size = 40,
    y_size = 36,
    base_font_size = 7,
    reference_plot = NULL) {
  require_columns(
    df_beh,
    c(
      "subject",
      "condition",
      "tree_size",
      "trial_id",
      "accuracy",
      "norm_reward"
    ),
    "Tree-size behavioral data"
  )
  require_columns(
    df_model,
    c(
      "fitted_subject",
      "condition",
      "tree_size",
      "trial_id",
      "accuracy",
      "norm_reward"
    ),
    "Tree-size fitted-model data"
  )
  combined_rows <- dplyr::bind_rows(
    df_beh %>%
      dplyr::transmute(
        source = "participants",
        participant_id = as.character(subject),
        condition = tolower(as.character(condition)),
        tree_size = as.integer(tree_size),
        trial_id = as.character(trial_id),
        accuracy = as.numeric(accuracy),
        norm_reward = as.numeric(norm_reward)
      ),
    df_model %>%
      dplyr::transmute(
        source = "model",
        participant_id = as.character(fitted_subject),
        condition = tolower(as.character(condition)),
        tree_size = as.integer(tree_size),
        trial_id = as.character(trial_id),
        accuracy = as.numeric(accuracy),
        norm_reward = as.numeric(norm_reward)
      )
  ) %>%
    dplyr::filter(
      condition == "planning",
      tree_size %in% c(7L, 13L, 31L)
    )
  accuracy_cells <- combined_rows %>%
    dplyr::filter(is.finite(accuracy)) %>%
    dplyr::group_by(source, participant_id, tree_size) %>%
    dplyr::summarise(
      participant_value = mean(accuracy),
      observation_count = dplyr::n(),
      trial_count = dplyr::n_distinct(trial_id),
      .groups = "drop"
    ) %>%
    dplyr::mutate(metric = "Recall accuracy")
  reward_trials <- combined_rows %>%
    dplyr::filter(is.finite(norm_reward)) %>%
    dplyr::group_by(source, participant_id, tree_size, trial_id) %>%
    dplyr::summarise(
      value_count = dplyr::n_distinct(norm_reward),
      norm_reward = dplyr::first(norm_reward),
      .groups = "drop"
    )
  if (any(reward_trials$value_count != 1L)) {
    stop(
      "A tree-size trial contains inconsistent normalized rewards.",
      call. = FALSE
    )
  }
  reward_cells <- reward_trials %>%
    dplyr::group_by(source, participant_id, tree_size) %>%
    dplyr::summarise(
      participant_value = mean(norm_reward),
      observation_count = dplyr::n(),
      trial_count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::mutate(metric = "Normalized reward")
  participant_cells <- dplyr::bind_rows(
    accuracy_cells,
    reward_cells
  )
  plot_summary <- participant_cells %>%
    dplyr::group_by(metric, source, tree_size) %>%
    dplyr::summarise(
      mean_value = mean(participant_value),
      se = ifelse(
        dplyr::n() > 1L,
        stats::sd(participant_value) / sqrt(dplyr::n()),
        NA_real_
      ),
      participant_count = dplyr::n_distinct(participant_id),
      observation_count = sum(observation_count),
      trial_count = sum(trial_count),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      source = factor(source, levels = c("participants", "model")),
      metric = factor(
        metric,
        levels = c("Recall accuracy", "Normalized reward")
      ),
      lower = pmax(0, mean_value - se),
      upper = pmin(1, mean_value + se)
    )
  expected_cells <- tidyr::expand_grid(
    metric = c("Recall accuracy", "Normalized reward"),
    source = c("participants", "model"),
    tree_size = c(7L, 13L, 31L)
  )
  missing_cells <- dplyr::anti_join(
    expected_cells,
    plot_summary %>%
      dplyr::transmute(
        metric = as.character(metric),
        source = as.character(source),
        tree_size
      ),
    by = c("metric", "source", "tree_size")
  )
  if (nrow(missing_cells) > 0L) {
    stop(
      "Tree-size plot is missing a metric, participant/model, or tree-size cell.",
      call. = FALSE
    )
  }

  plot <- ggplot2::ggplot(
    plot_summary,
    ggplot2::aes(x = tree_size, y = mean_value, group = 1)
  ) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = lower, ymax = upper),
      width = 0,
      linewidth = 0.3,
      color = fitted_participant_color,
      na.rm = TRUE
    ) +
    ggplot2::geom_line(
      linewidth = 0.8,
      color = fitted_participant_color,
      alpha = 0.8
    ) +
    ggplot2::geom_point(
      size = 0.5,
      color = fitted_participant_color
    ) +
    ggplot2::facet_wrap(~source, nrow = 1, drop = FALSE) +
    ggplot2::scale_x_continuous(
      breaks = c(7, 13, 31),
      limits = c(5, 33),
      expand = ggplot2::expansion(add = 0)
    ) +
    ggplot2::coord_cartesian(ylim = c(0, 1)) +
    ggplot2::labs(x = "Tree size", y = NULL) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      strip.clip = "off",
      panel.spacing = grid::unit(1.31, "mm"),
      plot.margin = ggplot2::margin(0, 1, 0, 1, unit = "mm")
    )

  output_paths <- character()
  for (metric_name in c("Recall accuracy", "Normalized reward")) {
    metric_summary <- plot_summary %>% dplyr::filter(as.character(metric) == metric_name)
    metric_plot <- ggplot2::`%+%`(plot, metric_summary) + ggplot2::labs(y = metric_name)
    stem <- if (metric_name == "Recall accuracy") "tree_size_accuracy" else "tree_size_norm_reward"
    output_path <- analysis_path("figures", save_dir, paste0(stem, ".pdf"))
    summary_path <- analysis_path("stats", "tests", save_dir, paste0(stem, ".csv"))
    ensure_parent_directory(output_path)
    ensure_parent_directory(summary_path)
    if (is.null(reference_plot)) {
      ggsave_pdf(output_path, plot = metric_plot, width = x_size, height = y_size)
    } else {
      save_fitted_paired_reference_plot(metric_plot, output_path, reference_plot, x_size, y_size)
    }
    readr::write_csv(metric_summary, summary_path)
    output_paths <- c(output_paths, output_path)
    cat(sprintf("Saved tree-size performance plot to: %s\n", output_path))
  }
  invisible(list(
    plot = plot,
    summary = plot_summary,
    output_paths = output_paths
  ))
}

extract_planning_condition_coefs <- function(coef_tab, term_map,
                                             source_label = "participants") {
  row_names <- rownames(coef_tab)
  p_col <- if ("Pr(>|t|)" %in% colnames(coef_tab)) "Pr(>|t|)" else "Pr(>|z|)"

  out <- lapply(names(term_map), function(term_name) {
    term_index <- which(row_names == term_name)

    estimate <- if (length(term_index)) {
      coef_tab[term_index, "Estimate"]
    } else {
      NA_real_
    }
    se <- if (length(term_index)) {
      coef_tab[term_index, "Std. Error"]
    } else {
      NA_real_
    }
    p_value <- if (length(term_index) && p_col %in% colnames(coef_tab)) {
      coef_tab[term_index, p_col]
    } else {
      NA_real_
    }

    star <- dplyr::case_when(
      is.na(p_value) ~ "",
      p_value < 0.001 ~ "***",
      p_value < 0.01 ~ "**",
      p_value < 0.05 ~ "*",
      TRUE ~ ""
    )

    data.frame(
      source = source_label,
      predictor = term_name,
      condition = "planning",
      estimate = estimate,
      se = se,
      star = star,
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(out)
}
