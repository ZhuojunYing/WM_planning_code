# ==============================================================================
# Shared Figure Utilities
# ==============================================================================

save_custom_plot <- function(p, figure_name, y_var, save_dir, x_size, y_size,
                             show_exposure = FALSE, prefix = "") {
  fname <- sprintf(
    "%s%s_%s%s.pdf",
    prefix,
    figure_name,
    y_var,
    ifelse(show_exposure && y_var == "accuracy", "_exposure", "")
  )
  outdir <- analysis_path("figures", save_dir)
  ensure_dir(outdir)

  ggsave_pdf(
    file.path(outdir, fname),
    plot = p,
    width = x_size,
    height = y_size
  )
}

plot_theme_axes <- function(base_font_size = 7, aspect_ratio = 1.0) {
  ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      legend.position = "none",
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(color = "black", linewidth = 0.5),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.15, "cm"),
      aspect.ratio = aspect_ratio
    )
}

prepare_binned_plot_data <- function(df_beh, df_model = NULL,
                                     dependent_var, y_var = "accuracy",
                                     show_exposure = FALSE,
                                     model_se_unit = "graph") {
  if (is.null(df_model)) {
    df <- df_beh %>%
      dplyr::mutate(source = "participants")
  } else {
    df <- dplyr::bind_rows(
      df_beh %>% dplyr::mutate(source = "participants"),
      df_model %>% dplyr::mutate(source = "model")
    )
  }

  if (!"graph" %in% names(df)) {
    df <- df %>% dplyr::mutate(graph = NA)
  }

  df <- df %>%
    dplyr::mutate(
      source = factor(source, levels = c("participants", "model")),
      se_unit = dplyr::case_when(
        source == "model" & model_se_unit == "graph" & !is.na(graph) ~
          paste0("graph_", graph),
        TRUE ~ paste0("subject_", subject)
      )
    )

  if (!show_exposure) {
    df <- df %>% dplyr::filter(condition == "planning")
  }

  df %>%
    dplyr::group_by(source, condition, !!rlang::sym(dependent_var), se_unit) %>%
    dplyr::summarise(
      avg_error = mean(.data[[y_var]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::group_by(source, condition, !!rlang::sym(dependent_var)) %>%
    dplyr::summarise(
      mean_error = mean(avg_error, na.rm = TRUE),
      se_error = stats::sd(avg_error, na.rm = TRUE) / sqrt(dplyr::n()),
      count = dplyr::n(),
      .groups = "drop"
    ) %>%
    dplyr::filter(count >= 5 | source == "model") %>%
    dplyr::rename(x_value = !!rlang::sym(dependent_var))
}

plot_binned_data_bar <- function(df_beh, df_model = NULL, dependent_var,
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

  discrete_data <- prepare_binned_plot_data(
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
    p <- p + ggplot2::facet_wrap(~ source, nrow = 1)
  }

  if (!is.null(y_step)) {
    p <- p + ggplot2::scale_y_continuous(
      breaks = seq(y_min, y_max, by = y_step)
    )
  }

  if (!combined_plot) {
    save_custom_plot(
      p, figure_name, y_var, save_dir, x_size, y_size,
      show_exposure, "bar_"
    )
  }

  return(p)
}

plot_binned_data_shade <- function(df_beh, df_model = NULL, dependent_var,
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
                                   save_dir = "plots") {
  y_var <- match.arg(y_var)

  if (y_var == "estimated_reward") {
    y_min <- -4
    y_max <- 4
    show_exposure <- TRUE
  }

  if (is.null(df_model) && is.null(y_step)) {
    y_step <- 0.1
  }

  discrete_data <- prepare_binned_plot_data(
    df_beh = df_beh,
    df_model = df_model,
    dependent_var = dependent_var,
    y_var = y_var,
    show_exposure = show_exposure
  )

  if (return_data) {
    return(discrete_data)
  }

  x_breaks <- if (!is.null(x_step)) {
    seq(from = x_min, to = x_max, by = x_step)
  } else {
    -4:4
  }

  p <- ggplot2::ggplot(
    discrete_data,
    ggplot2::aes(x = x_value, y = mean_error, color = condition)
  ) +
    ggplot2::geom_line(ggplot2::aes(group = condition), linewidth = 0.8, alpha = 0.8) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = mean_error - se_error, ymax = mean_error + se_error),
      width = 0.5,
      linewidth = 0.3,
      alpha = 0.95,
      na.rm = TRUE
    ) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::scale_color_manual(
      values = c("planning" = "#6e3934", "exposure" = "#c2beb5")
    ) +
    ggplot2::scale_x_continuous(breaks = x_breaks, expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(x = x_lab, y = y_lab, title = title) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.2) +
    ggplot2::theme(
      strip.clip = "off",
      panel.spacing = grid::unit(0.75, "lines"),
      plot.margin = ggplot2::margin(t = 0, r = 1, b = 0, l = 1, unit = "mm")
    )

  if (!is.null(df_model)) {
    p <- p + ggplot2::facet_wrap(~ source, nrow = 1)
  }

  if (!is.null(y_step)) {
    p <- p + ggplot2::scale_y_continuous(
      breaks = seq(y_min, y_max, by = y_step)
    )
  }

  if (!combined_plot) {
    save_custom_plot(
      p, figure_name, y_var, save_dir, x_size, y_size, show_exposure
    )
  }

  return(p)
}

plot_binned_data_discrete <- function(df_beh, dependent_var, x_lab = "Value",
                                      y_lab = "Average Accuracy",
                                      title = NULL,
                                      figure_name,
                                      x_size = 40,
                                      y_size = 33,
                                      base_font_size = 7,
                                      y_min = 0,
                                      y_max = 0.5,
                                      x_step = NULL,
                                      y_step = NULL,
                                      combined_plot = FALSE,
                                      show_exposure = FALSE,
                                      save_dir = "plots") {
  discrete_data <- prepare_binned_plot_data(
    df_beh = df_beh,
    dependent_var = dependent_var,
    y_var = "accuracy",
    show_exposure = show_exposure
  )

  x_min <- min(discrete_data$x_value, na.rm = TRUE)
  x_max <- max(discrete_data$x_value, na.rm = TRUE)
  x_breaks <- if (!is.null(x_step)) {
    seq(from = x_min, to = x_max, by = x_step)
  } else {
    unique(discrete_data$x_value)
  }

  p <- ggplot2::ggplot() +
    ggplot2::geom_errorbar(
      data = discrete_data,
      ggplot2::aes(
        x = x_value,
        ymin = mean_error - se_error,
        ymax = mean_error + se_error,
        color = condition
      ),
      width = 0.5,
      linewidth = 0.3,
      alpha = 0.95,
      position = ggplot2::position_dodge(width = 0.2)
    ) +
    ggplot2::geom_point(
      data = discrete_data,
      ggplot2::aes(x = x_value, y = mean_error, color = condition, fill = condition),
      size = 1.0,
      shape = 21,
      stroke = 0.3,
      position = ggplot2::position_dodge(width = 0.2)
    ) +
    ggplot2::scale_color_manual(values = c("planning" = "#6e3934", "exposure" = "#c2beb5")) +
    ggplot2::scale_fill_manual(values = c("planning" = "#6e3934", "exposure" = "#c2beb5")) +
    ggplot2::scale_x_continuous(breaks = x_breaks, expand = c(0.1, 0)) +
    ggplot2::coord_cartesian(xlim = c(x_min, x_max), ylim = c(y_min, y_max)) +
    ggplot2::labs(x = x_lab, y = y_lab, title = title) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.0)

  if (!is.null(y_step)) {
    p <- p + ggplot2::scale_y_continuous(
      breaks = seq(floor(y_min / y_step) * y_step, ceiling(y_max / y_step) * y_step, by = y_step)
    )
  }

  if (!combined_plot) {
    save_custom_plot(
      p, figure_name, "accuracy", save_dir, x_size, y_size, show_exposure
    )
  }

  return(p)
}

plot_binned_data_boxplot <- function(df_beh, dependent_var, x_lab = "Value",
                                     y_lab = "Average Accuracy",
                                     title = NULL,
                                     figure_name,
                                     x_size = 40,
                                     y_size = 33,
                                     base_font_size = 7,
                                     y_min = 0,
                                     y_max = 1,
                                     x_step = NULL,
                                     y_step = NULL,
                                     combined_plot = FALSE,
                                     x_min = -5,
                                     x_max = 5,
                                     show_exposure = FALSE,
                                     save_dir = "plots") {
  df <- df_beh
  if (!show_exposure) {
    df <- df %>% dplyr::filter(condition == "planning")
  }

  group_vars <- c("condition", dependent_var, "subject")
  if ("graph" %in% names(df)) group_vars <- c(group_vars, "graph")
  if ("path" %in% names(df)) group_vars <- c(group_vars, "path")

  subject_value_data <- df %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_vars))) %>%
    dplyr::summarise(y_outcome = mean(accuracy, na.rm = TRUE), .groups = "drop") %>%
    dplyr::rename(x_value = !!rlang::sym(dependent_var))

  x_labels <- sort(unique(subject_value_data$x_value))

  p <- ggplot2::ggplot(
    subject_value_data,
    ggplot2::aes(x = factor(x_value), y = y_outcome, fill = condition)
  ) +
    ggplot2::geom_point(
      ggplot2::aes(color = condition),
      position = ggplot2::position_jitterdodge(dodge.width = 0.7, jitter.width = 0.2),
      size = 0.5,
      alpha = 0.35,
      stroke = 0
    ) +
    ggplot2::geom_boxplot(
      outlier.shape = NA,
      alpha = 0.6,
      linewidth = 0.4,
      width = 0.5,
      position = ggplot2::position_dodge(width = 0.7),
      color = "black"
    ) +
    ggplot2::scale_color_manual(values = c("planning" = "#6e3934", "exposure" = "#c2beb5")) +
    ggplot2::scale_fill_manual(values = c("planning" = "#6e3934", "exposure" = "#c2beb5")) +
    ggplot2::scale_x_discrete(breaks = x_labels, labels = x_labels, name = x_lab) +
    ggplot2::coord_cartesian(ylim = c(y_min, y_max)) +
    ggplot2::labs(y = y_lab, title = title) +
    plot_theme_axes(base_font_size = base_font_size, aspect_ratio = 1.0) +
    ggplot2::theme(
      panel.grid.major.y = ggplot2::element_line(color = "grey90", linewidth = 0.2)
    )

  if (!is.null(y_step)) {
    p <- p + ggplot2::scale_y_continuous(
      breaks = seq(floor(y_min / y_step) * y_step, ceiling(y_max / y_step) * y_step, by = y_step)
    )
  }

  if (!combined_plot) {
    save_custom_plot(
      p, figure_name, "accuracy", save_dir, x_size, y_size, show_exposure,
      prefix = "boxplot_"
    )
  }

  return(p)
}

extract_condition_coefs <- function(coef_tab, term_map, source_label = "participants") {
  row_names <- rownames(coef_tab)
  out <- list()

  for (term_name in names(term_map)) {
    idx_main <- which(row_names == term_name)
    beta_main <- if (length(idx_main)) coef_tab[idx_main, "Estimate"] else NA_real_
    se_main <- if (length(idx_main)) coef_tab[idx_main, "Std. Error"] else NA_real_

    idx_int <- which(
      row_names == paste0(term_name, ":conditionexposure") |
        row_names == paste0("conditionexposure:", term_name)
    )

    if (length(idx_int)) {
      beta_int <- coef_tab[idx_int, "Estimate"]
      se_int <- coef_tab[idx_int, "Std. Error"]
      p_col <- if ("Pr(>|t|)" %in% colnames(coef_tab)) "Pr(>|t|)" else "Pr(>|z|)"
      p_int <- if (p_col %in% colnames(coef_tab)) coef_tab[idx_int, p_col] else NA_real_
    } else {
      beta_int <- NA_real_
      se_int <- NA_real_
      p_int <- NA_real_
    }

    stars <- dplyr::case_when(
      is.na(p_int) ~ "",
      p_int < 0.001 ~ "***",
      p_int < 0.01 ~ "**",
      p_int < 0.05 ~ "*",
      TRUE ~ ""
    )

    out[[length(out) + 1]] <- data.frame(
      source = source_label,
      predictor = term_name,
      condition = "planning",
      estimate = beta_main,
      se = se_main,
      star = "",
      stringsAsFactors = FALSE
    )
    out[[length(out) + 1]] <- data.frame(
      source = source_label,
      predictor = term_name,
      condition = "exposure",
      estimate = if (!is.na(beta_main) && !is.na(beta_int)) beta_main + beta_int else NA_real_,
      se = if (!is.na(se_main) && !is.na(se_int)) sqrt(se_main^2 + se_int^2) else NA_real_,
      star = stars,
      stringsAsFactors = FALSE
    )
  }

  dplyr::bind_rows(out)
}

plot_coef_df <- function(coef_df, terms_model, x_lab, y_lab, figure_name,
                         save_dir, x_size, y_size, base_font_size,
                         facet_source = TRUE, file_name = NULL,
                         predictor_levels = NULL) {
  if (is.null(predictor_levels)) {
    predictor_levels <- rev(unique(unname(terms_model)))
  }

  coef_df <- coef_df %>%
    dplyr::mutate(
      ci_low = estimate - 1.96 * se,
      ci_high = estimate + 1.96 * se,
      condition = factor(condition, levels = c("exposure", "planning")),
      predictor_label = factor(
        terms_model[predictor],
        levels = predictor_levels
      )
    ) %>%
    dplyr::arrange(condition)

  star_keys <- c("predictor")
  if ("source" %in% names(coef_df)) {
    star_keys <- c("source", "predictor")
    coef_df <- coef_df %>%
      dplyr::mutate(source = factor(source, levels = c("participants", "model")))
  }

  stars_map <- coef_df %>%
    dplyr::filter(condition == "exposure", star != "") %>%
    dplyr::select(dplyr::all_of(star_keys), star_exp = star)
  coef_df <- coef_df %>%
    dplyr::left_join(stars_map, by = star_keys) %>%
    dplyr::mutate(star = ifelse(condition == "planning", star_exp, "")) %>%
    dplyr::select(-star_exp)

  x_min <- min(coef_df$ci_low, na.rm = TRUE)
  x_max <- max(coef_df$ci_high, na.rm = TRUE)
  x_pad <- 0.1 * (x_max - x_min + 1e-8)

  p <- ggplot2::ggplot(
    coef_df,
    ggplot2::aes(x = estimate, y = predictor_label, color = condition)
  ) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3, color = "grey60") +
    ggplot2::geom_errorbarh(ggplot2::aes(xmin = ci_low, xmax = ci_high), height = 0, linewidth = 0.4) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::geom_text(
      data = coef_df %>% dplyr::filter(condition == "planning", star != ""),
      ggplot2::aes(label = star),
      vjust = -0.4,
      color = "black",
      size = 1.5,
      show.legend = FALSE
    ) +
    ggplot2::scale_color_manual(values = c("exposure" = "#c2beb5", "planning" = "#6e3934")) +
    ggplot2::scale_x_continuous(
      name = x_lab,
      breaks = scales::pretty_breaks(n = 3),
      limits = c(x_min - x_pad, x_max + x_pad)
    ) +
    ggplot2::labs(y = y_lab) +
    ggplot2::theme_minimal(base_size = base_font_size) +
    ggplot2::theme(
      legend.position = "none",
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.line = ggplot2::element_line(color = "black", linewidth = 0.4),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.15, "cm"),
      aspect.ratio = 0.8,
      panel.spacing = grid::unit(1, "lines"),
      plot.margin = ggplot2::margin(2, 2, 2, 2, "mm")
    )

  if (facet_source && "source" %in% names(coef_df)) {
    p <- p + ggplot2::facet_wrap(~ source, nrow = 1)
  }

  outdir <- analysis_path("figures", save_dir)
  ensure_dir(outdir)
  if (is.null(file_name)) {
    file_name <- sprintf("coef_%s_accuracy.pdf", figure_name)
  }

  ggsave_pdf(
    file.path(outdir, file_name),
    plot = p,
    width = x_size,
    height = y_size,
    limitsize = FALSE
  )

  return(p)
}

plot_combined_coefs <- function(df_beh, df_model = NULL, y_var = "accuracy",
                                x_lab = "Regression Weight",
                                y_lab = "",
                                figure_name = "coef_by_source",
                                save_dir = "plots",
                                x_size = ifelse(is.null(df_model), 40, 80),
                                y_size = 36,
                                base_font_size = 7) {
  df_beh$condition <- factor(df_beh$condition, levels = c("planning", "exposure"))

  if (is.null(df_model)) {
    df_beh <- df_beh %>%
      dplyr::mutate(
        preceding_sum_reward = ifelse(!is_leaf, 0, preceding_sum_reward),
        aunt_reward = ifelse(!is_leaf, 0, aunt_reward)
      )
    terms_model <- c(
      "scale(actual_reward)" = "Probed",
      "scale(sibling_reward)" = "Sibling",
      "scale(aunt_reward)" = "Aunt",
      "scale(preceding_sum_reward)" = "Preceding Sum"
    )
    base_fml <- paste(
      "scale(actual_reward) + scale(actual_reward^2) +",
      "scale(sibling_reward) + scale(sibling_reward^2) +",
      "scale(aunt_reward) + scale(preceding_sum_reward)"
    )
    fml_str <- paste0(
      y_var,
      " ~ (",
      base_fml,
      ") * condition"
    )
    fml_str_beh <- paste0(fml_str, " + ((", base_fml, ") * condition || subject)")
    model_beh <- if (y_var == "accuracy") {
      lme4::glmer(
        stats::as.formula(fml_str_beh),
        family = stats::binomial,
        data = df_beh,
        control = lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )
    } else {
      lmerTest::lmer(
        stats::as.formula(fml_str_beh),
        data = df_beh,
        REML = FALSE
      )
    }
    coef_df <- extract_condition_coefs(
      summary(model_beh)$coefficients,
      terms_model,
      source_label = "participants"
    ) %>%
      dplyr::select(-source)

    return(plot_coef_df(
      coef_df = coef_df,
      terms_model = terms_model,
      x_lab = x_lab,
      y_lab = y_lab,
      figure_name = figure_name,
      save_dir = save_dir,
      x_size = x_size,
      y_size = y_size,
      base_font_size = base_font_size,
      facet_source = FALSE,
      file_name = sprintf("%s_coef_accuracy.pdf", figure_name),
      predictor_levels = c("Aunt", "Preceding Sum", "Sibling", "Probed")
    ))
  }

  df_model$condition <- factor(df_model$condition, levels = c("planning", "exposure"))
  df_model <- df_model %>%
    dplyr::mutate(
      parent_reward = ifelse(!is_leaf, 0, parent_reward),
      aunt_reward = ifelse(!is_leaf, 0, aunt_reward)
    )
  df_beh <- df_beh %>%
    dplyr::mutate(
      parent_reward = ifelse(!is_leaf, 0, parent_reward),
      aunt_reward = ifelse(!is_leaf, 0, aunt_reward)
    )

  terms_model <- c(
    "scale(actual_reward)" = "Probed",
    "scale(sibling_reward)" = "Sibling",
    "scale(aunt_reward)" = "Aunt",
    "scale(parent_reward)" = "Parent"
  )
  base_fml <- paste(
    "scale(actual_reward) + scale(actual_reward^2) +",
    "scale(sibling_reward) + scale(sibling_reward^2) +",
    "scale(aunt_reward) + scale(parent_reward)"
  )
  fml_str_mod <- paste0(y_var, " ~ (", base_fml, ") * condition")
  fml_str_beh <- paste0(fml_str_mod, " + ((", base_fml, ") * condition || subject)")

  if (y_var == "accuracy") {
    model_beh <- lme4::glmer(
      stats::as.formula(fml_str_beh),
      family = stats::binomial,
      data = df_beh,
      control = lme4::glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
    model_mod <- stats::glm(
      stats::as.formula(fml_str_mod),
      family = stats::binomial,
      data = df_model
    )
  } else {
    model_beh <- lmerTest::lmer(stats::as.formula(fml_str_beh), data = df_beh, REML = FALSE)
    model_mod <- stats::lm(stats::as.formula(fml_str_mod), data = df_model)
  }

  coef_df <- dplyr::bind_rows(
    extract_condition_coefs(summary(model_beh)$coefficients, terms_model, "participants"),
    extract_condition_coefs(summary(model_mod)$coefficients, terms_model, "model")
  )

  plot_coef_df(
    coef_df = coef_df,
    terms_model = terms_model,
    x_lab = x_lab,
    y_lab = y_lab,
    figure_name = figure_name,
    save_dir = save_dir,
    x_size = x_size,
    y_size = y_size,
    base_font_size = base_font_size,
    facet_source = TRUE,
    predictor_levels = c("Aunt", "Parent", "Sibling", "Probed")
  )
}

generate_interaction_plot <- function(data, x_var = "order", group_var = "tree",
                                      y_var = "norm_reward",
                                      name = "interaction_plot",
                                      stat_type = "mean",
                                      error_type = "se",
                                      x_label = "Order",
                                      y_label = NULL,
                                      y_min = NULL,
                                      y_max = NULL,
                                      y_step = NULL,
                                      height = 30,
                                      width = 40,
                                      show_legend = TRUE,
                                      dpi = 300,
                                      custom_colors = NULL,
                                      dot_size = 0.2,
                                      line_size = 0.5,
                                      error_bar_width = 0.1,
                                      dodge_width = 0.1,
                                      save_dir = "exp3") {
  if (!x_var %in% names(data)) stop(sprintf("X variable '%s' not found in dataframe.", x_var))
  if (!group_var %in% names(data)) stop(sprintf("Group variable '%s' not found in dataframe.", group_var))
  if (!y_var %in% names(data)) stop(sprintf("Y variable '%s' not found in dataframe.", y_var))

  x_var_sym <- rlang::sym(x_var)
  group_var_sym <- rlang::sym(group_var)
  y_var_sym <- rlang::sym(y_var)

  if (is.null(y_label)) y_label <- paste(tools::toTitleCase(stat_type), y_var)

  data[[x_var]] <- as.factor(data[[x_var]])
  data[[group_var]] <- as.factor(data[[group_var]])
  unique_groups <- levels(data[[group_var]])

  if (is.null(custom_colors)) {
    custom_colors <- stats::setNames(c("#6e3934", "#c2beb5"), unique_groups)
  }

  df_summary <- data %>%
    dplyr::filter(!is.na(!!y_var_sym)) %>%
    dplyr::group_by(!!x_var_sym, !!group_var_sym) %>%
    dplyr::summarise(
      n = dplyr::n(),
      summary_value = if (stat_type == "median") {
        stats::median(!!y_var_sym, na.rm = TRUE)
      } else {
        mean(!!y_var_sym, na.rm = TRUE)
      },
      sd_val = stats::sd(!!y_var_sym, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      error_lower = dplyr::case_when(
        error_type == "se" ~ summary_value - sd_val / sqrt(n),
        error_type == "sd" ~ summary_value - sd_val,
        error_type == "ci" ~ summary_value - stats::qt(0.975, df = n - 1) * sd_val / sqrt(n)
      ),
      error_upper = dplyr::case_when(
        error_type == "se" ~ summary_value + sd_val / sqrt(n),
        error_type == "sd" ~ summary_value + sd_val,
        error_type == "ci" ~ summary_value + stats::qt(0.975, df = n - 1) * sd_val / sqrt(n)
      )
    )

  position <- ggplot2::position_dodge(dodge_width)

  g <- ggplot2::ggplot(
    df_summary,
    ggplot2::aes(
      x = !!x_var_sym,
      y = summary_value,
      color = !!group_var_sym,
      group = !!group_var_sym
    )
  ) +
    ggplot2::geom_line(position = position, linewidth = line_size) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = error_lower, ymax = error_upper),
      width = error_bar_width,
      position = position,
      linewidth = 0.4
    ) +
    ggplot2::geom_point(position = position, size = dot_size) +
    ggplot2::scale_color_manual(values = custom_colors) +
    ggplot2::labs(x = x_label, y = y_label, color = tools::toTitleCase(group_var)) +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      plot.title = ggplot2::element_text(size = 7),
      axis.title.x = ggplot2::element_text(size = 7),
      axis.title.y = ggplot2::element_text(size = 7),
      axis.text.x = ggplot2::element_text(size = 7, color = "black"),
      axis.text.y = ggplot2::element_text(size = 7, color = "black"),
      legend.text = ggplot2::element_text(size = 7),
      legend.title = ggplot2::element_text(size = 7),
      legend.position = if (show_legend) "right" else "none",
      panel.background = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.border = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = "black", linewidth = 0.5),
      axis.line.y = ggplot2::element_line(color = "black", linewidth = 0.5),
      axis.ticks = ggplot2::element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = grid::unit(0.15, "cm")
    )

  if (!is.null(y_step)) {
    if (is.null(y_min)) y_min <- floor(min(df_summary$error_lower, na.rm = TRUE))
    if (is.null(y_max)) y_max <- ceiling(max(df_summary$error_upper, na.rm = TRUE))
    g <- g + ggplot2::scale_y_continuous(
      limits = c(y_min, y_max),
      breaks = seq(y_min, y_max, by = y_step),
      labels = function(x) sprintf("%.2f", x)
    )
  } else if (!is.null(y_min) || !is.null(y_max)) {
    g <- g + ggplot2::scale_y_continuous(
      limits = c(y_min, y_max),
      labels = function(x) sprintf("%.2f", x)
    )
  } else {
    g <- g + ggplot2::scale_y_continuous(labels = function(x) sprintf("%.2f", x))
  }

  outdir <- analysis_path("figures", save_dir)
  ensure_dir(outdir)
  ggsave_pdf(
    file.path(outdir, sprintf("%s.pdf", name)),
    plot = g,
    width = width,
    height = height,
    dpi = dpi,
    limitsize = FALSE
  )

  return(g)
}
