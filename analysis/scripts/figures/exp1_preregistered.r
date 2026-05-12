# ==============================================================================
# Script: Plotting and Model Fitting (Experiment 1 & 2)
# Description: Generates binned behavioral vs model plots (bars/shaded lines),
#              fits mixed-effects models, and plots coefficient comparisons.
# ==============================================================================

library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(grid)
library(rlang)
library(lmerTest) # Used for generating p-values in mixed models
library(here)     # For safe, relative pathing

source(here::here("analysis", "scripts", "utils.R"))
source(here::here("analysis", "scripts", "plot_utils.R"))

# ==============================================================================
# Figure-Specific Coefficient Plot
# ==============================================================================

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

plot_exp1_preregistered_planning_coef_df <- function(coef_df, terms_model,
                                                     x_lab, y_lab,
                                                     figure_name, save_dir,
                                                     x_size = 80,
                                                     y_size = 36,
                                                     base_font_size = 7) {
  predictor_levels <- c("Aunt", "Parent", "Sibling", "Probed")

  coef_df <- coef_df %>%
    dplyr::mutate(
      ci_low = estimate - 1.96 * se,
      ci_high = estimate + 1.96 * se,
      source = factor(source, levels = c("participants", "model")),
      condition = factor(condition, levels = "planning"),
      predictor_label = factor(
        terms_model[predictor],
        levels = predictor_levels
      )
    )

  x_min <- min(coef_df$ci_low, na.rm = TRUE)
  x_max <- max(coef_df$ci_high, na.rm = TRUE)
  x_pad <- 0.1 * (x_max - x_min + 1e-8)

  p <- ggplot2::ggplot(
    coef_df,
    ggplot2::aes(x = estimate, y = predictor_label, color = condition)
  ) +
    ggplot2::geom_vline(
      xintercept = 0,
      linetype = "dashed",
      linewidth = 0.3,
      color = "grey60"
    ) +
    ggplot2::geom_errorbarh(
      ggplot2::aes(xmin = ci_low, xmax = ci_high),
      height = 0,
      linewidth = 0.4
    ) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::geom_text(
      data = coef_df %>% dplyr::filter(star != ""),
      ggplot2::aes(label = star),
      vjust = -0.4,
      color = "black",
      size = 1.5,
      show.legend = FALSE
    ) +
    ggplot2::scale_color_manual(values = c("planning" = "#6e3934")) +
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
    ) +
    ggplot2::facet_wrap(~ source, nrow = 1)

  outdir <- analysis_path("figures", save_dir)
  ensure_dir(outdir)

  ggsave_pdf(
    file.path(outdir, sprintf("coef_%s_accuracy.pdf", figure_name)),
    plot = p,
    width = x_size,
    height = y_size,
    limitsize = FALSE
  )

  return(p)
}

plot_exp1_preregistered_planning_coefs <- function(df_beh, df_model,
                                                   y_var = "accuracy",
                                                   x_lab = "Regression Weight",
                                                   y_lab = "",
                                                   figure_name = "coef_by_source",
                                                   save_dir = "plots",
                                                   x_size = 80,
                                                   y_size = 36,
                                                   base_font_size = 7) {
  df_beh <- df_beh %>%
    dplyr::filter(condition == "planning") %>%
    dplyr::mutate(
      parent_reward = ifelse(!is_leaf, 0, parent_reward),
      aunt_reward = ifelse(!is_leaf, 0, aunt_reward)
    )
  df_model <- df_model %>%
    dplyr::filter(condition == "planning") %>%
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
  fml_str_mod <- paste0(y_var, " ~ ", base_fml)
  fml_str_beh <- paste0(
    fml_str_mod,
    " + (",
    base_fml,
    " || subject)"
  )

  if (y_var == "accuracy") {
    model_beh <- lme4::glmer(
      stats::as.formula(fml_str_beh),
      family = stats::binomial,
      data = df_beh,
      control = lme4::glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
      )
    )
    model_mod <- stats::glm(
      stats::as.formula(fml_str_mod),
      family = stats::binomial,
      data = df_model
    )
  } else {
    model_beh <- lmerTest::lmer(
      stats::as.formula(fml_str_beh),
      data = df_beh,
      REML = FALSE
    )
    model_mod <- stats::lm(stats::as.formula(fml_str_mod), data = df_model)
  }

  coef_df <- dplyr::bind_rows(
    extract_planning_condition_coefs(
      summary(model_beh)$coefficients,
      terms_model,
      "participants"
    ),
    extract_planning_condition_coefs(
      summary(model_mod)$coefficients,
      terms_model,
      "model"
    )
  )

  plot_exp1_preregistered_planning_coef_df(
    coef_df = coef_df,
    terms_model = terms_model,
    x_lab = x_lab,
    y_lab = y_lab,
    figure_name = figure_name,
    save_dir = save_dir,
    x_size = x_size,
    y_size = y_size,
    base_font_size = base_font_size
  )
}

# ==============================================================================
# Execution Wrapper
# ==============================================================================

# Function to process all plots for a specific version/experiment
process_version <- function( ) {
  
  # Dynamic Data Loading via here()
  beh_file <- analysis_path("data", "processed_data",   "exp1_beh.csv")
  model_file <- analysis_path("data", "processed_data",   "exp1_model_preregistered.csv")
  
  df_beh <- read.csv(beh_file)
  df_model <- read.csv(model_file)
  
  # FIX APPLIED: Standardize 'subject' to character to prevent bind_rows() type mismatch
  df_beh$subject <- as.character(df_beh$subject)
  
  if (!"subject" %in% names(df_model)) {
    df_model$subject <- "model_agent"
  } else {
    df_model$subject <- as.character(df_model$subject)
  }
  
  # Set up axis boundaries based on experiment version
    y_min <- 0.0
    y_max <- 0.6
 
  save_dir <- "exp1_preregistered"
  
  # Generate Plots
  plot_exp1_preregistered_planning_coefs(df_beh, df_model, y_var = "accuracy", x_lab = "Coefficient", figure_name = "coef_by_source", save_dir = save_dir)
  
  plot_binned_data_shade(df_beh, df_model, dependent_var = "actual_reward", x_lab = "Probed Reward", y_lab = "Accuracy", figure_name = "node_reward", y_min = y_min, y_max = y_max, x_step = 2, y_var = "accuracy", save_dir = save_dir)
  
  plot_binned_data_shade(df_beh %>% filter(is_leaf == TRUE), df_model %>% filter(is_leaf == TRUE), dependent_var = "aunt_reward", x_lab = "Average Aunt Reward", y_lab = "Accuracy", figure_name = "aunt_reward", y_min = y_min, y_max = y_max, x_step = 2, y_var = "accuracy", save_dir = save_dir)
  
  plot_binned_data_shade(df_beh, df_model, dependent_var = "sibling_reward", x_lab = "Average Sibling Reward", y_lab = "Accuracy", figure_name = "sibling_reward", y_min = y_min, y_max = y_max, x_step = 2, y_var = "accuracy", save_dir = save_dir)
  
  plot_binned_data_shade(df_beh %>% filter(is_leaf == TRUE), df_model %>% filter(is_leaf == TRUE), dependent_var = "parent_reward", x_lab = "Parent Reward", y_lab = "Accuracy", figure_name = "parent_reward", y_min = y_min, y_max = y_max, y_var = "accuracy", x_step = 2, save_dir = save_dir)
  
  # Generate and Plot Rankings

    df_beh_rank <- calculate_path_rank_exp1_beh(df_beh %>% filter(condition == "planning"))
    df_model_rank <- calculate_path_rank_exp1_model(df_model %>% filter(condition == "planning"))
    x_max_val <- 4.5
    x_step_val <- NULL

  
  # Standardize rank column names
  df_beh_rank$path_rank <- df_beh_rank$p_rank
  df_model_rank$path_rank <- df_model_rank$p_rank
  
  # FIX APPLIED: Removed 'path' from group_by
  df_beh_rank_grouped <- df_beh_rank %>%
    group_by(subject, condition, graph, path_rank) %>%
    summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")
  
  # FIX APPLIED: Removed 'path' from group_by
  df_model_rank_grouped <- df_model_rank %>%
    group_by(subject, condition, graph, path_rank) %>%
    summarise(accuracy = mean(accuracy, na.rm = TRUE), .groups = "drop")
  
  plot_binned_data_bar(df_beh_rank_grouped, df_model_rank_grouped, dependent_var = "path_rank", x_lab = "Path Rank", y_lab = "Accuracy", figure_name = "path_rank", y_min = y_min, y_max = y_max, y_var = "accuracy", x_min = 1, x_max = x_max_val, x_step = x_step_val, save_dir = save_dir)
  }

# ------------------------------------------------------------------------------
# Execute Script
# ------------------------------------------------------------------------------
process_version( )
