# ==============================================================================
# Script: Interaction Plots (Experiment 3)
# Description: Generates customized line plots with error bars to visualize 
#              interactions between tree structure and search order.
# ==============================================================================

library(dplyr)
library(ggplot2)
library(rlang)
library(readr)
library(here)

source(here::here("analysis", "scripts", "utils.R"))

# ==============================================================================
# Data Loading
# ==============================================================================

# Data are loaded in separate behavioral and simulation blocks below because the
# simulation file defensively re-derives task-structure labels.

# ==============================================================================
# Data Preprocessing / Transformation
# ==============================================================================

# Experiment-specific recoding is applied immediately after each file is loaded.

# ==============================================================================
# Analysis / Plotting
# ==============================================================================

generate_interaction_plot <- function(data, x_var = "order", group_var = "tree", y_var = "norm_reward",
                                      name = "interaction_plot", stat_type = "mean", error_type = "se",
                                      x_label = "Order", y_label = NULL, y_min = NULL, y_max = NULL,
                                      y_step = NULL, height = 30, width = 40, show_legend = TRUE,
                                      dpi = 300, custom_colors = NULL, dot_size = 0.2, line_size = 0.5,
                                      error_bar_width = 0.1, dodge_width = 0.1, save_dir = "exp3") {
  
  # Validate variables
  if (!x_var %in% names(data)) stop(sprintf("X variable '%s' not found in dataframe.", x_var))
  if (!group_var %in% names(data)) stop(sprintf("Group variable '%s' not found in dataframe.", group_var))
  if (!y_var %in% names(data)) stop(sprintf("Y variable '%s' not found in dataframe.", y_var))
  
  x_var_sym <- sym(x_var)
  group_var_sym <- sym(group_var)
  y_var_sym <- sym(y_var)
  
  if (is.null(y_label)) y_label <- paste(tools::toTitleCase(stat_type), y_var)
  
  # Ensure factors for correct plotting order
  data[[x_var]] <- as.factor(data[[x_var]])
  data[[group_var]] <- as.factor(data[[group_var]])
  unique_groups <- levels(data[[group_var]])
  
  # Handle colors
  if (is.null(custom_colors)) {
    custom_colors <- setNames(c("#6e3934", "#c2beb5"), unique_groups) # Fallback colors
  }
  
  # Calculate summary statistics (Mean/Median + Error Bounds)
  df_summary <- data %>%
    filter(!is.na(!!y_var_sym)) %>%
    group_by(!!x_var_sym, !!group_var_sym) %>%
    summarise(
      n = n(),
      summary_value = if (stat_type == "median") median(!!y_var_sym, na.rm = TRUE) else mean(!!y_var_sym, na.rm = TRUE),
      sd_val = sd(!!y_var_sym, na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    mutate(
      error_lower = case_when(
        error_type == "se" ~ summary_value - sd_val / sqrt(n),
        error_type == "sd" ~ summary_value - sd_val,
        error_type == "ci" ~ summary_value - qt(0.975, df = n - 1) * sd_val / sqrt(n)
      ),
      error_upper = case_when(
        error_type == "se" ~ summary_value + sd_val / sqrt(n),
        error_type == "sd" ~ summary_value + sd_val,
        error_type == "ci" ~ summary_value + qt(0.975, df = n - 1) * sd_val / sqrt(n)
      )
    )
  
  # Position dodge to prevent perfectly overlapping error bars
  pd <- position_dodge(dodge_width)
  
  # Build Plot
  g <- ggplot(df_summary, aes(x = !!x_var_sym, y = summary_value, color = !!group_var_sym, group = !!group_var_sym)) +
    geom_line(position = pd, linewidth = line_size) +
    geom_errorbar(aes(ymin = error_lower, ymax = error_upper), width = error_bar_width, position = pd, linewidth = 0.4) +
    geom_point(position = pd, size = dot_size) +
    scale_color_manual(values = custom_colors) +
    labs(x = x_label, y = y_label, color = tools::toTitleCase(group_var)) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 7),
      axis.title.x = element_text(size = 7),
      axis.title.y = element_text(size = 7),
      axis.text.x = element_text(size = 7, color = "black"),
      axis.text.y = element_text(size = 7, color = "black"),
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 7),
      legend.position = if(show_legend) "right" else "none",
      panel.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank(),
      axis.line.x = element_line(color = "black", linewidth = 0.5),
      axis.line.y = element_line(color = "black", linewidth = 0.5),
      axis.ticks = element_line(color = "black", linewidth = 0.3),
      axis.ticks.length = unit(0.15, "cm")
    )
  
  # Handle Y-axis scaling
  if (!is.null(y_step)) {
    if (is.null(y_min)) y_min <- floor(min(df_summary$error_lower, na.rm = TRUE))
    if (is.null(y_max)) y_max <- ceiling(max(df_summary$error_upper, na.rm = TRUE))
    breaks <- seq(y_min, y_max, by = y_step)
    g <- g + scale_y_continuous(limits = c(y_min, y_max), breaks = breaks, labels = function(x) sprintf("%.2f", x))
  } else if (!is.null(y_min) || !is.null(y_max)) {
    g <- g + scale_y_continuous(limits = c(y_min, y_max), labels = function(x) sprintf("%.2f", x))
  } else {
    g <- g + scale_y_continuous(labels = function(x) sprintf("%.2f", x))
  }
  
  # Save plot
  outdir <- analysis_path("figures", save_dir)
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  
  ggsave(
    file.path(outdir, sprintf("%s.pdf", name)), 
    plot = g, width = width, height = height, dpi = dpi, units = "mm", 
    bg = "white", device = ifelse("cairo_pdf" %in% names(formals(ggplot2::ggsave)), cairo_pdf, "pdf")
  )
  
  return(g)
}

# ==============================================================================
# PART 2: Execution (Behavioral & Simulation Data)
# ==============================================================================

# Define colors for the 'tree' interaction lines
interaction_colors <- c(
  "deep" = "#755374", # Grey-ish
  "wide" = "#5b705e"  # Red-ish
)

# ------------------------------------------------------------------------------
# 1. Behavioral Data Plots
# ------------------------------------------------------------------------------

df_beh <- read_csv(analysis_path("data", "processed_data", "exp3_beh.csv"), show_col_types = FALSE) %>%
  mutate(order = ifelse(order == "depth", "DFS", "BFS"))

# Quick sanity check printout
print(df_beh %>% group_by(tree_type) %>% summarise(mean_acc = mean(accuracy, na.rm = TRUE)))

# Normalized Reward Plot (Behavioral)
g_interact_beh_reward <- generate_interaction_plot(
  data = df_beh, 
  x_var = "order", group_var = "tree", y_var = "norm_reward", 
  name = "norm_reward_beh", stat_type = "mean",
  y_label = "Normalized\nReward", x_label = "Order",
  custom_colors = interaction_colors, show_legend = FALSE,
  y_min = 0.8, y_max = 0.95, y_step = 0.05, dodge_width = 0.1
)

# Accuracy Plot (Behavioral)
g_interact_beh_acc <- generate_interaction_plot(
  data = df_beh, 
  x_var = "order", group_var = "tree", y_var = "accuracy", 
  name = "accuracy_beh", stat_type = "mean",
  y_label = "\nAccuracy", x_label = "Order",
  custom_colors = interaction_colors, show_legend = FALSE,
  y_min = 0.1, y_max = 0.35, y_step = 0.05, dodge_width = 0.1
)

# ------------------------------------------------------------------------------
# 2. Simulation Data Plots
# ------------------------------------------------------------------------------

df_sim <- read_csv(analysis_path("data", "processed_data", "exp3_model.csv"), show_col_types = FALSE) %>%
  mutate(
    # Defensively ensure tree and order exist, deriving them from tree_type
    tree = ifelse(tree_type %in% c("deep_depth", "deep_breadth"), "deep", "wide"),
    order = ifelse(tree_type %in% c("deep_depth", "wide_depth"), "depth", "breadth"),
    order = ifelse(order == "depth", "DFS", "BFS")
  )

# Normalized Reward Plot (Simulation)
g_interact_sim_reward <- generate_interaction_plot(
  data = df_sim, 
  x_var = "order", group_var = "tree", y_var = "norm_reward", 
  name = "norm_reward_model", stat_type = "mean",
  y_label = "Normalized\nReward", x_label = "Order",
  custom_colors = interaction_colors, show_legend = FALSE,
  y_min = 0.8, y_max = 0.95, y_step = 0.05, dodge_width = 0.1
)

# Accuracy Plot (Simulation)
g_interact_sim_acc <- generate_interaction_plot(
  data = df_sim, 
  x_var = "order", group_var = "tree", y_var = "accuracy", 
  name = "accuracy_model", stat_type = "mean",
  y_label = "\nAccuracy", x_label = "Order",
  custom_colors = interaction_colors, show_legend = FALSE,
  y_min = 0.1, y_max = 0.35, y_step = 0.05, dodge_width = 0.1
)
