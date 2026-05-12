# ==============================================================================
# Script: Rate-Distortion Bounds Plotting (2-Node and 6-Node)
# Description: Calculates empirical regret and plots it against mutual 
#              information costs, including theoretical optimal bounds for 2n.
# ==============================================================================

library(readr)
library(dplyr)
library(ggplot2)
library(here)

source(here::here("analysis", "scripts", "utils.R"))

# -------------------------------------------------------------------
# Shared Plotting Aesthetics
# -------------------------------------------------------------------

# Define a shared theme to keep the code DRY and plots perfectly consistent
rd_base_theme <- theme_minimal() +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 7),
    axis.title.y = element_text(size = 7),
    axis.text.x  = element_text(size = 7),
    axis.text.y  = element_text(size = 7),
    aspect.ratio = 1,
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line.x  = element_line(color = "black", linewidth = 0.5),
    axis.line.y  = element_line(color = "black", linewidth = 0.5),
    axis.ticks   = element_line(color = "black", linewidth = 0.3)
  )

# -------------------------------------------------------------------
# THEORETICAL SOLUTION GENERATION (2-Node)
# -------------------------------------------------------------------

# Binary entropy function
h2 <- function(theta) {
  ifelse(theta == 0 | theta == 1, 0, -theta * log2(theta) - (1 - theta) * log2(1 - theta))
}

theta_vals <- seq(0.5, 1.0, length.out = 100)
theoretical_MI_raw <- 1 - h2(theta_vals)
theoretical_V_raw <- 0.25 + 0.5 * theta_vals

optimal_df <- tibble(
  opt_MI = theoretical_MI_raw,
  opt_V  = (theoretical_V_raw - 0.5) / 0.25,
  opt_regret = 1 - ((theoretical_V_raw - 0.5) / 0.25)
)

# ==============================================================================
# PLOT 1: 2-Node Network
# ==============================================================================

# Load and prepare data
df_2n <- read_csv(analysis_path("data", "raw_data", "rd_2n_model.csv"), show_col_types = FALSE) %>%
  mutate(avg_regret = 1 - avg_V) %>%
  filter(avg_MI_cost <= 1)

# Generate Plot
g_2n <- ggplot() +
  geom_errorbar(
    data = df_2n, 
    aes(x = avg_MI_cost, ymin = avg_regret - se_V, ymax = avg_regret + se_V), 
    color = "#b59994", alpha = 0.5, width = 0, linewidth = 0.3
  ) +
  geom_point(
    data = df_2n, 
    aes(x = avg_MI_cost, y = avg_regret), 
    color = "#b59994", alpha = 0.8
  ) +
  geom_line(
    data = optimal_df, 
    aes(x = opt_MI, y = opt_regret), 
    linetype = "dashed", color = "black", linewidth = 0.8
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mutual Information", y = "Normalized Regret") +
  rd_base_theme

# Save Plot
outdir <- analysis_path("figures", "rate_distortion")
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

ggsave(
  file.path(outdir, "rd_2n.pdf"), plot = g_2n,
  width = 50, height = 40, units = "mm", dpi = 300, 
  bg = "white", device = ifelse("cairo_pdf" %in% names(formals(ggplot2::ggsave)), cairo_pdf, "pdf"), 
  scale = 1, limitsize = FALSE
)


# ==============================================================================
# PLOT 2: 6-Node Network
# ==============================================================================

# Load and prepare data
df_6n <- read_csv(analysis_path("data", "raw_data", "rd_6n_model.csv"), show_col_types = FALSE) %>%
  mutate(avg_regret = 1 - avg_V)

# Generate Plot
g_6n <- ggplot(df_6n, aes(x = avg_MI_cost, y = avg_regret)) +
  geom_errorbar(
    aes(ymin = avg_regret - se_V, ymax = avg_regret + se_V), 
    color = "#b59994", alpha = 0.5, width = 0, linewidth = 0.3
  ) +
  geom_point(color = "#b59994", alpha = 0.8) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(x = "Mutual Information", y = "Normalized Regret") +
  rd_base_theme

# Save Plot
ggsave(
  file.path(outdir, "rd_6n.pdf"), plot = g_6n,
  width = 50, height = 40, units = "mm", dpi = 300, 
  bg = "white", device = ifelse("cairo_pdf" %in% names(formals(ggplot2::ggsave)), cairo_pdf, "pdf"), 
  scale = 1, limitsize = FALSE
)
