library(here)

figure_scripts <- c(
  "rate_distortion.R",
  "exp3.R"
)

for (script in figure_scripts) {
  message("Running ", script)
  source(here::here("analysis", "scripts", "figures", script))
}

sink("analysis/sessionInfo.txt")
sessionInfo()
sink()