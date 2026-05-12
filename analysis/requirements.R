packages <- c(
  "broom.mixed",
  "dplyr",
  "formula.tools",
  "ggplot2",
  "glue",
  "here",
  "lme4",
  "lmerTest",
  "purrr",
  "readr",
  "rlang",
  "scales",
  "stringr",
  "tidyr"
)

missing_packages <- packages[!packages %in% rownames(installed.packages())]

if (length(missing_packages) > 0) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}

invisible(lapply(packages, require, character.only = TRUE))
