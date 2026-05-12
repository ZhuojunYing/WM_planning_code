library(here)

preprocessing_scripts <- c(
  "exp1_beh.R",
  "exp1_model.R",
  "exp1_model_preregistered.R",
  "exp2_beh.R",
  "exp2_model.R",
  "exp2_40n_beh.R",
  "exp3_beh.R",
  "exp3_model.R"
)

for (script in preprocessing_scripts) {
  message("Running ", script)
  source(here::here("analysis", "preprocessing", script))
}
