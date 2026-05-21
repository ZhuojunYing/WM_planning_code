# ==============================================================================
# Script: Statistical Analysis & Modeling Exports
# Description: Generates descriptive statistics, runs regressions and ANOVAs, 
#              and exports APA-formatted results to LaTeX (.tex) files.
# ==============================================================================

library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(glue)
library(here)
library(lme4)
library(lmerTest)
library(broom.mixed)
library(formula.tools)

source(here::here("analysis", "scripts", "utils.R"))

# ==============================================================================
# Statistical Modeling Functions
# ==============================================================================

is.binary <- function(x) { all(na.omit(x) %in% 0:1) }
zscore <- function(x) { as.numeric(scale(x, center = TRUE, scale = TRUE)) }

sprintf_transformer <- function(text, envir) {
  m <- regexpr(":.+$", text)
  if (m != -1) {
    format <- substring(regmatches(text, m), 2)
    regmatches(text, m) <- ""
    res <- eval(parse(text = text, keep.source = FALSE), envir)
    do.call(sprintf, list(glue("%{format}f"), res))
  } else {
    eval(parse(text = text, keep.source = FALSE), envir)
  }
}

fmt <- function(..., .envir = parent.frame()) { glue(..., .transformer = sprintf_transformer, .envir = .envir) }

pval <- function(p) {
  if (is.na(p)) return("NA")
  if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3), nsmall=3), 2)}")
}

format_number <- function(x, digits = 3) {
  if (is.na(x)) return("NA")
  sprintf(paste0("%.", digits, "f"), x)
}

format_ci <- function(low, high, digits = 3) {
  if (is.na(low) || is.na(high)) return("NA")
  sprintf(
    "[%s, %s]",
    format_number(low, digits),
    format_number(high, digits)
  )
}

partial_eta_squared <- function(f_value, df_effect, df_error) {
  if (any(is.na(c(f_value, df_effect, df_error)))) return(NA_real_)
  (f_value * df_effect) / ((f_value * df_effect) + df_error)
}

# ------------------------------------------------------------------------------
# Core Regression Wrapper
# ------------------------------------------------------------------------------
regress <- function(data, form, logistic = FALSE, mixed = TRUE, add_random = FALSE,
                    intercept = TRUE, standardize = FALSE, name = "", data_type = "participants",
                    reference_level = NULL, reference_levels = NULL,
                    random_effects = c("full", "intercept")) {
  
  preds <- paste(get.vars(rhs(form)), collapse = " + ")
  data <- as_tibble(data) 
  random_effects <- match.arg(random_effects)
  
  # Set reference level for column "condition"
  if (!is.null(reference_level) && "condition" %in% names(data)) {
    data$condition <- relevel(as.factor(data$condition), ref = reference_level)
  }
  
  if (!is.null(reference_levels)) {
    if (is.vector(reference_levels) && !is.list(reference_levels)) reference_levels <- as.list(reference_levels)
    for (col in names(reference_levels)) {
      if (col %in% names(data)) {
        ref_val <- reference_levels[[col]]
        data[[col]] <- as.factor(data[[col]])
        if (!ref_val %in% levels(data[[col]])) {
          warning(sprintf("Reference level '%s' not found in column '%s'", ref_val, col))
        } else {
          data[[col]] <- stats::relevel(data[[col]], ref = ref_val)
        }
      }
    }
  }
  
  if (standardize) {
    for (k in get.vars(form)) {
      if (k %in% names(data) && is.numeric(data[[k]]) && !is.binary(data[[k]])) {
        data[[k]] <- zscore(data[[k]])
      }
    }
  }
  
  model <- if (mixed) {
    form_parts <- as.character(form)
    lhs_str <- form_parts[2]
    rhs_str <- form_parts[3]
    random_term <- if (random_effects == "full") {
      random_rhs_str <- rhs_str
      rhs_vars <- setdiff(all.vars(stats::as.formula(glue("~ {rhs_str}"))), "subject")
      for (var_name in rhs_vars) {
        if (var_name %in% names(data) && (is.factor(data[[var_name]]) || is.character(data[[var_name]]))) {
          data[[var_name]] <- as.factor(data[[var_name]])
          if (nlevels(data[[var_name]]) != 2) {
            stop(glue("Full random-effects expansion with || currently supports binary factors only: {var_name}"))
          }
          contrast_name <- paste0(".re_", var_name)
          data[[contrast_name]] <- as.numeric(data[[var_name]]) - 1
          random_rhs_str <- gsub(
            glue("(?<![[:alnum:]_.]){var_name}(?![[:alnum:]_.])"),
            contrast_name,
            random_rhs_str,
            perl = TRUE
          )
        }
      }
      glue("(({random_rhs_str}) || subject)")
    } else {
      "(1 | subject)"
    }
    mixed_form <- as.formula(glue("{lhs_str} ~ {rhs_str} + {random_term}"))
    if (logistic) {
      glmer(
        mixed_form,
        family = binomial,
        data = data,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )
    } else {
      lmer(mixed_form, data = data, REML = FALSE)
    }
  } else {
    if (logistic) glm(form, family = binomial, data = data) else lm(form, data = data)
  }
  
  print(summary(model))
  cat("regress data type:", data_type, "\n")
  if (name != "") write_model_separate(model = model, name = name, data_type = data_type)
  return(model)
}

# ------------------------------------------------------------------------------
# Export Model Summaries
# ------------------------------------------------------------------------------
write_model_separate <- function(model, name, data_type = "participants") {
  coef_data <- broom.mixed::tidy(model, conf.int = TRUE)
  
  if ("effect" %in% names(coef_data)) {
    coef_data <- coef_data %>% filter(effect == "fixed", term != "(Intercept)")
  } else {
    coef_data <- coef_data %>% filter(term != "(Intercept)")
  }

  if (!"df" %in% names(coef_data)) {
    coef_data$df <- stats::df.residual(model)
  }
  if (!"conf.low" %in% names(coef_data)) {
    coef_data$conf.low <- NA_real_
  }
  if (!"conf.high" %in% names(coef_data)) {
    coef_data$conf.high <- NA_real_
  }
  
  base_dir <- analysis_path("stats", name)
  dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
  
  format_p <- function(p) {
    if (is.na(p)) return("NA")
    if (p < 0.001) return("< .001")
    return(sub("^0", "", sprintf("%.3f", p))) 
  }
  
  for (i in seq_len(nrow(coef_data))) {
    row <- coef_data[i, ]
    term_name <- row$term
    
    beta_tex <- sprintf("%.3f", row$estimate)
    p_tex <- format_p(row$p.value)
    df_tex <- if ("df" %in% names(row) && !is.na(row$df)) sprintf("%d", round(row$df)) else "NA"
    ci_low_tex <- format_number(row$conf.low)
    ci_high_tex <- format_number(row$conf.high)
    ci_tex <- format_ci(row$conf.low, row$conf.high)
    
    beta_file <- file.path(base_dir, paste0(term_name, "_beta.tex"))
    p_file    <- file.path(base_dir, paste0(term_name, "_p.tex"))
    df_file   <- file.path(base_dir, paste0(term_name, "_df.tex"))
    ci_low_file <- file.path(base_dir, paste0(term_name, "_ci_low.tex"))
    ci_high_file <- file.path(base_dir, paste0(term_name, "_ci_high.tex"))
    ci_file <- file.path(base_dir, paste0(term_name, "_ci.tex"))
    
    writeLines(beta_tex, beta_file)
    writeLines(p_tex, p_file)
    writeLines(df_tex, df_file)
    writeLines(ci_low_tex, ci_low_file)
    writeLines(ci_high_tex, ci_high_file)
    writeLines(ci_tex, ci_file)
  }
}

write_model <- function(model, name, data_type = "participants") {
  coef_data <- broom.mixed::tidy(model, conf.int = TRUE)
  
  if ("effect" %in% names(coef_data)) {
    coef_data <- coef_data %>% dplyr::filter(effect == "fixed", term != "(Intercept)")
  } else {
    coef_data <- coef_data %>% dplyr::filter(term != "(Intercept)")
  }
  
  if (!"df" %in% names(coef_data)) {
    coef_data$df <- stats::df.residual(model)
  }
  if (!"conf.low" %in% names(coef_data)) {
    coef_data$conf.low <- NA_real_
  }
  if (!"conf.high" %in% names(coef_data)) {
    coef_data$conf.high <- NA_real_
  }
  
  base_dir <- analysis_path("stats", name)
  ensure_dir(base_dir)
  
  format_p <- function(p) {
    if (is.na(p)) return("$p = NA$")
    if (p < 0.001) return("$p < .001$")
    sprintf("$p = %s$", sub("^0", "", sprintf("%.3f", p)))
  }
  
  format_df <- function(df) {
    if (is.na(df)) return("$\\text{df} = NA$")
    sprintf("$\\text{df} = %d$", round(df))
  }
  
  for (i in seq_len(nrow(coef_data))) {
    row <- coef_data[i, ]
    df_value <- if ("df" %in% names(row)) row$df else NA_real_
    tex_output <- sprintf(
      "$\\beta = %.3f$, 95\\%% CI %s, %s, %s",
      row$estimate,
      format_ci(row$conf.low, row$conf.high),
      format_p(row$p.value),
      format_df(df_value)
    )
    
    writeLines(tex_output, file.path(base_dir, paste0(row$term, ".tex")))
    writeLines(format_number(row$conf.low), file.path(base_dir, paste0(row$term, "_ci_low.tex")))
    writeLines(format_number(row$conf.high), file.path(base_dir, paste0(row$term, "_ci_high.tex")))
    writeLines(format_ci(row$conf.low, row$conf.high), file.path(base_dir, paste0(row$term, "_ci.tex")))
  }
  
  invisible(coef_data)
}

# ------------------------------------------------------------------------------
# Export Descriptive Stats
# ------------------------------------------------------------------------------
write_grouped_descriptive_stats <- function(data, variable, group_var = NULL, folder_name, data_type = "participants", format_digits = 3) {
  data <- as_tibble(data)
  if (!variable %in% names(data)) stop(paste("Variable", variable, "not found in data"))
  if (!is.null(group_var) && !group_var %in% names(data)) stop(paste("Group variable", group_var, "not found in data"))
  
  base_path <- analysis_path("stats",folder_name)
  dir.create(base_path, recursive = TRUE, showWarnings = FALSE)
  
  if (is.null(group_var)) {
    stats <- data %>%
      summarise(mean_val = mean(.data[[variable]], na.rm = TRUE), sd_val = sd(.data[[variable]], na.rm = TRUE),
                n_val = sum(!is.na(.data[[variable]])), median_val = median(.data[[variable]], na.rm = TRUE),
                min_val = min(.data[[variable]], na.rm = TRUE), max_val = max(.data[[variable]], na.rm = TRUE), .groups = "drop")
    
    format_str <- paste0("%.", format_digits, "f")
    writeLines(sprintf(paste0("$M = ", format_str, "$"), stats$mean_val), file.path(base_path, paste0(variable, "_mean.tex")))
    writeLines(sprintf(paste0("$SD = ", format_str, "$"), stats$sd_val), file.path(base_path, paste0(variable, "_sd.tex")))
  } else {
    stats <- data %>%
      group_by(.data[[group_var]]) %>%
      summarise(mean_val = mean(.data[[variable]], na.rm = TRUE), sd_val = sd(.data[[variable]], na.rm = TRUE),
                n_val = sum(!is.na(.data[[variable]])), median_val = median(.data[[variable]], na.rm = TRUE),
                min_val = min(.data[[variable]], na.rm = TRUE), max_val = max(.data[[variable]], na.rm = TRUE), .groups = "drop")
    
    for (i in 1:nrow(stats)) {
      group_name <- stats[[group_var]][i]
      group_stats <- stats[i, ]
      format_str <- paste0("%.", format_digits, "f")
      
      writeLines(sprintf(paste0("$M = ", format_str, "$"), group_stats$mean_val), file.path(base_path, paste0(variable, "_", group_name, "_mean.tex")))
      writeLines(sprintf(paste0("$SD = ", format_str, "$"), group_stats$sd_val), file.path(base_path, paste0(variable, "_", group_name, "_sd.tex")))
    }
  }
  invisible(stats)
}

# ------------------------------------------------------------------------------
# Compare Adjacent Path Ranks (ANOVA)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Compare Adjacent Path Ranks (ANOVA)
# ------------------------------------------------------------------------------
compare_adjacent_path_ranks <- function(data, y_var = "accuracy", folder_name, mixed = TRUE, data_type = "participants") {
  base_dir <- analysis_path("stats", folder_name, "")
  dir.create(base_dir, recursive = TRUE, showWarnings = FALSE)
  
  # NEW: Format exact p-values to 3 decimal places (APA style)
  format_p <- function(p) {
    if (is.na(p)) return("NA")
    if (p < 0.001) return("< .001")
    return(sub("^0", "", sprintf("%.3f", p))) 
  }
  
  if (!"path_rank" %in% names(data)) stop("Column 'path_rank' not found in data.")
  if (!y_var %in% names(data)) stop(sprintf("Column '%s' not found in data.", y_var))
  
  ranks <- sort(unique(data$path_rank))
  cat(sprintf("\n=== ANOVA ADJACENT RANKS: %s (%s) ===\n", folder_name, y_var))
  
  for (i in 1:(length(ranks) - 1)) {
    rank_A <- ranks[i]
    rank_B <- ranks[i+1]
    sub_data <- data %>% filter(path_rank %in% c(rank_A, rank_B))
    
    if (nrow(sub_data) < 5) next
    
    form_str <- paste0(y_var, " ~ as.factor(path_rank)")
    
    if (mixed) {
      model <- tryCatch({ lmer(as.formula(paste0(form_str, " + (1|subject)")), data = sub_data, REML = FALSE) }, error = function(e) NULL)
    } else {
      model <- tryCatch({ lm(as.formula(form_str), data = sub_data) }, error = function(e) NULL)
    }
    
    if (is.null(model)) { cat(sprintf("Model failed to converge for rank %d vs %d\n", rank_A, rank_B)); next }
    
    anova_res <- anova(model)
    
    if (mixed) {
      F_val <- anova_res$"F value"[1]; p_val <- anova_res$"Pr(>F)"[1]
      numDF <- anova_res$"NumDF"[1]; denDF <- anova_res$"DenDF"[1]
    } else {
      F_val <- anova_res$"F value"[1]; p_val <- anova_res$"Pr(>F)"[1]
      numDF <- anova_res$"Df"[1]; denDF <- anova_res$"Df"[2]
    }
    
    prefix <- paste0(y_var, "_rank_", rank_A, "_vs_", rank_B)
    
    # Format the isolated p-value and the full string
    p_exact_tex <- format_p(p_val)
    partial_eta_sq <- partial_eta_squared(F_val, numDF, denDF)
    partial_eta_tex <- format_number(partial_eta_sq)
    p_str <- if(is.na(p_val)) "ns" else if(p_val < .001) "p < .001" else sprintf("p = %s", p_exact_tex)
    full_stat_tex <- sprintf(
      "$F(%.0f, %.0f) = %.3f$, $\\eta_p^2 = %s$, %s",
      numDF,
      denDF,
      F_val,
      partial_eta_tex,
      p_str
    )
    
    # Write files
    writeLines(sprintf("%.3f", F_val), file.path(base_dir, paste0(prefix, "_F.tex")))
    writeLines(p_exact_tex, file.path(base_dir, paste0(prefix, "_p.tex"))) # Changed from _p_stars.tex
    writeLines(partial_eta_tex, file.path(base_dir, paste0(prefix, "_partial_eta_sq.tex")))
    writeLines(full_stat_tex, file.path(base_dir, paste0(prefix, "_full_stat.tex")))
    
    cat(sprintf("  %d vs %d -> %s\n", rank_A, rank_B, full_stat_tex))
  }
}
# ==============================================================================
# PART 3: Execution Block
# ==============================================================================

# Load Data
df_exp1_beh <- read_csv(analysis_path("data", "processed_data", "exp1_beh.csv"), show_col_types = FALSE)
df_exp2_beh <- read_csv(analysis_path("data", "processed_data", "exp2_beh.csv"), show_col_types = FALSE)
df_exp1_model <- read_csv(analysis_path("data", "processed_data", "exp1_model.csv"), show_col_types = FALSE)
df_exp2_model <- read_csv(analysis_path("data", "processed_data", "exp2_model.csv"), show_col_types = FALSE)
df_exp2_40n_beh <- read_csv(analysis_path("data", "processed_data", "exp2_40n_beh.csv"), show_col_types = FALSE)
df_exp3_beh <- read_csv(analysis_path("data", "processed_data", "exp3_beh.csv"), show_col_types = FALSE)
df_exp3_model <- read_csv(analysis_path("data", "processed_data", "exp3_model.csv"), show_col_types = FALSE)
df_exp1_model_preregistered <- read_csv(analysis_path("data", "processed_data", "exp1_model_preregistered.csv"), show_col_types = FALSE)

# 1. Write Descriptive Stats
write_grouped_descriptive_stats(df_exp1_model, "accuracy", "condition", "exp1_model")
write_grouped_descriptive_stats(df_exp1_beh, "accuracy", "condition", "exp1_beh")
write_grouped_descriptive_stats(df_exp2_model, "accuracy", "condition", "exp2_model")
write_grouped_descriptive_stats(df_exp2_beh, "accuracy", "condition", "exp2_beh")
write_grouped_descriptive_stats(df_exp2_40n_beh, "accuracy", "condition", "exp2_40n_beh")

write_grouped_descriptive_stats(df_exp1_model %>% filter(condition == "planning"), "norm_reward", folder_name = "exp1_model")
write_grouped_descriptive_stats(df_exp1_beh %>% filter(condition == "planning"), "norm_reward",  folder_name = "exp1_beh")
write_grouped_descriptive_stats(df_exp2_model %>% filter(condition == "planning"), "norm_reward", folder_name =  "exp2_model")
write_grouped_descriptive_stats(df_exp2_beh %>% filter(condition == "planning"), "norm_reward",  folder_name ="exp2_beh")
write_grouped_descriptive_stats(df_exp2_40n_beh %>% filter(condition == "planning"), "norm_reward",  folder_name ="exp2_40n_beh")

# Prep relational reward columns for regression (set non-leaves to 0)
prep_for_regression <- function(df) {
  df %>% mutate(node_reward = actual_reward, parent_reward = ifelse(!is_leaf, 0, parent_reward), aunt_reward = ifelse(!is_leaf, 0, aunt_reward))
}


prep_for_regression_40n <- function(df) {
  df %>% mutate(node_reward = actual_reward, preceding_sum_reward = ifelse(!is_leaf, 0, preceding_sum_reward), aunt_reward = ifelse(!is_leaf, 0, aunt_reward))
}

fit_behavioral_recall_model <- function(data) {
  data <- data %>%
    mutate(condition = stats::relevel(as.factor(condition), ref = "planning"))

  lmer(
    accuracy ~
      (
        scale(node_reward) +
          scale(node_reward^2) +
          scale(sibling_reward) +
          scale(sibling_reward^2) +
          scale(aunt_reward) +
          scale(parent_reward)
      ) * condition +
      (
        (
          scale(node_reward) +
            scale(node_reward^2) +
            scale(sibling_reward) +
            scale(sibling_reward^2) +
            scale(aunt_reward) +
            scale(parent_reward)
        ) * condition || subject
      ),
    data = data,
    REML = FALSE
  )
}

fit_behavioral_recall_model_40n <- function(data) {
  data <- data %>%
    mutate(condition = stats::relevel(as.factor(condition), ref = "planning"))

  lmer(
    accuracy ~
      (
        scale(node_reward) +
          scale(node_reward^2) +
          scale(sibling_reward) +
          scale(sibling_reward^2) +
          scale(aunt_reward) +
          scale(preceding_sum_reward)
      ) * condition +
      (
        (
          scale(node_reward) +
            scale(node_reward^2) +
            scale(sibling_reward) +
            scale(sibling_reward^2) +
            scale(aunt_reward) +
            scale(preceding_sum_reward)
        ) * condition || subject
      ),
    data = data,
    REML = FALSE
  )
}

fit_exp3_behavioral_action_model <- function(data) {
  data <- data %>%
    mutate(
      order = stats::relevel(as.factor(order), ref = "breadth"),
      tree = stats::relevel(as.factor(tree), ref = "wide")
    )

  lmer(
    norm_reward ~ order * tree + (order * tree || subject),
    data = data,
    REML = FALSE
  )
}

fit_exp3_behavioral_recall_model <- function(data) {
  data <- data %>%
    mutate(
      order = stats::relevel(as.factor(order), ref = "breadth"),
      tree = stats::relevel(as.factor(tree), ref = "wide")
    )

  lmer(
    accuracy ~ order * tree + (order * tree || subject),
    data = data,
    REML = FALSE
  )
}

fit_behavioral_path_rank_model <- function(data) {
  glmer(
    accuracy ~ path_rank + (path_rank || subject),
    family = binomial,
    data = data,
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
}


df_exp1_model <- prep_for_regression(df_exp1_model)
df_exp1_beh <- prep_for_regression(df_exp1_beh)
df_exp2_model <- prep_for_regression(df_exp2_model)
df_exp2_beh <- prep_for_regression(df_exp2_beh)
df_exp2_40n_beh <- prep_for_regression_40n(df_exp2_40n_beh)

# 2. Run Main Regressions
reg_formula <- accuracy ~ (scale(node_reward) + scale(node_reward^2) + scale(sibling_reward) + scale(sibling_reward^2) + scale(aunt_reward) + scale(parent_reward)) * condition
reg_formula_40n <- accuracy ~ (scale(node_reward) + scale(node_reward^2) + scale(sibling_reward) + scale(sibling_reward^2) + scale(aunt_reward) + scale(preceding_sum_reward)) * condition

df_exp1_model %>% regress(reg_formula, reference_level = "planning", add_random = FALSE, mixed = FALSE, logistic = TRUE) %>% write_model_separate("exp1_model") 
fit_behavioral_recall_model(df_exp1_beh  ) %>% write_model_separate("exp1_beh") 
df_exp2_model %>% regress(reg_formula, reference_level = "planning", add_random = FALSE, mixed = FALSE, logistic = TRUE) %>% write_model_separate("exp2_model") 
fit_behavioral_recall_model(df_exp2_beh) %>% write_model_separate("exp2_beh") 

fit_behavioral_recall_model_40n(df_exp2_40n_beh) %>% write_model_separate("exp2_40n_beh") 

# Exp3 Regressions
fit_exp3_behavioral_action_model(df_exp3_beh) %>% write_model_separate("exp3_beh_action")
df_exp3_model %>% regress(norm_reward ~ order * tree, reference_levels = c("order" = "breadth", "tree" = "wide"), add_random = FALSE, mixed = FALSE) %>% write_model_separate("exp3_model_action")
fit_exp3_behavioral_recall_model(df_exp3_beh) %>% write_model_separate("exp3_beh_recall")
df_exp3_model %>% regress(accuracy ~ order * tree, reference_levels = c("order" = "breadth", "tree" = "wide"), add_random = FALSE, mixed = FALSE) %>% write_model_separate("exp3_model_recall")

# 3. Path Rank Calculations & Comparisons
df_exp1_beh_rank <- calculate_path_rank_exp1_beh(df_exp1_beh) %>% mutate(path_rank = p_rank)
df_exp1_model_rank <- calculate_path_rank_exp1_model(df_exp1_model) %>% mutate(path_rank = p_rank)
df_exp2_beh_rank <- calculate_path_rank_exp2_beh(df_exp2_beh) %>% mutate(path_rank = p_rank)
df_exp2_model_rank <- calculate_path_rank_exp2_model(df_exp2_model) %>% mutate(path_rank = p_rank)
df_exp2_40n_beh_rank <- calculate_path_rank_exp2_40n_beh(df_exp2_40n_beh) %>% mutate(path_rank = p_rank)


df_exp1_model_rank %>% filter(condition == "planning") %>% regress(accuracy ~ path_rank, add_random = FALSE, mixed = FALSE, logistic = TRUE) %>% write_model("exp1_model_planning")  
fit_behavioral_path_rank_model(df_exp1_beh_rank %>% filter(condition == "planning")) %>% write_model("exp1_beh_planning")
df_exp2_model_rank %>% filter(condition == "planning") %>% regress(accuracy ~ path_rank, add_random = FALSE, mixed = FALSE, logistic = TRUE) %>% write_model("exp2_model_planning")  
fit_behavioral_path_rank_model(df_exp2_beh_rank %>% filter(condition == "planning")) %>% write_model("exp2_beh_planning")
fit_behavioral_path_rank_model(df_exp2_40n_beh_rank %>% filter(condition == "planning")) %>% write_model("exp2_40n_beh_planning")

# Compare Adjacent Ranks
compare_adjacent_path_ranks(df_exp1_beh_rank %>% filter(condition == "planning"), y_var = "accuracy", folder_name = "exp1_beh_planning", mixed = TRUE, data_type = "participants")
compare_adjacent_path_ranks(df_exp1_model_rank %>% filter(condition == "planning"), y_var = "accuracy", folder_name = "exp1_model_planning", mixed = FALSE, data_type = "model")

compare_adjacent_path_ranks(df_exp2_beh_rank %>% filter(condition == "planning"), y_var = "accuracy", folder_name = "exp2_beh_planning", mixed = TRUE, data_type = "participants")
compare_adjacent_path_ranks(df_exp2_model_rank %>% filter(condition == "planning"), y_var = "accuracy", folder_name = "exp2_model_planning", mixed = FALSE, data_type = "model")

compare_adjacent_path_ranks(df_exp2_40n_beh_rank %>% filter(condition == "planning"), y_var = "accuracy", folder_name = "exp2_40n_beh_planning", mixed = TRUE, data_type = "participants")
 

# Exp1 S2 Sim Specific Analysis
write_grouped_descriptive_stats(df_exp1_model_preregistered, "accuracy", "condition", "exp1_model_preregistered")
write_grouped_descriptive_stats(df_exp1_model_preregistered %>% filter(condition == "planning"), "norm_reward", folder_name = "exp1_model_preregistered")


 
df_exp1_model_preregistered <- prep_for_regression(df_exp1_model_preregistered)
df_exp1_model_rank_preregistered <- calculate_path_rank_exp1_model(df_exp1_model_preregistered) %>% mutate(path_rank = p_rank)

df_exp1_model_rank_preregistered %>% filter(condition == "planning") %>% regress(accuracy ~ path_rank, add_random = FALSE, mixed = FALSE, logistic = TRUE) %>% write_model("exp1_model_preregistered_planning")  

# compare_adjacent_path_ranks(df_exp1_model_rank_preregistered %>% filter(condition == "planning"), y_var = "accuracy", folder_name = "exp1_model_preregistered_planning", mixed = FALSE, data_type = "model")
df_exp1_model_preregistered %>% regress(reg_formula, reference_level = "planning", add_random = FALSE, mixed = FALSE, logistic = TRUE) %>% write_model_separate("exp1_model_preregistered") 


sink("analysis/sessionInfo.txt")
sessionInfo()
sink()
