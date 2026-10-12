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
# Shared Analysis Utilities
# ==============================================================================

source(project_path("analysis", "scripts", "project_paths.R"))

analysis_path <- function(...) {
  path_parts <- list(...)
  if (length(path_parts) == 0) {
    stop("analysis_path() requires an artifact category.")
  }

  category <- as.character(path_parts[[1]])
  relative_parts <- path_parts[-1]
  if (category == "data" && length(relative_parts)) {
    folder <- as.character(relative_parts[[1]])
    if (folder %in% c("raw_data", "raw_beh_data", "processed_data")) {
      root <- if (folder == "processed_data") PROCESSED_DATA_DIR else RAW_DATA_DIR
      return(do.call(file.path, c(list(root), relative_parts[-1])))
    }
  }
  category_root <- switch(
    category,
    data = ANALYSIS_DATA_DIR,
    figures = ANALYSIS_PLOT_DIR,
    stats = ANALYSIS_SUMMARY_DIR,
    logs = ANALYSIS_LOG_DIR,
    stop(sprintf("Unknown analysis artifact category: %s", category))
  )

  do.call(file.path, c(list(category_root), relative_parts))
}

read_analysis_csv <- function(...) {
  input_path <- analysis_path(...)
  require_existing_path(input_path, "Analysis input CSV")
  readr::read_csv(input_path, show_col_types = FALSE)
}

write_analysis_csv <- function(x, ...) {
  output_path <- analysis_path(...)
  ensure_parent_directory(output_path)
  readr::write_csv(x, output_path)
}

ensure_dir <- function(path) {
  ensure_directory(path)
}

ggsave_pdf <- function(filename, plot, width, height, units = "mm", dpi = 300,
                       bg = "white", limitsize = FALSE, ...) {
  device_arg <- if ("cairo_pdf" %in% names(formals(ggplot2::ggsave))) {
    grDevices::cairo_pdf
  } else {
    "pdf"
  }

  ggplot2::ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    units = units,
    dpi = dpi,
    bg = bg,
    device = device_arg,
    limitsize = limitsize,
    ...
  )
}

convert_true_false_strings <- function(df) {
  df %>%
    dplyr::mutate(
      dplyr::across(
        where(is.character),
        ~ ifelse(.x == "True", TRUE, ifelse(.x == "False", FALSE, .x))
      )
    )
}

convert_boolean_like_columns <- function(df) {
  df %>%
    dplyr::mutate(
      dplyr::across(
        where(is.character),
        ~ if (all(toupper(stats::na.omit(.x)) %in% c("TRUE", "FALSE", "T", "F"))) {
          as.logical(toupper(.x))
        } else {
          .x
        }
      )
    )
}

print_retention_summary <- function(title, n_original, n_kept, extra_lines = character()) {
  cat(sprintf("\n--- %s ---\n", title))
  for (line in extra_lines) {
    cat(line, "\n")
  }
  cat("Original subjects: ", n_original, "\n")
  cat("Subjects kept: ", n_kept, "\n")
  cat("Subjects removed: ", n_original - n_kept, "\n")
  cat("Retention rate: ", round(n_kept / n_original * 100, 1), "%\n\n")
}

bootstrap_inverse_MAE <- function(errors, n_bootstrap = 1000) {
  bootstrap_estimates <- replicate(n_bootstrap, {
    sampled_errors <- sample(errors, size = length(errors), replace = TRUE)
    mae <- mean(abs(sampled_errors), na.rm = TRUE)
    1 / mae
  })

  list(
    mean_inverse_MAE = mean(bootstrap_estimates),
    se_inverse_MAE = stats::sd(bootstrap_estimates)
  )
}

# ==============================================================================
# Path Ranking
# ==============================================================================

calculate_path_rank_exp1_beh <- function(df) {
  df %>%
    dplyr::filter(!is.na(subject), !is.na(graph)) %>%
    dplyr::mutate(
      path1_reward = n1_reward + n2_reward,
      path2_reward = n1_reward + n3_reward,
      path3_reward = n4_reward + n5_reward,
      path4_reward = n4_reward + n6_reward
    ) %>%
    dplyr::distinct(subject, graph, node, condition, .keep_all = TRUE) %>%
    dplyr::group_by(subject, graph, condition) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      path_rewards_vector = list(c(path1_reward, path2_reward, path3_reward, path4_reward)),
      path_ranks_vector = list(rank(-path_rewards_vector, ties.method = "min")),
      rank1 = unlist(path_ranks_vector)[1],
      rank2 = unlist(path_ranks_vector)[2],
      rank3 = unlist(path_ranks_vector)[3],
      rank4 = unlist(path_ranks_vector)[4]
    ) %>%
    dplyr::ungroup() %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      p_rank = {
        node_value <- as.integer(node)
        if (is.na(node_value)) {
          NA_real_
        } else if (node_value == 2) {
          as.numeric(rank1)
        } else if (node_value == 3) {
          as.numeric(rank2)
        } else if (node_value == 5) {
          as.numeric(rank3)
        } else if (node_value == 6) {
          as.numeric(rank4)
        } else if (node_value == 1) {
          min(as.numeric(c(rank1, rank2)), na.rm = TRUE)
        } else if (node_value == 4) {
          min(as.numeric(c(rank3, rank4)), na.rm = TRUE)
        } else {
          NA_real_
        }
      }
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(is.finite(p_rank)) %>%
    dplyr::select(subject, graph, node, condition, accuracy, p_rank) %>%
    dplyr::arrange(subject, graph, node, condition)
}

# Unpooled first-stage contextual slopes for the exploratory performance analyses.
# The same outputs feed plots and the second-stage joint performance regression.
individual_context_columns <- function() {
  paste0(c("node", "sibling", "parent", "aunt"), "_reward_coefficient")
}

fit_individual_context_glms <- function(rows, include_node_quadratic = FALSE,
                                        include_sibling_quadratic = FALSE) {
  required <- c("subject", "accuracy", "actual_reward", "sibling_reward",
                "parent_reward", "aunt_reward", "is_leaf")
  stopifnot(all(required %in% names(rows)))
  rows <- as.data.frame(rows)
  rows$subject <- as.character(rows$subject)
  if (anyNA(rows$subject)) stop("Missing participant identity in contextual fits.")
  leaf <- as.character(rows$is_leaf)
  if (any(!is.na(leaf) & !leaf %in% c("TRUE", "FALSE", "1", "0"))) stop("Invalid leaf coding.")
  rows$parent_reward <- ifelse(is.na(leaf), NA_real_, ifelse(leaf %in% c("TRUE", "1"), rows$parent_reward, 0))
  rows$aunt_reward <- ifelse(is.na(leaf), NA_real_, ifelse(leaf %in% c("TRUE", "1"), rows$aunt_reward, 0))
  raw <- data.frame(node_z=rows$actual_reward,
    sibling_z=rows$sibling_reward,
    parent_z=rows$parent_reward, aunt_z=rows$aunt_reward)
  if (include_node_quadratic) raw$node_squared_z <- rows$actual_reward^2
  if (include_sibling_quadratic) raw$sibling_squared_z <- rows$sibling_reward^2
  complete <- is.finite(rows$accuracy) & apply(raw, 1, function(x) all(is.finite(x)))
  if (any(!rows$accuracy[complete] %in% c(0,1))) stop("Contextual accuracy must be binary.")
  scaling <- data.frame(term=names(raw), center=vapply(raw[complete,,drop=FALSE], mean, numeric(1)),
    sd=vapply(raw[complete,,drop=FALSE], sd, numeric(1)))
  if (any(!is.finite(scaling$sd) | scaling$sd <= 0)) stop("Invalid/constant contextual predictor across source.")
  # A common scale within experiment/source makes independently fitted slopes comparable.
  for (j in seq_along(raw)) raw[[j]] <- (raw[[j]]-scaling$center[j])/scaling$sd[j]
  formula <- reformulate(names(raw), response = "accuracy")
  output <- lapply(unique(rows$subject), function(id) {
    selected <- rows$subject == id & complete
    data <- data.frame(accuracy=rows$accuracy[selected], raw[selected,,drop=FALSE])
    warnings <- character(); fit <- NULL
    status <- if (!nrow(data)) "no_complete_observations" else if (length(unique(data$accuracy)) < 2) "constant_response" else "ok"
    if (status == "ok") {
      fit <- tryCatch(withCallingHandlers(
        stats::glm(formula, data=data, family=stats::binomial("logit"),
          control=stats::glm.control(maxit=100)),
        warning=function(w) { warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning") }),
        error=function(e) { warnings <<- c(warnings, conditionMessage(e)); NULL })
      status <- if (is.null(fit)) "fit_error" else if (!isTRUE(fit$converged)) "nonconverged" else
        if (fit$rank < ncol(raw) + 1L || any(!is.finite(coef(fit)))) "rank_deficient" else
        if (isTRUE(fit$boundary) || any(grepl("0 or 1|did not converge", warnings))) "boundary_or_separation_warning" else
        if (any(!is.finite(summary(fit)$coefficients))) "nonfinite_inference" else "ok"
    }
    terms <- names(raw)
    estimates <- if (status == "ok") unname(coef(fit)[terms]) else rep(NA_real_,length(terms))
    result <- data.frame(subject=id, context_coefficient_method=if (include_sibling_quadratic)
        "separate_binomial_glm_sibling_quadratic_v1" else if (include_node_quadratic)
        "separate_binomial_glm_node_quadratic_v1" else "separate_binomial_glm_linear_v2",
      context_fit_status=status, context_observations=nrow(data),
      context_excluded_observations=sum(rows$subject == id)-nrow(data),
      context_fit_warnings=paste(unique(warnings),collapse="; "))
    output_columns <- c(individual_context_columns(), if (include_node_quadratic) "node_reward_squared_coefficient",
      if (include_sibling_quadratic) "sibling_reward_squared_coefficient")
    result[output_columns] <- as.list(estimates)
    if (include_node_quadratic) result$node_reward_squared_se <- if (status == "ok")
      unname(summary(fit)$coefficients["node_squared_z", "Std. Error"]) else NA_real_
    for (term in c("(Intercept)", names(raw))) {
      key <- if (term == "(Intercept)") "intercept" else term
      result[[paste0("context_estimate_",key)]] <- if (!is.null(fit)) unname(coef(fit)[term]) else NA_real_
    }
    result
  })
  result <- do.call(rbind, output)
  attr(result,"context_coefficient_scaling") <- scaling
  message("Separate contextual GLMs: ", sum(result$context_fit_status == "ok"), "/", nrow(result), " accepted; no pooling.")
  result
}

context_performance_sample <- function(data, outcome) {
  needed <- c("context_coefficient_method", "context_fit_status", individual_context_columns(), outcome)
  if (!all(needed %in% names(data)) || anyNA(data$context_coefficient_method) ||
      any(data$context_coefficient_method != "separate_binomial_glm_linear_v2")) {
    stop("Outdated contextual coefficients: regenerate Exp1/Exp2 reviewer plots/metrics using separate four-linear-predictor participant GLMs.")
  }
  keep <- !is.na(data$context_fit_status) & data$context_fit_status == "ok" &
    apply(as.matrix(data[c(individual_context_columns(),outcome)]),1,function(x) all(is.finite(x)))
  data[keep,,drop=FALSE]
}

save_individual_context_audit <- function(data, directory, source) {
  dir.create(directory, recursive=TRUE, showWarnings=FALSE)
  columns <- c("subject", grep("^context_|_reward_coefficient$", names(data), value=TRUE))
  utils::write.csv(data[columns], file.path(directory,paste0(source,"_context_glm_diagnostics.csv")), row.names=FALSE)
  utils::write.csv(attr(data,"context_coefficient_scaling"),
    file.path(directory,paste0(source,"_context_glm_scaling.csv")), row.names=FALSE)
}

calculate_path_rank_exp1_model <- function(df) {
  if (!"subject" %in% names(df)) {
    df$subject <- "model_agent"
  }

  df_rewards_wide <- df %>%
    dplyr::filter(!is.na(subject), !is.na(graph), node %in% 1:6) %>%
    dplyr::mutate(node_char = as.character(node), actual_reward = as.numeric(actual_reward)) %>%
    dplyr::select(subject, graph, condition, node_char, actual_reward) %>%
    dplyr::distinct() %>%
    tidyr::pivot_wider(
      names_from = node_char,
      values_from = actual_reward,
      names_prefix = "n",
      values_fn = list(actual_reward = max)
    ) %>%
    dplyr::rename(
      n1_reward = !!rlang::sym("n1"),
      n2_reward = !!rlang::sym("n2"),
      n3_reward = !!rlang::sym("n3"),
      n4_reward = !!rlang::sym("n4"),
      n5_reward = !!rlang::sym("n5"),
      n6_reward = !!rlang::sym("n6")
    )

  df %>%
    dplyr::select(subject, graph, condition, node, accuracy) %>%
    dplyr::distinct() %>%
    dplyr::left_join(df_rewards_wide, by = c("subject", "graph", "condition")) %>%
    dplyr::filter(
      !is.na(n1_reward), !is.na(n2_reward), !is.na(n3_reward),
      !is.na(n4_reward), !is.na(n5_reward), !is.na(n6_reward)
    ) %>%
    dplyr::mutate(
      path1_reward = n1_reward + n2_reward,
      path2_reward = n1_reward + n3_reward,
      path3_reward = n4_reward + n5_reward,
      path4_reward = n4_reward + n6_reward
    ) %>%
    dplyr::group_by(subject, graph, condition) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      path_rewards_vector = list(c(path1_reward, path2_reward, path3_reward, path4_reward)),
      path_ranks_vector = list(rank(-path_rewards_vector, ties.method = "min")),
      rank1 = unlist(path_ranks_vector)[1],
      rank2 = unlist(path_ranks_vector)[2],
      rank3 = unlist(path_ranks_vector)[3],
      rank4 = unlist(path_ranks_vector)[4]
    ) %>%
    dplyr::ungroup() %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      p_rank = {
        node_value <- as.integer(node)
        if (is.na(node_value)) {
          NA_real_
        } else if (node_value == 2) {
          as.numeric(rank1)
        } else if (node_value == 3) {
          as.numeric(rank2)
        } else if (node_value == 5) {
          as.numeric(rank3)
        } else if (node_value == 6) {
          as.numeric(rank4)
        } else if (node_value == 1) {
          min(as.numeric(c(rank1, rank2)), na.rm = TRUE)
        } else if (node_value == 4) {
          min(as.numeric(c(rank3, rank4)), na.rm = TRUE)
        } else {
          NA_real_
        }
      }
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(is.finite(p_rank)) %>%
    dplyr::select(subject, graph, node, accuracy, condition, p_rank) %>%
    dplyr::arrange(graph, node, condition)
}

calculate_path_rank_exp2_beh <- function(df) {
  df %>%
    dplyr::filter(!is.na(subject), !is.na(graph)) %>%
    dplyr::mutate(
      path1_reward = n1_reward + n2_reward,
      path2_reward = n1_reward + n3_reward,
      path3_reward = n1_reward + n4_reward,
      path4_reward = n5_reward + n6_reward,
      path5_reward = n5_reward + n7_reward,
      path6_reward = n5_reward + n8_reward,
      path7_reward = n9_reward + n10_reward,
      path8_reward = n9_reward + n11_reward,
      path9_reward = n9_reward + n12_reward
    ) %>%
    dplyr::distinct(subject, graph, node, condition, .keep_all = TRUE) %>%
    dplyr::group_by(subject, graph, condition) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      path_rewards_vector = list(c(
        path1_reward, path2_reward, path3_reward, path4_reward, path5_reward,
        path6_reward, path7_reward, path8_reward, path9_reward
      )),
      path_ranks_vector = list(rank(-path_rewards_vector, ties.method = "min")),
      rank1 = unlist(path_ranks_vector)[1],
      rank2 = unlist(path_ranks_vector)[2],
      rank3 = unlist(path_ranks_vector)[3],
      rank4 = unlist(path_ranks_vector)[4],
      rank5 = unlist(path_ranks_vector)[5],
      rank6 = unlist(path_ranks_vector)[6],
      rank7 = unlist(path_ranks_vector)[7],
      rank8 = unlist(path_ranks_vector)[8],
      rank9 = unlist(path_ranks_vector)[9]
    ) %>%
    dplyr::ungroup() %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      p_rank = {
        node_value <- as.integer(node)
        if (is.na(node_value)) {
          NA_real_
        } else if (node_value == 2) {
          as.numeric(rank1)
        } else if (node_value == 3) {
          as.numeric(rank2)
        } else if (node_value == 4) {
          as.numeric(rank3)
        } else if (node_value == 6) {
          as.numeric(rank4)
        } else if (node_value == 7) {
          as.numeric(rank5)
        } else if (node_value == 8) {
          as.numeric(rank6)
        } else if (node_value == 10) {
          as.numeric(rank7)
        } else if (node_value == 11) {
          as.numeric(rank8)
        } else if (node_value == 12) {
          as.numeric(rank9)
        } else if (node_value == 1) {
          min(as.numeric(c(rank1, rank2, rank3)), na.rm = TRUE)
        } else if (node_value == 5) {
          min(as.numeric(c(rank4, rank5, rank6)), na.rm = TRUE)
        } else if (node_value == 9) {
          min(as.numeric(c(rank7, rank8, rank9)), na.rm = TRUE)
        } else {
          NA_real_
        }
      }
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(is.finite(p_rank)) %>%
    dplyr::select(subject, graph, node, accuracy, condition, p_rank) %>%
    dplyr::arrange(subject, graph, node, condition)
}

calculate_path_rank_exp2_model <- function(df) {
  if (!"subject" %in% names(df)) {
    df$subject <- "model_agent"
  }

  df_rewards_wide <- df %>%
    dplyr::filter(!is.na(subject), !is.na(graph), node %in% 1:12) %>%
    dplyr::mutate(node_char = as.character(node)) %>%
    dplyr::select(subject, graph, condition, node_char, actual_reward) %>%
    dplyr::distinct() %>%
    tidyr::pivot_wider(names_from = node_char, values_from = actual_reward, names_prefix = "n") %>%
    dplyr::rename_with(~ paste0(., "_reward"), starts_with("n"))

  df %>%
    dplyr::select(subject, graph, condition, node, accuracy) %>%
    dplyr::distinct() %>%
    dplyr::left_join(df_rewards_wide, by = c("subject", "graph", "condition")) %>%
    dplyr::filter(dplyr::if_all(matches("^n[0-9]+_reward$"), ~ !is.na(.))) %>%
    dplyr::mutate(
      path1_reward = n1_reward + n2_reward,
      path2_reward = n1_reward + n3_reward,
      path3_reward = n1_reward + n4_reward,
      path4_reward = n5_reward + n6_reward,
      path5_reward = n5_reward + n7_reward,
      path6_reward = n5_reward + n8_reward,
      path7_reward = n9_reward + n10_reward,
      path8_reward = n9_reward + n11_reward,
      path9_reward = n9_reward + n12_reward
    ) %>%
    dplyr::group_by(subject, graph, condition) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      path_rewards_vector = list(c(
        path1_reward, path2_reward, path3_reward, path4_reward, path5_reward,
        path6_reward, path7_reward, path8_reward, path9_reward
      )),
      path_ranks_vector = list(rank(-path_rewards_vector, ties.method = "min")),
      rank1 = unlist(path_ranks_vector)[1],
      rank2 = unlist(path_ranks_vector)[2],
      rank3 = unlist(path_ranks_vector)[3],
      rank4 = unlist(path_ranks_vector)[4],
      rank5 = unlist(path_ranks_vector)[5],
      rank6 = unlist(path_ranks_vector)[6],
      rank7 = unlist(path_ranks_vector)[7],
      rank8 = unlist(path_ranks_vector)[8],
      rank9 = unlist(path_ranks_vector)[9]
    ) %>%
    dplyr::ungroup() %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      p_rank = {
        node_value <- as.integer(node)
        if (is.na(node_value)) {
          NA_real_
        } else if (node_value == 2) {
          as.numeric(rank1)
        } else if (node_value == 3) {
          as.numeric(rank2)
        } else if (node_value == 4) {
          as.numeric(rank3)
        } else if (node_value == 6) {
          as.numeric(rank4)
        } else if (node_value == 7) {
          as.numeric(rank5)
        } else if (node_value == 8) {
          as.numeric(rank6)
        } else if (node_value == 10) {
          as.numeric(rank7)
        } else if (node_value == 11) {
          as.numeric(rank8)
        } else if (node_value == 12) {
          as.numeric(rank9)
        } else if (node_value == 1) {
          min(as.numeric(c(rank1, rank2, rank3)), na.rm = TRUE)
        } else if (node_value == 5) {
          min(as.numeric(c(rank4, rank5, rank6)), na.rm = TRUE)
        } else if (node_value == 9) {
          min(as.numeric(c(rank7, rank8, rank9)), na.rm = TRUE)
        } else {
          NA_real_
        }
      }
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(is.finite(p_rank)) %>%
    dplyr::select(subject, graph, node, accuracy, condition, p_rank) %>%
    dplyr::arrange(graph, node, condition)
}

calculate_path_rank_exp2_40n_beh <- function(df) {
  df_ranks_calc <- df %>%
    dplyr::filter(!is.na(subject), !is.na(graph), !is.na(node)) %>%
    dplyr::mutate(
      path1_reward = n1_reward + n2_reward + n3_reward,
      path2_reward = n1_reward + n2_reward + n4_reward,
      path3_reward = n1_reward + n2_reward + n5_reward,
      path4_reward = n1_reward + n6_reward + n7_reward,
      path5_reward = n1_reward + n6_reward + n8_reward,
      path6_reward = n1_reward + n6_reward + n9_reward,
      path7_reward = n1_reward + n10_reward + n11_reward,
      path8_reward = n1_reward + n10_reward + n12_reward,
      path9_reward = n1_reward + n10_reward + n13_reward,
      path10_reward = n14_reward + n15_reward + n16_reward,
      path11_reward = n14_reward + n15_reward + n17_reward,
      path12_reward = n14_reward + n15_reward + n18_reward,
      path13_reward = n14_reward + n19_reward + n20_reward,
      path14_reward = n14_reward + n19_reward + n21_reward,
      path15_reward = n14_reward + n19_reward + n22_reward,
      path16_reward = n14_reward + n23_reward + n24_reward,
      path17_reward = n14_reward + n23_reward + n25_reward,
      path18_reward = n14_reward + n23_reward + n26_reward,
      path19_reward = n27_reward + n28_reward + n29_reward,
      path20_reward = n27_reward + n28_reward + n30_reward,
      path21_reward = n27_reward + n28_reward + n31_reward,
      path22_reward = n27_reward + n32_reward + n33_reward,
      path23_reward = n27_reward + n32_reward + n34_reward,
      path24_reward = n27_reward + n32_reward + n35_reward,
      path25_reward = n27_reward + n36_reward + n37_reward,
      path26_reward = n27_reward + n36_reward + n38_reward,
      path27_reward = n27_reward + n36_reward + n39_reward
    ) %>%
    dplyr::group_by(subject, graph, condition) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      path_rewards_vector = list(c(
        path1_reward, path2_reward, path3_reward, path4_reward, path5_reward,
        path6_reward, path7_reward, path8_reward, path9_reward, path10_reward,
        path11_reward, path12_reward, path13_reward, path14_reward,
        path15_reward, path16_reward, path17_reward, path18_reward,
        path19_reward, path20_reward, path21_reward, path22_reward,
        path23_reward, path24_reward, path25_reward, path26_reward,
        path27_reward
      )),
      path_ranks_vector = list(rank(-path_rewards_vector, na.last = "keep", ties.method = "min"))
    ) %>%
    dplyr::ungroup()

  node_path_map <- list(
    "1" = 1:9, "2" = 1:3, "3" = 1, "4" = 2, "5" = 3,
    "6" = 4:6, "7" = 4, "8" = 5, "9" = 6,
    "10" = 7:9, "11" = 7, "12" = 8, "13" = 9,
    "14" = 10:18, "15" = 10:12, "16" = 10, "17" = 11,
    "18" = 12, "19" = 13:15, "20" = 13, "21" = 14,
    "22" = 15, "23" = 16:18, "24" = 16, "25" = 17,
    "26" = 18, "27" = 19:27, "28" = 19:21, "29" = 19,
    "30" = 20, "31" = 21, "32" = 22:24, "33" = 22,
    "34" = 23, "35" = 24, "36" = 25:27, "37" = 25,
    "38" = 26, "39" = 27
  )

  df_ranks_calc %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      relevant_paths = list(node_path_map[[as.character(as.integer(node))]]),
      p_rank = {
        rank_values <- unlist(path_ranks_vector)[relevant_paths]
        rank_values <- rank_values[!is.na(rank_values)]

        if (length(rank_values) == 0) {
          NA_real_
        } else {
          min(rank_values)
        }
      }
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(is.finite(p_rank)) %>%
    dplyr::select(subject, graph, node, condition, accuracy, p_rank) %>%
    dplyr::arrange(subject, graph, node, condition)
}
