# ==============================================================================
# Shared Analysis Utilities
# ==============================================================================

analysis_path <- function(...) {
  here::here("analysis", ...)
}

read_analysis_csv <- function(...) {
  readr::read_csv(analysis_path(...), show_col_types = FALSE)
}

write_analysis_csv <- function(x, ...) {
  readr::write_csv(x, analysis_path(...))
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }
  invisible(path)
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
