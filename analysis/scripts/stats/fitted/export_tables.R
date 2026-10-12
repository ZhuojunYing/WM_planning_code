#!/usr/bin/env Rscript
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

# Base R only. Read completed statistics; never source common.R or refit models.
table_groups <- c(participants = "Participants", sequential = "Sequential model",
  simultaneous = "Simultaneous model", independent = "Independent model")
table_inference_note <- paste0("Coefficients ($b$) are unstandardized log-odds from binomial-logit glmer fits (Laplace ML), with 95\\% normal-Wald confidence intervals and unadjusted Wald z-test $p$ values. ",
  "Degrees of freedom are not applicable (---). Continuous predictors were standardized where specified in the fitted model; binary outcomes were not standardized. ")
table_model_note <- paste0("Sequential model denotes the sequential shared-representation model. ")
final_exports_only <- TRUE

table_row <- function(...) paste0(paste(c(...), collapse = " & "), " \\\\")
table_number <- function(x, digits = 3L) {
  ifelse(is.finite(x), sprintf(paste0("%.", digits, "f"), x), "NA")
}
table_p <- function(x) {
  if (!is.finite(x)) return("NA")
  if (x < .001) return("$< .001$")
  paste0("$", sub("^0", "", sprintf("%.3f", x)), "$")
}

new_table_reader <- function(root, experiment,
    expected_estimator = if (experiment == "exp3") "bam" else "glmer") {
  cache <- new.env(parent = emptyenv())
  inputs <- character()
  read <- function(source, relative) {
    path <- file.path(root, experiment, source, relative)
    if (!file.exists(path)) stop("Required statistics file is missing: ", path, call. = FALSE)
    inputs <<- unique(c(inputs, path))
    if (!exists(path, envir = cache, inherits = FALSE)) {
      assign(path, read.csv(path, stringsAsFactors = FALSE, check.names = FALSE), envir = cache)
    }
    get(path, envir = cache, inherits = FALSE)
  }
  coefficient <- function(source, test, term, allow_constant = FALSE) {
    data <- read(source, file.path(test, "coefficients.csv"))
    required <- c("term", "beta", "ci_low", "ci_high", "df", "p")
    if (!all(required %in% names(data))) stop("Invalid coefficient schema: ", source, "/", test)
    row <- data[!is.na(data$term) & data$term == term, , drop = FALSE]
    if (nrow(row) != 1L) stop("Expected exactly one coefficient: ", source, "/", test, "/", term)
    if (allow_constant && "status" %in% names(row) && identical(row$status, "not_estimable_constant")) {
      if (!"constant_value" %in% names(row) || !is.finite(row$constant_value) ||
          !all(is.na(row[1, required[-1]]))) stop("Invalid constant-predictor result: ", term)
      return(row)
    }
    logistic <- is.na(row$df) &&
      all(c("z", "odds_ratio") %in% names(row)) &&
      is.finite(row$z) && is.finite(row$odds_ratio) && row$odds_ratio > 0
    if (grepl("^(context_|path_rank|rank_change|branch_timing|observation_timestep|tree_order_accuracy)", test) && !logistic)
      stop("Obsolete Gaussian accuracy results: rerun ", experiment, "/", source, "/", test)
    if (logistic && (!"estimator" %in% names(row) || is.na(row$estimator) || row$estimator != expected_estimator))
      stop("Missing ", expected_estimator, " provenance: rerun ", experiment, "/", source, "/", test)
    checked <- if (logistic) setdiff(required[-1], "df") else required[-1]
    if (any(!is.finite(as.numeric(row[1, checked])))) {
      stop("Non-finite required coefficient: ", source, "/", test, "/", term)
    }
    if (row$p < 0 || row$p > 1 || (!logistic && row$df <= 0) || row$ci_low > row$ci_high) stop("Invalid coefficient values: ", term)
    row
  }
  list(read = read, coefficient = coefficient, inputs = function() inputs)
}

coefficient_cells <- function(row) c(table_number(row$beta),
  sprintf("[%s, %s]", table_number(row$ci_low), table_number(row$ci_high)),
  ifelse(is.na(row$df), "---", table_number(row$df, 1)), table_p(row$p))

render_stats_table <- function(caption, label, headers, rows, columns = NULL) {
  if (is.null(columns)) columns <- paste0("@{}", paste(rep("l", length(headers)), collapse = ""), "@{}")
  c("% Generated from fitted-statistics CSVs; requires booktabs and tabularx.",
    "% Self-contained table: no external statistic fragments are required.",
    "\\begin{table}[htbp]", "\\centering", "\\small",
    paste0("\\caption{", caption, "}"), paste0("\\label{", label, "}"),
    paste0("\\begin{tabularx}{\\linewidth}{", columns, "}"), "\\toprule",
    do.call(table_row, as.list(headers)), "\\midrule", rows,
    "\\bottomrule", "\\end{tabularx}", "\\end{table}")
}

build_experiment_tables <- function(root, experiment) {
  reader <- new_table_reader(root, experiment)
  result <- list()
  number <- sub("exp", "", experiment, fixed = TRUE)
  primary <- c("participants", "sequential")
  headers <- c("Group", "Effect", "$b$", "95\\% CI", "df", "$p$")
  columns <- "@{}lXllll@{}"
  add <- function(name, caption, label, rows, head = headers, cols = columns) {
    stem <- paste0(experiment, "_", name)
    if (!identical(label, paste0("tab:", stem))) stop("Table label must match filename: ", stem)
    result[[paste0(stem, ".tex")]] <<-
      render_stats_table(caption, label, head, rows, cols)
  }
  effect_row <- function(source, test, term, label) {
    do.call(table_row, as.list(c(table_groups[[source]], label,
      coefficient_cells(reader$coefficient(source, test, term)))))
  }
  if (experiment != "exp3") {
    descriptive <- function(source, measure, condition) {
      filename <- if (measure == "accuracy") "participant_accuracy.csv" else "participant_norm_reward.csv"
      data <- reader$read(source, filename)
      if (!all(c("participant", measure) %in% names(data))) stop("Invalid descriptive schema: ", filename)
      if (measure == "accuracy") {
        if (!"condition" %in% names(data)) stop("Missing descriptive condition: ", filename)
        data <- data[!is.na(data$condition) & data$condition == condition, , drop = FALSE]
      }
      if (!nrow(data) || anyDuplicated(data$participant) || any(!is.finite(data[[measure]]))) {
        stop("Missing/invalid participant summaries: ", source, "/", filename, "/", condition)
      }
      sprintf("%s (%s)", table_number(mean(data[[measure]])), table_number(sd(data[[measure]])))
    }
    descriptive_row <- function(measure, condition) {
      values <- vapply(names(table_groups), function(source) {
        if (condition == "exposure" && source %in% c("simultaneous", "independent")) return("---")
        descriptive(source, measure, condition)
      }, character(1))
      do.call(table_row, as.list(c(if (measure == "accuracy") "Recall accuracy" else "Normalized reward",
        if (condition == "planning") "Planning" else "Exposure", values)))
    }
    rows <- c(descriptive_row("norm_reward", "planning"), descriptive_row("accuracy", "planning"),
      descriptive_row("accuracy", "exposure"))
    add("engagement", sub("Experiment~1", paste0("Experiment~", number), "Normalized chosen path reward and recall accuracy in Experiment~1. Values are means (SDs) across participant-level means, with participants equally weighted. Dashes indicate unavailable exposure simulations for the simultaneous and independent variants.", fixed = TRUE),
      paste0("tab:", experiment, "_engagement"), rows,
      c("Measure", "Condition", unname(table_groups)),
      "@{}l l *{4}{>{\\raggedright\\arraybackslash}X}@{}")

    # Adjacent logistic contrasts are no longer exported; retain classical ANOVAs.
    rows <- unlist(lapply(primary, function(source) {
      d <- reader$read(source, "adjacent_path_ranks_anova/comparisons.csv")
      required <- c("rank_lower", "rank_higher", "participants", "F", "df1", "df2",
        "partial_eta_squared", "p", "status", "scope", "difference", "difference_ci_low", "difference_ci_high")
      if (!all(required %in% names(d)) || anyNA(d$scope) ||
          any(d$scope != "planning_all_probes_best_containing_path") ||
          !identical(as.integer(d$rank_lower), seq_len(if (experiment == "exp1") 3L else 8L)))
        stop("Invalid all-probe adjacent-rank ANOVA export; rerun path_rank: ", experiment, "/", source)
      vapply(seq_len(nrow(d)), function(i) {
        r <- d[i, ]
        if (!r$status %in% c("ok", "not_estimable")) stop("Invalid ANOVA status")
        if (r$status == "ok" && (any(!is.finite(unlist(r[c("F", "df1", "df2", "partial_eta_squared", "p", "difference", "difference_ci_low", "difference_ci_high")])) ) ||
            r$difference_ci_low > r$difference || r$difference_ci_high < r$difference ||
            r$F < 0 || r$df1 != 1 || r$df2 != r$participants - 1 || r$df2 < 1 ||
            r$p < 0 || r$p > 1 ||
            r$partial_eta_squared < 0 || r$partial_eta_squared > 1)) stop("Invalid ANOVA result")
        values <- if (r$status == "ok") c(table_number(r$difference),
          sprintf("[%s, %s]", table_number(r$difference_ci_low), table_number(r$difference_ci_high)), table_number(r$F),
          sprintf("%d, %d", r$df1, r$df2), table_number(r$partial_eta_squared), table_p(r$p)) else
          c("Not estimable", "---", "---", "---", "---", "---")
        do.call(table_row, as.list(c(table_groups[[source]],
          sprintf("%d vs.\\ %d", r$rank_lower, r$rank_higher), r$participants, values)))
      }, character(1))
    }))
    add("adjacent_path_ranks_anova", sub("Experiment~1", paste0("Experiment~", number), "Adjacent path-rank comparisons in Experiment~1 (planning condition; leaf and non-leaf probes). Each probe is assigned the lowest numerical rank among complete paths containing it. Each participant contributes one mean accuracy per rank. Each comparison includes participants observed at both ranks; model trials and simulation seeds are pooled within participant and rank. $\\Delta$ is accuracy at the larger numerical rank minus accuracy at the smaller rank, so negative values favor the better-ranked path.", fixed = TRUE),
      paste0("tab:", experiment, "_adjacent_path_ranks_anova"), rows,
      c("Group", "Ranks", "$N$", "$\\Delta$", "95\\% CI", "$F$", "df", "$\\eta_p^2$", "$p$"), "@{}lXlllllll@{}")

    effects <- c(node_reward_z = "Node reward", node_reward_squared_z = "Node reward$^2$",
      "node_reward_z:conditionexposure" = "Node reward $\\times$ condition",
      sibling_reward_z = "Sibling reward", sibling_reward_squared_z = "Sibling reward$^2$",
      "sibling_reward_z:conditionexposure" = "Sibling reward $\\times$ condition",
      parent_reward_z = "Parent reward", "parent_reward_z:conditionexposure" = "Parent reward $\\times$ condition",
      aunt_reward_z = "Aunt reward", "aunt_reward_z:conditionexposure" = "Aunt reward $\\times$ condition")
    rows <- unlist(lapply(names(table_groups), function(source) {
      joint <- source %in% primary
      selected_effects <- if (joint) effects else effects[!grepl(":conditionexposure", names(effects), fixed = TRUE)]
      test <- if (joint) "context_joint_conditions" else "context_planning"
      vapply(names(selected_effects), function(term)
        effect_row(source, test, term, selected_effects[[term]]), character(1))
    }))
    add("context_reward_effects", sub("Experiment~1", paste0("Experiment~", number), "Reward-predictor effects on recall accuracy in Experiment~1. Participants and the sequential model are fit jointly across planning and exposure, with planning as the reference condition; main effects are planning slopes and predictor $\\times$ condition terms are exposure-minus-planning slope differences. The simultaneous and independent variants are fit to planning only.", fixed = TRUE),
      paste0("tab:", experiment, "_context_reward_effects"), rows)

    rows <- vapply(names(table_groups), function(source) effect_row(source, "branch_timing",
      "context_reward_z:relative_branch_timingbefore", "Context reward $\\times$ timing"), character(1))
    add("branch_timing", sub("Experiment~1", paste0("Experiment~", number), "Competing-branch reward $\\times$ timing interaction in Experiment~1. Context reward is the maximum full path reward outside the probed root branch. The interaction is the earlier-minus-later difference in the context-reward slope, controlling for observation timestep. Ties in which maximizing branches span both timing conditions are excluded. For the simultaneous variant, timing indexes task order rather than a sequential encoding event.", fixed = TRUE),
      paste0("tab:", experiment, "_branch_timing"), rows)

    if (experiment == "exp2") {
      rows <- vapply(names(table_groups), function(source) {
        d <- reader$read(source, "best_branch_timing/coefficients.csv")
        required <- c("term", "beta", "ci_low", "ci_high", "z", "p")
        if (!all(required %in% names(d))) stop("Invalid best-branch timing schema")
        if (!"estimator" %in% names(d) || any(d$estimator != "glmer")) stop("Rerun best_branch_timing with glmer for ", source)
        d <- d[d$term == "best_branch_timingbefore", , drop = FALSE]
        if (nrow(d) != 1L || any(!is.finite(as.numeric(d[1, required[-1]]))) ||
            d$p < 0 || d$p > 1 || d$ci_low > d$beta || d$ci_high < d$beta)
          stop("Invalid best-branch timing coefficient for ", source)
        table_row(table_groups[[source]], "Best branch first vs.\\ worst branch first",
          table_number(d$beta), sprintf("[%s, %s]", table_number(d$ci_low), table_number(d$ci_high)),
          table_number(d$z), table_p(d$p))
      }, character(1))
      add("best_branch_timing", "Best-versus-worst branch presentation timing in Experiment~2. Only middle-branch leaf probes (positions 6--8) are included, on trials where the globally best and worst root-to-leaf paths lie outside the middle branch and in different branches. Worst-branch-first is the reference condition; positive coefficients indicate higher recall when the best branch appears first.",
        "tab:exp2_best_branch_timing", rows, head = c("Group", "Effect", "$b$", "95\\% CI", "$z$", "$p$"))
    }
    for (source in names(table_groups)) {
      temporal <- reader$read(source, "observation_timestep/coefficients.csv")
      if (!all(c("is_leafTRUE", "observation_step_z") %in% temporal$term))
        stop("Rerun leaf-adjusted observation_timestep for ", experiment, "/", source)
    }
    timestep_effects <- c(observation_step_z = "Observation timestep", is_leafTRUE = "Leaf vs. non-leaf")
    rows <- unlist(lapply(names(table_groups), function(source)
      vapply(names(timestep_effects), function(term)
        effect_row(source, "observation_timestep", term, timestep_effects[[term]]), character(1))))
    add("observation_timestep", sub("Experiment~1", paste0("Experiment~", number), "Observation-timestep effects on planning-condition recall accuracy in Experiment~1, adjusting for leaf status (non-leaf reference). For the simultaneous variant, timestep indexes task order rather than a sequential encoding event.", fixed = TRUE), paste0("tab:", experiment, "_observation_timestep"), rows,
      c("Group", "Effect", "$b$", "95\\% CI", "df", "$p$"))
  } else {
    accuracy_results <- lapply(primary, function(source) reader$read(source, "tree_order_accuracy/coefficients.csv"))
    logistic_accuracy <- vapply(accuracy_results, function(d) all(is.na(d$df)) &&
      all(c("z", "odds_ratio") %in% names(d)), logical(1))
    if (any(logistic_accuracy) && !all(logistic_accuracy))
      stop("Mixed old Gaussian and new logistic Exp3 accuracy results; rerun both sources")
    bam_note <- paste0("Unstandardized coefficients ($b$) and unadjusted $p$ values are from BAM fits (fREML; discrete=TRUE) for accuracy and mgcv BAM fits (fREML) for normalized reward. ",
      "Recall accuracy uses binomial-logit models: coefficients and 95\\% normal-Wald intervals are on the log-odds scale, with Wald z tests; df is not applicable (---). ",
      "Normalized reward uses Gaussian models, approximate t tests, residual effective df and 95\\% t intervals.")
    effects <- c(order_c = "Presentation order", tree_c = "Tree structure", "tree_c:order_c" = "Order $\\times$ tree")
    rows <- character()
    for (source in primary) for (outcome in c("norm_reward", "accuracy")) for (term in names(effects)) {
      rows <- c(rows, do.call(table_row, as.list(c(table_groups[[source]],
        if (outcome == "norm_reward") "Normalized reward" else "Recall accuracy", effects[[term]],
        coefficient_cells(reader$coefficient(source, paste0("tree_order_", outcome), term))))))
    }
    add("tree_order_effects", "Presentation-order and tree-structure effects in Experiment~3. Tree is coded deep $=-0.5$, wide $=+0.5$, and order is coded BFS $=-0.5$, DFS $=+0.5$. The order coefficient is DFS minus BFS averaged across tree structures, the tree coefficient is wide minus deep averaged across orders, and the interaction is the DFS--BFS difference in wide versus deep trees.", "tab:exp3_tree_order_effects", rows,
      c("Group", "Outcome", "Predictor", "$b$", "95\\% CI", "df", "$p$"), "@{}lXXllll@{}")
    rank_results <- lapply(primary, function(source) reader$read(source, "rank_change/coefficients.csv"))
    rank_labels <- c("(Intercept)" = "Intercept",
      "final_rank_better_than_encoding_rank" = "Rank improvement",
      "tree_c" = "Tree", "order_c" = "Order",
      "final_rank_better_than_encoding_rank:tree_c" = "Rank improvement $\\times$ tree",
      "final_rank_better_than_encoding_rank:order_c" = "Rank improvement $\\times$ order",
      "tree_c:order_c" = "Tree $\\times$ order",
      "final_rank_better_than_encoding_rank:tree_c:order_c" = "Rank improvement $\\times$ tree $\\times$ order",
      "final_rank" = "Final rank",
      "tree_c:final_rank" = "Final rank $\\times$ tree",
      "order_c:final_rank" = "Final rank $\\times$ order",
      "tree_c:order_c:final_rank" = "Final rank $\\times$ tree $\\times$ order")
    if (!all(vapply(rank_results, function(d) all(names(rank_labels) %in% d$term) &&
        all(is.na(d$df)) && all(c("z", "odds_ratio") %in% names(d)), logical(1))))
      stop("Exp3 rank-change tables require the current improvement and final-rank interaction glmer results; rerun both sources")
    rows <- unlist(lapply(primary, function(source) vapply(names(rank_labels),
      function(term) effect_row(source, "rank_change", term, rank_labels[[term]]), character(1))))
    add("rank_change", "Rank-change effects in Experiment~3. Rank improvement is coded improved $=1$ and worsened $=0$; unchanged ranks are excluded. All eligible nodes and trials are included. Final rank is numeric and unscaled; tree/order are coded deep/BFS $=-0.5$ and wide/DFS $=+0.5$. Tree and order main effects are conditional on worsened ranks and final rank $=0$, whereas rank-improvement effects and interactions hold final rank constant.", "tab:exp3_rank_change", rows)
  }
  if (experiment %in% c("exp1", "exp2")) {
    variants <- c("simultaneous", "independent")
    comparisons <- lapply(variants, function(source) {
      r <- reader$read(source, "nll_vs_sequential/comparisons.csv")
      numeric <- c("participants", "mean_full_nll", "mean_variant_nll", "beta", "t", "df", "p", "p_holm", "ci_low", "ci_high", "reward_nll_scale")
      if (nrow(r) != 1L || !all(c(numeric, "term", "variant", "reference", "alternative") %in% names(r)) ||
          any(!is.finite(as.numeric(r[1, numeric])))) stop("Invalid NLL comparison: ", source)
      if (r$term != paste0(source, "_minus_sequential") || r$variant != source || r$reference != "sequential" ||
          r$alternative != "variant NLL > sequential NLL" || abs(r$reward_nll_scale - .5) > 1e-8 ||
          r$participants < 2 || r$participants != floor(r$participants) || r$df != r$participants - 1 ||
          r$p < 0 || r$p > 1 || r$p_holm < r$p || r$p_holm > 1 ||
          r$ci_low > r$beta || r$ci_high < r$beta ||
          abs(r$beta - (r$mean_variant_nll - r$mean_full_nll)) > 1e-8 * max(1, abs(r$beta)))
        stop("Invalid NLL coding/values: ", source)
      r
    })
    nll <- do.call(rbind, comparisons)
    if (length(unique(nll$participants)) != 1L ||
        !isTRUE(all.equal(nll$mean_full_nll[1], nll$mean_full_nll[2])) ||
        !isTRUE(all.equal(nll$p_holm, p.adjust(nll$p, "holm"), tolerance = 1e-12)))
      stop("NLL comparison pair has inconsistent participants, full-model mean, or Holm correction")
    rows <- vapply(seq_len(nrow(nll)), function(i) {
      r <- nll[i, ]
      table_row(table_groups[[r$variant]], r$participants,
        table_number(r$mean_full_nll), table_number(r$mean_variant_nll), table_number(r$beta),
        sprintf("[%s, %s]", table_number(r$ci_low), table_number(r$ci_high)),
        table_number(r$t), table_number(r$df, 0), table_p(r$p), table_p(r$p_holm))
    }, character(1))
    add("nll_comparisons", sub("Experiment~1", paste0("Experiment~", number), "Model-fit comparison in Experiment~1. Each participant contributes one selected-fit weighted NLL (path NLL $+\\,0.5\\times$ recall NLL) per model. $\\Delta$NLL is variant minus full model, so positive values favor the full model. One-sided paired $t$ tests are Holm-adjusted across the two variant comparisons; confidence intervals are unadjusted two-sided intervals.", fixed = TRUE),
      paste0("tab:", experiment, "_nll_comparisons"), rows,
      c("Variant", "$N$", "Full mean NLL", "Variant mean NLL", "$\\Delta\\mathrm{NLL}$",
        "95\\% CI", "$t$", "df", "$p$ (one-sided)", "$p_{\\mathrm{Holm}}$"), "@{}Xrrrrrrrrr@{}")
  }
  list(tables = result, inputs = reader$inputs())
}

build_performance_tables <- function(root, experiment, pairwise_only = FALSE) {
  reader <- new_table_reader(root, experiment)
  result <- list()
  primary <- c("participants", "sequential")
  anova_cells <- function(source, test) {
    row <- reader$read(source, file.path(test, "anova.csv"))
    required <- c("F", "df_num", "df_den", "p")
    if (nrow(row) != 1L || !all(required %in% names(row)) ||
        any(!is.finite(as.numeric(row[1, required])))) stop("Invalid ANOVA: ", source, "/", test)
    if (row$F < 0 || row$df_num <= 0 || row$df_den <= 0 || row$p < 0 || row$p > 1) stop("Invalid ANOVA values")
    c(table_number(row$F), table_number(row$df_num, 0), table_number(row$df_den, 0), table_p(row$p))
  }
  add <- function(name, caption, headers, rows, columns) {
    stem <- paste0(experiment, "_", name)
    result[[paste0(stem, ".tex")]] <<- render_stats_table(caption, paste0("tab:", stem), headers, rows, columns)
  }
  if (experiment == "cross_experiment") {
    if (!pairwise_only) {
    rows <- vapply(primary, function(source) do.call(table_row, as.list(c(table_groups[[source]],
      anova_cells(source, "planning_accuracy_anova")))), character(1))
    add("planning_accuracy_anova", paste0("Accuracy in the planning condition across Experiments 1--3. ",
      "Separate one-way ANOVAs use one mean accuracy per participant per experiment, with equal participant weighting. ",
      "Independent experiment cohorts and common residual variance are assumed. Tests are omnibus experiment effects, not pairwise contrasts. ",
      table_model_note), c("Group", "$F$", "Numerator df", "Denominator df", "$p$"), rows, "@{}Xllll@{}")
    }
    effects <- c(exp2_minus_exp1 = "13 minus 7 nodes", exp3_minus_exp1 = "31 minus 7 nodes",
      exp3_minus_exp2 = "31 minus 13 nodes")
    rows <- unlist(lapply(primary, function(source) {
      vapply(names(effects), function(term) {
        r <- reader$coefficient(source, "planning_accuracy_pairwise", term)
        do.call(table_row, as.list(c(table_groups[[source]], effects[[term]], coefficient_cells(r))))
      }, character(1))
    }))
    add("planning_accuracy_pairwise", "Cross-experiment pairwise comparisons of planning-condition recall accuracy. Each participant contributes one mean per experiment. $\\Delta$ is accuracy in the larger minus smaller tree, so negative values indicate lower recall in larger trees. Contrasts use the pooled residual variance from a one-way ANOVA; confidence intervals and $p$ values are unadjusted.",
      c("Group", "Contrast", "$b$", "95\\% CI", "df", "$p$"), rows, "@{}lXllll@{}")
  } else {
    coefficient_table <- function(name, test, sources, effects, description) {
      notes <- character()
      rows <- unlist(lapply(sources, function(source) {
        included <- character()
        slopes <- vapply(names(effects), function(term) {
          r <- reader$coefficient(source, test, term, allow_constant = test == "memory_allocation_performance")
          if (test == "memory_allocation_performance" &&
              (!"context_coefficient_method" %in% names(r) ||
               !identical(r$context_coefficient_method, "separate_binomial_glm_linear_v2")))
            stop("Outdated memory-allocation results: regenerate reviewer metrics and --tests memory_allocation for ", experiment, "/", source)
          constant <- "status" %in% names(r) && identical(r$status, "not_estimable_constant")
          if (constant) {
            notes <<- c(notes, paste0(table_groups[[source]], ": ", effects[[term]],
              " was constant across participants (", table_number(r$constant_value, 5),
              "; SD = 0) and excluded from the regression."))
            cells <- c("Not estimable", "---", "---", "---")
          } else {
            included <<- c(included, effects[[term]])
            cells <- coefficient_cells(r)
          }
          do.call(table_row, as.list(c(table_groups[[source]], effects[[term]], cells)))
        }, character(1))
        a <- anova_cells(source, test)
        c(slopes, table_row(table_groups[[source]], "Overall model", paste0("$F = ", a[[1]], "$"),
          "---", paste0(a[[2]], ", ", a[[3]]), a[[4]]))
      }))
      add(name, description,
        c("Group", "Effect", "$b$ / $F$", "95\\% CI", "df", "$p$"), rows, "@{}lXllll@{}")
    }
    if (experiment %in% c("exp1", "exp2")) coefficient_table("memory_allocation_performance",
      "memory_allocation_performance", primary,
      c(node_reward_coefficient_z = "Node-reward coefficient", sibling_reward_coefficient_z = "Sibling-reward coefficient",
        parent_reward_coefficient_z = "Parent-reward coefficient", aunt_reward_coefficient_z = "Aunt-reward coefficient"),
      sub("Experiment~1", paste0("Experiment~", sub("exp", "", experiment)), "Associations between participant-specific contextual memory effects and normalized planning reward in Experiment~1. For each participant or participant-specific model simulation, the probed-, sibling-, parent-, and aunt-reward effects are estimated jointly and then entered together into an ordinary least-squares regression predicting mean normalized reward, with one equally weighted row per participant. Contextual-effect predictors are standardized before the second-stage regression, so $b$ is the normalized-reward change per one-SD increase in the corresponding effect. Overall-model rows give the omnibus $F$ test against an intercept-only model.", fixed = TRUE))
  }
  list(tables = result, inputs = reader$inputs())
}

build_combined_parameter_table <- function(root, experiments) {
  rows <- character(); inputs <- character()
  effects <- c(fitted_lambda = "Fitted $\\lambda$", fitted_alpha = "Fitted $\\alpha$",
    "fitted_lambda:fitted_alpha" = "Fitted $\\lambda \\times \\alpha$")
  for (experiment in experiments) {
    reader <- new_table_reader(root, experiment)
    for (outcome in c("norm_reward", "accuracy")) {
      test <- paste0("fitted_parameters_", outcome)
      prefix <- c(sub("exp", "", experiment), if (outcome == "accuracy") "Recall accuracy" else "Normalized reward")
      for (term in names(effects)) {
        cells <- coefficient_cells(reader$coefficient("participants", test, term))
        rows <- c(rows, do.call(table_row, as.list(c(prefix, effects[[term]], cells))))
      }
      a <- reader$read("participants", file.path(test, "anova.csv"))
      required <- c("F", "df_num", "df_den", "p")
      if (nrow(a) != 1L || !all(required %in% names(a)) ||
          any(!is.finite(as.numeric(a[1, required]))) || a$F < 0 ||
          a$df_num <= 0 || a$df_den <= 0 || a$p < 0 || a$p > 1) stop("Invalid parameter-performance ANOVA: ", experiment, "/", test)
      rows <- c(rows, do.call(table_row, as.list(c(prefix, "Overall model",
        paste0("$F = ", table_number(a$F), "$"), "---",
        paste0(table_number(a$df_num, 0), ", ", table_number(a$df_den, 0)), table_p(a$p)))))
    }
    inputs <- c(inputs, reader$inputs())
  }
  caption <- sub("Experiments~1--3", if (identical(experiments, c("exp1", "exp2", "exp3"))) "Experiments~1--3" else paste0("Experiments~", paste(sub("exp", "", experiments), collapse = ", ")), "Associations between fitted parameters and participant planning performance across Experiments~1--3. Separate Gaussian BAM regressions predict normalized reward or recall accuracy from fitted $\\lambda$, fitted $\\alpha$, and their interaction; parameters and outcomes remain in their original units. Main effects are conditional on the other parameter equaling zero. Overall-model rows give the omnibus $F$ test against an intercept-only model.", fixed = TRUE)
  list(tables = list(fitted_parameters_performance.tex = render_stats_table(caption,
    "tab:fitted_parameters_performance", c("Exp.", "Outcome", "Effect", "$b$ / $F$", "95\\% CI", "df", "$p$"), rows, "@{}lXXllll@{}")), inputs = inputs)
}

build_exposure_memory_table <- function(root, experiments) {
  experiments <- intersect(experiments, c("exp1", "exp2"))
  if (!length(experiments)) stop("Exposure-memory associations are available for Exp1/Exp2 only")
  rows <- character(); inputs <- character()
  effects <- c(sigma_z = "Memory spread ($\\sigma$)")
  for (experiment in experiments) {
    path <- file.path(root, experiment, "all_identifiable_fits", "coefficients.csv")
    if (!file.exists(path)) stop("Required statistics file is missing: ", path)
    data <- read.csv(path, stringsAsFactors = FALSE)
    fields <- c("outcome", "term", "n", "b", "ci_low", "ci_high", "df", "p", "status")
    if (!all(fields %in% names(data))) stop("Invalid exposure-memory coefficient schema: ", path)
    if (!setequal(data$term, names(effects)))
      stop("Obsolete exposure-memory model: rerun the sigma-only regression: ", path)
    inputs <- c(inputs, path)
    for (outcome in c("norm_reward", "accuracy")) for (term in names(effects)) {
      r <- data[!is.na(data$outcome) & !is.na(data$term) & data$outcome == outcome & data$term == term, , drop = FALSE]
      if (nrow(r) != 1L) stop("Expected one exposure-memory coefficient: ", experiment, "/", outcome, "/", term)
      if (!is.finite(r$n) || r$n < 0 || r$n != floor(r$n) || is.na(r$status)) stop("Invalid sample/status: ", path)
      if (r$status %in% c("not_estimable", "rank_deficient")) {
        cells <- c("Not estimable", "---", "---", "---")
      } else {
        values <- unlist(r[c("b", "ci_low", "ci_high", "df", "p")], use.names = FALSE)
        if (r$status != "estimated" || any(!is.finite(values)) || r$df <= 0 ||
            r$p < 0 || r$p > 1 || r$ci_low > r$ci_high)
          stop("Invalid exposure-memory coefficient: ", path)
        cells <- c(table_number(r$b), sprintf("[%s, %s]", table_number(r$ci_low), table_number(r$ci_high)),
          table_number(r$df, 0), table_p(r$p))
      }
      rows <- c(rows, do.call(table_row, as.list(c(sub("exp", "", experiment),
        if (outcome == "accuracy") "Recall accuracy" else "Normalized reward",
        effects[[term]], table_number(r$n, 0), cells))))
    }
  }
  caption <- sub("Experiments~1 and~2", paste0("Experiments~", paste(sub("exp", "", experiments), collapse = " and~")), "Exposure-based memory precision and planning performance in Experiments~1 and~2. Separate ordinary least-squares regressions predict participant mean normalized reward and recall accuracy from standardized exposure-fitted memory spread ($\\sigma$), with lower $\\sigma$ indicating greater precision. Each participant contributes one equally weighted mean; $b$ is the outcome change per one-SD increase in $\\sigma$. Identifiable fits, including boundary fits, are retained; failed or missing fits and all-guess solutions with unidentified $\\sigma$ are excluded. These exploratory regressions do not propagate uncertainty in the fitted memory parameter. The table reports 95\\% $t$ confidence intervals and unadjusted two-sided $p$ values.", fixed = TRUE)
  list(tables = list(exposure_memory_performance.tex = render_stats_table(caption,
    "tab:exposure_memory_performance", c("Exp.", "Outcome", "Predictor", "$N$", "$b$", "95\\% CI", "df", "$p$"),
    rows, "@{}lXXrllrl@{}")), inputs = inputs)
}

build_exposure_lambda_table <- function(root, experiments) {
  experiments <- intersect(experiments, c("exp1", "exp2"))
  if (!length(experiments)) stop("Select Exp1 or Exp2 for exposure-lambda regressions")
  rows <- character(); inputs <- character()
  labels <- c(sigma_z = "Memory spread ($\\sigma$)")
  for (experiment in experiments) {
    path <- file.path(root, experiment, "all_identifiable_fits/coefficients.csv")
    if (!file.exists(path)) stop("Required statistics file is missing: ", path)
    data <- read.csv(path)
    if (!all(c("term", "n", "b", "df", "ci_low", "ci_high", "p", "status") %in% names(data)))
      stop("Invalid exposure-lambda regression schema: ", path)
    if (!setequal(data$term, names(labels)))
      stop("Obsolete or unexpected lambda model: rerun the sigma-only regression: ", path)
    inputs <- c(inputs, path)
    for (predictor in names(labels)) {
      r <- data[!is.na(data$term) & data$term == predictor, ]
      if (nrow(r) != 1L || !is.finite(r$n) || r$n < 0 || r$n != floor(r$n) || is.na(r$status)) stop("Invalid regression row: ", path)
      if (r$status %in% c("not_estimable", "rank_deficient")) cells <- c("Not estimable", "---", "---", "---") else {
        if (r$status != "estimated" || any(!is.finite(c(r$b, r$ci_low, r$ci_high, r$df, r$p))) ||
            r$p < 0 || r$p > 1 || r$n < 5 || r$df <= 0 || r$ci_low > r$ci_high)
          stop("Invalid regression coefficient: ", path)
        cells <- c(table_number(r$b), sprintf("[%s, %s]", table_number(r$ci_low), table_number(r$ci_high)),
          table_number(r$df, 0), table_p(r$p))
      }
      rows <- c(rows, do.call(table_row, as.list(c(sub("exp", "", experiment), labels[[predictor]],
        table_number(r$n, 0), cells))))
    }
  }
  caption <- sub("Experiments~1 and~2", paste0("Experiments~", paste(sub("exp", "", experiments), collapse = " and~")), "Association between exposure-based memory precision and planning-fitted $\\lambda$ in Experiments~1 and~2. Separate ordinary least-squares regressions predict raw planning-fitted $\\lambda$ from standardized exposure-fitted memory spread ($\\sigma$), with lower $\\sigma$ indicating greater precision. Each participant contributes one observation; $b$ is the change in $\\lambda$ per one-SD increase in $\\sigma$. Identifiable fits, including boundary fits, are retained; missing or failed fits and all-guess solutions with unidentified $\\sigma$ are excluded. These exploratory regressions do not propagate estimation uncertainty in either fitted parameter. The table reports 95\\% $t$ confidence intervals and unadjusted two-sided $p$ values.", fixed = TRUE)
  list(tables = list(exposure_memory_lambda.tex = render_stats_table(caption,
    "tab:exposure_memory_lambda", c("Exp.", "Predictor", "$N$", "$b$", "95\\% CI", "df", "$p$"),
    rows, "@{}lXrllrl@{}")), inputs = inputs)
}

build_quadratic_performance_table <- function(root, experiments) {
  inputs <- rows <- character()
  for (experiment in intersect(experiments, c("exp1", "exp2"))) {
    directory <- file.path(dirname(dirname(root)), "tests", "quadratic_node_reward_performance", experiment, "participants")
    path <- file.path(directory, "coefficients.csv")
    audit <- read.csv(file.path(directory, "participant_coefficients_and_performance.csv"))
    if (!"sibling_reward_squared_coefficient" %in% names(audit)) stop("Rerun six-predictor quadratic test: ", directory)
    model <- readLines(file.path(directory,"model.txt"))
    if (!any(grepl("node_squared_z + sibling_squared_z",model,fixed=TRUE))) stop("Outdated quadratic model: ",directory)
    d <- read.csv(path)
    r <- d[d$term == "quadratic_coefficient_z",]
    if (nrow(r)!=1L || any(!is.finite(unlist(r[c("b","se","df","p","ci_low","ci_high","n")])))) stop("Invalid coefficient: ",path)
    if (r$n < 5 || r$df != r$n-2 || r$p < 0 || r$p > 1 || r$ci_low > r$ci_high) stop("Invalid inference: ",path)
    rows <- c(rows, table_row(sub("exp","",experiment), "Quadratic probed reward", table_number(r$n,0),
      table_number(r$b), sprintf("[%s, %s]",table_number(r$ci_low),table_number(r$ci_high)),table_number(r$df,0),table_p(r$p)))
    inputs <- c(inputs,path,file.path(directory,"model.txt"),file.path(directory,"participant_coefficients_and_performance.csv"))
  }
  if (!length(rows)) stop("Select exp1 and/or exp2")
  caption <- "Participant quadratic probed-reward effects and planning performance. Separate participant binomial-logit regressions jointly include probed reward, probed reward squared, parent reward, sibling reward, aunt reward, and sibling reward squared, standardized within experiment. The quadratic probed-reward coefficient is standardized across accepted participants and is the sole predictor of mean normalized reward in a separate ordinary least-squares regression for each experiment. Each participant is equally weighted; reward is averaged once per trial. Flagged first-stage fits are excluded. Intervals are 95\\% t confidence intervals; p values are unadjusted and two-sided. First-stage estimation uncertainty is not propagated."
  list(tables=list(quadratic_node_reward_performance.tex=render_stats_table(caption,
    "tab:quadratic_node_reward_performance",c("Exp.","Predictor","$N$","$b$","95\\% CI","df","$p$"),rows,"@{}lXrllrl@{}")),inputs=inputs)
}

build_pilot40_table <- function(root) {
  reader <- new_table_reader(root, "exp2_40n", expected_estimator = "bam")
  test <- "context_joint_conditions"
  model_path <- file.path(root, "exp2_40n", "participants", test, "model.txt")
  metadata <- readLines(model_path)
  if (!all(c("Estimator: mgcv::bam fREML, discrete=TRUE; NOT ML",
             "Family: binomial Link: logit", "Converged: TRUE") %in% metadata))
    stop("Pilot requires converged binomial-logit BAM (fREML, discrete=TRUE)")
  outer <- metadata[startsWith(metadata, "Outer convergence:")]
  if (length(outer) != 1L || !trimws(sub("Outer convergence:", "", outer, fixed = TRUE)) %in%
      c("", "full convergence")) stop("Pilot BAM outer convergence check failed")
  effects <- c(node_reward_z = "Probed reward", node_reward_squared_z = "Probed reward squared",
    sibling_reward_z = "Sibling reward", sibling_reward_squared_z = "Sibling reward squared",
    preceding_sum_reward_z = "Preceding reward sum", aunt_reward_z = "Aunt reward")
  terms <- c("conditionexposure", names(effects), paste0(names(effects), ":conditionexposure"))
  labels <- c("Exposure vs. planning", unname(effects), paste0(unname(effects), " $\\times$ exposure"))
  rows <- vapply(seq_along(terms), function(i) {
    r <- reader$coefficient("participants", test, terms[i])
    if (!all(c("family", "link") %in% names(r)) ||
        !isTRUE(r$family == "binomial") || !isTRUE(r$link == "logit"))
      stop("Pilot requires binomial-logit BAM")
    table_row(labels[i], table_number(r$beta),
      sprintf("[%s, %s]", table_number(r$ci_low), table_number(r$ci_high)),
      table_number(r$z), table_p(r$p))
  }, character(1))
  caption <- paste0("Reward-predictor effects on participant recall accuracy in the 40-node pilot. ",
    "A joint planning/exposure binomial-logit mixed model is fitted using mgcv::bam (fREML; discrete=TRUE). ",
    "Planning is the reference; reward main effects are planning slopes and interactions are exposure-minus-planning slope differences. ",
    "Linear and squared reward predictors are standardized separately across the included observations. ",
    "Preceding reward sum and aunt reward are zero for non-leaf probes. ",
    "Random effects comprise an independent participant intercept and slopes for all fixed-effect contrasts. ",
    "Entries are log-odds coefficients ($b$), approximate 95\\% normal-Wald confidence intervals, Wald $z$, and unadjusted two-sided $p$ values. ",
    "No participant-fitted model simulations are included.")
  list(tables = list(exp2_40n_context_reward_effects.tex = render_stats_table(caption,
    "tab:exp2_40n_context_reward_effects", c("Effect", "$b$", "95\\% CI", "$z$", "$p$"),
    rows, "@{}Xllll@{}")), inputs = c(reader$inputs(), model_path))
}

build_engagement_table <- function(root, experiment) {
  stopifnot(experiment %in% c("exp3", "exp2_40n"))
  reader <- new_table_reader(root, experiment)
  sources <- if (experiment == "exp3") c("participants", "sequential") else "participants"
  measures <- c("norm_reward", "accuracy", if (experiment == "exp2_40n") "accuracy")
  conditions <- c("planning", "planning", if (experiment == "exp2_40n") "exposure")
  rows <- vapply(seq_along(measures), function(i) {
    measure <- measures[i]
    values <- vapply(sources, function(source) {
      d <- reader$read(source, paste0("participant_", measure, ".csv"))
      if (!all(c("participant", measure) %in% names(d))) stop("Invalid engagement schema")
      if (measure == "accuracy") {
        if (!"condition" %in% names(d)) stop("Missing engagement condition")
        d <- d[!is.na(d$condition) & d$condition == conditions[i], , drop = FALSE]
      }
      if (nrow(d) < 2 || anyNA(d$participant) || anyDuplicated(d$participant) ||
          any(!is.finite(d[[measure]]))) stop("Invalid engagement summaries: ", experiment, "/", source)
      sprintf("%s (%s)", table_number(mean(d[[measure]])), table_number(sd(d[[measure]])))
    }, character(1))
    do.call(table_row, as.list(c(if (measure == "accuracy") "Recall accuracy" else "Normalized reward",
      if (conditions[i] == "planning") "Planning" else "Exposure", values)))
  }, character(1))
  caption <- paste0("Normalized chosen path reward and recall accuracy in ",
    if (experiment == "exp3") "Experiment~3" else "the 40-node variant of Experiment~2",
    ". Values are means (SDs) across participant-level means, with participants equally weighted. Normalized reward is averaged once per trial.")
  label <- if (experiment == "exp3") "tab:exp3_engagement" else "tab:40n_engagement"
  tables <- setNames(list(render_stats_table(caption, label,
    c("Measure", "Condition", unname(table_groups[sources])), rows)), paste0(experiment, "_engagement.tex"))
  list(tables = tables, inputs = reader$inputs())
}

export_fitted_tables <- function(root, output, experiments = c("exp1", "exp2", "exp3"), table_set = "manuscript",
    exposure_root = file.path(dirname(dirname(root)), "tests", if (table_set == "exposure_lambda") "exposure_memory_lambda" else "exposure_memory_performance")) {
  if (any(!experiments %in% c("exp1", "exp2", "exp3")) || anyDuplicated(experiments)) stop("Invalid experiments")
  if (!table_set %in% c("manuscript", "performance", "all", "engagement", "exposure_memory", "exposure_lambda", "pilot_40n", "quadratic_performance")) stop("Invalid table set")
  # Validate/render everything before writing; missing results never become blanks.
  built <- list()
  if (table_set == "quadratic_performance") built <- list(build_quadratic_performance_table(root,experiments))
  if (table_set == "pilot_40n") built <- list(build_pilot40_table(root))
  if (table_set == "engagement" && "exp3" %in% experiments) built <- list(build_engagement_table(root, "exp3"))
  if (final_exports_only && table_set == "pilot_40n") built <- c(built, list(build_engagement_table(root, "exp2_40n")))
  # This exploratory test is opt-in, separate from the manuscript table sets.
  if (table_set == "exposure_memory") built <- list(build_exposure_memory_table(exposure_root, experiments))
  if (table_set == "exposure_lambda") built <- list(build_exposure_lambda_table(exposure_root, experiments))
  if (table_set %in% c("manuscript", "all")) built <- lapply(experiments, function(experiment) build_experiment_tables(root, experiment))
  if (table_set %in% c("manuscript", "all") && "exp3" %in% experiments)
    built <- c(built, list(build_engagement_table(root, "exp3")))
  if (table_set %in% c("performance", "all")) {
    scopes <- c(if (!final_exports_only) experiments, if (all(c("exp1", "exp2", "exp3") %in% experiments)) "cross_experiment")
    built <- c(built, lapply(scopes, function(experiment) build_performance_tables(root, experiment, pairwise_only = final_exports_only)))
    built <- c(built, list(build_combined_parameter_table(root, experiments)))
  }
  if (final_exports_only && table_set == "all") {
    exps12 <- intersect(experiments, c("exp1", "exp2"))
    if (length(exps12)) built <- c(built, list(
      build_exposure_memory_table(file.path(dirname(dirname(root)), "tests", "exposure_memory_performance"), exps12),
      build_exposure_lambda_table(file.path(dirname(dirname(root)), "tests", "exposure_memory_lambda"), exps12),
      build_quadratic_performance_table(root, exps12)))
    if ("exp2" %in% experiments) built <- c(built, list(build_pilot40_table(root), build_engagement_table(root, "exp2_40n")))
  }
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  if (table_set %in% c("manuscript", "all")) {
    for (experiment in intersect(experiments, c("exp1", "exp2"))) {
      old <- file.path(output, paste0(experiment, "_adjacent_path_ranks.tex"))
      if (file.exists(old)) {
        archive <- file.path(output, "retired_logistic_rank")
        dir.create(archive, recursive = TRUE, showWarnings = FALSE)
        target <- tempfile(paste0(experiment, "_adjacent_path_ranks_"), tmpdir = archive, fileext = ".tex")
        if (!file.rename(old, target)) stop("Cannot archive retired table: ", old)
        message("Archived retired logistic table: ", target)
      }
    }
  }
  for (batch in built) for (name in names(batch$tables)) {
    writeLines(batch$tables[[name]], file.path(output, name))
    message("Saved ", file.path(output, name))
  }
  paths <- unique(unlist(lapply(built, `[[`, "inputs")))
  write.csv(data.frame(path = paths, bytes = file.info(paths)$size,
    modified = as.character(file.info(paths)$mtime), md5 = unname(tools::md5sum(paths))),
    file.path(output, "table_inputs.csv"), row.names = FALSE)
  invisible(unlist(lapply(built, function(x) names(x$tables))))
}

table_export_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  storage <- STORAGE_ROOT
  root <- STATS_ROOT
  output <- NULL
  experiments <- c("exp1", "exp2", "exp3")
  table_set <- if (final_exports_only) "all" else "manuscript"
  exposure_root <- NULL
  while (length(args)) {
    if (args[[1]] %in% c("--help", "-h")) {
      cat("Usage: Rscript --vanilla analysis/scripts/stats/fitted/export_tables.R\n",
        "  [--stats-root PATH] [--output-dir PATH] [--experiments exp1,exp2,exp3]\n",
        "  [--table-set manuscript|performance|all|engagement|exposure_memory|exposure_lambda|pilot_40n|quadratic_performance] (default: all)\n",
        "  all exports only the selected final tables, including pilot and exploratory tables.\n",
        "  engagement exports Exp3 descriptives without loading inferential model results.\n",
        "  [--exposure-stats-root PATH] (optional root for exploratory exposure associations)\n",
        "Reads completed CSV statistics only; no model fits or plotting.\n")
      return(invisible(NULL))
    }
    flag <- args[[1]]
    if (length(args) < 2 || !flag %in% c("--stats-root", "--output-dir", "--experiments", "--table-set", "--exposure-stats-root")) stop("Unknown/incomplete option: ", flag)
    if (flag == "--stats-root") root <- args[[2]]
    if (flag == "--output-dir") output <- args[[2]]
    if (flag == "--experiments") experiments <- strsplit(args[[2]], ",", fixed = TRUE)[[1]]
    if (flag == "--table-set") table_set <- args[[2]]
    if (flag == "--exposure-stats-root") exposure_root <- args[[2]]
    args <- args[-c(1, 2)]
  }
  if (is.null(output)) output <- TABLE_DIR
  if (is.null(exposure_root)) exposure_root <- file.path(dirname(dirname(root)), "tests",
    if (table_set == "exposure_lambda") "exposure_memory_lambda" else "exposure_memory_performance")
  export_fitted_tables(root, output, experiments, table_set, exposure_root)
}

if (sys.nframe() == 0L) table_export_main()
