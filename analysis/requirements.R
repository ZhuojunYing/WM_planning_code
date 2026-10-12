packages <- c(
  "broom.mixed",
  "cowplot",
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
  "tibble",
  "tidyr"
)

minimum_versions <- c(
  "rlang" = "1.1.7"
)

installed_package_table <- installed.packages()
missing_packages <- setdiff(packages, rownames(installed_package_table))
outdated_packages <- names(minimum_versions)[vapply(
  names(minimum_versions),
  function(package) {
    if (!package %in% rownames(installed_package_table)) {
      return(FALSE)
    }
    utils::compareVersion(
      installed_package_table[package, "Version"],
      minimum_versions[[package]]
    ) < 0L
  },
  logical(1)
)]
packages_to_install <- unique(c(missing_packages, outdated_packages))

if (length(packages_to_install) > 0L) {
  install.packages(
    packages_to_install,
    repos = "https://cloud.r-project.org",
    lib = .libPaths()[[1]]
  )
}

installed_package_table <- installed.packages()
still_missing <- setdiff(packages, rownames(installed_package_table))
still_outdated <- names(minimum_versions)[vapply(
  names(minimum_versions),
  function(package) {
    if (!package %in% rownames(installed_package_table)) {
      return(FALSE)
    }
    utils::compareVersion(
      installed_package_table[package, "Version"],
      minimum_versions[[package]]
    ) < 0L
  },
  logical(1)
)]
if (length(still_missing) > 0L || length(still_outdated) > 0L) {
  missing_label <- if (length(still_missing) > 0L) {
    paste(still_missing, collapse = ", ")
  } else {
    "none"
  }
  outdated_label <- if (length(still_outdated) > 0L) {
    paste(still_outdated, collapse = ", ")
  } else {
    "none"
  }
  stop(
    paste0(
      "R package requirements remain unsatisfied. Missing: ",
      missing_label,
      "; below minimum version: ",
      outdated_label
    ),
    call. = FALSE
  )
}

# Package loading is deliberately left to a fresh R process. Loading an older
# namespace before upgrading it prevents dependent packages from seeing the
# newly installed version until R is restarted.
invisible(packages)
