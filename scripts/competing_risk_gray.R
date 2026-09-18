#!/usr/bin/env Rscript
# Cumulative incidence by phenotype and Gray's tests (cmprsk).
# Usage: Rscript competing_risk_gray.R cohort.csv phenotypes.csv output_dir

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript competing_risk_gray.R cohort.csv phenotypes.csv output_dir")
}
if (!requireNamespace("cmprsk", quietly = TRUE)) {
  stop("R package 'cmprsk' is required.")
}

cohort_path <- args[[1]]
phenotype_path <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cohort <- read.csv(cohort_path, check.names = FALSE, fileEncoding = "UTF-8-BOM")
phenotypes <- read.csv(phenotype_path, check.names = FALSE, fileEncoding = "UTF-8-BOM")
if (length(setdiff(c("ID", "TIME", "MACET"), names(cohort))) > 0) stop("Cohort is missing ID, TIME or MACET.")
if (length(setdiff(c("ID", "Cluster"), names(phenotypes))) > 0) stop("Phenotype table is missing ID or Cluster.")
if (anyDuplicated(cohort$ID) || anyDuplicated(phenotypes$ID)) stop("ID must be unique.")
if (!setequal(as.character(cohort$ID), as.character(phenotypes$ID))) {
  stop("Cohort and phenotype files must contain the same patient IDs.")
}

cohort$Cluster <- NULL
data <- merge(cohort, phenotypes[, c("ID", "Cluster")], by = "ID", all.x = TRUE, sort = FALSE)
data <- data[match(cohort$ID, data$ID), ]
if (anyNA(data[, c("ID", "TIME", "MACET", "Cluster")])) stop("ID, TIME, MACET and Cluster must be complete.")
if (!all(data$MACET %in% c(0, 1, 2))) stop("MACET must be coded 0, 1 or 2.")
if (!setequal(unique(data$Cluster), c(1, 2, 3))) stop("All three clusters must be present.")

fit <- cmprsk::cuminc(
  ftime = as.numeric(data$TIME),
  fstatus = as.integer(data$MACET),
  group = factor(data$Cluster, levels = c(1, 2, 3)),
  cencode = 0
)
if (is.null(fit$Tests)) stop("cmprsk did not return Gray tests.")
tests <- as.data.frame(fit$Tests)
tests$event_code <- suppressWarnings(as.integer(rownames(tests)))
tests$endpoint <- ifelse(tests$event_code == 1, "Hard_MACE", ifelse(tests$event_code == 2, "Soft_MACE", NA_character_))
tests <- tests[!is.na(tests$endpoint), c("endpoint", "event_code", "stat", "df", "pv")]
names(tests)[names(tests) == "stat"] <- "gray_chisq"
names(tests)[names(tests) == "pv"] <- "p_value"
write.csv(tests, file.path(output_dir, "gray_tests.csv"), row.names = FALSE)

event_counts <- aggregate(list(n = data$ID), by = list(Cluster = data$Cluster, event_code = data$MACET), FUN = length)
event_counts$endpoint <- ifelse(event_counts$event_code == 0, "Censored", ifelse(event_counts$event_code == 1, "Hard_MACE", "Soft_MACE"))
write.csv(event_counts[, c("Cluster", "event_code", "endpoint", "n")], file.path(output_dir, "event_counts_by_cluster.csv"), row.names = FALSE)

curve_rows <- list()
for (name in names(fit)) {
  component <- fit[[name]]
  if (!is.list(component) || is.null(component$time) || is.null(component$est)) next
  pieces <- strsplit(name, " ", fixed = TRUE)[[1]]
  if (length(pieces) < 2) next
  curve_rows[[length(curve_rows) + 1]] <- data.frame(
    Cluster = as.integer(pieces[[1]]),
    event_code = as.integer(pieces[[2]]),
    time = component$time,
    cumulative_incidence = component$est,
    variance = component$var
  )
}
if (length(curve_rows) > 0) {
  curves <- do.call(rbind, curve_rows)
  curves$endpoint <- ifelse(curves$event_code == 1, "Hard_MACE", "Soft_MACE")
  write.csv(curves, file.path(output_dir, "cumulative_incidence_source_data.csv"), row.names = FALSE)
}
cat("Gray tests written to", normalizePath(output_dir), "\n")
