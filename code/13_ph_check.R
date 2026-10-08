#!/usr/bin/env Rscript
# 13_ph_check.R - proportional-hazards check for the M1 Cox model
# (CS + ER, PR, HER2, grade, age) in METABRIC, via Schoenfeld residuals
# (survival::cox.zph). Outputs results/ph_check.json.

suppressMessages({
  library(survival)
  library(jsonlite)
})

base <- getwd()
data_dir <- file.path(base, "data")
res_dir  <- file.path(base, "results")

expr  <- read.csv(file.path(data_dir, "metabric_processed.csv"), row.names = 1,
                  check.names = FALSE)
anal  <- read.csv(file.path(data_dir, "metabric_analysis.csv"),
                  check.names = FALSE)
driver <- read.csv(file.path(res_dir, "driver_genes.csv"), check.names = FALSE)
node  <- read.csv(file.path(res_dir, "node_metrics.csv"), check.names = FALSE)

anal$rfs_time  <- suppressWarnings(as.numeric(anal$p_RFS_MONTHS))
anal$rfs_event <- suppressWarnings(as.numeric(
  sub(":.*", "", as.character(anal$p_RFS_STATUS))))
anal$age  <- suppressWarnings(as.numeric(anal$p_AGE_AT_DIAGNOSIS))
anal$grade <- suppressWarnings(as.numeric(anal$s_GRADE))
anal$er   <- as.integer(grepl("pos", tolower(anal$s_ER_STATUS)))
anal$pr   <- as.integer(grepl("pos", tolower(anal$s_PR_STATUS)))
anal$her2 <- as.integer(grepl("pos", tolower(anal$s_HER2_STATUS)))

sub <- anal[!is.na(anal$rfs_time) & !is.na(anal$rfs_event) &
              anal$sampleId %in% rownames(expr), ]

genes <- driver$gene
present <- genes[genes %in% colnames(expr)]
w <- node$control_score[match(present, node$gene)]
Z <- as.matrix(expr[sub$sampleId, present, drop = FALSE])

cs <- as.vector(Z %*% w / sum(w))
cs <- (cs - mean(cs)) / sd(cs)

g <- sub$grade
gz <- (g - mean(g, na.rm = TRUE)) / (sd(g, na.rm = TRUE) + 1e-8)
agez <- (sub$age - mean(sub$age, na.rm = TRUE)) /
  (sd(sub$age, na.rm = TRUE) + 1e-8)
gz[is.na(gz)] <- 0

dat <- data.frame(
  rfs_time = sub$rfs_time, rfs_event = sub$rfs_event,
  CS = cs, ER = sub$er, PR = sub$pr, HER2 = sub$her2,
  grade_z = gz, age_z = agez)

fit <- coxph(Surv(rfs_time, rfs_event) ~ CS + ER + PR + HER2 + grade_z + age_z,
             data = dat)
zp <- cox.zph(fit)

pvals <- zp$table[, "p"]

# ---- time-varying structure of the CS effect ----
# (a) CS x log(1+t) interaction (log1p avoids -Inf at t = 0 months)
dat$lt <- log1p(dat$rfs_time)
fit_tv <- coxph(Surv(rfs_time, rfs_event) ~ CS + CS:lt + ER + PR +
                  HER2 + grade_z + age_z, data = dat)
tv <- summary(fit_tv)$coefficients["CS:lt", c("coef", "se(coef)", "Pr(>|z|)")]

# (b) period-specific HRs via time split at 24 and 60 months
#     (t = 0 month events are shifted to 0.01 so that the left boundary is
#      strictly below all observed times, as required by survSplit).
#     The model is CS * strata(period), so the raw interaction coefficients
#     test "period-k CS effect vs period-1 CS effect".  The period-specific
#     p-values reported here are instead Wald tests of the period-k CS
#     coefficient against HR = 1 (z = coef / se), where se is obtained from
#     the full coefficient covariance matrix.
dat_s <- dat
dat_s$rfs_time[dat_s$rfs_time == 0] <- 0.01
dat2 <- survSplit(Surv(rfs_time, rfs_event) ~ ., data = dat_s,
                  cut = c(24, 60), episode = "period")
fit_period <- coxph(Surv(tstart, rfs_time, rfs_event) ~ CS * strata(period) +
                      ER + PR + HER2 + grade_z + age_z, data = dat2)
pcoef <- summary(fit_period)$coefficients
b <- coef(fit_period)
V <- vcov(fit_period)
nms <- names(b)
cs_names <- c("CS", "CS:strata(period)period=2", "CS:strata(period)period=3")
stopifnot(all(cs_names %in% nms))

# period-1 CS coefficient: b["CS"] (strata(period) period=1 is the reference)
hr_p1 <- unname(exp(b["CS"]))
z_p1 <- b["CS"] / sqrt(V["CS", "CS"])
p_p1 <- 2 * pnorm(-abs(z_p1))

# period k (k = 2, 3): combined coefficient b["CS"] + b[paste0(...)]
per2_name <- cs_names[2]; per3_name <- cs_names[3]
coef_p2 <- b["CS"] + b[per2_name]
se_p2 <- sqrt(V["CS", "CS"] + 2 * V["CS", per2_name] + V[per2_name, per2_name])
z_p2 <- coef_p2 / se_p2
p_p2 <- 2 * pnorm(-abs(z_p2))
coef_p3 <- b["CS"] + b[per3_name]
se_p3 <- sqrt(V["CS", "CS"] + 2 * V["CS", per3_name] + V[per3_name, per3_name])
z_p3 <- coef_p3 / se_p3
p_p3 <- 2 * pnorm(-abs(z_p3))

period_rows <- list(
  period1_0_24m = c(HR = hr_p1, p = p_p1, z = z_p1),
  period2_24_60m = c(HR = unname(exp(coef_p2)), p = p_p2, z = z_p2),
  period3_gt60m = c(HR = unname(exp(coef_p3)), p = p_p3, z = z_p3))

out <- list(
  n = nrow(dat),
  events = sum(dat$rfs_event),
  coxph_global_p = unname(pvals["GLOBAL"]),
  schoenfeld_p = as.list(pvals[setdiff(names(pvals), "GLOBAL")]),
  cs_logtime_interaction = c(
    coef = unname(tv["coef"]),
    se = unname(tv["se(coef)"]),
    p = unname(tv["Pr(>|z|)"])),
  period_specific_CS_HR = period_rows,
  note = paste0(
    "Schoenfeld residual tests (survival::cox.zph) reject the PH assumption ",
    "for every covariate except HER2 and age; for CS p = ",
    formatC(unname(pvals["CS"]), digits = 3, format = "e"),
    " and for the global test p = ",
    formatC(unname(pvals["GLOBAL"]), digits = 3, format = "e"), ". ",
    "The CS x log(1+t) interaction and period-specific HRs (period 1: 0-24, ",
    "2: 24-60, 3: >60 months) characterise the time-dependence of the CS ",
    "effect; period p-values are Wald tests of the period-specific CS ",
    "coefficient against HR = 1. HRs above 1 indicate that higher CS ",
    "carries higher hazard in that period."))

write_json(out, file.path(res_dir, "ph_check.json"), pretty = TRUE,
           auto_unbox = TRUE)
cat(sprintf("PH check: n=%d events=%d global p=%.4g\n",
            nrow(dat), sum(dat$rfs_event), pvals["GLOBAL"]))
cat(sprintf("CS x log(t): coef=%.4f p=%.4g\n", tv["coef"], tv["Pr(>|z|)"]))
print(round(zp$table, 4))
print(period_rows)
