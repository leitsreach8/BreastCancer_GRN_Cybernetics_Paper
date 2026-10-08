# 11_tcga_validation.R - TCGA-BRCA external validation (overall survival).
# Reads results/tcga_input.csv (prepared by 11_tcga_validation.py: CS, ER, PR,
# HER2, age, t, e), fits Cox models with survival::coxph (Efron ties),
# computes Harrell C-index, Mantel-Cox log-rank test and 5-year KM survival,
# and writes results/tcga_validation.json.
#
# The endpoint is overall survival (days). TCGA-BRCA has no curated
# relapse-free-survival endpoint; this is a prognostic-association validation.

suppressMessages(library(survival))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
inp <- if (length(args) >= 1) args[1] else "results/tcga_input.csv"
outpath <- if (length(args) >= 2) args[2] else "results/tcga_validation.json"

cc <- read.csv(inp, stringsAsFactors = FALSE)
cc <- cc[complete.cases(cc), ]
cat(sprintf("analysis n = %d, events = %d\n", nrow(cc), sum(cc$e)))

fit_summary <- function(fit) {
  s <- summary(fit)
  cf <- s$coefficients
  ci <- s$conf.int
  nms <- rownames(cf)
  list(
    Covariates = nms,
    beta = as.numeric(cf[, "coef"]),
    SE = as.numeric(cf[, "se(coef)"]),
    HR = as.numeric(ci[, "exp(coef)"]),
    CI = lapply(seq_along(nms), function(i) c(ci[i, "lower .95"], ci[i, "upper .95"])),
    p = as.numeric(cf[, "Pr(>|z|)"]),
    Cindex = as.numeric(concordance(fit)$concordance)
  )
}

fit_m0 <- coxph(Surv(t, e) ~ ER + PR + HER2 + scale(age), data = cc)
fit_m1 <- coxph(Surv(t, e) ~ ER + PR + HER2 + scale(age) + CS, data = cc)
fit_cs <- coxph(Surv(t, e) ~ CS, data = cc)

# log-rank on a median split of CS
med <- median(cc$CS)
cc$grp <- ifelse(cc$CS > med, 1L, 0L)
lr <- survdiff(Surv(t, e) ~ grp, data = cc)
lr_p <- pchisq(lr$chisq, length(lr$n) - 1, lower.tail = FALSE)

# 5-year (1825 days) KM survival per stratum
km <- survfit(Surv(t, e) ~ grp, data = cc)
s5 <- summary(km, times = 1825)

# full KM curves exported for Fig. 10 (analysis cohort only); km$strata is a
# named vector of per-stratum row counts, so the strata labels are expanded
# to align with the time/survival vectors
strata_names <- rep(sub("grp=", "", names(km$strata)), km$strata)
km_rows <- data.frame(stratum = strata_names, time_days = km$time,
                      survival = km$surv, n_at_risk = km$n.risk)
write.csv(km_rows, "results/tcga_km_data.csv", row.names = FALSE)

# subgroup analyses (Section 3.7): ER+/ER- strata and CS tertiles
er_pos <- cc[cc$ER == 1, ]
er_neg <- cc[cc$ER == 0, ]
fit_erp <- tryCatch(coxph(Surv(t, e) ~ CS, data = er_pos),
                    error = function(e) NULL)
fit_ern <- tryCatch(coxph(Surv(t, e) ~ CS, data = er_neg),
                    error = function(e) NULL)
sub_hr <- function(fit) {
  if (is.null(fit)) return(list(HR = NA, p = NA))
  s <- summary(fit)
  list(HR = unname(s$conf.int[1, "exp(coef)"]),
       p = unname(s$coefficients[1, "Pr(>|z|)"]))
}
q3 <- quantile(cc$CS, c(1/3, 2/3))
tert <- cut(cc$CS, breaks = c(-Inf, q3[1], q3[2], Inf),
            labels = c("T1", "T2", "T3"))
lr_tert <- survdiff(Surv(t, e) ~ tert, data = cc)
tert_p <- pchisq(lr_tert$chisq, length(lr_tert$n) - 1, lower.tail = FALSE)

res <- list(
  subgroups = list(
    ER_pos = c(n = nrow(er_pos), sub_hr(fit_erp)),
    ER_neg = c(n = nrow(er_neg), sub_hr(fit_ern)),
    CS_tertile_logrank_p = tert_p
  ),
  cohort = "TCGA-BRCA",
  endpoint = "overall survival (days)",
  analysis_n = nrow(cc),
  events = sum(cc$e),
  event_rate = sum(cc$e) / nrow(cc),
  software = "R survival",
  survival_version = as.character(packageVersion("survival")),
  M0 = fit_summary(fit_m0),
  M1 = fit_summary(fit_m1),
  CS_univariable = {
    x <- fit_summary(fit_cs)
    list(
      Covariates = "CS",
      beta = x$beta[1],
      SE = x$SE[1],
      HR = x$HR[1],
      CI = x$CI[[1]],
      p = x$p[1],
      Cindex = x$Cindex
    )
  },
  logrank = list(
    chi2 = as.numeric(lr$chisq),
    p = lr_p,
    median_CS = med,
    n_high = sum(cc$grp),
    n_low = sum(1 - cc$grp)
  ),
  km_5yr_survival = list(
    CS_low = as.numeric(s5$surv[1]),
    CS_high = as.numeric(s5$surv[2])
  )
)

write(toJSON(res, digits = 10, auto_unbox = TRUE), outpath)
cat("saved", outpath, "\n")
cat(toJSON(res, digits = 6, auto_unbox = TRUE), "\n")
