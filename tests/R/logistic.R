#!/usr/bin/env Rscript
# Generate reference data for comprehensive logistic regression tests.
# Run from tests/R/ directory: Rscript logistic.R

options(digits = 17)

# Use near-machine-precision convergence so reference targets are as accurate as possible.
# R's default epsilon is 1e-8 (relative deviance change); we use 1e-14 to get close to
# machine precision without the oscillation issues that arise at .Machine$double.eps.
ctrl <- glm.control(epsilon = 1e-14, maxit = 10000)

set.seed(99)
n <- 30
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- 0.5 * x1 + rnorm(n, sd = 0.5)
eta_true <- 0.5 + 0.8 * x1 - 1.2 * x2 + 0.3 * x3
p_true <- 1 / (1 + exp(-eta_true))

# Boolean response: binary 0/1
y_bool <- as.integer(rbinom(n, 1, p_true))
# Float response in (0,1): use true probabilities (clamped to avoid log(0))
y_float <- pmax(0.0001, pmin(0.9999, rbeta(n, p_true, 1. - p_true)))

var_wt <- runif(n, 0.5, 2.0)
freq_wt <- sample(1:3, n, replace = TRUE)

# Save dataset with header
write.csv(
  data.frame(y_bool, y_float, x1, x2, x3, var_wt, freq_wt),
  file = "../data/logistic.csv", row.names = FALSE
)

# Helper: expand rows by frequency weight
expand_by_freq <- function(df, freq_col) {
  idx <- rep(seq_len(nrow(df)), df[[freq_col]])
  list(data = df[idx, ], orig_idx = cumsum(df[[freq_col]]) - df[[freq_col]] + 1)
}

df <- data.frame(y_bool, y_float, x1, x2, x3, var_wt, freq_wt)
exp_result <- expand_by_freq(df, "freq_wt")
df_exp <- exp_result$data
orig_idx <- exp_result$orig_idx

# Fixed held-out test observations for prediction checks (3 rows, not in training set).
# These values are hard-coded in the Rust test.
x_test <- data.frame(x1 = c(0.5, -1.0, 2.0), x2 = c(0.3, -0.5, 1.2), x3 = c(0.1, 0.8, -0.3))
off_test <- 0.5 * x_test$x1       # c(0.25, -0.50, 1.00), matches Rust's offset_from_x
x_test_sub <- x_test[, c("x2", "x3")]  # for no-intercept+offset scenarios (2 predictors)

# Helper: export all quantities for a logistic glm fit.
# If orig_idx is provided, only export per-obs quantities at those indices (freq-expanded).
export_scenario <- function(model, dir_name, orig_idx = NULL, model_no_int = NULL,
                             pred_data = NULL, pred_off = NULL) {
  dir.create(dir_name, showWarnings = FALSE, recursive = TRUE)
  ms <- summary(model)

  sub <- function(x) {
    if (!is.null(orig_idx)) x[orig_idx] else x
  }
  sub_mat <- function(x) {
    if (!is.null(orig_idx)) x[orig_idx, , drop = FALSE] else x
  }

  write(model$coefficients, file.path(dir_name, "coefficients.csv"), ncolumns = 1)
  # For binomial/logistic, cov.scaled == cov.unscaled (dispersion = 1)
  write(ms$cov.scaled,      file.path(dir_name, "covariance.csv"),    ncolumns = 1)
  write(ms$dispersion,      file.path(dir_name, "dispersion.csv"),     ncolumns = 1)
  write(model$deviance,     file.path(dir_name, "deviance.csv"),       ncolumns = 1)
  write(model$null.deviance,file.path(dir_name, "null_deviance.csv"),  ncolumns = 1)
  write(model$aic,          file.path(dir_name, "aic.csv"),            ncolumns = 1)
  write(BIC(model),         file.path(dir_name, "bic.csv"),            ncolumns = 1)

  # Residuals (per original observation)
  write(sub(residuals(model, type = "response")), file.path(dir_name, "resid_resp.csv"),     ncolumns = 1)
  write(sub(residuals(model, type = "pearson")),  file.path(dir_name, "resid_pear.csv"),     ncolumns = 1)
  write(sub(residuals(model, type = "deviance")), file.path(dir_name, "resid_dev.csv"),      ncolumns = 1)
  write(sub(residuals(model, type = "working")),  file.path(dir_name, "resid_work.csv"),     ncolumns = 1)
  # Partial residuals: write row-major (each row is one observation)
  write(t(sub_mat(residuals(model, type = "partial"))),
        file.path(dir_name, "resid_partial.csv"), ncolumns = 1)
  write(sub(rstandard(model, type = "pearson")),  file.path(dir_name, "resid_pear_std.csv"), ncolumns = 1)
  write(sub(rstandard(model, type = "deviance")), file.path(dir_name, "resid_dev_std.csv"),  ncolumns = 1)
  write(sub(rstudent(model)),                     file.path(dir_name, "resid_student.csv"),  ncolumns = 1)

  # Leverage & influence (per original observation)
  infl <- influence(model, do.coef = TRUE)
  write(sub(infl$hat),                         file.path(dir_name, "leverage.csv"), ncolumns = 1)
  # influence coefficients: write column-major via t()
  write(t(sub_mat(infl$coefficients)),         file.path(dir_name, "loo_coef.csv"), ncolumns = 1)
  write(sub(cooks.distance(model)),            file.path(dir_name, "cooks.csv"),    ncolumns = 1)

  # Wald z-values (binomial uses z, not t)
  write(ms$coefficients[, "z value"],          file.path(dir_name, "wald_z.csv"),   ncolumns = 1)

  # --- P-values ---

  # pvalue_lr_test: chi-squared omnibus test
  lr_df <- model$df.null - model$df.residual
  pvalue_lr <- pchisq(model$null.deviance - model$deviance, df = lr_df, lower.tail = FALSE)
  write(pvalue_lr, file.path(dir_name, "pvalue_lr_test.csv"), ncolumns = 1)

  # pvalue_wald: Pr(>|z|) for binomial
  write(ms$coefficients[, "Pr(>|z|)"],         file.path(dir_name, "pvalue_wald.csv"), ncolumns = 1)

  # pvalue_exact: chi-squared drop-one test (test = "Chisq" for logistic)
  d1 <- drop1(model, test = "Chisq")
  pred_p <- d1[["Pr(>Chi)"]][-1]  # drop the <none> row
  if (!is.null(model_no_int)) {
    intercept_p <- anova(model_no_int, model, test = "Chisq")[["Pr(>Chi)"]][2]
    write(c(intercept_p, pred_p), file.path(dir_name, "pvalue_exact.csv"), ncolumns = 1)
  } else {
    write(pred_p, file.path(dir_name, "pvalue_exact.csv"), ncolumns = 1)
  }

  # Prediction on held-out test observations.
  if (!is.null(pred_data)) {
    if (!is.null(pred_off)) {
      # No-intercept + offset: compute manually to avoid R's newdata+offset recycling issue
      new_lp <- as.vector(as.matrix(pred_data) %*% model$coefficients + pred_off)
      new_pred <- model$family$linkinv(new_lp)
    } else {
      new_pred <- predict(model, newdata = pred_data, type = "response")
    }
    write(new_pred, file.path(dir_name, "predict_resp.csv"), ncolumns = 1)
  }

  cat("Exported scenario:", dir_name, "\n")
}

# --- Bool response scenarios ---

m_bool_none <- glm(y_bool ~ x1 + x2 + x3, data = df, family = binomial(), control = ctrl)
m_bool_none_noint <- glm(y_bool ~ x1 + x2 + x3 - 1, data = df, family = binomial(), control = ctrl)
export_scenario(m_bool_none, "logistic_results/bool_none",
                model_no_int = m_bool_none_noint, pred_data = x_test)

m_bool_var <- glm(y_bool ~ x1 + x2 + x3, data = df, family = binomial(), weights = var_wt, control = ctrl)
m_bool_var_noint <- glm(y_bool ~ x1 + x2 + x3 - 1, data = df, family = binomial(), weights = var_wt, control = ctrl)
export_scenario(m_bool_var, "logistic_results/bool_var",
                model_no_int = m_bool_var_noint, pred_data = x_test)

m_bool_freq <- glm(y_bool ~ x1 + x2 + x3, data = df_exp, family = binomial(), control = ctrl)
m_bool_freq_noint <- glm(y_bool ~ x1 + x2 + x3 - 1, data = df_exp, family = binomial(), control = ctrl)
export_scenario(m_bool_freq, "logistic_results/bool_freq", orig_idx,
                model_no_int = m_bool_freq_noint, pred_data = x_test)

m_bool_both <- glm(y_bool ~ x1 + x2 + x3, data = df_exp, family = binomial(), weights = var_wt, control = ctrl)
m_bool_both_noint <- glm(y_bool ~ x1 + x2 + x3 - 1, data = df_exp, family = binomial(), weights = var_wt, control = ctrl)
export_scenario(m_bool_both, "logistic_results/bool_both", orig_idx,
                model_no_int = m_bool_both_noint, pred_data = x_test)

# bool_off: no intercept, offset = 0.5 * x1, fit y ~ x2 + x3 - 1
off <- 0.5 * x1
m_bool_off <- glm(y_bool ~ x2 + x3 - 1, offset = off, data = df, family = binomial(), control = ctrl)
export_scenario(m_bool_off, "logistic_results/bool_off",
                pred_data = x_test_sub, pred_off = off_test)

# --- Float response scenarios (y in (0,1)) ---
# R gives a "non-integer successes" warning; this is expected.

m_float_none <- glm(y_float ~ x1 + x2 + x3, data = df, family = binomial(), control = ctrl)
m_float_none_noint <- glm(y_float ~ x1 + x2 + x3 - 1, data = df, family = binomial(), control = ctrl)
export_scenario(m_float_none, "logistic_results/float_none",
                model_no_int = m_float_none_noint, pred_data = x_test)

m_float_var <- glm(y_float ~ x1 + x2 + x3, data = df, family = binomial(), weights = var_wt, control = ctrl)
m_float_var_noint <- glm(y_float ~ x1 + x2 + x3 - 1, data = df, family = binomial(), weights = var_wt, control = ctrl)
export_scenario(m_float_var, "logistic_results/float_var",
                model_no_int = m_float_var_noint, pred_data = x_test)

cat("Done. All logistic scenarios exported.\n")
