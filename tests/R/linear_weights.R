#!/usr/bin/env Rscript
# Generate reference data for linear model weight tests.
# Run from tests/R/ directory: Rscript linear_weights.R

# Use full double precision in output
options(digits = 17)

set.seed(42)
n <- 25
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- 0.5 * x1 + rnorm(n, sd = 0.5)
y <- 1.5 + 0.8 * x1 - 1.2 * x2 + 0.3 * x3 + rnorm(n, sd = 0.5)
var_wt <- runif(n, 0.5, 2.0)
freq_wt <- sample(1:3, n, replace = TRUE)

# Save dataset with header
write.csv(data.frame(y, x1, x2, x3, var_wt, freq_wt),
          file = "../data/linear_weights.csv", row.names = FALSE)

# Helper: export all quantities for a glm fit.
# If orig_idx is provided, only export quantities at those indices (for freq-expanded scenarios).
export_scenario <- function(model, dir_name, orig_idx = NULL) {
  dir.create(dir_name, showWarnings = FALSE, recursive = TRUE)
  ms <- summary(model)

  # Subset helper: extract values at original observation indices
  sub <- function(x) {
    if (!is.null(orig_idx)) x[orig_idx] else x
  }
  sub_mat <- function(x) {
    if (!is.null(orig_idx)) x[orig_idx, , drop = FALSE] else x
  }

  write(model$coefficients, file.path(dir_name, "coefficients.csv"), ncolumns = 1)
  write(ms$cov.scaled, file.path(dir_name, "covariance.csv"), ncolumns = 1)
  write(ms$dispersion, file.path(dir_name, "dispersion.csv"), ncolumns = 1)
  write(model$deviance, file.path(dir_name, "deviance.csv"), ncolumns = 1)
  write(model$null.deviance, file.path(dir_name, "null_deviance.csv"), ncolumns = 1)
  write(model$aic, file.path(dir_name, "aic.csv"), ncolumns = 1)
  write(BIC(model), file.path(dir_name, "bic.csv"), ncolumns = 1)

  # Residuals (per original observation)
  write(sub(residuals(model, type = "response")), file.path(dir_name, "resid_resp.csv"), ncolumns = 1)
  write(sub(residuals(model, type = "pearson")), file.path(dir_name, "resid_pear.csv"), ncolumns = 1)
  write(sub(residuals(model, type = "deviance")), file.path(dir_name, "resid_dev.csv"), ncolumns = 1)
  write(sub(residuals(model, type = "working")), file.path(dir_name, "resid_work.csv"), ncolumns = 1)
  write(sub(rstandard(model, type = "pearson")), file.path(dir_name, "resid_pear_std.csv"), ncolumns = 1)
  write(sub(rstandard(model, type = "deviance")), file.path(dir_name, "resid_dev_std.csv"), ncolumns = 1)
  write(sub(rstudent(model)), file.path(dir_name, "resid_student.csv"), ncolumns = 1)

  # Leverage & influence (per original observation)
  infl <- influence(model, do.coef = TRUE)
  write(sub(infl$hat), file.path(dir_name, "leverage.csv"), ncolumns = 1)
  # influence coefficients: write column-major via t()
  write(t(sub_mat(infl$coefficients)), file.path(dir_name, "loo_coef.csv"), ncolumns = 1)

  # Wald t-values
  write(ms$coefficients[, "t value"], file.path(dir_name, "wald_z.csv"), ncolumns = 1)

  cat("Exported scenario:", dir_name, "\n")
}

# Scenario: no weights
m_none <- glm(y ~ x1 + x2 + x3, family = gaussian())
export_scenario(m_none, "linear_weights/none")

# Scenario: variance weights only
m_var <- glm(y ~ x1 + x2 + x3, family = gaussian(), weights = var_wt)
export_scenario(m_var, "linear_weights/var")

# Expand rows by frequency for freq-weight scenarios
expand_by_freq <- function(df, freq_col) {
  idx <- rep(seq_len(nrow(df)), df[[freq_col]])
  list(data = df[idx, ], orig_idx = cumsum(df[[freq_col]]) - df[[freq_col]] + 1)
}
df <- data.frame(y, x1, x2, x3, var_wt, freq_wt)
exp_result <- expand_by_freq(df, "freq_wt")
df_exp <- exp_result$data
# Indices of first occurrence of each original observation in expanded data
orig_idx <- exp_result$orig_idx

# Scenario: frequency weights only
m_freq <- glm(y ~ x1 + x2 + x3, data = df_exp, family = gaussian())
export_scenario(m_freq, "linear_weights/freq", orig_idx)

# Scenario: both weights
m_both <- glm(y ~ x1 + x2 + x3, data = df_exp, family = gaussian(), weights = var_wt)
export_scenario(m_both, "linear_weights/both", orig_idx)

# --- Ridge scenarios via glmnet ---
library(glmnet)

x_mat <- as.matrix(data.frame(x1, x2, x3))

# glmnet's lambda is scaled differently from our l2_reg parameter.
# Our library penalizes with l2 * ||beta||^2 (no intercept penalty).
# glmnet penalizes with lambda * ||beta||^2 / (2*n) when alpha=0.
# So to match our l2_reg = 0.1, we set lambda = 2 * 0.1 / n = 0.2 / n.
# Actually, glmnet uses: (1/2n) * ||y - X*beta||^2 + lambda * alpha * ||beta||_1
#                         + lambda * (1-alpha)/2 * ||beta||^2
# Our library uses: -loglike + l2 * ||beta||^2 (excluding intercept)
# For gaussian, -loglike = (1/2) * sum(w * (y - X*beta)^2)
# So glmnet's objective (times n) = our objective when lambda*(1-alpha)/2*n = l2
# i.e. lambda = 2*l2/n for alpha=0.
l2_param <- 0.1
lambda_glmnet <- 2 * l2_param / n

dir.create("linear_weights/ridge_none", showWarnings = FALSE, recursive = TRUE)
m_ridge <- glmnet(x_mat, y, alpha = 0, lambda = lambda_glmnet, standardize = FALSE,
                   family = "gaussian", thresh = 1e-14)
write(as.vector(coef(m_ridge)), "linear_weights/ridge_none/coefficients.csv", ncolumns = 1)
cat("Exported scenario: linear_weights/ridge_none\n")

dir.create("linear_weights/ridge_var", showWarnings = FALSE, recursive = TRUE)
m_ridge_var <- glmnet(x_mat, y, alpha = 0, lambda = lambda_glmnet, standardize = FALSE,
                       weights = var_wt / sum(var_wt) * n,
                       family = "gaussian", thresh = 1e-14)
write(as.vector(coef(m_ridge_var)), "linear_weights/ridge_var/coefficients.csv", ncolumns = 1)
cat("Exported scenario: linear_weights/ridge_var\n")

cat("Done. All scenarios exported.\n")
