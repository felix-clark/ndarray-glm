#!/usr/bin/env Rscript

infile <- "../data/log_weights.csv"
data <- read.csv(infile, header = FALSE)
print(data)
model <- glm(V1 ~ V2 + V3, data, family = "binomial", weights = V4)
# model <- glm(V1 ~ V2 + V3, data, family = "binomial")
print(model)
coefs <- model$coefficients
write(coefs, file = "log_weights/coefficients.csv", ncolumns = 1)

mod_sum <- summary(model)
# For logistic the scaled and unscaled covariances are the same
cov_mat <- mod_sum$cov.unscaled
write(cov_mat, file = "log_weights/covariance.csv", ncolumns = 1)

write(model$deviance, file = "log_weights/deviance.csv", ncolumns = 1)

write(model$null.deviance, file = "log_weights/null_dev.csv", ncolumns = 1)

infl <- influence(model, do.coef = TRUE)
print(infl)
# These aren't the coefficients per se, but the contribution to each
# coefficient from each observation. Subtracting these from the results
# approximates the coefficients that would arise from leaving out the
# observation.
write(
  t(infl$coef),
  file = "log_weights/loo_coef.csv", ncolumns = 1
)

write(
  infl$hat,
  file = "log_weights/hat.csv", ncolumns = 1
)

# check unstandardized for consistency
write(
  infl$pear.res,
  file = "log_weights/pearson_resid.csv", ncolumns = 1
)
# these two are equivalent:
# mod_sum$deviance.resid and infl$dev.res
write(mod_sum$deviance.resid, file = "log_weights/dev_resid.csv", ncolumns = 1)

write(
  rstandard(model, type = "pearson"),
  file = "log_weights/standard_pearson_resid.csv", ncolumns = 1
)
write(
  rstandard(model, type = "deviance"),
  file = "log_weights/standard_deviance_resid.csv", ncolumns = 1
)
write(rstudent(model), file = "log_weights/student_resid.csv", ncolumns = 1)

# TODO: cooks distance?

# AIC and BIC do not match; these are unique only to an additive constant which
# is possibly responsible for the descrepancy. The odd bit is that they seem to
# agree in the unweighted case.
write(model$aic, file = "log_weights/aic.csv", ncolumns = 1)
write(BIC(model), file = "log_weights/bic.csv", ncolumns = 1)

# TODO: wald and score tests, etc.
# wald_score <- wald.test(cov_mat, coefs)
# write(wald_score, file="log_weights/wald.csv")
