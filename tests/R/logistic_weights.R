#!/usr/bin/env Rscript

infile <- "../data/log_weights.csv"
data <- read.csv(infile, header = FALSE)
print(data)
model <- glm(V1 ~ V2 + V3, data, family = "binomial", weights = V4)
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
# the leave-one-out coefficients
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
