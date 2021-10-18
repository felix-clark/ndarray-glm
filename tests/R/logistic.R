#!/usr/bin/env Rscript

infile <- "../data/log_termination_0.csv"
data <- read.csv(infile, header=FALSE)
model <- glm(V1 ~ V2, data, family="binomial", offset=V3)
print(model)
coefs <- model$coefficients
write(coefs, file="log_termination_0/coefficients.csv", ncolumns=1)
mod_sum <- summary(model)
write(mod_sum$deviance.resid, file="log_termination_0/dev_resid.csv", ncolumns=1)
# For logistic the scaled and unscaled covariances are the same
# print(mod_sum$cov.scaled)
cov_mat <- mod_sum$cov.unscaled
write(cov_mat, file="log_termination_0/covariance.csv", ncolumns=1)
write(model$deviance, file="log_termination_0/deviance.csv", ncolumns=1)
write(model$null.deviance, file="log_termination_0/null_dev.csv", ncolumns=1)
write(model$aic, file="log_termination_0/aic.csv", ncolumns=1)
write(BIC(model), file="log_termination_0/bic.csv", ncolumns=1)
write(rstandard(model, type="pearson"), file="log_termination_0/standard_pearson_resid.csv", ncolumns=1)
write(rstandard(model, type="deviance"), file="log_termination_0/standard_deviance_resid.csv", ncolumns=1)
write(rstudent(model), file="log_termination_0/student_resid.csv", ncolumns=1)

# TODO: wald and score tests, bic, etc.
# wald_score <- wald.test(cov_mat, coefs)
# write(wald_score, file="log_termination_0/wald.csv")
