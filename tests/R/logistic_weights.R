#!/usr/bin/env Rscript

infile <- "../data/log_weights.csv"
data <- read.csv(infile, header = FALSE)
model <- glm(V1 ~ V2 + V3, data, family = "binomial", weights = V4)
coefs <- model$coefficients
write(coefs, file = "log_weights/coefficients.csv", ncolumns = 1)
