#!/usr/bin/env Rscript

# This script uses glmnet:
# https://glmnet.stanford.edu/articles/glmnet.html
library(glmnet)

infile <- "../data/iris.csv"
data <- read.csv(infile, header = TRUE)
x_data <- data[
  ,
  c("sepal_length", "sepal_width", "petal_length", "petal_width")
]
# This class is seperable, so this will test regularization
y_data <- data["class"] == "setosa"
# y_data <- data["class"] == "versicolor"
# The glmnet package divides the squared errors by N, so to match our
# convention we need to scale their lambda too.
l2 <- 1e-2 / length(y_data)
model <- glmnet(
  x_data, y_data,
  # Standardization is recommended particularly for L1, but it doesn't change
  # the result because the result is re-scaled
  # standardize = TRUE,
  standardize = FALSE,
  # the ridge penalty
  alpha = 0,
  # With this dataset, a large lambda zeros out all coefficients
  lambda = l2,
  # the tolerance has to be much smaller to make this result more precise.
  thresh = 1e-10,
  family = "binomial",
)
print(model)
beta <- coef(model)
print(beta)
beta <- beta[, "s0"]
# There are convenience functions to read array from single-column files
write(beta, file = "log_regularization/iris_setosa_l2_1e-2.csv", sep = "\n")
