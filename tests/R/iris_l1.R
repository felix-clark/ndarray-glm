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
# scale externally so that we can match the operation
x_data <- scale(x_data)
# This class is seperable, so this will test regularization
# y_data <- data["class"] == "setosa"
y_data <- data["class"] == "versicolor"
l1 <- 1e-2 / length(y_data)
# NOTE: We had trouble with convergence for larger lambda
# l1 <- 0.1 / length(y_data)
model <- glmnet(
  x_data, y_data,
  # Standardization is recommended particularly for L1, although the result is
  # re-scaled anyway.
  standardize = FALSE,
  # the LASSO penalty
  alpha = 1,
  # With this dataset, a large lambda zeros out all coefficients
  lambda = l1,
  # lambda = 0,
  thresh = 1e-20,
  family = "binomial",
)
beta <- coef(model)
print(beta)
beta <- beta[, "s0"]
write(beta, file = "log_regularization/iris_setosa_l1_1e-2.csv", sep = "\n")
