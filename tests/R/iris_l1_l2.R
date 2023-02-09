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
y_data <- data["class"] == "setosa"
l1 <- 1e-2 / length(y_data)
l2 <- 1e-2 / length(y_data)
lambda <- l1 + l2
alpha <- l1 / lambda
model <- glmnet(
  x_data, y_data,
  # Standardization is recommended particularly for L1, although the result is
  # re-scaled anyway.
  standardize = FALSE,
  alpha = alpha,
  lambda = lambda,
  thresh = 1e-20,
  family = "binomial",
)
beta <- coef(model)
print(beta)
beta <- beta[, "s0"]
write(
  beta,
  file = "log_regularization/iris_setosa_l1_l2_1e-2.csv",
  sep = "\n",
)
