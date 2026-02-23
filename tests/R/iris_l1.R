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
# y_data <- data["class"] == "setosa"
y_data <- data["class"] == "versicolor"
l1 <- 1e-2 / length(y_data)
alpha = 1

# Do the model with standardization
model <- glmnet(
  x_data, y_data,
  standardize = TRUE,
  # the LASSO penalty
  alpha = alpha,
  # With this dataset, a large lambda zeros out all coefficients
  lambda = l1,
  thresh = 1e-20,
  family = "binomial",
)
beta <- coef(model)
print(beta)
beta <- beta[, "s0"]
write(beta, file = "log_regularization/iris_versicolor_l1_1e-2.csv", sep = "\n")

# Then do the model without standardization
model <- glmnet(
  x_data, y_data,
  standardize = FALSE,
  # the LASSO penalty
  alpha = alpha,
  # With this dataset, a large lambda zeros out all coefficients
  lambda = l1,
  thresh = 1e-20,
  family = "binomial",
)
beta <- coef(model)
print(beta)
beta <- beta[, "s0"]
write(beta, file = "log_regularization/iris_versicolor_l1_1e-2_nostd.csv", sep = "\n")
