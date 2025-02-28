# Load necessary libraries
library(ISLR)
library(leaps)
library(lars)

# Load and prepare the data
Auto[1:5,]
Auto.mat <- as.matrix(Auto[,1:8])
Auto.df <- as.data.frame(Auto.mat)

# Function to create second-order polynomial terms
matrix.2ndorder.make <- function(x, only.quad = F) {
  x0 <- x
  dimn <- dimnames(x)[[2]] # Extract column names
  num.col <- length(x[1,]) # Number of columns
  for(i in 1:num.col) {
    if (!only.quad) {
      for(j in i:num.col) {
        x0 <- cbind(x0, x[, i] * x[, j]) # Interaction term
        dimn <- c(dimn, paste(dimn[i], dimn[j], sep = ":"))
      }
    } else {
      # Squared terms
      x0 <- cbind(x0, x[, i] * x[, i]) # Squared term
      dimn <- c(dimn, paste(dimn[i], "2", sep = ":"))
    }
  }
  dimnames(x0)[[2]] <- dimn
  x0
}

# Create second-order terms for the Auto dataset
Auto.mat <- as.matrix(Auto[, 1:8])  # Original dataset matrix
Auto.mat2nd <- matrix.2ndorder.make(Auto.mat[, -1])  # Remove 'mpg' for predictors

# Set seed for reproducibility
set.seed(123)

# Create a random split for training and testing data
train_indices <- sample(1:nrow(Auto), size = 0.7 * nrow(Auto))
train_data <- Auto[train_indices, ]
test_data <- Auto[-train_indices, ]

# Use regsubsets for subset selection with really.big=T for large number of predictors
regsubsets.model <- regsubsets(mpg ~ ., data = Auto, nvmax = 30, really.big = TRUE)

# View the summary of regsubsets
regsubsets_summary <- summary(regsubsets.model)

# Plot Cp vs number of predictors
plot(regsubsets_summary$cp, type = "b", xlab = "Number of Predictors", ylab = "Cp")

# Get the best model by Cp
best_model_index <- which.min(regsubsets_summary$cp)
best_model_index

# Fit the best model using the selected predictors from regsubsets
best_predictors <- regsubsets_summary$which[best_model_index, ]
best_predictors

# Fit the model using the selected features (based on regsubsets output)
best_model <- lm(mpg ~ ., data = train_data[, c(best_predictors, "mpg")])

# Fit the model using LARS (Least Angle Regression)
lars_model <- lars(Auto.mat2nd[train_indices, ], Auto$mpg[train_indices])

# Plot Cp vs df for model selection using LARS
plot(lars_model$df, lars_model$Cp, log = "y", main = "LARS: Cp vs df")

# Get predictions from LARS
lars_predictions <- predict(lars_model, Auto.mat2nd[-train_indices, ])

# Predictions from the best Leaps model
leaps_predictions <- predict(best_model, newdata = test_data)

# Compare predictions with actual values (test data)
par(mfrow = c(1, 2))  # Plot side by side
plot(leaps_predictions, test_data$mpg, main = "Leaps: Predictions vs Actual", xlab = "Predictions", ylab = "Actual")
plot(lars_predictions, test_data$mpg, main = "LARS: Predictions vs Actual", xlab = "Predictions", ylab = "Actual")

# Filter data by countries: US, Germany, Japan
Auto.usa <- Auto[Auto$origin == 1, ]
Auto.germany <- Auto[Auto$origin == 2, ]
Auto.japan <- Auto[Auto$origin == 3, ]

# Create second-order terms for each country
Auto.mat.usa2nd <- matrix.2ndorder.make(Auto.usa[, -1])
Auto.mat.germany2nd <- matrix.2ndorder.make(Auto.germany[, -1])
Auto.mat.japan2nd <- matrix.2ndorder.make(Auto.japan[, -1])

# Fit models and compare them for each country (use steps from above)

# Analyze the non-zero coefficients in the LARS and selected coefficients in Leaps
lars_coeffs <- lars_model$beta
leaps_coeffs <- coef(best_model)

# Compare non-zero coefficients
lars_coeffs_nonzero <- lars_coeffs[lars_coeffs != 0]
leaps_coeffs_nonzero <- leaps_coeffs[leaps_coeffs != 0]

# Interpret differences
# For example, if certain features have non-zero coefficients for one country but not others,
# it might indicate which features are more important for designing to increase mpg in that country.

# Function to calculate PRESS for a given model
calculate_PRESS <- function(model, X, y) {
  # Predicted values (fitted values) from the model
  y_hat <- predict(model, newdata = X)
  
  # Calculate leverage (hii)
  hii <- hatvalues(model)
  
  # Compute PRESS
  press <- sum(((y - y_hat) / (1 - hii))^2)
  
  return(press)
}

# Calculate PRESS for the LARS model (using test data)
press_lars <- calculate_PRESS(lars_model, Auto.mat2nd[train_indices, ], Auto$mpg[train_indices])
press_lars