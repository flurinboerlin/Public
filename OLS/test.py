import numpy as np
from numpy.linalg import inv

# Prepare your data
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Independent variable matrix
y = np.array([2, 4, 5, 4, 5])  # Dependent variable

# Compute the regression coefficients
beta_hat = inv(X.T @ X) @ X.T @ y

# Compute the residuals
residuals = y - X @ beta_hat

# Compute the residual sum of squares (RSS)
RSS = residuals.T @ residuals

# Compute the degrees of freedom
n = len(y)  # Number of observations
k = X.shape[1] - 1  # Number of independent variables (excluding the intercept)
df = n - k - 1  # Degrees of freedom

# Compute the variance-covariance matrix of the residuals
variance_covariance = RSS / df * inv(X.T @ X)

# Extract the standard errors from the diagonal of the variance-covariance matrix
standard_errors = np.sqrt(np.diag(variance_covariance))

# Print the standard errors
print(standard_errors)