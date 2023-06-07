import numpy as np
import pandas as pd


# Assumptions of OLS
# 1. Linear form: The data-generating process must be based on a linear functional form
# 2. Strict exogeneity: The errors in the regression have a conditional mean of zero E[\epsilon | X] = 0 -> Error does not depend on the explanatory variable.
# Implies errors having a mean of zero E[\epsilon] = 0, errors not depending on explanatory variables cna also be stated as E[X^T \epsilon] = 0
# 3. Linear independence of regressors. The matrix of explanatory variables X must have full column rank: rank(X) = p where X is a p x p matrix.
# 4. Homoscedasticity: Var[\epsilon | X] = \sigma^2 I_n where \sigma^2 is independent of the dependent variable observation x_i and multiplied with the
# identity matrix I_n, meaning we have the same value on the diagonals and zeros on the off-diagonals, meaning we have no correlation anywhere between
# any of the error terms.
# 4a. The assumption of \sigma^2 being identical across observations can be relaxed to allow for variances that are dependent on the explanatory variable
# -> \sigma^2_i
# 4b. The assumption of no autocorrelation, or Covariance matrix being zero in the off-diagonals, can be relaxed for regressions such as time series analysis
# where we allow for E[\epsilon_i \epsilon_j | X] != 0 (off-diagonals being zero implies E[\epsilon_i \epsilon_j | X] = 0)
# 5. Observations are independent and identically distributed (iid) -> (x_i, y_i) is independent from, and has same distribution as (x_j, y_j) for all i!=j
# This implies no perfect multicollinearity where Q_XX = [x_i x_i^T] is a positive-definite matrix in is equal to (3), exogeneity (2) and homoscedasticity (4)


class my_regression:
    def __init__(self, data:pd.DataFrame, regression_equation:str):

        assert type(data).__name__ == 'DataFrame', f'Explanatory variable argument is not a pandas data frame.'

        # TODO: Operations such as the .append() for the p-values cause issues if we were to reshuffle the order of coefficient names. Need to either
        # add a test that this was not done or implement a functionality so coefficient names can be shuffled but are matched.

        # TODO: Check whether degrees of freedom adjustments are applied correctly for standard error calculation.

        # FB: Remove empty spaces improved by the user. Means function will not work for column names with empty spaces (e.g. 'column name' instead of
        # 'column_name')
        regression_equation = regression_equation.replace(' ', '')
        self.regression_equation = regression_equation

        dependent_variable = regression_equation.split('=')[0]
        explanatory_variables = regression_equation.split('=')[1]
        explanatory_variables = explanatory_variables.split('+')

        intercept = pd.Series(np.ones(shape = (data.shape[0])), name = 'intercept', index = data.index)
        self.X = pd.concat([pd.DataFrame(intercept), data[explanatory_variables]], axis = 1)
        self.Y = data[dependent_variable]


        # Running regression
        # Derivation of OLS. \y signifies a matrix of y_i where \y = Sum_i=1^N y_i. \y has dimensions 1xN, \X has dimensions MxN where M<N, \X^T is the transpose of \X
        # S(\beta) = ||\y - \X \\beta||^2 
        # = (\y - \X \\beta)^T (\y - \X \\beta)
        # = \y^T \y - \\beta^T \X^T \y - \y^T \X \\beta + \\beta^T \X^T \X \\beta
        # = \y^T \y - 2 \\beta^T \X^T \y + \beta^T \X^T \X \\beta
        # Differentiate w.r.t \beta and set to zero:
        # -\X^T \y + (\X^T \X) \\beta = 0
        # => \\beta = (\X^T \X)^-1 \X^T \y 
        # Note that here we can directly see the necessity of OLS assumption (3). If (3) does not hold, (\X^T \X)^-1 cannot be computed because the inverse of a matrix
        # can only be computed if your matrix is a square matrix. If \X has full column rank, this implies that \X^T \X is a square matrix.
        X = np.array(self.X)
        Y = np.array(self.Y)
        betas_hat = np.linalg.inv(X.T @ X) @ X.T @ Y



        # Derivation of beta variance:
        # Start with definition of \hat{\beta}:
        # \hat{\\beta} = (\X^T \X)^-1 \X^T (\X \\beta + \u)
        # = \\beta + (\X^T \X)^-1 \X^T \u 
        # -> Decomposition of \hat{\\beta} into true \\beta and error term. Variance originating from error term is:
        # V\hat{\\beta} = E(\hat{\\beta} - \\beta) (\hat{\\beta} - \\beta)^T
        # = E(\X^T \X)^-1 \X^T \u \u^T \X (\X^T \X)^-1
        # = (\X^T \X)^-1 \X^T (E(\u \u^T)) \X(\X^T \X)^-1
        # Due to our assumptions of iid observations (cov(u_i, u_j) = 0 for i != j) and spherical error terms (cov(u_i, u_i) = sigma for all i),
        # we can simplify E(\u \u^T) to \sigma^2 \I where \I is the identity matrix. These two assumptions are absolutely crucial and drastically
        # affect the empirical estimate for the standard error.
        # = \sigma^2 (\X^T \X)^-1


        # FB: Model expected y given explanatory variable values as per regression betas
        y_hat = X @ betas_hat
        # FB: Get the empirical error terms
        u_hat = y_hat - Y
        
        # FB: Compute inverse of design matrix X'X once and then re-use it since it is computationally expensive.
        xInv = np.linalg.inv(X.T @ X)
        # FB: Compute empirical variance given by 1/(n_observations - n_regressors) * residuals. This is \sigma^2.
        var_hat = 1/(len(u_hat)-X.shape[1])*np.sum(u_hat**2)
        var_betas = np.multiply(var_hat, xInv)
        standard_errors = np.sqrt(np.diag(var_betas))

        t_values = betas_hat / standard_errors


        # Implementation of CDFs for t distribution up to 30 degrees of freedom and then normal distribution. Samples 1,000,000 random draws
        # to get a representative CDF.
        if len(u_hat)-X.shape[1] < 30:
            degrees_of_freedom = len(u_hat)-X.shape[1]
            CDF = np.random.standard_t(df = degrees_of_freedom, size = int(1e+6))
        else:
            CDF = np.random.normal(size = int(1e+6))
        
        # FB: Here we first sort the random draws to get a CDF and then sort by reverse order to get the correct p-values.
        CDF.sort()
        CDF = CDF[::-1]

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        p_values = []
        for i in range(len(t_values)):
            pos = find_nearest(CDF, t_values[i])
            p_value = pos/1e+6
            p_values.append(p_value)

        # Sample statistics 
        summary_statistics = pd.DataFrame(None, index=self.X.columns, columns = ['beta_coefficient', 'standard_error', 't-value', 'p-value'])
        summary_statistics['beta_coefficient'] = betas_hat
        summary_statistics['standard_error'] = standard_errors
        summary_statistics['t-value'] = t_values
        summary_statistics['p-value'] = p_values

        self.summary_statistics = summary_statistics
        