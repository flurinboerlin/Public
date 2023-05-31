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
    def __init__(self, regression_equation:str, explanatory_variables:pd.DataFrame):

        assert type(dependent_variable).__name__ == 'DataFrame', f'Dependent variable argument is not a pandas data frame.'
        assert type(explanatory_variables).__name__ == 'DataFrame', f'Explanatory variable argument is not a pandas data frame.'

        self.dependent_variable_name = dependent_variable_name
        self.y = dependent_variable
        self.explanatory_variables = explanatory_variables

        if data_cleaning == 'yes':
            # Data cleaning
            # Check for NAs. NA entries are deleted, dependent_variable and explanatory_variables data frames are aligned afterwards.
            self.y = self.y.dropna()
            self.explanatory_variables.dropna()

            interesction_index = self.y.index.intersection(self.explanatory_variables.index)
            self.y = self.y.loc[interesction_index, :]
            self.explanatory_variables = self.explanatory_variables.loc[interesction_index, :]

        self.intercept = pd.Series(np.ones(shape = (self.explanatory_variables.shape[0])), name = 'intercept', index = self.explanatory_variables.index)
        self.X = pd.concat([pd.DataFrame(self.intercept), self.explanatory_variables], axis = 1)

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
        betas_hat = np.squeeze(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), np.array(self.y))))




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
        y_hat = np.matmul(X, betas_hat)
        # FB: Get the empirical error terms
        u_hat = np.array(y_hat - self.y)
        
        # FB: Compute inverse of design matrix X'X once and then re-use it since it is computationally expensive.
        xInv = np.linalg.inv(np.matmul(np.transpose(X), X))
        # FB: Compute empirical variance given by 1/(n_observations - n_regressors) * residuals. This is \sigma^2.
        var_hat = np.array(1/(len(u_hat)-X.shape[1])*np.sum(u_hat**2))
        var_betas = np.multiply(var_hat, xInv)
        standard_errors = np.sqrt(np.diag(var_betas))

        t_values = betas_hat / standard_errors


        # TODO: Implement CDFs for t distribution up to 30 degrees of freedom and then normal distribution.
        if len(u_hat)-X.shape[1] < 30:
            df = len(u_hat)-X.shape[1]
            CDF = np.random.standard_t(df = df, size = int(1e+6))
        else:
            CDF = np.random.normal(size = int(1e+6))

        normalCDF = np.random.normal(loc = 0, scale = 1, size = int(1e+6))
        normalCDF = np.sort(normalCDF)

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
        summary_statistics = pd.DataFrame(None, index = ['mean', 'variance', 'standard error', 'estimator_variance'], columns=self.X.columns)
        sigma_squared = np.var(self.y)
        # Compute sample average, variance and standard errors for every explanatory variable
        for column in self.X.columns:
            summary_statistics.loc['mean', column] = np.mean(self.X.loc[:, column])
            summary_statistics.loc['variance', column] = np.var(self.X.loc[:, column])
            summary_statistics.loc['standard error', column] = np.sqrt(summary_statistics.loc['variance', column])/np.sqrt(N)
            # if (not column == 'intercept'):
                # TODO: Implement calculation of estimator variance
                # summary_statistics.loc['estimator_variance', column] = sigma_squared[0] / summary_statistics.loc['variance', column]
        



        print(f'The name of the dependent variable is {dependent_variable_name}.')
    def calculate_betas(self):
        return(self.dependent_variable + self.explanatory_variables)

# unemployment_rate = fred.get_series('UNRATE')
# ON_rate = fred.get_series('DFF')
# spx = fred.get_series('SP500')


unemployment_rate = pd.read_csv('C:/temp/unemployment.csv')
ON_rate = pd.read_csv('C:/temp/ON_rate.csv', index_col=0)
spx = pd.read_csv('C:/temp/spx.csv', index_col=0)
spx_diff = spx.diff(axis = 0)

# FB: Align series based on index which contains daily dates
spx_diff = spx_diff[[x in ON_rate.index for x in spx_diff.index]]
ON_rate = ON_rate[[x in spx_diff.index for x in ON_rate.index]]


dependent_variable = pd.DataFrame([1,2,3,4,5,6,7,8,9,10])
explanatory_variable = pd.DataFrame([3,5,7,9,11,12.5,15,17,20,21])
regression = my_regression(dependent_variable_name='test', dependent_variable=dependent_variable, explanatory_variables=explanatory_variable)


print(regression.calculate_betas())
print(regression.dependent_variable_name)
print(regression.__dict__)


# test

print('hello')