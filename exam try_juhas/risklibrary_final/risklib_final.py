from scipy.stats import moment
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, lognorm, kurtosis
from scipy.optimize import minimize, fsolve
from scipy.integrate import quad
import pandas as pd
import time
from tqdm import tqdm
import math
from numpy.linalg import eig
from datetime import datetime
import scipy
from statsmodels.tsa.arima.model import ARIMA
import copy
from typing import List

# The covariance_matrix function calculates the covariance matrix of a given DataFrame input_df,
# offering two modes: if skipna is True, it calculates covariance after dropping rows with any missing values;
# if skipna is False, it calculates covariance while considering all rows, handling missing values pairwise.
# The function then sets the column and index names of the covariance matrix to match those of the input DataFrame.


# This function is used to calculate the covariance matrix in two ways: pairwise or skipping rows
def covariance_matrix(input_df, skipna=True):
    # calculate the covariance matrix either pairwise or skipping rows

    if skipna:
        cov_matrix = input_df.dropna().cov()
    else:
        cov_matrix = input_df.cov()

    # Set the column and index names to match the input
    cov_matrix.columns = input_df.columns
    cov_matrix.index = input_df.columns

    return cov_matrix


# Per Dom's Notes (Does the same job as covariance_matrix up top)
# calculate either the covariance or correlation function when there are missing values
def missing_cov(x, skipMiss=True, fun=np.cov):
    # Check for missing values
    if not x.isnull().values.any():
        return fun(x, rowvar=False)

    if skipMiss:
        # Skip rows with any missing values
        valid_rows = x.dropna()
        return fun(valid_rows, rowvar=False)
    else:
        # Pairwise calculation
        m = x.shape[1]
        out = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                valid_rows = x.iloc[:, [i, j]].dropna()
                out[i, j] = (
                    fun(valid_rows, rowvar=False)[0, 1]
                    if len(valid_rows) > 1
                    else np.nan
                )
                if i != j:
                    out[j, i] = out[i, j]
        return out


# this function is used to calculate the correlation matrix in two ways: pairwise or skipping rows
def missing_corr(x, skipMiss=True):
    # Check for missing values
    if not x.isnull().values.any():
        return np.corrcoef(x, rowvar=False)

    if skipMiss:
        # Skip rows with any missing values
        valid_rows = x.dropna()
        return np.corrcoef(valid_rows, rowvar=False)
    else:
        # Pairwise calculation
        m = x.shape[1]
        out = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                valid_rows = x.iloc[:, [i, j]].dropna()
                if len(valid_rows) > 1:
                    out[i, j] = np.corrcoef(valid_rows, rowvar=False)[0, 1]
                else:
                    out[i, j] = np.nan
                if i != j:
                    out[j, i] = out[i, j]
        return out


# another way to do so
def correlation_matrix(input_df, skipna=True):
    # calculate the correlation matrix either pairwise or skipping rows
    if skipna:
        corr_matrix = input_df.dropna().corr()
    else:
        corr_matrix = input_df.corr()

    # Set the column and index names to match the input
    corr_matrix.columns = input_df.columns
    corr_matrix.index = input_df.columns

    return corr_matrix


# example usage - week03.jl
# skipMiss = missing_cov(x)
# pairwise = missing_cov(x,skipMiss=false)
# eigvals(pairwise)


# The `first4Moments` function calculates and returns the first four statistical moments (mean, variance, skewness, and either regular or excess kurtosis) of a given sample,
# with an option to compute either regular kurtosis or excess kurtosis (kurtosis minus 3).


# this calculates the four kurtosis per first class
def first4Moments(sample, excess_kurtosis=True):
    # Calculate the raw moments
    mean_hat = moment(sample, moment=1)
    var_hat = moment(sample, moment=2, nan_policy="omit")

    # Calculate skewness and kurtosis without dividing
    skew_hat = moment(sample, moment=3)
    kurt_hat = moment(sample, moment=4)

    # Calculate excess kurtosis if excess_kurtosis is True, otherwise return regular kurtosis
    if excess_kurtosis:
        excessKurt_hat = kurt_hat - 3  # Excess kurtosis
        return mean_hat, var_hat, skew_hat, excessKurt_hat
    else:
        return mean_hat, var_hat, skew_hat, kurt_hat  # Regular kurtosis


# A raw OLS function but can be done using statsmodels with their built in function. Not necessary bc sm is being called
def perform_ols(X, y, visualize_error=False):
    # Add a constant term to X matrix for the intercept
    X = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    # Calculate error vector
    error_vector = model.resid

    # visualize error if desired
    if visualize_error:
        # Visualize the error distribution
        plt.figure(figsize=(8, 6))
        sns.histplot(error_vector, kde=True, color="blue", bins=100)
        plt.title("Error Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

    print("Error Vector: ", error_vector)
    averaged_error_vector = np.mean(error_vector)
    print("Averaged Error Vector: ", averaged_error_vector)
    variance_error_vector = np.var(error_vector)
    print("Variance Error Vector: ", variance_error_vector)
    return error_vector


# The `mle_t_distribution` function fits a t-distribution to data `X` using Maximum Likelihood Estimation (MLE) to find the optimized mean, standard deviation,
# and degrees of freedom, and optionally performs a hypothesis test to check if `X` comes from a standard t-distribution,
# reporting the test statistic, p-value, and whether the null hypothesis is rejected.


def mle_t_distribution(X, y, perform_hypothesis_test=False):
    # Define the likelihood function for the t-distribution
    def log_likelihood(mean, var, df, X):
        adjusted_X = X - mean
        var2 = var**2
        log_likeli = np.sum(t.logpdf(adjusted_X / np.sqrt(var2), df))
        return -log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    df_guess = len(X) - 1  # You can adjust the initial guess for degrees of freedom

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess, df_guess]

    # Perform optimization through minimization of log likelihood
    result = minimize(
        lambda params: log_likelihood(params[0], params[1], params[2], X),
        initial_params,
    )

    # Extract optimized parameters
    optimized_mean, optimized_std_dev, optimized_df = result.x

    # Print optimized parameters
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)
    print("Optimized Degrees of Freedom:", optimized_df)

    # Perform hypothesis test if specified
    if perform_hypothesis_test:
        # Calculate test statistic and p-value against standard t-distribution (0, 1, df)
        test_statistic = (optimized_mean - 0) / (
            optimized_std_dev / np.sqrt(optimized_df)
        )
        p_value = 2 * (
            1 - t.cdf(abs(test_statistic), df=optimized_df)
        )  # Two-tailed test

        # Determine if the null hypothesis (X is from a standard t-distribution) is rejected
        reject_null = p_value < 0.05  # Using a significance level of 0.05

        # Print hypothesis test results
        print("Test Statistic:", test_statistic)
        print("P-Value:", p_value)
        print("Reject Null Hypothesis:", reject_null)


# The `mle_normal_distribution_one_input` function fits a normal distribution to the data `X` using Maximum Likelihood Estimation (MLE),
# calculating and returning the optimized mean and standard deviation of the distribution by minimizing the negative log likelihood.


# Fixed MLE for t distribution for just the dataframe values input
def mle_t_distribution_one_input(y):
    # Define the likelihood function for the t-distribution
    def neg_log_likelihood(params, y):
        mean, var, df = params
        adjusted_y = y - mean
        log_likeli = -np.sum(t.logpdf(adjusted_y, df, loc=mean, scale=var))
        return log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    _, _, df_guess = t.fit(y)

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess, df_guess]

    # Perform optimization through minimization of negative log likelihood
    result = minimize(
        neg_log_likelihood, initial_params, args=(y,), method="Nelder-Mead"
    )

    # Extract optimized parameters
    optimized_mean, optimized_std_dev, optimized_df = result.x

    return optimized_mean, optimized_std_dev, optimized_df


# The `simulate_MA` function in Python simulates a Moving Average (MA) process of order `N` over a specified number of steps (`num_steps`), with a noise series `e`,
# a burn-in period (`burn_in`), and a specified mean. It caps values exceeding a `max_threshold`, computes and
# prints the mean and variance of the simulated series (excluding the burn-in period), optionally plots the time series, and returns the simulated data, its mean, and variance.


def simulate_MA(N, num_steps, e, burn_in, mean, plot_y=False, max_threshold=1e4):
    # Initialize y MA preds
    y = np.empty(num_steps)

    # Simulate the MA(N) process
    for i in range(1, num_steps + burn_in):
        y_t = mean + np.sum([0.05 * e[i - j] for j in range(1, N + 1)]) + e[i]
        if i > burn_in:
            # Check if y_t is beyond a certain threshold
            if abs(y_t) > max_threshold:
                y_t = np.sign(y_t) * mean
            y[i - burn_in - 1] = y_t

    # Calculate the mean and variance only for the non-burn-in period
    mean_y = np.mean(y)
    var_y = np.var(y)
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    if plot_y == True:
        # Plot the time series
        plt.figure(figsize=(10, 4))
        plt.plot(y)
        plt.title(f"MA({N}) Time Series")
        plt.xlabel("Timestep")
        plt.ylabel("Y")
        plt.savefig(f"plots/MA_{N}_Steps.png")
        plt.show()

    return y, mean_y, var_y


def simulate_AR(N, num_steps, e, burn_in, mean, plot_y=True):
    # Initialize variables
    n = num_steps
    y = np.empty(n)

    # Simulate the AR(N) process
    for i in range(n + burn_in):
        y_t = mean  # Initialize y_t to the mean

        # Compute the AR(N) value for y_t
        for j in range(1, N + 1):
            if i - j >= 0:
                y_t += (
                    0.5**j * y[i - j - burn_in]
                )  # take a look at removing the burn in

        # Add the white noise
        y_t += e[i]

        # Store the value in the y array if not in the burn-in period
        if i >= burn_in:
            y[i - burn_in] = y_t

    # Optionally plot the time series
    if plot_y:
        plt.figure(figsize=(10, 4))
        plt.plot(y)
        plt.title(f"AR({N}) Time Series")
        plt.xlabel("Timestep")
        plt.ylabel("Y")
        plt.savefig(f"plots/AR_{N}_Steps.png")
        plt.show()

    # Calculate the mean and variance only for the non-burn-in period
    mean_y = np.mean(y[burn_in:])
    var_y = np.var(y[burn_in:])
    print(f"Mean of Y: {mean_y:.4f}")
    print(f"Var of Y: {var_y:.4f}")

    return y, mean_y, var_y


def plot_acf_pacf(y, N, plot_type="AR", save_plots=False):
    # Set custom styling for the plots
    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = "black"
    plt.rcParams["axes.edgecolor"] = "white"
    plt.rcParams["xtick.color"] = "red"
    plt.rcParams["ytick.color"] = "red"
    plt.rcParams["text.color"] = "white"

    # Create a directory to save plots if it doesn't exist
    if save_plots:
        import os

        if not os.path.exists("plots"):
            os.makedirs("plots")

    # Plot the ACF and PACF with red lines
    plt.figure(figsize=(12, 6))

    # ACF plot
    ax1 = plt.subplot(121)
    plot_acf(y, lags=40, ax=ax1, color="red")
    ax1.set_title("Autocorrelation Function (ACF)")

    # PACF plot
    ax2 = plt.subplot(122)
    plot_pacf(y, lags=40, ax=ax2, color="red")
    ax2.set_title("Partial Autocorrelation Function (PACF)")

    # Add an overall title including the plot_type
    plt.suptitle(f"{plot_type}({N}) - ACF and PACF Plots", color="white", fontsize=16)

    plt.savefig(f"plots/{plot_type}_{N}_ACF_PACF.png")

    plt.tight_layout()

    # Display the plots
    plt.show()


def VaR_historical(data, alpha=0.05):
    """Given a dataset(array), calculate the its historical VaR"""
    data.sort()
    n = round(data.shape[0] * alpha)
    return -data[n - 1]


# Generated similar to the historical VaR above but with a different formula for normal distribution
def VaR_normal(data, alpha=0.05):
    """Given a dataset(array), calculate the its normal VaR"""
    mean = np.mean(data)
    std = np.std(data)
    return -(mean + norm.ppf(alpha) * std)


# Calculate VaR Normal Distribution: (Value at Risk)


def calc_var_normal(mean, std_dev, alpha=0.05):
    VaR = norm.ppf(alpha, loc=mean, scale=std_dev)

    return -VaR


# Calculte VaR T Distribution: (Value at Risk)
def calc_var_t_dist(mean, std_dev, df, alpha=0.05):
    VaR = t.ppf(q=alpha, df=df, loc=mean, scale=std_dev)

    return -VaR


# Calculate ES for Normal (expected shortfall)
def calc_expected_shortfall_normal(mean, std_dev, alpha=0.05):
    # Calculate ES using the formula
    es = -1 * mean + (std_dev * norm.pdf(norm.ppf(alpha, mean, std_dev)) / alpha)

    return es


# Calculate ES for Generalized T (expected shortfall)
def calc_expected_shortfall_t(mean, std_dev, df, alpha=0.05):
    # VaR for t dist
    var = -1 * calc_var_t_dist(mean, std_dev, df, alpha=alpha)

    # PDF fucntion for t dist
    def t_pdf(x):
        return t.pdf(x, df, loc=mean, scale=std_dev)

    # Integrand for es
    def integrand(x):
        return x * t_pdf(x)

    # Calc ES using integration
    es, _ = quad(integrand, float("-inf"), var)

    return es / alpha


# The `calculate_ewma_covariance_matrix` function computes the Exponentially Weighted Moving Average (EWMA) covariance matrix for a given DataFrame `df`
# and a decay factor `lambd`, by assigning more weight to more recent observations and less to older ones, using normalized exponentially decreasing weights.
# The function returns the EWMA covariance matrix, which reflects the time-varying relationships between the variables in `df`.


def calculate_ewma_covariance_matrix(df, lambd):
    # Calculate exponentially weighted covariance matrix provided a dataframe and lambda

    # Get the number of time steps n and vars m
    n, m = df.shape

    # Initialize the exponentially weighted covariance matrix as a square matrix with dimensions (m, m)
    ewma_cov_matrix = np.zeros((m, m))

    # Calculate the weights and normalized weights for each time step
    # w_{t_i} = (1-lambda)*lambda^{i-1}
    weights = [(1 - lambd) * lambd ** (i) for i in range(n)]
    weights = weights[::-1]
    #### Flip the weights

    # Calculate the sum of weights to normalize them
    total_weight = sum(weights)  # sum w_{t-j}

    # Normalize the weights by dividing each weight by the total weight
    # w_{t_i}^hat = w_{t_i} / sum w_{t-j}
    normalized_weights = [w / total_weight for w in weights]

    # Calculate the means for each variable across all time steps
    means = df.mean(axis=0)

    # Calculate the exponentially weighted covariance matrix
    for t in range(n):
        # Calculate the deviation of each variable at time t from its mean
        deviation = df.iloc[t, :] - means

        # weighted deviation from means for x and y
        ewma_cov_matrix += (
            normalized_weights[t]
            * deviation.values.reshape(-1, 1)
            @ deviation.values.reshape(1, -1)
        )
    ewma_cov_matrix = pd.DataFrame(ewma_cov_matrix)
    return ewma_cov_matrix


# The `calculate_ewma_correlation_matrix` function computes the Exponentially Weighted Moving Average (EWMA) correlation matrix for a given DataFrame `df`,
# using specified decay factors `lambda_corr` for correlation and `lambda_cov` for covariance (defaulting to `lambda_corr` if not provided), which emphasizes more recent data. It calculates the correlation matrix by normalizing the EWMA covariance matrix with the standard deviations of each variable.


def calculate_ewma_correlation_matrix(df, lambda_corr, lambda_cov=None):
    # Calculate exponentially weighted correlation matrix provided a dataframe and lambda

    if lambda_cov is None:
        lambda_cov = lambda_corr

    ewma_cov_matrix = calculate_ewma_covariance_matrix(df, lambda_cov)

    # Calculate the standard deviations for each variable across all time steps
    std_devs = np.sqrt(np.diag(ewma_cov_matrix))

    # Calculate the exponentially weighted correlation matrix
    ewma_corr_matrix = ewma_cov_matrix / np.outer(std_devs, std_devs)

    return pd.DataFrame(ewma_corr_matrix)
