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

# These are all the functions used in the homework

# Proj_w2


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


def calc_estimated_kurtosis(sample_size=sample_size):
    # Array to keep kurtosis values for
    estimated_kurtosis_vals = []
    estimated_skew_vals = []

    # Test a new kurtosis for the number of sample distributions available
    for _ in tqdm(range(num_samples), desc="Generating Samples"):
        # Create random normal sample distribution
        sample = np.random.normal(mean, std_dev, sample_size)

        # Calculate kurtosis using your function (first4Moments)
        _, _, skew, kurtosis = first4Moments(sample, excess_kurtosis=False)

        estimated_kurtosis_vals.append(kurtosis)
        estimated_skew_vals.append(skew)

    # Average the estimated kurtosis using your function
    averaged_estimated_kurtosis = np.mean(estimated_kurtosis_vals)
    averaged_estimated_skew = np.mean(estimated_skew_vals)
    return (
        averaged_estimated_kurtosis,
        estimated_kurtosis_vals,
        averaged_estimated_skew,
        estimated_skew_vals,
    )


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


def mle_normal_distribution(X, y, perform_hypothesis_test=False):
    # Define the likelihood function for the normal distribution
    def log_likelihood(mean, var, X):
        # Number of values in X
        n = len(X)
        # Adjust x to be centered around the mean
        adjusted_X = X - mean
        # Get squared variance
        var2 = var**2
        # Calculate log likelihood
        log_likeli = -(n / 2) * np.log(var2 * 2 * np.pi) - np.dot(
            adjusted_X, adjusted_X
        ) / (2 * var2)

        return -log_likeli

    # Calculate initial guess for mean and standard deviation
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess]

    # Perform optimization through minimization of log likelihood
    result = minimize(
        lambda params: log_likelihood(params[0], params[1], X), initial_params
    )

    # Extract optimized parameters
    optimized_mean, optimized_std_dev = result.x

    # Print optimized mean and standard deviation
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)

    # Perform hypothesis test if specified
    if perform_hypothesis_test:
        # Calculate test statistic and p-value against standard normal (0, 1)
        test_statistic = (optimized_mean - 0) / optimized_std_dev  # Z-score
        p_value = 2 * (1 - norm.cdf(abs(test_statistic)))  # Two-tailed test

        # Determine if the null hypothesis (X is from a standard normal distribution) is rejected
        reject_null = p_value < 0.05  # Using a significance level of 0.05

        # Print hypothesis test results
        print("Test Statistic:", test_statistic)
        print("P-Value:", p_value)
        print("Reject Null Hypothesis:", reject_null)

    return optimized_mean, optimized_std_dev


def mle_t_distribution(X, y, perform_hypothesis_test=False):
    # Define the likelihood function for the t-distribution
    def log_likelihood(mean, var, df, X):
        adjusted_X = X - mean
        var2 = var**2
        log_likeli = np.sum(t.logpdf(adjusted_X / np.sqrt(var2), df))
        return -log_likeli


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
        # plt.savefig(f'plots/MA_{N}_Steps.png')
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
        # plt.savefig(f'plots/AR_{N}_Steps.png')
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
    # if save_plots:
    #     import os
    #     if not os.path.exists('plots'):
    #         os.makedirs('plots')

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

    # plt.savefig(f'plots/{plot_type}_{N}_ACF_PACF.png')

    plt.tight_layout()

    # Display the plots
    plt.show()


# Proj_w3


# q1
def calculate_ewma_covariance_matrix(df, lambd):
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
    means = df.mean()

    # Calculate the exponentially weighted covariance matrix
    for t in range(n):
        # Calculate the deviation of each variable at time t from its mean
        deviation = df.iloc[t, :] - means

        # weighted deviation from means for x and y
        ### NEED TO PERFORM ELEMENT WISE OPERATION, FIX THIS
        ewma_cov_matrix += (
            normalized_weights[t]
            * deviation.values.reshape(-1, 1)
            @ deviation.values.reshape(1, -1)
        )
    ewma_cov_matrix = pd.DataFrame(ewma_cov_matrix)
    return ewma_cov_matrix


# Compare to non-weighted covariance matrix
p = dailyreturn.cov()
frobenius_norm = np.linalg.norm(v_df.values - p.values, "fro")
print(
    f"Frobenius Norm between unweighted covar matrix and my exp. weighted covar matrix {frobenius_norm}"
)

# q2 are all the PSD matrices

# q3


# Implement a multivariate normal simulation that allows for simulation directly from a covar matrix or using PCA and parameter for % var explained
def multivariate_normal_simulation(
    mean, cov_matrix, num_samples, method="Direct", pca_explained_var=None
):
    if method == "Direct":
        cov_matrix = cov_matrix.values
        n = cov_matrix.shape[1]
        # Initialize an array to store the simulation results
        simulations = np.zeros((num_samples, n))
        # Initialize the root matrix
        root = np.zeros((n, n), dtype=np.float64)

        L = chol_psd(root, cov_matrix)

        Z = np.random.randn(n, num_samples)

        # Calculate simulated multivariate normal samples
        for i in range(num_samples):
            simulations[i, :] = mean + np.dot(L, Z[:, i])

        return simulations
    elif method == "PCA":
        if pca_explained_var is None:
            pca_explained_var = 1.0
        # Calculate eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues in descending order along with eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Normalize eigenvalues to get the proportion of explained variance
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

        # Determine the number of components needed to explain the desired variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        k = np.argmax(cumulative_variance_ratio >= pca_explained_var) + 1

        # Select the top k eigenvectors and their eigenvalues
        selected_eigenvalues = eigenvalues[:k]
        selected_eigenvectors = eigenvectors[:, :k]

        # Construct a new covariance matrix using the selected eigenvectors and eigenvalues
        new_cov_matrix = np.dot(
            selected_eigenvectors,
            np.dot(np.diag(selected_eigenvalues), selected_eigenvectors.T),
        )

        n = cov_matrix.shape[0]
        simulations = np.random.multivariate_normal(mean, new_cov_matrix, num_samples)

        return simulations


def simulate_and_print_norms(
    cov_matrices,
    mean_returns,
    num_samples,
    cov_matrix_names,
    method="Direct",
    pca_explained_var=None,
):
    for i, (cov_matrix, cov_matrix_name) in enumerate(
        zip(cov_matrices, cov_matrix_names)
    ):
        # Start timing
        start_time = time.time()

        # Direct Simulation
        simulated_data = multivariate_normal_simulation(
            mean_returns, cov_matrix, num_samples, method, pca_explained_var
        )

        # Calculate the covariance matrix of the simulated data
        simulated_covariance = np.cov(simulated_data, rowvar=False)

        # Calculate the Frobenius Norm
        frobenius_norm = np.linalg.norm(cov_matrix - simulated_covariance)

        # End timing
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        if method == "Direct":
            print("Method: ", method)
        else:
            print(f"Method: {method} Explained Variance: {pca_explained_var}")
        print(f"Simulation {i + 1} - Covariance Matrix: {cov_matrix_name}")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print(f"Frobenius Norm: {frobenius_norm:.4f}\n")


# proj_w4


# q1


def calculate_prices(
    returns, initial_price, method="classical_brownian", print_calc=True
):
    # initial price
    prices = [initial_price]

    for i in range(len(returns)):
        r_t = returns.iloc[i]

        if method == "classical_brownian":
            # Classical Brownian Motion: P_t = P_{t-1} + r_t
            p_t = prices[i] + r_t
        elif method == "arithmetic_return":
            # Arithmetic Return System: P_t = P_{t-1}(r_t + 1)
            p_t = prices[i] * (1 + r_t)
        elif method == "geometric_brownian":
            # Log Return or Geometric Brownian Motion: P_t = P_{t-1}*e^{r_t}
            p_t = prices[i] * np.exp(r_t)
        else:
            raise ValueError(
                "Invalid method. Supported methods are 'classical_brownian', 'arithmetic_return', and 'geometric_brownian'."
            )

        prices.append(p_t)

    expected_value = np.mean(prices)
    std_deviation = np.std(prices)
    if print_calc == True:
        print(f"Expected value of {method}: {expected_value}")
        print(f"Standard Deviation of {method}: {std_deviation}\n")

    return prices, expected_value, std_deviation


def simulate_price_return_formulas(
    std_dev_returns, initial_price, method_form, num_sims=1000
):
    num_samples = len(returns[best_col])
    # Initialize arrays to contain simulation results
    sim_means = np.zeros(num_sims)
    sim_stds = np.zeros(num_sims)

    for i in range(num_sims):
        # Simulate a random returns based on the std dev of returns for this column
        sim_returns = np.random.normal(0, std_dev_returns, num_samples)
        sim_returns = pd.Series(sim_returns)
        sim_prices, sim_mean, sim_std = calculate_prices(
            sim_returns, initial_price, method=method_form, print_calc=False
        )
        sim_means[i] = sim_mean
        sim_stds[i] = sim_std

    # Show the mean and std dev match your expectations
    print(f"Simulated Expected Value of {method_form}: {np.mean(sim_means)}")
    print(f"Simulated Standard Deviation of {method_form}: {np.mean(sim_stds)}\n")
    return np.mean(sim_means), np.mean(sim_stds)


# q2


def calculate_ewma_covariance_matrix(df, lambd):
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
    means = df.mean()

    # Calculate the exponentially weighted covariance matrix
    for t in range(n):
        # Calculate the deviation of each variable at time t from its mean
        deviation = df.iloc[t, :] - means

        # weighted deviation from means for x and y
        ### NEED TO PERFORM ELEMENT WISE OPERATION, FIX THIS
        ewma_cov_matrix += (
            normalized_weights[t]
            * deviation.values.reshape(-1, 1)
            @ deviation.values.reshape(1, -1)
        )
    ewma_cov_matrix = pd.DataFrame(ewma_cov_matrix)
    return ewma_cov_matrix


alpha = 0.05


def return_calculate(prices_df, method="DISCRETE", date_column="Date"):
    vars = prices_df.columns
    n_vars = len(vars)
    vars = [var for var in vars if var != date_column]

    if n_vars == len(vars):
        raise ValueError(f"date_column: {date_column} not in DataFrame: {vars}")

    n_vars = n_vars - 1

    p = prices_df[vars].values
    n = p.shape[0]
    m = p.shape[1]
    p2 = np.empty((n - 1, m))

    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]

    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f'method: {method} must be in ("LOG","DISCRETE")')

    dates = prices_df[date_column].iloc[1:]

    # Create a new DataFrame with all columns
    data = {date_column: dates}
    for i in range(n_vars):
        data[vars[i]] = p2[:, i]

    out = pd.DataFrame(data)

    return out


# Calculate VaR[NOT historical var]:
def calc_var_normal(mean, std_dev, alpha=0.05):
    VaR = norm.ppf(alpha, mean, std_dev)

    return VaR


### MLE fitted T distribution


# Fixed MLE for t distribution for just the dataframe values input
def mle_t_distribution(y):
    # Define the likelihood function for the t-distribution
    def neg_log_likelihood(params, y):
        mean, var, df = params
        adjusted_y = y - mean
        log_likeli = -np.sum(t.logpdf(adjusted_y, df, loc=mean, scale=var))
        return log_likeli

    # Calculate initial guess for mean, standard deviation, and degrees of freedom
    mean_guess = np.mean(y)
    std_dev_guess = np.std(y)
    df_guess = len(y)

    # Initial guess for optimization
    initial_params = [mean_guess, std_dev_guess, df_guess]

    # Perform optimization through minimization of negative log likelihood
    result = minimize(
        neg_log_likelihood, initial_params, args=(y,), method="Nelder-Mead"
    )

    # Extract optimized parameters
    optimized_mean, optimized_std_dev, optimized_df = result.x

    return optimized_mean, optimized_std_dev, optimized_df


### AR(1) method
def simulate_ar1_process(N, alpha, sigma, mu, num_steps):
    # Initialize variables
    y = np.empty((N, num_steps))

    for i in range(N):
        # Generate random noise
        epsilon = np.random.normal(0, sigma, num_steps)
        # initial value
        y[i, 0] = mu + epsilon[0]

        for t in range(1, num_steps):
            y[i, t] = mu + alpha * (y[i, t - 1] - mu) + epsilon[t]

    return y


# q3


def calculate_portfolio_var(portfolio, price_df, returns_df, lambd, alpha=0.05):
    # calculate total portfolio value
    portfolio_value = 0.0
    # create array to store each stock's value
    delta = []
    for _, row in portfolio.iterrows():
        stock_value = row["Holding"] * price_df[row["Stock"]].iloc[-1]
        portfolio_value += stock_value
        delta.append(stock_value)

    print(f"Portfolio Value: {portfolio_value}")
    delta = np.array(delta)
    normalized_delta = delta / portfolio_value

    exp_weighted_cov = calculate_ewma_covariance_matrix(returns_df, lambd)
    exp_weighted_std = np.sqrt(np.diagonal(exp_weighted_cov))

    # Create a dictionary to store column titles and corresponding exp_weighted_std values
    result_dict = {
        column: std for column, std in zip(returns_df.columns, exp_weighted_std)
    }

    exp_weighted_std_portfolio = np.array(
        [result_dict[stock] for stock in portfolio["Stock"]]
    )

    p_sig = np.sqrt(
        np.dot(np.dot(normalized_delta, exp_weighted_std_portfolio), normalized_delta)
    )

    VaR = -delta * norm.ppf(1 - alpha) * p_sig
    total_VaR = sum(VaR)

    print(f"Porftolio Value at Risk: ${total_VaR}\n")
    return total_VaR


# Using Portfolio and DailyPrices assume the expected return on all stocks is 0
# portfolio = pd.read_csv("Week04/Project/portfolio.csv")
portfolio = pd.read_csv(
    "/Users/ahmedibrahim/Desktop/Mids/Fall24/Quantitative Risk Management/exam prep/try_juhas/Week04_answers/Project/Portfolio.csv"
)

port_a = portfolio[portfolio["Portfolio"] == "A"]
port_b = portfolio[portfolio["Portfolio"] == "B"]
port_c = portfolio[portfolio["Portfolio"] == "C"]

# example usage below

# calculate_portfolio_var(port_a, prices, returns, lambd=0.94)

# proj_w5

# q1


def mle_normal_distribution_one_input(X):
    # Define the likelihood function for the normal distribution
    def log_likelihood(params, X):
        mean, var = params
        n = len(X)
        # Adjust X to be centered around the mean
        adjusted_X = X - mean
        # Get squared variance
        var2 = var**2
        # Calculate log likelihood
        log_likeli = -(n / 2) * np.log(var2 * 2 * np.pi) - np.sum(adjusted_X**2) / (
            2 * var2
        )

        return -log_likeli

    # Calculate initial guess for mean and standard deviation
    mean_guess = np.mean(X, axis=0)
    std_dev_guess = np.std(X, axis=0)

    # Initial guess for optimization as a 1D array
    initial_params = np.array([mean_guess, std_dev_guess])

    # Perform optimization through minimization of log likelihood
    result = minimize(log_likelihood, initial_params, args=(X,))

    # Extract optimized parameters
    optimized_mean, optimized_std_dev = result.x

    # Print optimized mean and standard deviation
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)

    return optimized_mean, optimized_std_dev


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


# Calculate VaR Normal Distribution:
def calc_var_normal(mean, std_dev, alpha=0.05):
    VaR = norm.ppf(alpha, loc=mean, scale=std_dev)

    return -VaR


# Calculte VaR T Distribution:
def calc_var_t_dist(mean, std_dev, df, alpha=0.05):
    VaR = t.ppf(q=alpha, df=df, loc=mean, scale=std_dev)

    return -VaR


# Calculate ES for Normal
def calc_expected_shortfall_normal(mean, std_dev, alpha=0.05):
    # Calculate ES using the formula
    es = -1 * mean + (std_dev * norm.pdf(norm.ppf(alpha, mean, std_dev)) / alpha)

    return es


# Calculate ES for Generalized T Distribution
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


# q2 - All functions to be in the library

# q3


def portfolio_es(portfolio, stock_dict, dist="T"):
    if dist == "T":
        portfolio_es_individual = []
        for stock in portfolio["Stock"]:
            mean = stock_dict[stock]["mean"]
            std_dev = stock_dict[stock]["std_dev"]
            df = stock_dict[stock]["df"]
            stock_es = calc_expected_shortfall_t(mean, std_dev, df, alpha=0.05)
            stock_es *= portfolio.loc[portfolio["Stock"] == stock, "Holding"]
            portfolio_es_individual.append(stock_es)
        return np.mean(portfolio_es_individual)


def calculate_ewma_covariance_matrix(df, lambd):
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


def calculate_portfolio_var(portfolio, price_df, returns_df, lambd, alpha=0.05):
    # calculate total portfolio value
    portfolio_value = 0.0
    # create array to store each stock's value
    delta = []
    for _, row in portfolio.iterrows():
        stock_value = row["Holding"] * price_df[row["Stock"]].iloc[-1]
        portfolio_value += stock_value
        delta.append(stock_value)

    print(f"Portfolio Value: {portfolio_value}")
    delta = np.array(delta)
    normalized_delta = delta / portfolio_value

    exp_weighted_cov = calculate_ewma_covariance_matrix(returns_df, lambd)
    exp_weighted_std = np.sqrt(np.diagonal(exp_weighted_cov))

    # Create a dictionary to store column titles and corresponding exp_weighted_std values
    result_dict = {
        column: std for column, std in zip(returns_df.columns, exp_weighted_std)
    }

    exp_weighted_std_portfolio = np.array(
        [result_dict[stock] for stock in portfolio["Stock"]]
    )

    p_sig = np.sqrt(
        np.dot(np.dot(normalized_delta, exp_weighted_std_portfolio), normalized_delta)
    )

    VaR = -delta * norm.ppf(1 - alpha) * p_sig
    total_VaR = sum(VaR)

    print(f"Porftolio Value at Risk: ${total_VaR}\n")
    return total_VaR


# proj_w6

# q1


def integral_bsm_with_coupons(
    call,
    underlying,
    strike,
    days,
    rf,
    ivol,
    tradingDayYear,
    couponRate,
    function_type="Black Scholes",
    q=None,
):
    if function_type == "Black Scholes":
        b = rf
    if function_type == "Merton":
        b = rf - q

    # time to maturity
    ttm = days / tradingDayYear

    # daily volatility with continuously compounded implied volatility
    dailyVol = ivol / np.sqrt(tradingDayYear)

    # std dev and mean for log normal distribution
    sigma = np.sqrt(days) * dailyVol
    mu = np.log(underlying) + ttm * b - 0.5 * sigma**2

    # log normal distribution
    d = lognorm(scale=np.exp(mu), s=sigma)

    # calculate the present value of coupons
    couponPV = 0.0
    for day in range(int(ttm * tradingDayYear)):
        # present value of the coupon payment for each day,
        couponPV += couponRate * np.exp(-rf * (day / tradingDayYear))

    if call:
        # option value for call
        def f(x):
            return (max(0, x - strike) + couponPV) * d.pdf(x)

        val, _ = quad(f, 0, underlying * 2)
    else:
        # option value for put
        def g(x):
            return (max(0, strike - x) + couponPV) * d.pdf(x)

        val, _ = quad(g, 0, underlying * 2)

    return val * np.exp(-rf * ttm)


# example usage
# provided values
# underlying_price = 165
# strike = 170
# start_date = datetime(2023, 3, 3)
# options_expiration_date = datetime(2023, 3, 17)
# days_to_maturity = (options_expiration_date - start_date).days
# trading_days_in_year = 365
# risk_free_rate = 0.0525
# continuously_compounding_coupon_rate = 0.0053

# implied_volatilities = np.linspace(0.1, 0.8, 100)

# call_option_values = []
# put_option_values = []

# # calculate option values for each implied volatility
# for ivol in implied_volatilities:
#     # calculate the European call option value
#     call_option_value = integral_bsm_with_coupons(True, underlying_price, strike, days_to_maturity, risk_free_rate, ivol, trading_days_in_year, continuously_compounding_coupon_rate)
#     call_option_values.append(call_option_value)

#     # calculate the European put option value
#     put_option_value = integral_bsm_with_coupons(False, underlying_price, strike, days_to_maturity, risk_free_rate, ivol, trading_days_in_year, continuously_compounding_coupon_rate)
#     put_option_values.append(put_option_value)

# plt.figure(figsize=(12, 6))

# plt.plot(implied_volatilities, call_option_values, label='Call Option')
# plt.plot(implied_volatilities, put_option_values, label='Put Option', color='red')

# plt.title('European Call and Put Option Values with Continuously Compounding Coupon')
# plt.xlabel('Implied Volatility')
# plt.ylabel('Option Value')
# plt.legend()

# plt.tight_layout()
# plt.show()

# q2


def options_price(S, X, T, sigma, r, b, option_type="call"):
    """
    S: Underlying Price
    X: Strike
    T: Time to Maturity(in years)
    sigma: implied volatility
    r: risk free rate
    b: cost of carry -> r if black scholes, r-q if merton
    """
    d1 = (math.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp((b - r) * T) * norm.cdf(d1) - X * math.exp(
            -r * T
        ) * norm.cdf(d2)
    else:
        return X * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(
            (b - r) * T
        ) * norm.cdf(-d1)


def option_price_error(sigma, S, X, T, r, b, option_type, market_price):
    option_price = options_price(S, X, T, sigma, r, b, option_type)
    return abs(option_price - market_price)


# q3


def calculate_portfolio_value(underlying_value, portfolio):
    portfolio_value = 0.0

    for _, asset in portfolio.iterrows():
        if asset["Type"] == "Option":
            S = underlying_value
            X = asset["Strike"]
            T = (
                datetime.strptime(asset["ExpirationDate"], "%m/%d/%Y") - curr_date
            ).days / 365.0
            option_type = asset["OptionType"]
            market_price = asset["CurrentPrice"]
            b = risk_free - dividend_rate if option_type == "Call" else risk_free

            result = minimize_scalar(
                lambda sigma: option_price_error(
                    sigma, S, X, T, risk_free, b, option_type, market_price
                ),
                bounds=(0.001, 5.0),  # Adjust the bounds as needed
            )
            implied_volatility = result.x

            # Calculate the option value using implied volatility
            option_value = options_price(
                S, X, T, implied_volatility, risk_free, b, option_type
            )

            # Add or subtract option value to the portfolio based on Holding (1 or -1)
            portfolio_value += asset["Holding"] * option_value
        elif asset["Type"] == "Stock":
            # If it's a stock, just add its current price to the portfolio value
            portfolio_value += asset["Holding"] * (
                asset["CurrentPrice"] - underlying_value
            )

    return portfolio_value


# Function to fit an AR(1) model to data
def fit_ar1(data):
    n = len(data)
    x_t = data[:-1]
    x_t1 = data[1:]
    alpha = np.cov(x_t, x_t1)[0, 1] / np.var(x_t)
    epsilon = x_t1 - alpha * x_t
    sigma = np.std(epsilon)
    return alpha, sigma


initial_price = 170.15


# calculate prices from returns
def calculate_prices(initial_price, returns):
    prices = [initial_price]
    for r in returns:
        price_t = prices[-1] * (1 + r)
        prices.append(price_t)
    return prices[1:]


# proj_w7


# an example

# calculate the covariance matrix


def cov_matrix(df):
    return df.cov()


corr = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
sd = np.array([0.2, 0.1, 0.05])
er = np.array([0.05, 0.04, 0.03])

# Calculate covariance matrix
covar = np.diag(sd) @ corr @ np.diag(sd)


# q1


def options_price(S, X, T, sigma, r, b, option_type="call"):
    """
    S: Underlying Price
    X: Strike
    T: Time to Maturity(in years)
    sigma: implied volatility
    r: risk free rate
    b: cost of carry -> r if black scholes, r-q if merton
    """
    d1 = (math.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp((b - r) * T) * norm.cdf(d1) - X * math.exp(
            -r * T
        ) * norm.cdf(d2)
    else:
        return X * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(
            (b - r) * T
        ) * norm.cdf(-d1)


# function to calculate the option price error
def option_price_error(sigma, S, X, T, r, b, option_type, market_price):
    option_price = options_price(S, X, T, sigma, r, b, option_type)
    return abs(option_price - market_price)


def calculate_implied_volatility(
    curr_stock_price,
    strike_price,
    current_date,
    options_expiration_date,
    risk_free_rate,
    continuously_compounding_coupon,
    option_type,
    tol=1e-4,
    max_iter=300,
):
    S = curr_stock_price
    X = strike_price
    T = (options_expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuously_compounding_coupon
    b = r - q

    def calc_option_price(sigma):
        option_price = options_price(S, X, T, sigma, r, b, option_type)
        return option_price

    iteration = 0
    lower_vol = 0.001
    upper_vol = 15.0

    while iteration <= max_iter:
        mid_vol = (lower_vol + upper_vol) / 2
        option_price = calc_option_price(mid_vol)

        if abs(option_price) < tol:
            return mid_vol

        if option_price > 0:
            upper_vol = mid_vol
        else:
            lower_vol = mid_vol

        iteration += 1

    raise ValueError("Implied volatility calculation did not converge")


def greeks(
    underlying_price,
    strike_price,
    risk_free_rate,
    implied_volatility,
    continuous_dividend_rate,
    current_date,
    expiration_date,
    option_type,
):
    T = (expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    d1 = (np.log(S / X) + (b + (0.5 * sigma**2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = np.exp((b - r) * T) * norm.cdf(d1)
        theta = (
            -1 * (S * np.exp((b - r) * T) * norm.pdf(d1) * sigma) / (0.5 * np.sqrt(T))
            - (b - r) * S * np.exp((b - r) * T) * norm.cdf(d1)
            - r * X * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = T * X * np.exp(-r * T) * norm.cdf(d2)
        carry_rho = T * S * np.exp((b - r) * T) * norm.cdf(d2)

    else:
        delta = np.exp((b - r) * T) * (norm.cdf(d1) - 1)
        theta = (
            -1 * (S * np.exp((b - r) * T) * norm.pdf(d1) * sigma) / (0.5 * np.sqrt(T))
            + (b - r) * S * np.exp((b - r) * T) * norm.cdf(-1 * d1)
            + r * X * np.exp(-r * T) * norm.cdf(-1 * d2)
        )
        rho = -1 * T * X * np.exp(-r * T) * norm.cdf(-d2)
        carry_rho = -1 * T * S * np.exp((b - r) * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) * np.exp((b - r) * T) / (S * sigma * (np.sqrt(T)))
    vega = S * np.exp((b - r) * T) * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, vega, theta, rho, carry_rho


# finite difference derivative calculation greeks
def greeks_df(
    underlying_price,
    strike_price,
    risk_free_rate,
    implied_volatility,
    continuous_dividend_rate,
    current_date,
    expiration_date,
    option_type,
    epsilon=0.01,
):
    T = (expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    # options_price(S, X, T, sigma, r, b, option_type='call')

    def derivative(variable=None):
        if variable == "underlying":  # delta
            up_price = options_price(S + epsilon, X, T, sigma, r, b, option_type)
            down_price = options_price(S - epsilon, X, T, sigma, r, b, option_type)
            return (up_price - down_price) / (2 * epsilon)
        if variable == "double_underlying":  # gamma
            up_price = options_price(S + epsilon, X, T, sigma, r, b, option_type)
            down_price = options_price(S - epsilon, X, T, sigma, r, b, option_type)
            reg_price = options_price(S, X, T, sigma, r, b, option_type)
            return (up_price + down_price - 2 * reg_price) / (epsilon**2)
        if variable == "implied_volatility":  # vega
            up_price = options_price(S, X, T, sigma + epsilon, r, b, option_type)
            down_price = options_price(S, X, T, sigma - epsilon, r, b, option_type)
            return (up_price - down_price) / (2 * epsilon)
        if variable == "time_to_maturity":  # theta
            up_price = options_price(S, X, T + epsilon, sigma, r, b, option_type)
            down_price = options_price(S, X, T - epsilon, sigma, r, b, option_type)
            return -(up_price - down_price) / (2 * epsilon)
        if variable == "risk_free_rate":  # rho
            up_price = options_price(S, X, T, sigma, r + epsilon, b, option_type)
            down_price = options_price(S, X, T, sigma, r - epsilon, b, option_type)
            return (up_price - down_price) / (2 * epsilon)
        if variable == "cost_of_carry":  # carry rho
            up_price = options_price(S, X, T, sigma, r, b + epsilon, option_type)
            down_price = options_price(S, X, T, sigma, r, b - epsilon, option_type)
            return (up_price - down_price) / (2 * epsilon)

    delta = derivative("underlying")
    gamma = derivative("double_underlying")
    vega = derivative("implied_volatility")
    theta = derivative("time_to_maturity")
    rho = derivative("risk_free_rate")
    carry_rho = derivative("cost_of_carry")

    return delta, gamma, vega, theta, rho, carry_rho


def greeks_with_dividends(
    underlying_price,
    strike_price,
    risk_free_rate,
    implied_volatility,
    continuous_dividend_rate,
    current_date,
    expiration_date,
    option_type,
    div_dates,
    div_amounts,
):
    T = (expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    # Calculate present value of dividends
    pv_dividends = 0
    for div_date, div_amount in zip(div_dates, div_amounts):
        if div_date > current_date and div_date < expiration_date:
            pv_dividends += div_amount * np.exp(
                -r * (div_date - current_date).days / 365
            )

    # Adjust underlying price for dividends
    S_adj = S - pv_dividends

    d1 = (np.log(S_adj / X) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        delta = np.exp((b - r) * T) * norm.cdf(d1)
        theta = (
            -1
            * (S_adj * np.exp((b - r) * T) * norm.pdf(d1) * sigma)
            / (0.5 * np.sqrt(T))
            - (b - r) * S_adj * np.exp((b - r) * T) * norm.cdf(d1)
            - r * X * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = T * X * np.exp(-r * T) * norm.cdf(d2)
        carry_rho = T * S_adj * np.exp((b - r) * T) * norm.cdf(d2)

    else:
        delta = np.exp((b - r) * T) * (norm.cdf(d1) - 1)
        theta = (
            -1
            * (S_adj * np.exp((b - r) * T) * norm.pdf(d1) * sigma)
            / (0.5 * np.sqrt(T))
            + (b - r) * S_adj * np.exp((b - r) * T) * norm.cdf(-1 * d1)
            + r * X * np.exp(-r * T) * norm.cdf(-1 * d2)
        )
        rho = -1 * T * X * np.exp(-r * T) * norm.cdf(-d2)
        carry_rho = -1 * T * S_adj * np.exp((b - r) * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) * np.exp((b - r) * T) / (S_adj * sigma * (np.sqrt(T)))
    vega = S_adj * np.exp((b - r) * T) * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, vega, theta, rho, carry_rho


def binomial_tree_option_pricing_american(
    underlying_price,
    strike_price,
    ttm,
    risk_free_rate,
    b,
    implied_volatility,
    num_steps,
    option_type,
):
    S = underlying_price
    X = strike_price
    T = ttm
    r = risk_free_rate
    sigma = implied_volatility
    N = num_steps

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    if option_type == "call":
        z = 1
    else:
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N) - 1

    optionValues = np.empty(nNodes + 1)  # Increase the size by 1

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = S * (u**i) * (d ** (j - i))
            optionValues[idx] = max(0, z * (price - X))

            if j < N:
                optionValues[idx] = max(
                    optionValues[idx],
                    df
                    * (
                        pu * optionValues[idxFunc(i + 1, j + 1)]
                        + pd * optionValues[idxFunc(i, j + 1)]
                    ),
                )

    return optionValues[0]


def binomial_tree_option_pricing_american_complete(
    underlying_price,
    strike_price,
    ttm,
    risk_free_rate,
    implied_volatility,
    num_steps,
    option_type,
    div_amounts=None,
    div_times=None,
):
    S = underlying_price
    X = strike_price
    T = ttm
    r = risk_free_rate
    sigma = implied_volatility
    N = num_steps

    if (
        (div_amounts is None)
        or (div_times is None)
        or len(div_amounts) == 0
        or len(div_times) == 0
        or div_times[0] > N
    ):
        return binomial_tree_option_pricing_american(
            S,
            X,
            T,
            risk_free_rate,
            risk_free_rate,
            implied_volatility,
            num_steps,
            option_type,
        )

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(r * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-r * dt)
    if option_type == "call":
        z = 1
    else:
        z = -1

    def nNodeFunc(n):
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i + 1

    nDiv = len(div_times)
    n_nodes = nNodeFunc(N)
    option_values = np.empty(n_nodes + 1)  # Increase the size by 1

    for j in range(div_times[0], -1, -1):  # Use a float range for j
        for i in range(j, -1, -1):  # Use a float range for i
            idx = idxFunc(i, j)
            price = S * (u**i) * (d ** (j - i))

            if j < div_times[0]:
                # times before or at the dividend working backward induction
                option_values[idx] = max(0, z * (price - X))
                option_values[idx] = max(
                    option_values[idx],
                    df
                    * (
                        pu * option_values[idxFunc(i + 1, j + 1)]
                        + pd * option_values[idxFunc(i, j + 1)]
                    ),
                )
            else:
                # time after the dividend
                val_no_exercise = binomial_tree_option_pricing_american_complete(
                    price - div_amounts[0],
                    X,
                    ttm - div_times[0] * dt,
                    risk_free_rate,
                    implied_volatility,
                    N - div_times[0],
                    option_type,
                    div_amounts[1:nDiv],
                    div_times[1:nDiv] - div_times[0],
                )
                val_exercise = max(0, z * (price - X))
                option_values[idx] = max(val_no_exercise, val_exercise)

    return option_values[0]


def mle_normal_distribution_one_input(X):
    # Define the likelihood function for the normal distribution
    def log_likelihood(params, X):
        mean, var = params
        n = len(X)
        # Adjust X to be centered around the mean
        adjusted_X = X - mean
        # Get squared variance
        var2 = var**2
        # Calculate log likelihood
        log_likeli = -(n / 2) * np.log(var2 * 2 * np.pi) - np.sum(adjusted_X**2) / (
            2 * var2
        )

        return -log_likeli

    # Calculate initial guess for mean and standard deviation
    mean_guess = np.mean(X, axis=0)
    std_dev_guess = np.std(X, axis=0)

    # Initial guess for optimization as a 1D array
    initial_params = np.array([mean_guess, std_dev_guess])

    # Perform optimization through minimization of log likelihood
    result = minimize(log_likelihood, initial_params, args=(X,))

    # Extract optimized parameters
    optimized_mean, optimized_std_dev = result.x

    # Print optimized mean and standard deviation
    print("Optimized Mean:", optimized_mean)
    print("Optimized Standard Deviation:", optimized_std_dev)

    return optimized_mean, optimized_std_dev


def calc_var_normal(mean, std_dev, alpha=0.05):
    VaR = norm.ppf(alpha, loc=mean, scale=std_dev)

    return -VaR


# Calculate ES for Normal
def calc_expected_shortfall_normal(mean, std_dev, alpha=0.05):
    # Calculate ES using the formula
    es = -1 * mean + (std_dev * norm.pdf(norm.ppf(alpha, mean, std_dev)) / alpha)

    return es


# Calculate ES for Generalized T
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


def calculate_portfolio_value_american(
    underlying_value, portfolio, current_date, dividend_payment_date, risk_free_rate
):
    portfolio_value = 0.0

    for _, asset in portfolio.iterrows():
        if asset["Type"] == "Option":
            S = underlying_value
            X = asset["Strike"]
            expiration_date = datetime.strptime(asset["ExpirationDate"], "%m/%d/%Y")
            T = (expiration_date - current_date).days / 365.0
            option_type = asset["OptionType"]
            dividend_payment_time = np.array(
                [(dividend_payment_date - current_date).days]
            )
            dividend_payment = np.array([1])

            implied_volatility = calculate_implied_volatility_newton(
                S, X, current_date, expiration_date, risk_free_rate, 0, option_type
            )

            # Calculate the american option price using tree method
            option_value = binomial_tree_option_pricing_american_complete(
                S,
                X,
                T,
                risk_free_rate,
                implied_volatility,
                (dividend_payment_date - current_date).days + 1,
                option_type,
                dividend_payment,
                dividend_payment_time,
            )

            # Add or subtract option value to the portfolio based on Holding (1 or -1)
            portfolio_value += asset["Holding"] * option_value
        elif asset["Type"] == "Stock":
            # If it's a stock, just add its current price to the portfolio value
            portfolio_value += asset["Holding"] * (
                asset["CurrentPrice"] - underlying_value
            )

    return portfolio_value


# separate implied volatility function to help puts converge
def calculate_implied_volatility_newton(
    curr_stock_price,
    strike_price,
    current_date,
    options_expiration_date,
    risk_free_rate,
    continuously_compounding_coupon,
    option_type,
    tol=1e-5,
    max_iter=500,
):
    S = curr_stock_price
    X = strike_price
    T = (options_expiration_date - current_date).days / 365
    r = risk_free_rate
    q = continuously_compounding_coupon
    b = r - q

    def calc_option_price(sigma):
        option_price = options_price(S, X, T, sigma, r, b, option_type)
        return option_price

    def calc_vega(sigma):
        d1 = (math.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * math.exp((b - r) * T) * norm.pdf(d1) * math.sqrt(T)
        return vega

    iteration = 0
    volatility = 0.2  # Initial guess

    while iteration <= max_iter:
        option_price = calc_option_price(volatility)
        vega = calc_vega(volatility)

        if abs(option_price) < tol:
            return volatility

        volatility = volatility - option_price / vega

        iteration += 1

    raise ValueError("Implied volatility calculation did not converge")


# q3


# Simulate AAPL returns 10 days ahead
def fit_ar1(data):
    n = len(data)
    x_t = data[:-1]
    x_t1 = data[1:]
    alpha = np.cov(x_t, x_t1)[0, 1] / np.var(x_t)
    epsilon = x_t1 - alpha * x_t
    sigma = np.std(epsilon)
    return alpha, sigma


# final solution


# q1


# returns = return_calculate(problem1, method="LOG", date_column="Date")
# print(returns.head())


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


def greeks(
    underlying_price,
    strike_price,
    risk_free_rate,
    implied_volatility,
    continuous_dividend_rate,
    current_date,
    expiration_date,
    option_type,
):
    T = (expiration_date - current_date) / 255
    r = risk_free_rate
    q = continuous_dividend_rate
    b = r - q
    S = underlying_price
    X = strike_price
    sigma = implied_volatility

    d1 = (np.log(S / X) + (b + (0.5 * sigma**2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # call_price = (underlying * np.exp(-div_rate * ttm) * norm.cdf(d1)) - (strike * np.exp(-rf * ttm) * norm.cdf(d2))

    if option_type == "call":
        # Calculate Call Price
        call_price = (underlying * np.exp(-div_rate * ttm) * norm.cdf(d1)) - (
            strike * np.exp(-rf * ttm) * norm.cdf(d2)
        )
        delta = np.exp((b - r) * T) * norm.cdf(d1)
        theta = (
            -1 * (S * np.exp((b - r) * T) * norm.pdf(d1) * sigma) / (0.5 * np.sqrt(T))
            - (b - r) * S * np.exp((b - r) * T) * norm.cdf(d1)
            - r * X * np.exp(-r * T) * norm.cdf(d2)
        )
        rho = T * X * np.exp(-r * T) * norm.cdf(d2)
        carry_rho = T * S * np.exp((b - r) * T) * norm.cdf(d2)

    else:
        delta = np.exp((b - r) * T) * (norm.cdf(d1) - 1)
        theta = (
            -1 * (S * np.exp((b - r) * T) * norm.pdf(d1) * sigma) / (0.5 * np.sqrt(T))
            + (b - r) * S * np.exp((b - r) * T) * norm.cdf(-1 * d1)
            + r * X * np.exp(-r * T) * norm.cdf(-1 * d2)
        )
        rho = -1 * T * X * np.exp(-r * T) * norm.cdf(-d2)
        carry_rho = -1 * T * S * np.exp((b - r) * T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) * np.exp((b - r) * T) / (S * sigma * (np.sqrt(T)))
    vega = S * np.exp((b - r) * T) * norm.pdf(d1) * np.sqrt(T)

    return call_price, delta, gamma, vega, theta, rho, carry_rho


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


# Calculate options price
def options_price(S, X, T, sigma, r, b, option_type="call"):
    """
    S: Underlying Price
    X: Strike
    T: Time to Maturity(in years)
    sigma: implied volatility
    r: risk free rate
    b: cost of carry -> r if black scholes, r-q if merton
    """
    d1 = (math.log(S / X) + (b + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp((b - r) * T) * norm.cdf(d1) - X * math.exp(
            -r * T
        ) * norm.cdf(d2)
    else:
        return X * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(
            (b - r) * T
        ) * norm.cdf(-d1)


# Function to calculate Black-Scholes call option price
def black_scholes_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# Calculate covariance matrix
covar = np.diag(sd) @ corr @ np.diag(sd)


def optimize_risk(R, er, covar):
    n = len(er)

    def objective(w):
        return w.T @ covar @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ er - R},
    ]
    bounds = [(0, 1) for _ in range(n)]
    init_guess = np.array([1 / n] * n)
    result = minimize(
        objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )

    return {"risk": result.fun, "weights": result.x, "R": R}


def optimize_risk_parity(covar):
    n = len(covar)

    def objective(w):
        portfolio_variance = np.dot(w, np.dot(covar, w))
        individual_risk_contributions = w * np.dot(covar, w) / portfolio_variance
        return np.sum((individual_risk_contributions - 1 / n) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    init_guess = np.array([1 / n] * n)
    result = minimize(
        objective, init_guess, method="SLSQP", bounds=bounds, constraints=constraints
    )

    return {"weights": result.x, "risk": np.dot(result.x, np.dot(covar, result.x))}


# risk_parity_portfolio = optimize_risk_parity(covar) example


# The sr function is modified to return the negative of the Sharpe Ratio because the minimize function in SciPy performs minimization,
# and we want to maximize the Sharpe Ratio.
# Sharpe Ratio Function
def sr(w, er, covar, rf):
    m = np.dot(w, er) - rf
    s = np.sqrt(np.dot(w, np.dot(covar, w)))
    return -m / s  # Negative for minimization


# Portfolio Volatility Function
def pvol(w, covar):
    return np.sqrt(np.dot(w, np.dot(covar, w)))


# Component Standard Deviation Function
def pCSD(w, covar):
    p_vol = pvol(w, covar)
    csd = w * np.dot(covar, w) / p_vol
    return csd


# Sum Square Error of Component Standard Deviation
def sseCSD(w, covar):
    csd = pCSD(w, covar)
    mCSD = np.mean(csd)
    dCsd = csd - mCSD
    return 1.0e5 * np.sum(dCsd**2)


def VaR_historical(data, alpha=0.05):
    """Given a dataset(array), calculate the its historical VaR"""
    data.sort()
    n = round(data.shape[0] * alpha)
    return -data[n - 1]


# Calculate Frobenius Norm
def fnorm(mtxa, mtxb):
    s = mtxa - mtxb
    norm = 0
    for i in range(len(s)):
        for j in range(len(s[0])):
            norm += s[i][j] ** 2
    return norm


# Risklib- part 2 - fix it in the morning


# 1. Covariance estimation techniques


# Generating expoentially weighted weights
def weight_gen(n, lambd=0.94):
    weight = np.zeros(n)
    for i in range(n):
        weight[i] = (1 - lambd) * (lambd) ** i
    normalized_weight = weight / np.sum(weight)
    return normalized_weight


def ewcov_gen(data, weight):
    data = data - data.mean(axis=0)
    weight = np.diag(weight)
    data_left = weight @ data
    data_right = np.dot(data.T, data_left)
    return data_right


# 2. Non PSD fixes for correlation matrices

# chol_psd function, return the lower half matrix


def chol_psd(a):
    n = a.shape[1]
    # Initialize the root matrix with 0 values
    root = np.zeros((n, n))
    # loop over columns
    for j in range(n):
        s = 0
        # if we are not on the first column, calculate the dot product of the preceeding row values.
        if j > 0:
            s = root[j, 0:j].T @ root[j, 0:j]
        temp = a[j, j] - s
        # here temp is the critical value, when temp>=-1e-3, there is no nan but still invalid answer, but it is close
        if temp <= 0 and temp >= -1e-3:
            temp = 0
        root[j, j] = np.sqrt(temp)
        # Check for the 0 eigan value.  Just set the column to 0 if we have one
        if root[j, j] == 0:
            for i in range(j, n):
                root[j, i] = 0
        else:
            ir = 1 / root[j, j]
            for i in range(j + 1, n):
                s = root[i, 0:j].T @ root[j, 0:j]
                root[i, j] = (a[i, j] - s) * ir
    return root


# fixing psd matrix


def near_psd(a, epsilon=0.0):
    is_cov = False
    for i in np.diag(a):
        if abs(i - 1) > 1e-8:
            is_cov = True
        else:
            is_cov = False
            break
    if is_cov:
        invSD = np.diag(1 / np.sqrt(np.diag(a)))
        a = invSD @ a @ invSD
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    T = 1 / (np.square(vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    if is_cov:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out


# Implement Higham’s 2002 nearest psd correlation function


def Frobenius_Norm(a):
    return np.sqrt(np.sum(np.square(a)))


def projection_u(a):
    np.fill_diagonal(a, 1.0)
    return a


# A note here, epsilon is the smallest eigenvalue, 0 does not work well here, will still generate very small negativa values, so I set it to 1e-7


def projection_s(a, epsilon=1e-7):
    vals, vecs = np.linalg.eigh(a)
    vals = np.array([max(i, epsilon) for i in vals])
    return vecs @ np.diag(vals) @ vecs.T


def Higham_method(a, tol=1e-8):
    s = 0
    gamma = np.inf
    y = a
    # iteration
    while True:
        r = y - s
        x = projection_s(r)
        s = x - r
        y = projection_u(x)
        gamma_next = Frobenius_Norm(y - a)
        if abs(gamma - gamma_next) < tol:
            break
        gamma = gamma_next
    return y


# if a matrix is psd


def is_psd(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= 0)


# 3. Simulation Methods


def sim_mvn_from_cov(cov, num_of_simulation=25000):
    return chol_psd(cov) @ np.random.normal(size=(cov.shape[0], num_of_simulation))


# variance matrix


def var(cov):
    return np.diag(cov)


# Correlation matrix


def corr(cov):
    return np.diag(1 / np.sqrt(var(cov))) @ cov @ np.diag(1 / np.sqrt(var(cov))).T


# Covariance matrix


def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T


# using PCA with an optional parameter for % variance explained.
# return the simulation result
def PCA_with_percent(cov, percent=0.95, num_of_simulation=25000):
    eigenvalue, eigenvector = np.linalg.eigh(cov)
    total = np.sum(eigenvalue)
    for i in range(cov.shape[0]):
        i = len(eigenvalue) - i - 1
        if eigenvalue[i] < 0:
            eigenvalue = eigenvalue[i + 1 :]
            eigenvector = eigenvector[:, i + 1 :]
            break
        if sum(eigenvalue[i:]) / total > percent:
            eigenvalue = eigenvalue[i:]
            eigenvector = eigenvector[:, i:]
            break
    simulate = np.random.normal(size=(len(eigenvalue), num_of_simulation))
    return eigenvector @ np.diag(np.sqrt(eigenvalue)) @ simulate


# direct simulation


# direct_simulation() used for multivariate simulation - directly from a covariance matrix
def direct_simulation(cov, n_samples=25000):
    B = chol_psd(cov)
    r = scipy.random.randn(len(B[0]), n_samples)
    return B @ r


# 4. VaR calculation methods


# calculate_var used to given data and alpha, return the VaR
def calculate_var(
    data, mean=0, alpha=0.05
):  # mean is the current expected return, so you should include actual mean if you assume 0 mean, or just subtract each data with mean
    return mean - np.quantile(data, alpha)


# normal_var() used to calculate VaR when returns are fitted using normal distribution and then simulated
def normal_var(data, mean=0, alpha=0.05, nsamples=10000):
    sigma = np.std(data)
    simulation_norm = np.random.normal(mean, sigma, nsamples)
    var_norm = calculate_var(simulation_norm, mean, alpha)
    return var_norm


# ewcov_normal_var() used to calculate VaR when returns are fitted using normal distribution with ew var and then simulated
def ewcov_normal_var(data, mean=0, alpha=0.05, nsamples=10000, lambd=0.94):
    ew_cov = ewcov_gen(data, weight_gen(len(data), lambd))
    ew_variance = ew_cov
    sigma = np.sqrt(ew_variance)
    simulation_ew = np.random.normal(mean, sigma, nsamples)
    var_ew = calculate_var(simulation_ew, mean, alpha)
    return var_ew


# t_var() used to calculate VaR when returns are fitted using T-distribution by MLE and then simulated
def t_var(data, mean=0, alpha=0.05, nsamples=10000):
    params = scipy.stats.t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = scipy.stats.t(df, loc, scale).rvs(nsamples)
    var_t = calculate_var(simulation_t, mean, alpha)
    return var_t


# ar1_var() used to calculate VaR when returns are fitted using ar1 and then simulated
def ar1_var(returns, alpha=0.05, num_sample=1000):
    result = ARIMA(returns, order=(1, 0, 0)).fit()
    t_a = result.params[0]  # constant term
    t_phi = result.params[1]  # coefficient of the lagged term
    resid_std = np.std(result.resid)  # the residual of the fit
    last_return = returns[len(returns)]  # obtain the last return in returns
    Rt = (
        t_a
        + t_phi * last_return
        + np.random.normal(loc=0, scale=resid_std, size=num_sample)
    )  # alpha + phi * Rt-1 (since it is AR(1)) + residual, which we use exact residual of the fit, to some extent an element of "simulation of historical distribution"
    var = calculate_var(Rt)
    return var


def historic_var(data, mean=0, alpha=0.05):
    return calculate_var(data, mean, alpha)


# 5. ES calculation
def calculate_es(data, mean=0, alpha=0.05):
    return -np.mean(data[data < -calculate_var(data, mean, alpha)])


# def return_calculate(price, method='discrete'):
#     returns = []
#     for i in range(len(price)-1):
#         returns.append(price[i+1]/price[i])
#     returns = np.array(returns)
#     if method == 'discrete':
#         return returns - 1
#     if method == 'log':
#         return np.log(returns)

# derivative


# gbsm() is used to calculate the theoretical price of a European call or put option based on various inputs using the Black-Scholes-Merton formula.
def gbsm(option_type, S, X, r, b, sigma, T):
    d1 = (np.log(S / X) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp((b - r) * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(
            d2
        )
    else:
        return X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(
            -d1
        )


# implied_vol() is used to calculate the implied volatility, or sigma, given other parameters of the option, basically reversed engineering of gbsm()
def implied_vol(option_type, S, X, T, r, b, market_price, x0=0.5):
    def equation(sigma):
        return gbsm(option_type, S, X, r, b, sigma, T) - market_price

    # Back solve the Black-Scholes formula to get the implied volatility
    return fsolve(equation, x0=x0, xtol=0.0001)[0]


# bt_no_div is used to calculate the option price, using non-gbsm, and with no dividend, could be applied to American options
def bt_no_div(call, underlying, strike, ttm, rf, b, ivol, N):
    # call: the type of option,
    # underlying: the current price of the underlying asset,
    # strike: the strike price of the option,
    # ttm: time to maturity,
    # rf: risk-free interest rate,
    # b: cost-of-carry rate,
    # ivol: implied volatility,
    # N: the number of time steps
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return (n + 1) * (n + 2) // 2

    def idxFunc(i, j):
        return nNodeFunc(j - 1) + i

    nNodes = nNodeFunc(N)

    optionValues = [0.0] * nNodes

    for j in range(N, -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * u**i * d ** (j - i)
            optionValues[idx] = max(0, z * (price - strike))

            if j < N:
                optionValues[idx] = max(
                    optionValues[idx],
                    df
                    * (
                        pu * optionValues[idxFunc(i + 1, j + 1)]
                        + pd * optionValues[idxFunc(i, j + 1)]
                    ),
                )

    return optionValues[0]


# bt_with_div() is used to calculate option price using non-gbsm, with dividends
def bt_with_div(call, underlying, strike, ttm, rf, b, divAmts, divTimes, ivol, N):
    # call: The type of option, where True indicates a call option and False indicates a put option.
    # underlying: The current price of the underlying asset (such as a stock) for which the option is written.
    # strike: The strike price of the option, which is the fixed price at which the holder can buy (for a call) or sell (for a put) the underlying asset.
    # ttm: Time to maturity of the option, expressed in years. It represents how much time is left until the option expires.
    # rf: The risk-free interest rate, typically representing the theoretical return of an investment with no risk of financial loss.
    # b: The cost-of-carry rate, which in the context of this function, is assumed to be equal to the risk-free interest rate (rf). This simplification is due to the discrete treatment of dividends.
    # divAmts: A list of dividend amounts, with each element in the list representing the amount of a single dividend payment.
    # divTimes: A list of times at which dividends are paid, expressed as the number of time steps (from the total N steps) until each dividend payment.
    # ivol: Implied volatility of the underlying asset, reflecting the market's expectation of the asset's future volatility over the life of the option.
    # N: The number of time steps used in the binomial tree model, affecting the granularity and accuracy of the option price calculation.

    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_no_div(call, underlying, strike, ttm, rf, b, ivol, N)

    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def nNodeFunc(n: int) -> int:
        return int((n + 1) * (n + 2) / 2)

    def idxFunc(i: int, j: int) -> int:
        return nNodeFunc(j - 1) + i

    nDiv = len(divTimes)
    nNodes = nNodeFunc(divTimes[0])

    optionValues = [0] * nNodes

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u**i) * (d ** (j - i))

            if j < divTimes[0]:
                # times before the dividend working backward induction
                optionValues[idx] = max(0, z * (price - strike))
                optionValues[idx] = max(
                    optionValues[idx],
                    df
                    * (
                        pu * optionValues[idxFunc(i + 1, j + 1)]
                        + pd * optionValues[idxFunc(i, j + 1)]
                    ),
                )
            else:
                # time of the dividend
                valNoExercise = bt_with_div(
                    call,
                    price - divAmts[0],
                    strike,
                    ttm - divTimes[0] * dt,
                    rf,
                    b,
                    divAmts[1:],
                    [t - divTimes[0] for t in divTimes[1:]],
                    ivol,
                    N - divTimes[0],
                )
                valExercise = max(0, z * (price - strike))
                optionValues[idx] = max(valNoExercise, valExercise)

    return optionValues[0]


def find_iv(
    call, underlying, strike, ttm, rf, b, divAmts, divTimes, N, price, guess=0.5
):
    def f(ivol):
        return (
            bt_with_div(
                call, underlying, strike, ttm, rf, b, divAmts, divTimes, ivol, N
            )
            - price
        )

    return fsolve(f, guess)[0]


# below are functions and codes to calculate the Greeks
def d1(S, K, b, sigma, T):
    return (np.log(S / K) + (b + sigma**2 / 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, b, sigma, T):
    return d1(S, K, b, sigma, T) - sigma * np.sqrt(T)


# delta_call = np.exp((b-r)*T)*norm.cdf(rml.d1(S, X, b, sigma, T))
# print('Delta of the call option is: ', round(delta_call,3))
# delta_put = np.exp((b-r)*T)*(norm.cdf(rml.d1(S, X, b, sigma, T)) - 1)
# print('Delta of the put option is: ', round(delta_put,3))
# gamma = np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))/(S*sigma*np.sqrt(T))
# print('Gamma of the call option is: ', round(gamma,3))
# print('Gamma of the put option is: ', round(gamma,3))
# vega = S*np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))*np.sqrt(T)
# print('Vega of the call option is: ', round(vega,3))
# print('Vega of the put option is: ', round(vega,3))
# theta_call = -S*np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))*sigma/(2*np.sqrt(T)) - (b-r)*S*np.exp((b-r)*T)*norm.cdf(rml.d1(S, X, b, sigma, T)) - r*X*np.exp(-r*T)*norm.cdf(rml.d2(S, X, b, sigma, T))
# print('Theta of the call option is: ', round(theta_call,3))
# theta_put = -S*np.exp((b-r)*T)*norm.pdf(rml.d1(S, X, b, sigma, T))*sigma/(2*np.sqrt(T)) + (b-r)*S*np.exp((b-r)*T)*norm.cdf(-rml.d1(S, X, b, sigma, T)) + r*X*np.exp(-r*T)*norm.cdf(-rml.d2(S, X, b, sigma, T))
# print('Theta of the put option is: ', round(theta_put,3))
# # because the textbook has an assumption that b = rf but it does not hold here, we need to calculate the rho seperately
# rho_call = -T*S*np.exp(b*T - r*T)*norm.cdf(rml.d1(S, X, b, sigma, T)) + X*T*np.exp(-r*T)*norm.cdf(rml.d2(S, X, b, sigma, T))
# print('Rho of the call option is: ', round(rho_call,3))
# rho_put = -X*T*np.exp(-r*T)*norm.cdf(-rml.d2(S, X, b, sigma, T))+T*S*np.exp(b*T - r*T)*norm.cdf(-rml.d1(S, X, b, sigma, T))
# print('Rho of the put option is: ', round(rho_put,3))
# carry_rho_call = S*T*np.exp((b-r)*T)*norm.cdf(rml.d1(S, X, b, sigma, T))
# print('Carry Rho of the call option is: ', round(carry_rho_call,3))
# carry_rho_put = -S*T*np.exp((b-r)*T)*norm.cdf(-rml.d1(S, X, b, sigma, T))
# print('Carry Rho of the put option is: ', round(carry_rho_put,3))


# portfolio_return() is used to calculate the portfolio return, for the whole portfolio, not individual asset
def portfolio_return(weights, expected_returns):
    return np.sum(weights * expected_returns)


# portfolio_volatility() is used to calculate the portfolio volatility, not individual asset
def portfolio_volatility(weights, correlation_matrix, volatilities):
    covariance_matrix = (
        np.diag(volatilities) @ correlation_matrix @ np.diag(volatilities)
    )
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility


# sharpe_ratio is used to calculate the sharpe ratio for a portfolio
def sharpe_ratio(weights, expected_returns, correlation_matrix, volatilities):
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(weights, correlation_matrix, volatilities)
    sharpe_ratio = (p_return - 0.04) / p_volatility
    return sharpe_ratio


# optimize_sharpe_ratio() is used to calculate the weights for each asset in the portfolio to achieve optimized sharpe ratio
def optimize_sharpe_ratio(expected_returns, volatilities, correlation_matrix):
    # expected_returns is an array or list of expected returns for each asset in the portfolio.
    # volatilities is an array or list of the volatilities (standard deviations) of each asset.
    # correlation_matrix is a matrix representing the correlation coefficients between the assets in the portfolio.
    n_assets = len(expected_returns)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    # The line below is restricted for positive value
    bounds = [(0, 1) for i in range(n_assets)]
    initial_weights = np.ones(n_assets) / n_assets
    result = minimize(
        lambda x: -sharpe_ratio(x, expected_returns, correlation_matrix, volatilities),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x, -result.fun


# result.x is the list of the optimal weights for each asset in the portfolio that maximize the Sharpe ratio.
# -result.fun is the maximum Sharpe ratio achievable with these optimal weights.


# risk_parity_weight() is used to calculate the weight for each asset in the portfolio to achieve risk parity
def risk_parity_weight(corr_matrix, vol):
    # corr_matrix is a correlation matrix representing the correlation coefficients between the assets in the portfolio. remember to use corr() to convert covariance matrix to correlation matrix
    # vol is a list representing the volatilities (standard deviations) of the individual assets in the portfolio
    covar = np.outer(vol, vol) * corr_matrix
    n = covar.shape[0]

    def pvol(w):
        return np.sqrt(w @ covar @ w)

    def pCSD(w):
        pVol = pvol(w)
        csd = w * (covar @ w) / pVol
        return csd

    def sseCSD(w):
        csd = pCSD(w)
        mCSD = np.sum(csd) / n
        dCsd = csd - mCSD
        se = dCsd * dCsd
        # Add a large multiplier for better convergence
        return 1.0e5 * np.sum(se)

    # Define optimization problem
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for i in range(n)]
    w0 = np.ones(n) / n

    # Solve optimization problem
    result = minimize(sseCSD, w0, method="SLSQP", bounds=bounds, constraints=cons)
    wrp = np.round(result.x, decimals=4)
    return wrp


# wrp is a list of the weights for asset to achieve risk parity


# ex_post_contribution() is used to calculate both the return contribution and risk contribution from each asset to the whole portfolio
def ex_post_contribution(w, stocks, upReturns):
    # w is a list of the assets' initial weights
    # stocks is the list of the assets' names
    # upReturns is a df of all the assets' returns

    # Calculate portfolio return and updated weights for each day
    n = upReturns.shape[0]
    pReturn = np.empty(n)
    weights = np.empty((n, len(w)))
    lastW = np.array(w)
    matReturns = upReturns[stocks].values
    for i in range(n):
        # Save current weights in matrix
        weights[i, :] = lastW

        # Update weights by return
        lastW = lastW * (1.0 + matReturns[i, :])

        # Portfolio return is the sum of the updated weights
        pR = np.sum(lastW)
        # Normalize the weights back so sum = 1
        lastW = lastW / pR
        # Store the return
        pReturn[i] = pR - 1

    # Set the portfolio return in the Update Return DataFrame
    upReturns["Portfolio"] = pReturn

    # Calculate the total return
    totalRet = np.exp(np.sum(np.log(pReturn + 1))) - 1
    # Calculate the Carino K
    k = np.log(totalRet + 1) / totalRet

    # Carino k_t is the ratio scaled by 1/K
    carinoK = np.log(1.0 + pReturn) / pReturn / k
    # Calculate the return attribution
    attrib = pd.DataFrame(
        matReturns * weights * carinoK[:, None], columns=stocks, index=upReturns.index
    )

    # Set up a DataFrame for output.
    Attribution = pd.DataFrame(index=["TotalReturn", "Return Attribution"])
    # Loop over the stocks
    for s in stocks + ["Portfolio"]:
        # Total Stock return over the period
        tr = np.exp(np.sum(np.log(upReturns[s] + 1))) - 1
        # Attribution Return (total portfolio return if we are updating the portfolio column)
        atr = attrib[s].sum() if s != "Portfolio" else tr
        # Set the values
        Attribution[s] = [tr, atr]
    Y = matReturns * weights
    # Set up X with the Portfolio Return
    X = np.column_stack((np.ones(n), pReturn))
    # Calculate the Beta and discard the intercept
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    B = B[1, :]
    # Component SD is Beta times the standard Deviation of the portfolio
    cSD = B * np.std(pReturn)

    # Check that the sum of component SD is equal to the portfolio SD
    np.isclose(np.sum(cSD), np.std(pReturn))

    # Add the Vol attribution to the output
    Attribution = Attribution.append(
        pd.DataFrame(
            {
                "Value": "Vol Attribution",
                **{stocks[i]: cSD[i] for i in range(len(stocks))},
                "Portfolio": np.std(pReturn),
            },
            index=[0],
        ),
        ignore_index=True,
    )
    Attribution.loc[0, "Value"] = "Total Return"
    Attribution.loc[1, "Value"] = "Return Attribution"
    return Attribution, weights


# Attribution: A DataFrame that includes: Total return of each asset over the period. Return attribution of each asset, showing how much each contributed to the portfolio's total return. Volatility attribution, indicating each asset's contribution to the portfolio's overall risk.
# weights is a matrix (numpy array) of the updated weights of the assets in the portfolio for each time period.


# cal_t_pVals() is used to simulate the total value of multiple assets
def cal_t_pVals(port, returns_port, price):
    # port is a df specifying the holdings of each individual asset
    # returns_port is a df with historical returns of each individual individual asset along time span
    # price is an array specifying the latest price of each individual asset
    return_cdf = []
    par = []
    for col in returns_port.columns:
        df, loc, scale = t.fit(returns_port[col].values)
        par.append([df, loc, scale])
        return_cdf.append(
            t.cdf(returns_port[col].values, df=df, loc=loc, scale=scale).tolist()
        )
    return_cdf = pd.DataFrame(return_cdf).T
    spearman_cor = return_cdf.corr(method="spearman")
    sample = pd.DataFrame(PCA_with_percent(spearman_cor)).T
    sample_cdf = []
    for col in sample.columns:
        sample_cdf.append(norm.cdf(sample[col].values, loc=0, scale=1).tolist())
    simu_return = []
    for i in range(len(sample_cdf)):
        simu_return.append(
            t.ppf(sample_cdf[i], df=par[i][0], loc=par[i][1], scale=par[i][2])
        )
    simu_return = np.array(simu_return)

    sim_price = (1 + simu_return.T) * price
    pVals = sim_price.dot(port["Holding"])
    pVals.sort()
    return pVals


# pVals is simulated portfolio values


# return_calculate() is used to calculate returns
def return_calculate(prices, method="DISCRETE", dateColumn="Date"):
    # prices is a table of prices for different assets on each column
    # method is to choose whether we want arithmatic "DISCRETE", or continuous "LOG"
    # dateColumn is to assume there is a date column
    vars = (
        prices.columns.values.tolist()
    )  # extract the index of columns (column names), and convert it into a list
    nVars = len(vars)  # number of columns
    vars.remove(dateColumn)  # remove date column to get pure data
    if nVars == len(
        vars
    ):  # if the number of columns does not change after we remove the date column, this means we don't have the date column from the start, and such time series analysis will be meaningless, thus we raise an error on it
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = (
        nVars - 1
    )  # update the number of columns by reflecting the removal of date column
    p = np.array(
        prices.drop(columns=[dateColumn])
    )  # drop the date column and convert to np date frame
    n = p.shape[0]  # num of rows
    m = p.shape[1]  # num of column
    p2 = np.empty((n - 1, m))  # creates an empty NumPy array p2 with shape (n-1, m)
    for i in range(n - 1):
        for j in range(m):
            p2[i, j] = p[i + 1, j] / p[i, j]
    if method.upper() == "DISCRETE":
        p2 = p2 - 1.0  # if it is discrete compounding, then r = pt / pt-1 - 1
    elif method.upper() == "LOG":
        p2 = np.log(p2)  # if it is continuous compounding, then r = ln(pt / pt-1)
    else:
        raise ValueError(
            f'method: {method} must be in ("LOG","DISCRETE")'
        )  # there is no method other than discrete or compounding, so input error
    dates = prices[dateColumn][
        1:
    ]  # get the date, as the first row corresponds to the first day, and first day has no return, so we start from the second row
    out = pd.DataFrame(
        {dateColumn: dates}
    )  # initialize an empty "out" df, with its dateColumn set to be dates extracted
    for i in range(
        nVars
    ):  # add all rows calculated values corresponding to the stock name in that column in vars, then input this matrix into df "out"
        out[vars[i]] = p2[:, i]
    return out  # "out" is the df having stock name, date and return
