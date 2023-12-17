from scipy.stats import moment
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t, lognorm, kurtosis
from scipy.optimize import minimize
from scipy.integrate import quad
import pandas as pd
import time
from tqdm import tqdm
import math
from numpy.linalg import eig
from datetime import datetime


def chol_psd(a):
    if isinstance(a, pd.DataFrame):
        a = a.to_numpy()

    n = a.shape[0]

    # Initialize the root matrix with 0 values
    root = np.zeros((n, n), dtype=np.float64)
    root.fill(0.0)

    # Loop over columns
    for j in range(n):
        s = 0.0

        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])

        # Diagonal Element
        temp = a[j, j] - s
        if 0.0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. Just set the column to 0 if we have one
        if root[j, j] == 0.0:
            root[j, j + 1 :] = 0.0
        else:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    return root


def is_psd(matrix):
    # Check if a matrix is positive semidefinite
    eigenvalues = np.linalg.eigvals(matrix)
    psd = np.all(eigenvalues >= 0)
    if psd:
        print("The matrix is positive semidefinite.")
    else:
        print("The matrix is not positive semidefinite.")
        # Print the negative eigenvalues
        negative_eigenvalues = eigenvalues[eigenvalues < 0]
        print("Negative Eigenvalues:", negative_eigenvalues)


def near_psd(a, epsilon=0.0):
    # Ensure that a given matrix is almost positive semi-definite (PSD).
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance
    if np.sum(np.isclose(np.diag(out), 1.0)) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(np.dot(invSD, out), invSD)

    # SVD, update the eigenvalue and scale
    vals, vecs = np.linalg.eigh(out)

    vals = np.maximum(vals, epsilon)

    T = 1.0 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(np.dot(invSD, out), invSD)

    # Convert the output to a DataFrame with original column names
    out_df = pd.DataFrame(out, columns=a.columns, index=a.index)

    return out_df


# The `_getAplus` function performs eigenvalue decomposition on a matrix `A`, sets any negative eigenvalues to zero, and reconstructs the matrix,
# ensuring it is positive semi-definite. As an example, if `A` is a covariance matrix with some negative eigenvalues due to rounding errors,
# `_getAplus(A)` will adjust `A` to be positive semi-definite, suitable for further financial analyses like portfolio optimization.


def _getAplus(A):
    vals, vecs = eig(A)
    vals = np.diag(np.maximum(vals, 0))
    return vecs @ vals @ np.conj(vecs).T


# The `_getPS` function adjusts a given square matrix `A` to be positive semi-definite using a weighting matrix `W`. It does so by scaling `A` with the square root of `W`, applying the
# `_getAplus` function to make it positive semi-definite, and then rescaling back. This function is useful, for instance, in risk management to adjust a covariance matrix `A` using weights `W` to ensure
# it is suitable for optimization problems.


def _getPS(A, W):
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW


# The `_getPu` function takes a square matrix `A` and modifies it by replacing its diagonal elements with 1.0, effectively standardizing the diagonal while preserving the off-diagonal elements.
# For example, if `A` is a correlation matrix, `_getPu(A)` will adjust it to have perfect (1.0) correlations along the diagonal,
# which can be useful in certain financial analyses where the diagonal needs to represent self-correlation or unity.


def _getPu(A, W):
    Aret = np.copy(A)
    np.fill_diagonal(Aret, 1.0)
    return Aret


# The `wgtNorm` function calculates the weighted norm of a matrix `A` using a weighting matrix `W`.
# It does this by first scaling `A` with the square root of `W`, performing a matrix multiplication, and then summing the squares of all elements in the resulting matrix.
# For example, in a financial context, `wgtNorm(A, W)` can be used to compute the weighted norm of a risk matrix `A`
# with weights `W` representing asset allocations or importance, providing a measure of overall risk.


def wgtNorm(A, W):
    W05 = np.sqrt(W)
    W05 = W05 @ A @ W05
    return np.sum(W05 @ W05)


def higham_nearestPSD(pc, epsilon=1e-9, maxIter=100, tol=1e-9):
    # find nearest PSD using Higham
    if isinstance(pc, pd.DataFrame):
        pc = pc.to_numpy()

    n = pc.shape[0]

    W = np.diag(np.ones(n))

    deltaS = 0
    Yk = np.copy(pc)
    invSD = None

    # calculate the correlation matrix if we got a covariance
    if np.count_nonzero(np.isclose(np.diag(Yk), 1.0)) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD

    norml = np.finfo(float).max
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W)
        norm = wgtNorm(Yk - pc, W)
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if np.abs(norm - norml) < tol and minEigVal > -epsilon:
            print(f"Converged in {i} iterations.")
            break

        norml = norm
        i += 1
    if i == maxIter:
        print(f"Convergence failed after {i-1} iterations")

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    # Create a DataFrame with the same column titles as the input pc
    result_df = pd.DataFrame(
        Yk, columns=pc.columns if isinstance(pc, pd.DataFrame) else None
    )

    return result_df


# Implement a multivariate normal simulation that allows for simulation directly from a covar matrix or using PCA and parameter for % var explained

# The `multivariate_normal_simulation` function generates simulations of multivariate normal data using either a direct method with Cholesky decomposition (`Direct`) or Principal Component Analysis (`PCA`)
# based on a specified mean, covariance matrix, and number of samples. As an example, for risk modeling in finance, this function can simulate asset returns given their historical mean and covariance,
# using either the full covariance matrix (`Direct`) or a reduced PCA representation for efficiency (`PCA`).


def multivariate_normal_simulation(
    mean, cov_matrix, num_samples, method="Direct", pca_explained_var=None
):
    if method == "Direct":
        cov_matrix = cov_matrix  # .values
        n = cov_matrix.shape[1]
        # Initialize an array to store the simulation results
        simulations = np.zeros((num_samples, n))

        L = chol_psd(cov_matrix)

        Z = np.random.randn(n, num_samples)

        # Calculate simulated multivariate normal samples
        for i in range(num_samples):
            simulations[i, :] = mean + np.dot(L, Z[:, i])

        # Convert the output to a DataFrame with original column names
        simulations_df = pd.DataFrame(simulations, columns=cov_matrix.columns)

        return simulations_df
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
        # Generate a mean vector based on the original mean
        mean_vector = np.full(n, mean)

        n = cov_matrix.shape[0]
        simulations = np.random.multivariate_normal(
            mean_vector, new_cov_matrix, num_samples
        )

        # Convert the output to a DataFrame with original column names
        simulations_df = pd.DataFrame(simulations, columns=cov_matrix.columns)

        return simulations_df


# The `simulate_and_print_norms` function simulates multivariate normal data for each covariance matrix in `cov_matrices` using a specified method (either `Direct` or `PCA`),
# computes the covariance matrix of the simulated data, and calculates the Frobenius norm difference between the original and simulated covariance matrices, timing and printing the results for each simulation.
# For example, in financial risk modeling, this function can be used to assess the accuracy of simulated asset return distributions against historical data,
# by comparing their covariance matrices and evaluating the simulation time and accuracy.


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


# The `calculate_portfolio_var` function calculates the Value at Risk (VaR) of a financial portfolio, given a portfolio DataFrame, price data, return data, a decay factor `lambd`, and a confidence level `alpha`.
# It does this by determining the portfolio's total value, calculating each asset's proportion of the portfolio, and using an Exponentially Weighted Moving Average (EWMA) covariance matrix
# to compute the portfolio's standard deviation, which is then used to estimate VaR. As an example, this function can be applied to
# assess the potential maximum loss of a stock portfolio at a given confidence level over a specified time period, considering the recent volatility and correlations of asset returns.


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


# The `return_calculate` function calculates either discrete or logarithmic returns for financial assets from a DataFrame of price data `prices_df`, based on the specified `method`.
# It excludes a specified `date_column` and outputs a new DataFrame with calculated returns and corresponding dates. For example, if you have daily stock price data in `prices_df`,
# this function can calculate the daily returns (either discrete or logarithmic) for each stock, excluding the date column, and return a DataFrame with these returns alongside their respective dates.


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


# The `portfolio_es` function calculates the Expected Shortfall (ES) of a financial portfolio under the assumption that asset returns follow a t-distribution.
# It computes the ES for each stock in the portfolio (using its mean, standard deviation, and degrees of freedom) weighted by its holding, and then returns the average of these individual ES values.
# For example, in a risk management context, this function is used to estimate the average potential loss expected from a portfolio of stocks,
# given the risk characteristics of each stock and their respective holdings in the portfolio.


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


# The `calculate_prices` function generates a series of asset prices from a given series of returns and an initial price, using one of three methods: "classical_brownian" (adds returns to the previous price),
# "arithmetic_return" (applies arithmetic returns), or "geometric_brownian" (also known as "log_return", applies exponential growth to the previous price based on returns).
# It calculates and optionally prints the expected value and standard deviation of these generated prices.
# For example, this function can be used to model stock price evolution over time based on historical return data, choosing the appropriate method to reflect different financial theories or market behaviors.


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
        elif method == "geometric_brownian" or method == "log_return":
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


# The `integral_bsm_with_coupons` function calculates the price of a call or put option with coupon payments using the Black-Scholes or Merton models,
# by integrating the option payoff function (considering the underlying asset price, strike price, coupons, and a log-normal distribution for asset prices) over a range.
# As an example, this function can be used to price a call or put option on a stock that pays regular coupons, accounting for factors like time to maturity, interest rate, and volatility, in accordance with either the Black-Scholes or Merton pricing frameworks.


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


# function to calculate the option price error
def option_price_error(sigma, S, X, T, r, b, option_type, market_price):
    option_price = options_price(S, X, T, sigma, r, b, option_type)
    return abs(option_price - market_price)


# AR(1) method
def simulate_ar1_process(N, alpha, sigma, mu, num_steps):
    # Initialize variables
    y = np.empty((N, num_steps))

    for i in range(N):
        # Generate random noise
        epsilon = np.random.normal(0, sigma, num_steps)
        # Initialize the process with a random value
        y[i, 0] = mu + epsilon[0]

        for t in range(1, num_steps):
            y[i, t] = mu + alpha * (y[i, t - 1] - mu) + epsilon[t]

    return y


# The `calculate_implied_volatility` function estimates the implied volatility of an option using the bisection method, based on known parameters such as current stock price, strike price, risk-free rate, and continuously compounding coupon rate,
# by iteratively narrowing the volatility range until the calculated option price is close to the observed market price. As an example, this function can be used to determine the implied volatility for a specific call or put option,
# given its market price, to understand market expectations of future asset price volatility.


# Calculate implied volatility using bisection
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


# The `calculate_implied_volatility_newton` function estimates the implied volatility of an option using Newton-Raphson's method, which is generally faster and more accurate than the bisection method used in the previous function.
# This approach iteratively adjusts the volatility guess based on the gradient (vega) of the option pricing formula, converging towards the volatility that matches the observed market price of the option.
# The initial guess for volatility is set at 0.2, and the function iterates up to `max_iter` times or until the calculated option price is within a specified tolerance (`tol`) of the market price.
# Unlike the bisection method which narrows down a range, Newton-Raphson's method uses the derivative (vega in this case) to make more informed steps towards the solution, often leading to faster convergence,
# especially useful for put options or options with complex payoffs.


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


# Closed form greeks
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


# The `greeks_df` function calculates the Greek values (Delta, Gamma, Vega, Theta, Rho, and Carry Rho) for an option, given its parameters like underlying price, strike price, risk-free rate, implied volatility, and continuous dividend rate.
# It does this by numerically approximating the partial derivatives of the option price with respect to each parameter using the finite difference method.
# For example, you can use this function to understand how sensitive an option's price is to changes in underlying factors like the stock price or volatility,
# which is crucial for risk management and hedging strategies in options trading.


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


# The `greeks_with_dividends` function calculates the option Greeks (Delta, Gamma, Vega, Theta, Rho, and Carry Rho) for an option with dividends, considering the effect of upcoming dividend payments
# (specified by `div_dates` and `div_amounts`) on the option's value. It adjusts the underlying stock price for the present value of these dividends before computing the Greeks using the Black-Scholes-Merton formula.
# For example, in options trading, this function can be used to assess the sensitivity of an option's price to various factors, accounting for dividends, which is crucial for options on dividend-paying stocks.


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


# The `binomial_tree_option_pricing_european` function calculates the price of a European-style option (either a call or put) using the binomial tree model,
# based on parameters like underlying stock price, strike price, risk-free rate, dividend yield, implied volatility, and number of steps in the tree.
# For example, this function can be used to price a European call or put option on a stock, factoring in its volatility and time to expiration,
# by constructing a binomial tree to model possible future stock prices and their corresponding option payoffs.


# Binomial tree European option
def binomial_tree_option_pricing_european(
    underlying_price,
    strike_price,
    current_date,
    expiration_date,
    risk_free_rate,
    dividend_yield,
    implied_volatility,
    num_steps,
    option_type,
):
    S = underlying_price
    X = strike_price
    T = (expiration_date - current_date).days / 365.0
    r = risk_free_rate
    q = dividend_yield
    b = r - q
    sigma = implied_volatility
    N = num_steps

    # parameters
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp((b) * dt) - d) / (u - d)
    pd = 1.0 - pu

    # initialize arrays
    ps = np.zeros(N + 1)
    paths = np.zeros(N + 1)
    prices = np.zeros(N + 1)

    # calculate factorials
    n_fact = math.factorial(N)

    # calculate stock prices at each node
    for i in range(N + 1):
        prices[i] = S * u**i * d ** (N - i)
        ps[i] = pu**i * pd ** (N - i)
        paths[i] = n_fact / (math.factorial(i) * math.factorial(N - i))

    # Calculate option payoff at each leaf
    if option_type == "call":
        prices = np.maximum(0, prices - X)
    else:
        prices = np.maximum(X - prices, 0)

    # calculate final option prices as the discounted expected payoff
    prices = prices * ps
    option_price = np.dot(prices, paths)
    return np.exp(-r * T) * option_price


# The `binomial_tree_option_pricing_american` function calculates the price of an American-style option (call or put) using the binomial tree model,
# considering parameters like the underlying stock price, strike price, time to maturity, risk-free rate, cost of carry (`b`), and implied volatility.
# The function builds a binomial tree for the option's life and accounts for the possibility of early exercise, typical of American options.
# For example, this function can be used to price an American call or put option by evaluating potential future prices and
# the option to exercise early at each step in the tree, factoring in stock volatility and time value.


# note:

# american differs from its European counterpart by incorporating the feature of early exercise, which is characteristic of American options.
# While the European model only considers option payoff at expiration, the American model evaluates the option's value at each node of the binomial tree,
# allowing for the possibility of exercising the option before its expiration date if it is financially beneficial to do so.


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


# The binomial_tree_option_pricing_american_complete function enhances the binomial tree model for pricing American options by incorporating scheduled dividend payments, which can significantly impact the option's value and the decision to exercise early.
# It adjusts the stock price for dividends at specified times (div_times) and amounts (div_amounts) and then calculates the option price using a backward induction method that considers both the value of exercising and holding the option at each step.


# note:

# This function is different from the basic binomial_tree_option_pricing_american as it takes into account the effect of dividends on the option pricing,
# which is particularly important for dividend-paying stocks where the dividends can influence early exercise decisions.
# For example, you can use this function to price an American call or put option on a dividend-paying stock, factoring in both the volatility of the stock and
# the impact of expected dividend payments on the option's exercise strategy.


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


# The calculate_portfolio_value_american function calculates the total value of a mixed portfolio containing American options and stocks.
# It prices each American option by considering dividends, implied volatility, and time to expiration, and adds the current value of stocks, adjusting for each holding's quantity (positive for long positions, negative for short positions).
# For example, in portfolio management, this function can be used to determine the current total value of a
# portfolio consisting of American options and stocks, where the options are priced using a binomial tree method that accounts for upcoming dividend payments and the specific characteristics of each option.


# Function to calculate the portfolio value for a given underlying value
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


class FittedModel:
    def __init__(self, beta, error_model, evaluate, u):
        self.beta = beta
        self.error_model = error_model
        self.evaluate = evaluate
        self.u = u


# The `general_t_ll` function calculates the negative log-likelihood of a dataset `x` given parameters `mu`, `s`, and `nu` for a generalized Student's t-distribution.
# This function is typically used in statistical analysis or optimization algorithms where the aim is to find the parameters (`mu` for location, `s` for scale, `nu` for degrees of freedom)
# that best fit a given dataset to a t-distribution, such as in financial modeling to estimate the distribution of asset returns.


def general_t_ll(params, x):
    mu, s, nu = params
    td = t(loc=mu, scale=s, df=nu)
    return -np.sum(np.log(td.pdf(x)))


# The `fit_general_t` function fits a generalized Student's t-distribution to a given data set `x`, using initial estimates for the mean, scale, and degrees of freedom, and then optimizes these parameters using the Nelder-Mead method to minimize the negative log-likelihood.
# As an example, in financial modeling, this function can be used to fit asset returns to a t-distribution, capturing the fat tails and skewness
# often observed in financial data, and providing a model (`fitted`) that includes methods for evaluating quantiles and cumulative distribution function (CDF) values.


def fit_general_t(x):
    # Fit a general T distribution given an x input
    start_m = np.mean(x)
    start_nu = 6.0 / kurtosis(x) + 4
    start_s = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)

    def _gtl(params):
        return general_t_ll(params, x)

    # Initial parameter values
    initial_params = np.array([start_m, start_s, start_nu])

    # Optimization using Nelder Mead
    result = minimize(_gtl, initial_params, method="Nelder-Mead")

    m, s, nu = result.x

    error_model = t(df=nu, loc=m, scale=s)

    def evaluate_u(quantile):
        return error_model.ppf(quantile)

    u = error_model.cdf(x)

    fitted = FittedModel(None, error_model, evaluate_u, u)

    return m, s, nu, fitted


# The `fit_regression_t` function performs a regression analysis where the error terms are assumed to follow a generalized Student's t-distribution, rather than the normal distribution commonly used in linear regression.
# It fits this t-distributed regression model to data `x` (predictors) and `y` (response), starting with initial parameter estimates based on ordinary least squares regression,
# and then optimizes these parameters (mean, scale, degrees of freedom of the t-distribution, and regression coefficients) using the Nelder-Mead method.
# This approach is particularly useful in modeling financial data where error terms may exhibit heavy tails, and the function returns the optimized parameters for the t-distribution and regression coefficients.
# For example, it can be used in financial econometrics to model relationships between economic variables where assumptions of normality do not hold.
# Fit regression model with T errors
def fit_regression_t(x, y):
    n = x.shape[0]

    __x = np.column_stack((np.ones(n), x))
    __y = y

    # Fit a general T distribution given an x input
    b_start = np.linalg.inv(__x.T @ __x) @ __x.T @ __y
    e = __y - __x @ b_start
    start_m = np.mean(e)
    start_nu = 6.0 / kurtosis(e) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    def _gtl(params):
        mu, s, nu, *beta = params

        xm = __y.values.reshape(-1, 1) - (__x @ beta).reshape(-1, 1)
        new_params = [mu, s, nu]
        return general_t_ll(new_params, xm)

    # Initial parameter values
    initial_params = np.concatenate(([start_m, start_s, start_nu], b_start))

    # Optimization using Nelder Mead
    result = minimize(_gtl, initial_params, method="Nelder-Mead")

    m, s, nu, *beta = result.x

    return m, s, nu, *beta


# The `fit_normal` function fits a normal distribution to a given dataset `x`, calculating the mean and standard deviation, and constructs a model for the normal distribution.
# It computes the cumulative distribution function (CDF) values `u` for `x`, and provides a function `evaluate_u` to calculate quantiles.
# For example, this function can be used in statistical analysis to model asset returns or other data that are assumed to follow a normal distribution,
# providing both a fit to the data and a way to assess probabilities and quantiles within that fit.


def fit_normal(x):
    # Mean and Std values
    m = np.mean(x)
    s = np.std(x)

    # Create the error model
    error_model = norm(loc=m, scale=s)

    # Calculate the errors and U
    u = error_model.cdf(x)

    def evaluate_u(quantile):
        return error_model.ppf(quantile)

    return FittedModel(None, error_model, evaluate_u, u)


# The `VaR_error_model` function calculates the Value at Risk (VaR) at a specified confidence level `alpha` for a given error distribution model `error_model` by returning the negative value of the percentile point function
# (inverse of the cumulative distribution function) at `alpha`. For example, in risk management, this function can be used to determine the VaR for a portfolio or asset returns distribution,
# quantifying the potential loss at a given confidence level based on the assumed distribution of returns or errors.


# value at risk from a provided error model, either a norm or t dist
def VaR_error_model(error_model, alpha=0.05):
    return -error_model.ppf(alpha)


# Expected shortfall from an error_model, either norm or t distributin
def ES_error_model(error_model, alpha=0.05):
    var_value = VaR_error_model(error_model, alpha)

    def integrand(x):
        return x * error_model.pdf(x)

    # Set the lower bound for integration to a very small value
    lower_bound = error_model.ppf(1e-12)

    # Integrate the function from the lower bound to the negative VaR
    es_value, _ = quad(
        integrand, lower_bound, -var_value, points=1000, epsabs=1e-8, epsrel=1e-8
    )

    # Divide by alpha to get the expected shortfall
    return -es_value / alpha


# The VaR_simulation function calculates the Value at Risk (VaR) at a specified confidence level alpha from a given array a of simulated returns or losses.
# It does this by sorting a, then averaging the values at the indices corresponding to the ceiling and floor of alpha times the length of a, and returning the negative of this average.


def VaR_simulation(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])
    return -v


# The ES_simulation function calculates the Expected Shortfall (ES) at a specified confidence level alpha from a given array a of simulated returns or losses.
# It first finds the Value at Risk (VaR) by averaging the values at the ceiling and floor indices of alpha times the length of a,
# then computes the ES as the mean of the sorted array values that are less than or equal to this VaR, and returns the negative of this mean.
# For example, in financial risk analysis, this function can be used to estimate the ES of a portfolio by providing a simulated array of portfolio returns or losses,
# giving a more comprehensive risk measure than VaR by averaging the worst losses up to the VaR threshold.


def ES_simulation(a, alpha=0.05):
    x = np.sort(a)
    nup = int(np.ceil(len(a) * alpha))
    ndn = int(np.floor(len(a) * alpha))
    v = 0.5 * (x[nup] + x[ndn])

    es = np.mean(x[x <= v])

    return -es


# The `simulate_pca` function simulates data using Principal Component Analysis (PCA) from a covariance matrix `a`. It decomposes `a` into eigenvalues and eigenvectors,
# selects a number of principal components based on a percentage of explained variance (`pctExp`), then generates `nsim` simulations using these components and adds the specified `mean`.
# For example, in financial modeling, this function can be used to simulate correlated asset returns based on their historical covariance matrix, capturing the key patterns in the data while reducing dimensionality.


# Simulation of PCA for usage in Gaussian copula
def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    # If the mean is missing then set to 0, otherwise use the provided mean
    _mean = np.zeros(n)
    if mean is not None:
        _mean = mean.copy()

    # Eigenvalue decomposition
    vals, vecs = eig(a)
    sorted_indices = np.argsort(vals)[
        ::-1
    ]  # Get indices for sorting in descending order
    vals = np.real(vals[sorted_indices])
    vecs = np.real(vecs[:, sorted_indices])

    tv = np.sum(vals)

    posv = np.where(vals >= 1e-8)[0]
    if pctExp < 1:
        nval = 0
        pct = 0.0
        # figure out how many factors we need for the requested percent explained
        for i in range(len(posv)):
            pct += vals[i] / tv
            nval += 1
            if pct >= pctExp:
                break
        if nval < len(posv):
            posv = posv[:nval]

    vals = vals[posv]
    vecs = vecs[:, posv]

    # print(f"Simulating with {len(posv)} PC Factors: {np.sum(vals)/tv*100}% total variance explained")
    B = vecs @ np.diag(np.sqrt(vals))

    np.random.seed(seed)
    m = len(vals)
    r = np.random.randn(m, nsim)

    out = (B @ r).T

    # Loop over iterations and add the mean
    for i in range(n):
        out[:, i] += _mean[i]

    return out


# Gaussian Copula


class GaussianCopula:
    """Construct the Gaussian Copula to simulate

    Parameters:
        dists(DataFrame) --- a group of distributions
        data(DataFrame) --- the data that fits the distributions will be used to generate the simulated sample

    Usage:
        copula=GaussianCopula(dists,data)
        sample=copula.simulate()
    """

    def __init__(self, dists, data):
        self.models = dists
        self.data = data

    def simulate(self, NSim=5000):
        transform_data = pd.DataFrame()
        for name in self.data.columns:
            rt = np.array(self.data[name])
            # Use the CDF to transform the data to uniform universe
            # Use the standard normal quantile function to transform the uniform to normal
            transform_data[name] = stats.norm.ppf(self.models[name][0].cdf(rt))
        # Spearman correlation
        corr_spearman = stats.spearmanr(transform_data, axis=0)[0]
        # Use PCA simulation
        simulator = Simulator(corr_spearman, NSim)
        # Simulate Normal & Transform to uniform
        SimU = stats.norm.cdf(simulator.PCA_Simulation(1), loc=0, scale=1)
        # Transform to Model Distribution
        simulatedResults = pd.DataFrame()
        for idx, name in enumerate(self.data.columns):
            simulatedResults[name] = self.models[name][0].ppf(SimU[idx, :])
        return simulatedResults.T
