# calculate VaR using a normal distribution
def calculate_VaR(data, alpha=0.05):
    return -np.quantile(data, alpha)


# The norm_VaR function calculates the Value at Risk (VaR) for a set of returns assumed to follow a normal distribution.
# It first computes the mean and standard deviation of the returns, then generates num_sample random samples from a normal distribution with these parameters.
# Finally, it calculates the VaR for these generated samples using a given confidence level alpha.
# For example, in risk management, this function can be used to estimate the VaR for a portfolio or asset returns, assuming normal distribution, by providing historical return data and specifying a confidence level, such as 95%.
def norm_VaR(returns, alpha=0.05, num_sample=1000):
    mean = returns.mean()
    std = returns.std()
    Rt = np.random.normal(mean, std, num_sample)
    var = calculate_VaR(Rt, alpha)
    return var, Rt


# the function to calculate Var based on normal dist with ew var
def norm_ew_VaR(returns, alpha=0.05, num_sample=1000, w_lambda=0.94):
    mean = returns.mean()
    std = np.sqrt(expo_weighted_cov(returns, w_lambda))
    Rt = np.random.normal(mean, std, num_sample)
    var = calculate_VaR(Rt, alpha)
    return var, Rt


# calculate VaR using a MLE fitted T distribution
def MLE_T_VaR(returns, alpha=0.05, num_sample=1000):
    result = t.fit(returns, method="MLE")  # fit the returns into MLE
    df = result[0]  # used to get the required parameters for T distribution simulation
    loc = result[1]
    scale = result[2]
    Rt = t(df, loc, scale).rvs(
        num_sample
    )  # generate num_sample random variates from the t-distribution of (df, loc, scale)
    var = calculate_VaR(Rt, alpha)
    return var, Rt


# calculate VaR using AR(1)
def ar1_VaR(returns, alpha=0.05, num_sample=1000):
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
    var = calculate_VaR(Rt, alpha)
    return var, Rt


# Purpose: This function is designed to calculate VaR based on the historical returns of a portfolio. It does not involve simulation.
# Input: It takes historical return data as input (e.g., daily returns of a portfolio).
# Calculation: It directly uses historical returns to calculate VaR by finding the value at risk corresponding to the desired quantile (alpha).

# historical_returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.02, 0.01, -0.01, -0.01, 0.02]
# alpha = 0.05
# var, returns = his_VaR(historical_returns, alpha)


# calculating VaR using historical distribution
def his_VaR(returns, alpha=0.05):
    Rt = (
        returns.values
    )  # no further simulation, just obtain all data to get historical distribution
    var = calculate_VaR(Rt, alpha)
    return var, Rt


# calculate VaR using Delta Normal
def cal_delta_VaR(aPortfolio, prices, alpha=0.05, w_lambda=0.94):
    daily_price, holdings, port_value = parsing_port(aPortfolio, prices)
    returns = return_calculate(daily_price).drop("Date", axis=1)
    latest_prices = (
        daily_price.drop("Date", axis=1).tail(1).values
    )  # tail() as opposite to head(), get last n rows
    dR_dr = (
        latest_prices.T * holdings.values.reshape(-1, 1) / port_value
    )  # transpose and re-organize into two columns, reshape(-1, 1) means reshape to be 1 column, -1 means num of rows will auto be set to required
    cov_mtx = expo_weighted_cov_valueOnly(returns, w_lambda)
    R_std = np.sqrt(np.transpose(dR_dr) @ cov_mtx @ dR_dr)
    var = (-1) * port_value * norm.ppf(alpha) * R_std
    return var[0][0]  # var is [[]]


# Purpose: This function is designed to estimate VaR for a portfolio through historical simulation. It simulates the portfolio's returns based on historical data.
# Input: It requires additional inputs, such as portfolio holdings and historical price data for the assets in the portfolio. It also involves simulation of returns.
# Calculation: It simulates daily returns for the portfolio, taking into account asset prices, holdings, and historical returns. It then calculates VaR based on the simulated portfolio returns.

# Portfolio holdings and historical prices
# portfolio = {
#    "Asset1": {"Holdings": 100, "PriceData": [50, 52, 51, 53, 55, 54, 56, 58, 59, 60]},
#    "Asset2": {"Holdings": 200, "PriceData": [30, 32, 31, 33, 35, 34, 36, 38, 37, 40]}
# }
# alpha = 0.05
# num_simulations = 1000
# var, simulated_returns = cal_his_VaR(portfolio, alpha=alpha, num_sample=num_simulations)


# calculate VaR using historic simulation
def cal_his_VaR(aPortfolio, prices, alpha=0.05, num_sample=1000):
    daily_price, holdings, port_value = parsing_port(aPortfolio, prices)
    returns = return_calculate(daily_price).drop("Date", axis=1)
    simu = returns.sample(num_sample, replace=True)
    latest_prices = daily_price.drop("Date", axis=1).tail(1).values.reshape(-1, 1)

    pchange = simu * latest_prices.T
    holdings = holdings.values.reshape(-1, 1)
    simu_change = pchange @ holdings
    var = calculate_VaR(simu_change, alpha)
    return var, simu_change


# calculate ES
def cal_ES(var, sim_data):
    return -np.mean(sim_data[sim_data <= -var])


# Function Explanation: It computes ES, which represents the average of losses exceeding the VaR threshold(above)


def ES_historical(data, alpha=0.05):
    """Given a dataset(array), calculate the its historical Expected Shortfall"""
    data.sort()
    n = round(data.shape[0] * alpha)
    return -data[:n].mean()
