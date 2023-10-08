import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize


# Calculate Var
def calculate_var(data, mean=0, alpha=0.05):
    return mean - np.quantile(data, alpha)


# Calculate ES
def calculate_es(data, mean=0, alpha=0.05):
    var = mean - np.quantile(data, alpha)
    return -data[data <= -var].mean()


def VAR(a, alpha=0.05):
    x = np.sort(a)
    n = len(a)
    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup] + x[ndn])
    return -v


def ES(a, alpha=0.05):
    x = np.sort(a)
    n = len(a)
    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup] + x[ndn])
    es = np.mean(x[x <= v])
    return -es


# 1. VaR using normal distribution
def var_normal(returns, confidence_level=0.95):
    std_dev = returns.std()
    z_score = stats.norm.ppf(confidence_level)
    var_normal = -(std_dev * z_score)
    return var_normal


# 2. VaR using normal distribution with EWM variance (lambda = 0.94)
def var_normal_ewm(returns, confidence_level=0.95):
    variance = returns.ewm(alpha=0.06).var().iloc[-1]
    std_dev_ewm = np.sqrt(variance)
    z_score_ewm = stats.norm.ppf(confidence_level)
    var_ewm = -(std_dev_ewm * z_score_ewm)
    return var_ewm


# 3. VaR using MLE fitted T distribution
def var_t(returns, confidence_level=0.95):
    params = stats.t.fit(returns, floc=0)
    t_dist = stats.t(*params)
    var_tdist = -t_dist.ppf(confidence_level)
    return var_tdist


# 4. VaR using fitted AR(1) model
def var_AR1(returns, confidence_level=0.95):
    model = sm.tsa.AR(returns)
    results = model.fit(maxlag=1)
    rho = results.params[1]
    var_ar1 = -(
        rho * returns.mean()
        + np.sqrt(1 - rho**2) * returns.std() * stats.norm.ppf(confidence_level)
    )
    return var_ar1


# 5. VaR using Historic Simulation
def var_historic(returns, confidence_level=0.95):
    data_sorted = returns.sort_values()
    var_hist = -data_sorted.quantile(1 - confidence_level)
    return var_hist


# 6. Calculate the returns
def return_calculate(prices, method="DISCRETE", date_column="Date"):
    vars = prices.columns
    n_vars = len(vars)
    vars = [v for v in vars if v != date_column]
    if n_vars == len(vars):
        raise ValueError(f"{date_column} not in DataFrame {vars}")
    n_vars = len(vars)
    p = prices[vars].to_numpy()
    n, m = p.shape
    p2 = np.empty((n - 1, m))
    for i in range(n - 1):
        p2[i, :] = p[i + 1, :] / p[i, :]
    if method.upper() == "DISCRETE":
        p2 -= 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f'method {method} must be in ("LOG", "DISCRETE")')
    dates = prices.iloc[1:, prices.columns.get_loc(date_column)]
    out = pd.DataFrame({date_column: dates})
    for i in range(n_vars):
        out[vars[i]] = p2[:, i]
    return out
