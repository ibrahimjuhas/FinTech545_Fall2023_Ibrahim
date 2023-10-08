import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd, eigh
from numpy.linalg import inv, LinAlgError


# Covariance Matrix
def generate_covariance_matrix(correlation_matrix, variance_vector):
    return np.array(
        np.outer(np.sqrt(variance_vector), np.sqrt(variance_vector))
        * correlation_matrix
    )


# Exponentially Weighted Covariance Matrix
def exp_weighted_cov(data, lambda_value=0.97):
    # Initialize the covariance matrix with zeros
    ewcov = np.zeros((len(data.columns), len(data.columns)))

    # Loop through each asset
    for i in range(len(data.columns)):
        for j in range(i, len(data.columns)):
            # Calculate the weights
            weights = [
                (1 - lambda_value) * (lambda_value ** (k - 1))
                for k in range(1, len(data) + 1)
            ]
            weights = np.array(weights) / np.sum(weights)
            weights = sorted(weights)

            # Calculate the mean of each asset
            mean_i = data.iloc[:, i].mean()
            mean_j = data.iloc[:, j].mean()

            # Calculate the weighted covariance
            ewcov[i][j] = (
                weights * ((data.iloc[:, i] - mean_i) * (data.iloc[:, j] - mean_j))
            ).sum()
            ewcov[j][i] = ewcov[i][j]

    return ewcov
