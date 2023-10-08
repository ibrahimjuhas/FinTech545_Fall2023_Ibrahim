import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd, eigh
from numpy.linalg import inv, LinAlgError


def direct_simulation(cov, num):
    result = chol_psd(cov) @ np.random.standard_normal(size=(len(cov), num))
    return result


num = 25000


# PCA simulation
def simulate_pca(a, nsim, perc):
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eig(a)

    flip = np.argsort(vals)[::-1]
    vals = vals[flip]
    vecs = vecs[:, flip]

    tv = np.sum(vals)
    start = 0
    while np.abs(np.sum(vals[:start]) / tv) < perc:
        start += 1
    vals = vals[:start]
    vecs = vecs[:, :start]
    print(
        "Simulating with",
        start,
        "PC Factors: {:.2f}".format(np.abs(sum(vals) / tv * 100)),
        "% total variance explained",
    )
    B = np.matmul(vecs, np.diag(np.sqrt(vals)))
    m = B.shape[1]
    r = np.random.randn(m, nsim)
    return np.matmul(B, r)
