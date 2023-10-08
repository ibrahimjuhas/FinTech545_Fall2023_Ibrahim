import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.linalg import svd, eigh
from numpy.linalg import inv, LinAlgError


# Cholesky Factorization
def cholesky_psd(A):
    n = A.shape[1]
    root = np.zeros((n, n))

    # loop over columns
    for i in range(n):
        s = 0.0
        if i > 0:
            s = root[i][:i].T @ root[i][:i]

        # Diagonal Element
        temp = A[i][i] - s
        if temp <= 0 and temp >= -1e-8:
            temp = 0.0
        root[i][i] = np.sqrt(temp)

        # check for eigen value. set the column to 0 if there is one
        if root[i][i] == 0.0:
            root[i][(i + 1) : n] = 0.0
        else:
            ir = 1.0 / root[i][i]
            for j in np.arange(i + 1, n):
                s = root[j][:i].T @ root[i][:i]
                root[j][i] = (A[j][i] - s) * ir
    return root


def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # calculate covariance matrix
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = np.dot(np.dot(invSD, out), invSD)

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (np.dot(np.dot(vecs, np.diag(vals)), vecs.T))
    T = np.diag(np.sqrt(np.diag(T)))
    l = np.diag(np.sqrt(vals))
    B = np.dot(np.dot(T, vecs), l)
    out = np.dot(B, B.T)

    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = np.dot(np.dot(invSD, out), invSD)
    return out


def Higham(A, num_iter):
    while num_iter > 0:
        eigvals, eigvecs = eigh(A)
        neg = 0
        for e in eigvals:
            if e < 0:
                D = np.diag(np.maximum(eigvals, 0))
                V = eigvecs @ np.sqrt(D)
                A = V @ V.T
                neg += 1
                break
        if neg == 0:
            break
        num_iter -= 1

    return (A + A.T) / 2
