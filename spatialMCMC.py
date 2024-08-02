#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:29:27 2024

@author: jacopo
"""


# %%import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.linalg import solve_triangular
import time
import pandas as pd

import netCDF4 as nc


# %% Import data

meta = pd.read_csv('COTavgInfoSubset_df.csv')
data = pd.read_csv('COTavgMAM.csv')


s = np.vstack((meta['lon'], meta['lat']))

# Response variable
z = (data['x'].to_numpy() / 100).reshape(-1, 1)
n = z.shape[0]

# Plot
fig, ax = plt.subplots()
sc = ax.scatter(s[0, :], s[1, :], c=z)
fig.colorbar(sc)
ax.set_title("Temperature")


# compute the Mathern covaraince function
dist = distance.cdist(s.T, s.T)
K = (1 + dist / 0.75) * np.exp(-dist / 0.75)

# Build the design matrix (intercept and coordinates)
X = np.vstack((np.ones(n), s[0, :], s[1, :])).T


# Buld the prediction grid
gridx, gridy = np.meshgrid(
    np.arange(-110, -100, step=0.2), np.arange(36, 42, step=0.12))

grid = np.vstack((gridx.flatten(), gridy.flatten())).T


# %% Chain properties

chain = {}
chain['len'] = 1000
chain['burnin'] = 500


# %% user functions
def sampleMVG(mean, cov):

    # Compute the Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov)

    # Generate standard normal samples
    standard_normal_samples = np.random.randn(mean.shape[0], 1)

    # Transform the standard normal samples
    return mean + L @ standard_normal_samples


def log_normalpdf(x, mean, cov):
    k = mean.shape[0]
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)
    diff = x - mean
    sol = L_inv @ diff

    log_det_cov = 2 * np.sum(np.log(np.diag(L)))
    term1 = -0.5 * k * np.log(2 * np.pi)
    term2 = -0.5 * log_det_cov
    term3 = -0.5 * (sol.T @ sol)

    return term1 + term2 + term3


def getLikelihood(z, mu, cov, lam, q1, r1):

    p1 = log_normalpdf(z, mu, cov)

    p2 = stat.invgamma.logpdf(lam, a=q1, scale=r1)

    return p1 + p2

# %% estimate the model


def estimateMHHierachical(z, n, psi, K):

    # psi**2
    tpsi = psi @ psi

    # Kinv
    invK = np.linalg.solve(K, np.eye(n))

    # set initial values
    beta = np.linalg.inv(X.T @ X) @ X.T @ z
    s2 = 1
    tao2 = 1

    # temp
    Xblock = np.linalg.solve(X.T @ X, np.eye(beta.shape[0])) @ X.T

    # firs step residual
    y = z - X@beta
    ypsi = psi @ y

    # lambda: weakly informative inverse gamma
    q1 = .001
    r1 = .001
    # s2: weakly informative inverse gamma
    q2 = .001
    r2 = .001

    tao2_result = np.zeros((chain['len'], 1))
    s2_result = np.zeros((chain['len'], 1))
    beta_result = np.zeros((3, chain['len']))
    y_result = np.zeros((n, chain['len']))
    beta_star_result = np.zeros((3, chain['len']))

    for i in range(0, chain['len']-1):

        # Compute the y

        # propose a new values beta
        A = (1 / tao2) * X.T @ X
        b = (1 / tao2) * X.T @ (z - ypsi)

        # sample form multivariate normal
        invA = np.linalg.inv(A)
        beta = sampleMVG(invA @ b, invA)

        # New value for y
        A = tpsi/tao2 + invK / s2
        b = (psi.T / tao2) @ (z - X @ beta)

        # sample form multivariate normal
        invA = np.linalg.inv(A)
        y = sampleMVG(invA @ b, invA)
        ypsi = psi @ y

        # new value for tao2
        shape = n/2 + q2
        temp = (z - X @ beta - ypsi)
        scale = (1/2) * temp.T @  temp + r2
        tao2 = stat.invgamma.rvs(a=shape, scale=scale)

        # propose a new value sigma2
        shape = n/2 + q1
        scale = (1/2) * y.T @ invK @ y + r1
        s2 = stat.invgamma.rvs(a=shape, scale=scale)

        beta_star_result[:, i] = beta.reshape(-1).copy()
        beta_result[:, i] = (beta - Xblock @ y).reshape(-1).copy()
        s2_result[i] = s2.copy()
        tao2_result[i] = tao2.copy()
        y_result[:, i] = y.reshape(-1).copy()

    return beta_star_result, beta_result, s2_result, tao2_result, y_result

# %% Estimate the model


# Compute the projection matrix
psi = np.eye(n) - X @ np.linalg.solve(X.T @ X, np.eye(X.shape[1])) @ X.T

tStart = time.time()
beta_start, beta, s2, tao2, y = estimateMHHierachical(
    z, n, psi, K)
tEnd = time.time() - tStart

print("Time: {time} (sec)".format(time=round(tEnd, 2)))

# %% plot the trace plot


fig, ax = plt.subplots(1, 2)

ax[0].plot(tao2[:-1])
ax[1].hist(tao2[:-1])
ax[0].set_title('Tao2 Trace plot (s2Lam = 0.2) (MH)')
ax[1].set_title(f'Tao2 Hist. (mu = {round(tao2.mean(),2)})')

fig, ax = plt.subplots(1, 2)
ax[0].plot(s2[:-1])
ax[1].hist(s2[:-1])
ax[0].set_title('s2 Trace plot')
ax[1].set_title(f's2 Hist. (mu = {round(s2.mean(),2)})')

fig, ax = plt.subplots(1, 2)
ax[0].plot(y[0, :-1])
ax[1].hist(y[0, :-1])


fig, ax = plt.subplots(3, 2)
ax[0, 0].plot(beta[0, :-1])
ax[0, 1].hist(beta[0, :-1])
ax[1, 0].plot(beta[1, :-1])
ax[1, 1].hist(beta[1, :-1])
ax[2, 0].plot(beta[2, :-1])
ax[2, 1].hist(beta[2, :-1])

beta.mean(axis=1)

# scatter plot of the mean of y
ymean = y.mean(axis=1)

fig, ax = plt.subplots()
sc = ax.scatter(s[0, :], s[1, :], c=ymean)
fig.colorbar(sc)
ax.set_title("Spatial pattern")


