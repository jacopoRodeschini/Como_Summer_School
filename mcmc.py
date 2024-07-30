#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:50:33 2024

@title: Bayesian Kriging

@author: Jacopo Rodeschini, Alessandro Fusta Moro 
"""

# %% Import package
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.linalg import solve_triangular
import time

# Fix the seed (reporducibility)
np.random.seed(1)

# %% User defined functions


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


# %% generate syntetic data

# Syntethic data object
synth = {}
synth['s2'] = 3.4
synth['tau2'] = 2.1
synth['lambda'] = synth['tau2']/synth['s2']
synth['beta'] = np.array([3, -0.7, 0.01]).reshape(-1, 1)
synth['n'] = 100
synth['coords'] = np.sort(np.random.uniform(
    low=0, high=100, size=(synth['n'],)))


# Design matrix
X = np.vstack((np.ones(shape=(synth['n'],)),
              synth['coords'], synth['coords']**2)).T

# Compute the mather covariance matrix
dist = distance.cdist(synth['coords'].reshape(-1, 1),
                      synth['coords'].reshape(-1, 1))
K = (1 + dist / 5) * np.exp(-dist / 5)


# Create syntetic data [true process and observed process]
csigma = np.linalg.cholesky(K)  # lower triangular

yTrue = X @ synth['beta'] + \
    csigma @ np.random.normal(0, synth['s2'], size=(synth['n'], 1))
z = yTrue + np.random.normal(0, synth['tau2'], size=(synth['n'], 1))

# plot
fig, ax = plt.subplots()
ax.plot(yTrue, '--', label="True process")
ax.plot(z, 'o', label="Observed process")
ax.legend()
ax.set_title("lambda = {v}".format(v=round(synth['lambda'], 2)))

# %%  chain properties

chain = {}
chain['len'] = 10000
chain['burnin'] = 5000


# %% Implementation of Method 1

def estimateMH(z, X, K, chain):

    # set initial values (OLS estimate)
    beta = np.linalg.inv(X.T @ X) @ X.T @ z
    s2 = 1
    lam = 100
    n = X.shape[0]

    # Compute the covariance
    cov = np.eye(n) * lam + K
    precision = np.linalg.inv(cov)

    # lambda: weakly prior
    q1 = .001
    r1 = .001

    # s2: weakly prior
    q2 = .001
    r2 = .001

    lam_result = np.zeros((chain['len'], 1))
    s2_result = np.zeros((chain['len'], 1))
    beta_result = np.zeros((3, chain['len']))

    # tuning parameter for lambda
    tun_lam = 0.2

    for i in range(0, chain['len']-1):

        # propose a new values beta
        A = (1 / s2) * X.T @ precision @ X
        b = (1 / s2) * X.T @ precision @ z

        # sample form multivariate normal
        invA = np.linalg.inv(A)
        beta = sampleMVG(invA @ b, invA)

        # propose a new value sigma2
        shape = n/2 + q2
        temp = (z - X@beta)
        scale = (1/2) * temp.T @ precision @ temp + r2

        s2 = stat.invgamma.rvs(a=shape, scale=scale)

        # Propose new value of lam2
        lam_star = np.random.normal(lam, tun_lam, 1)
        cov_star = np.eye(n) * lam_star + K

        if lam_star > 0:

            m1 = getLikelihood(z, X@beta, cov_star*s2, lam_star, q1, r1)

            m2 = getLikelihood(z, X@beta, cov*s2, lam, q1, r1)

            ratio = np.exp(m1 - m2)

            if (ratio > np.random.uniform()):
                lam = lam_star.copy()
                cov = cov_star.copy()
                precision = np.linalg.solve(cov, np.eye(n))

            beta_result[:, i] = beta.reshape(-1).copy()
            s2_result[i] = s2
            lam_result[i] = lam

    return beta_result, s2_result, lam_result


# %%  Estimate the model parameters (~ 20s)

tStart = time.time()
beta, s2, lam = estimateMH(z, X, K, chain)
tEnd = time.time() - tStart

print("Time: {time} (sec)".format(time=round(tEnd, 2)))

# %% Prot the results


fig, ax = plt.subplots(3, 2)

ax[0, 0].plot(lam[:-1])
ax[0, 1].hist(lam[:-1])

ax[1, 0].plot(s2[:-1])
ax[1, 1].hist(s2[:-1])

ax[2, 0].plot(beta[0, :-1])
ax[2, 1].hist(beta[0, :-1])


beta.mean(axis=1)
lam[chain['burnin']:].mean()
s2[chain['burnin']:].mean()

# %% predictions

n = synth['n']

# Compute the precision of the matern covaraince matrix
invK = np.linalg.solve(K, np.eye(n))

prediction = np.zeros((n, chain['len']-chain['burnin']))

for i in range(0, chain['len']-chain['burnin']):

    p_tao2 = lam[i] * s2[i]
    prediction[:, i] = sampleMVG(
        (X @ beta[:, i]).reshape(-1, 1), np.eye(n)*p_tao2 + K*s2[i]).reshape(-1)


# Compute the mean and the quantile interval
mu = prediction.mean(axis=1)
mu_ic = np.quantile(prediction, [0.05, 0.95], axis=1)

fig, ax = plt.subplots()
ax.plot(yTrue, '--', label="True process", linewidth=3)
ax.plot(z, 'o', label='Observed process')
ax.plot(mu, label='posterior mean')
ax.plot(mu_ic[0, :], '--r', label="Quantile intervel [0.05, 0.95]")
ax.plot(mu_ic[1, :], '--r')
ax.legend()

# %% Bayesian Hierarchical Model


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


# %% Estimate Hierarchical Model (~8 sec)

tStart = time.time()
hie_beta, _, hie_s2, hie_tao2, hie_y = estimateMHHierachical(
    z, synth['n'], np.eye(synth['n']), K)
tEnd = time.time() - tStart

print("Time: {time} (sec)".format(time=round(tEnd, 2)))


# %% Prot the results


fig, ax = plt.subplots(1, 2)

ax[0].plot(hie_tao2[:-1])
ax[1].hist(hie_tao2[:-1])
ax[0].set_title('Tao^2 Trace plot')
ax[1].set_title("Tao^2 Hist. mean={mu}".format(
    mu=round(hie_tao2[chain['burnin']:].mean(), 2)))


fig, ax = plt.subplots(1, 2)
ax[0].plot(hie_s2[:-1])
ax[1].hist(hie_s2[:-1])
ax[0].set_title('s^2 Trace plot')
ax[1].set_title("s^2 Hist. mean={mu}".format(
    mu=round(hie_s2[chain['burnin']:].mean(), 2)))

fig, ax = plt.subplots(1, 2)
ax[0].plot(hie_y[0, :-1])
ax[1].hist(hie_y[0, :-1])
ax[0].set_title('y[0] Trace plot')
ax[1].set_title("y[0] Hist. mean={mu}".format(
    mu=round(hie_y[0, chain['burnin']:].mean(), 2)))


fig, ax = plt.subplots(3, 2)
ax[0, 0].plot(hie_beta[0, :-1])
ax[0, 1].hist(hie_beta[0, :-1])
ax[1, 0].plot(hie_beta[1, :-1])
ax[1, 1].hist(hie_beta[1, :-1])
ax[2, 0].plot(hie_beta[2, :-1])
ax[2, 1].hist(hie_beta[2, :-1])

hie_beta.mean(axis=1)


# %% orthogonalized model (~5 sec)
n = synth['n']

# Compute the projection matrix
psi = np.eye(n) - X @ np.linalg.solve(X.T @ X, np.eye(X.shape[1])) @ X.T

tStart = time.time()
ort_beta_start, ort_beta, ort_s2, ort_tao2, ort_y = estimateMHHierachical(
    z, n, psi, K)
tEnd = time.time() - tStart

print("Time: {time} (sec)".format(time=round(tEnd, 2)))

# %% plot

fig, ax = plt.subplots(1, 2)

ax[0].plot(ort_tao2[:-1])
ax[1].hist(ort_tao2[:-1])
ax[0].set_title('Tao^2 Trace plot')
ax[1].set_title("Tao^2 Hist. {mu}".format(
    mu=round(ort_tao2[chain['burnin']].mean(), 2)))

fig, ax = plt.subplots(1, 2)
ax[0].plot(ort_s2[:-1])
ax[1].hist(ort_s2[:-1])
ax[0].set_title('s2 Trace plot')
ax[1].set_title(f's2 Hist. (mu = {round(ort_s2.mean(),2)})')


fig, ax = plt.subplots(3, 2)
ax[0, 0].plot(ort_beta[0, :-1])
ax[0, 1].hist(ort_beta[0, :-1])
ax[1, 0].plot(ort_beta[1, :-1])
ax[1, 1].hist(ort_beta[1, :-1])
ax[2, 0].plot(ort_beta[2, :-1])
ax[2, 1].hist(ort_beta[2, :-1])

ort_beta.mean(axis=1)


# prediction

ort_prediction = np.zeros((n, chain['len']-chain['burnin']))
tpsi = psi @ psi
invK = np.linalg.inv(K)

for i in range(0, chain['len']-chain['burnin']):

    A = tpsi/ort_tao2[i] + invK/ort_s2[i]
    b = (z - (X @ ort_beta_start[:, i]).reshape(-1, 1)) / ort_tao2[i]

    # sample form multivariate normal
    y = sampleMVG(b, np.linalg.solve(A, np.eye(n))).reshape(-1)

    ort_prediction[:, i] = X@ort_beta_start[:, i] + y


mu = ort_prediction.mean(axis=1)
mu_ic = np.quantile(ort_prediction, [0.05, 0.95], axis=1)

fig, ax = plt.subplots()
ax.plot(yTrue, '--', label="True process", linewidth=3)
ax.plot(z, 'o', label='Observed process')
ax.plot(mu, label='posterior mean')
ax.plot(mu_ic[0, :], '--r', label="Quantile intervel [0.05, 0.95]")
ax.plot(mu_ic[1, :], '--r')
ax.legend()

fig, ax = plt.subplots()
ax.plot(mu_ic[0, :] - mu, '--r')
ax.plot(mu_ic[1, :] - mu, '--r')
ax.set_title("Size of 90% credible interval")
