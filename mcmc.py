#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 17:50:33 2024

@author: jacopo
"""

# %%
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.linalg import solve_triangular
import time

np.random.seed(1)


# %% Bayesian methods

# def sampleMVG(A, b, n, e):
#     cholA = np.linalg.cholesky(A)
#     sample = np.linalg.solve(cholA.T, np.linalg.solve(cholA, b) + e)
#     return sample

def multi(z, mu, sigma):
    y = z.reshape(1, -1)
    p = y.shape[1]

    dec = np.linalg.cholesky(sigma)

    tmp = solve_triangular(dec, y.T - mu)

    rss = (tmp**2).sum()

    logretval = -np.log(np.diag(dec)).sum() - 0.5 * \
        p * np.log(2*np.pi) - 0.5*rss

    return logretval


def getLikelihood(z, X, b, lam, cov, q1, r1):
    p1 = stat.multivariate_normal.logpdf(
        z.reshape(-1), mean=(X@b).reshape(-1), cov=cov)

    # p1 = multi(z, X@b, cov)

    p2 = stat.invgamma.logpdf(lam, a=q1, scale=r1)
    return p1 + p2


# %% generate syntetic data


# coords = np.arange(1, 101, step=1).reshape(-1, 1)
# n = coords.shape[0]

# synth = {}
# synth['s2'] = 10
# synth['tau2'] = 5
# synth['lambda'] = synth['tau2']/synth['s2']
# synth['B'] = np.array([3, -0.7, 0.01]).reshape(-1, 1)

# X = np.hstack((np.ones(shape=(n, 1)), coords, coords**2))
# K = (1 + dist / 5) * np.exp(-dist / 5)

# dist = distance.cdist(coords, coords)
# csigma = np.linalg.cholesky(K)  # get the lower triangular

# yTrue = X @ synth['B'] + csigma @ np.random.normal(0, synth['s2'], size=(n, 1))
# z = yTrue + np.random.normal(0, synth['tau2'], size=(n, 1))

# # plot
# fig, ax = plt.subplots()
# ax.plot(yTrue, '--')
# ax.plot(z, 'o')

# %%
df = pd.read_csv('df.csv')

n = df.shape[0]

coords = np.arange(1, 101, step=1).reshape(-1, 1)
# n = coords.shape[0]
dist = distance.cdist(coords, coords)

synth = {}
synth['s2'] = 10
synth['tau2'] = 2
synth['lambda'] = synth['tau2']/synth['s2']
synth['B'] = np.array([3, -0.7, 0.01]).reshape(-1, 1)

# X = np.hstack((np.ones(shape=(n, 1)), coords, coords**2))

K = (1 + dist / 5) * np.exp(-dist / 5)

# generate syntetic data

# csigma = np.linalg.cholesky(sigma)  # get the lower triangular

# yTrue = X @ synth['B'] + csigma @ np.random.normal(0, synth['s2'], size=(n, 1))
# yObs = yTrue + np.random.normal(0, synth['tau2'], size=(n, 1))

# # plot
# fig, ax = plt.subplots()
# ax.plot(yTrue, '--')
# ax.plot(yObs, 'o')


z = df['z'].to_numpy().reshape(-1, 1)
X = df[['V1', 'x', 'V3']].to_numpy()

# %%  chain properties

chain = {}
chain['len'] = 10000
chain['burnin'] = 5000


# %% estimate model parameter


def estimateMH():

    # set initial values
    beta = np.linalg.inv(X.T @ X) @ X.T @ z
    s2 = 1
    lam = 100

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

    for i in range(0, chain['len']-1):

        # propose a new values beta
        A = (1 / s2) * X.T @ precision @ X
        b = (1 / s2) * X.T @ precision @ z

        # sample form multivariate normal
        invA = np.linalg.inv(A)
        beta = np.random.multivariate_normal(( @ b).reshape(-1), invA)
        beta = beta.reshape(-1, 1)

        # propose a new value sigma2

        shape = n/2 + q2
        temp = (z - X@beta)
        scale = (1/2) * temp.T @ precision @ temp + r2

        s2 = stat.invgamma.rvs(a=shape, scale=scale)

        # Propose new value of lam2
        tun_lam = 0.2
        lam_star = np.random.normal(lam, tun_lam, 1)
        cov_star = np.eye(n) * lam_star + K

        if lam_star > 0:

            # collo
            m1 = getLikelihood(z, X, beta, lam_star, cov_star*s2, q1, r1)

            m2 = getLikelihood(z, X, beta, lam, cov*s2, q1, r1)

            ratio = np.exp(m1 - m2)

            if (ratio > np.random.uniform()):
                lam = lam_star.copy()
                cov = cov_star.copy()
                precision = np.linalg.solve(cov, np.eye(n))

            beta_result[:, i] = beta.reshape(-1).copy()
            s2_result[i] = s2
            lam_result[i] = lam

    return beta_result, s2_result, lam_result


# Estimate the parameters (~)
tStart = time.time()
beta, s2, lam = estimateMH()
tEnd = time.time() - tStart


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

# stat.describe(lams)

# %% predictions

invK = np.linalg.solve(K, np.eye(n))

prediction = np.zeros((n, chain['len']-chain['burnin']))
for i in range(0, chain['len']-chain['burnin']):

    p_tao2 = lam[i] * s2[i]
    p_s2 = s2[i]

    A = np.eye(n)/p_tao2 + invK/p_s2
    b = (z - (X @ beta[:, i]).reshape(-1, 1)) / p_tao2

    noise = np.random.multivariate_normal(
        b.reshape(-1), np.linalg.solve(A, np.eye(n)))

    prediction[:, i] = X@beta[:, i] + noise


mu = prediction.mean(axis=1)
mu_ic = np.quantile(prediction, [0.05, 0.95], axis=1)

fig, ax = plt.subplots()
ax.plot(z, 'o')
ax.plot(mu)
ax.plot(mu_ic[0, :], '--r')
ax.plot(mu_ic[1, :], '--r')


# %% Hierarchical models


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
        beta = np.random.multivariate_normal((invA @ b).reshape(-1), invA)
        beta = beta.reshape(-1, 1)

        # New value for y
        A = tpsi/tao2 + invK / s2
        b = (psi.T / tao2) @ (z - X @ beta)

        # sample form multivariate normal
        invA = np.linalg.inv(A)
        y = np.random.multivariate_normal((invA @ b).reshape(-1), invA)
        y = y.reshape(-1, 1)
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


# %% Estimate model
tStart = time.time()
hie_beta, hie_s2, hie_tao2, hie_y = estimateMHHierachical(z, n, np.eye(n), K)
tEnd = time.time() - tStart
# %% Prot the results


fig, ax = plt.subplots(1, 2)

ax[0].plot(hie_tao2[:-1])
ax[1].hist(hie_tao2[:-1])
ax[0].set_title('Tao2 Trace plot (s2Lam = 0.2) (MH)')
ax[1].set_title(f'Tao2 Hist. (mu = {round(hie_tao2.mean(),2)})')

fig, ax = plt.subplots(1, 2)
ax[0].plot(hie_s2[:-1])
ax[1].hist(hie_s2[:-1])
ax[0].set_title('s2 Trace plot')
ax[1].set_title(f's2 Hist. (mu = {round(hie_s2.mean(),2)})')

fig, ax = plt.subplots(1, 2)
ax[0].plot(hie_y[0, :-1])
ax[1].hist(hie_y[0, :-1])


fig, ax = plt.subplots(3, 2)
ax[0, 0].plot(hie_beta[0, :-1])
ax[0, 1].hist(hie_beta[0, :-1])
ax[1, 0].plot(hie_beta[1, :-1])
ax[1, 1].hist(hie_beta[1, :-1])
ax[2, 0].plot(hie_beta[2, :-1])
ax[2, 1].hist(hie_beta[2, :-1])

hie_beta.mean(axis=1)


# %% orthogonalized model

psi = np.eye(n) - X @ np.linalg.solve(X.T @ X, np.eye(X.shape[1])) @ X.T

tStart = time.time()
ort_beta_start, ort_beta, ort_s2, ort_tao2, ort_y = estimateMHHierachical(
    z, n, psi, K)
tEnd = time.time() - tStart

# %% plot

fig, ax = plt.subplots(1, 2)

ax[0].plot(ort_tao2[:-1])
ax[1].hist(ort_tao2[:-1])
ax[0].set_title('Tao2 Trace plot (s2Lam = 0.2) (MH)')
ax[1].set_title(f'Tao2 Hist. (mu = {round(ort_tao2.mean(),2)})')

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


ort_prediction = np.zeros((n, chain['len']-chain['burnin']))
tpsi = psi @ psi

for i in range(0, chain['len']-chain['burnin']):

    p_tao2 = ort_tao2[i]
    p_s2 = ort_s2[i]

    tmp = X @ ort_beta[:, i] + ort_y[:, i]

    A = tpsi/p_tao2 + invK/p_s2
    b = (z - (X @ ort_beta_start[:, i]).reshape(-1, 1)) / p_tao2

    # sample form multivariate normal
    invA = np.linalg.inv(A)

    y = np.random.multivariate_normal(
        b.reshape(-1), np.linalg.solve(A, np.eye(n)))

    ort_prediction[:, i] = X@ort_beta_start[:, i] + y


mu = prediction.mean(axis=1)
mu_ic = np.quantile(prediction, [0.05, 0.95], axis=1)

fig, ax = plt.subplots()
ax.plot(z, 'o')
ax.plot(mu)
ax.plot(mu_ic[0, :], '--r')
ax.plot(mu_ic[1, :], '--r')

fig, ax = plt.subplots()
ax.plot(mu_ic[0, :] - mu, '--r')
ax.plot(mu_ic[1, :] - mu, '--r')
