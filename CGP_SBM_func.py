#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:08:29 2019

@author: Anonymous

simulate Causal Graph Processes following Stochastic block model
From "Signal Processing on Graphs: Estimating the structure of a graph"
"""
import numpy as np


# --------------------------------------------------------------------------- #
# for stochatic Block Model
def compSBM_A(Nc, Nn, lbde):
    """
    Compute the adjacency matrix of a Stochatic Block Model
    follow the set-up of Mei & Mourra 2017

    Nc: number of clusters
    Nn: number of nodes
    lbde: weight for the laplace distribution
    """
    # each node has a uniform probability of belonging to a cluster
    # list indicating to which cluster each node belongs to
    Cn = np.array([np.random.randint(low=0, high=Nc) for nn in np.arange(Nn)])

    # matrix of inter- & intra- clusters proba
    Cp = np.eye(Nc) * 0.05 + np.random.uniform(low=0.0, high=0.04,
                                               size=(Nc, Nc))
    Cp[Cp < 0.025] = 0

    # build edges for each cluster
    A = np.zeros((Nn, Nn))
    for i in np.arange(Nc):
        Cn_idx_i = np.where(Cn == i)[0]
        Nc_i = len(Cn_idx_i)
        Ac_i = np.random.binomial(1, Cp[i, i], size=(Nc_i, Nc_i)) * 1.0
        Ac_i *= np.random.laplace(0, lbde, size=(Nc_i, Nc_i))
        for ii in np.arange(Nc_i):
            A[Cn_idx_i[ii], Cn_idx_i] = Ac_i[ii, :]

        # build edges between clusters
        # for edges going from cluster j to cluster i
        for j in np.arange(Nc):
            Cn_idx_j = np.where(Cn == j)[0]
            Nc_j = len(Cn_idx_j)
            Ac_ij = np.random.binomial(1, Cp[i, j], size=(Nc_i, Nc_j)) * 1.0
            Ac_ij *= np.random.laplace(0, lbde, size=(Nc_i, Nc_j))
            for ii in np.arange(Nc_i):
                A[Cn_idx_i[ii], Cn_idx_j] = Ac_ij[ii, :]

    # normalised the adjacency matrix by its largest eigenvalue
    Aw, Av = np.linalg.eig(A)
    A /= 1.1 * np.abs(Aw).max()

    return A


# --------------------------------------------------------------------------- #
# foor CGP
def simulate_CGP(x0, P, A, N, M, K):
    """
    simulate a CGP starting from x0

    x0: initial state
    P: polynomial coefficient for each time lag
    A: adjacency matrix
    N: number of time series
    M: number of time lags
    """
    # x0 must have at least M time points
    x_o = np.zeros((N, K + M))
    x_o[:, :M] = x0

    for k in np.arange(M, K + M):
        wk_o = np.random.normal(loc=0.0, scale=1.0, size=N)  # white noise
        x_o[:, k] = wk_o + A.dot(x_o[:, [k - 1]]).flatten()

        kk = 2
        while kk <= M:
            x_o[:, k] += P[kk - 2].dot(x_o[:, [k - kk]]).flatten()
            kk += 1

    return x_o[:, M:]


def compCGP_C(M):
    """
    sample for polynomial coefficients C for the CGP

    We store the coefficients in a lower triangular matrix
    with the coefficients of lag i on line i

    M: number of lags
    """
    # we store the coedd as a lower triangular matrix
    # random polynomial coefficients
    c = 0.5 * np.random.uniform(-1.0, -0.45, size=(M + 1, M + 1)) +\
        0.5 * np.random.uniform(0.45, 1.0, size=(M + 1, M + 1))
    for i in np.arange(M + 1):
        c[i, :] /= 2**(np.arange(M + 1) + i)
    c /= 1.5
    c = np.tril(c)
    c[0, 0] = 0
    c[1, 0] = 0
    c[1, 1] = 1

    return c


def compCoeff_CGP(i, A, c, N):
    """
    Compute the CGP coefficients for lag i
    cA, c_2 A^2, ...

    i: time lag
    A: adjacency matrix
    C: polynomial coefficients
    N: number of time series
    """
    Ap = np.copy(A)
    out = c[i, 0] * np.eye(N)
    j = 1
    while j <= i:
        # compute A to the power p
        if j > 1:
            Ap = Ap.dot(A)

        # add to the polynome
        out += c[i, j] * Ap
        j += 1

    return out


# --------------------------------------------------------------------------- #
def compSBM_CGP(N, Nc, M, K, lbde, burn, L_test):
    """
    Compute a Causal Graph Process following a stochastic block model structure

    N: number of nodes
    Nc: number of clusters
    M: number of time lags
    K: number of time points
    lbde: weight for the laplace distribution
    burn: number of time points to burn
    L_test: number of time points for the out-of-sample set
    """

    # comp the Adjacency matrix
    A = compSBM_A(Nc, N, lbde)

    # comp the polynomial coefficients
    C = compCGP_C(M)
    P = [compCoeff_CGP(i, A, C, N) for i in np.arange(2, M + 1)]

    # ----------------------------------------------------------------------- #
    # simulate the time series
    x0 = np.zeros((N, M))
    x = simulate_CGP(x0, P, A, N, M, K + burn + L_test)

    # isolate in- and out-of-sample sets
    x = x[:, burn:]
    if L_test != 0:
        x_test = x[:, -L_test:]
        x = x[:, :-L_test]

    out = {'A': A,
           'C': C,
           'P': P,
           'x': x,
           'x_test': x_test}

    return out
