#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:26:29 2019

@author: theophile

script to show how to use the CGP-CCD algorithm

Will show the following functionalities:
    - Simulate a CGP-SBM graph
    - estimate the CGP coefficient with CGP-CCD algorithm
    - perform grid search to find lambda_1
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
import CGP_SBM_func as fc_cgp
import CGP_CCD_funcPerf as fc_perf

# global parameters for computation library
# True if we want to run the algorithm with pytorch instead of numpy
usePyTorch = False

if usePyTorch:
    import torch
    import CGP_CCD_func_pytorch as fc_ccd
    ftype = torch.float32  # float type to use
    device = 'cpu'  # device to use
else:
    import CGP_CCD_func_numpy as fc_ccd
    ftype = np.float32  # float type to use
    device = 'cpu'  # device to use

# --------------------------------------------------------------------------- #
doLog = False  # we want to use print not write on log file

###############################################################################
# ------------------------ Simulate a CGP-SBM graph ------------------------- #
###############################################################################
# parameters
Nc = 5  # nb of clusters
N = 500  # nb of nodes
lbde = 2.0  # lambda of the laplacian distribution use to sample the weights

# simulate CGP
M = 3  # number of lags to consider
K = 260 * 4  # time length
th_zero = 1e-10  # Threshold to add noise to zero elements of the diagonal to avoid singularity

# for simulation
L_test = 10  # length of time used as out-of-sample
burn = 500  # size of the burn-in

# --------------------------------------------------------------------------- #
# Build a CGP following a stochastic block model
out_SBM = fc_cgp.compSBM_CGP(N, Nc, M, K, lbde, burn, L_test)
A = out_SBM['A']  # The adjacency matrix
C = out_SBM['C']  # The polynom's coefficients
P = out_SBM['P']  # The CGP AR matrix coefficients for each lag
x = out_SBM['x']  # The simulated time series
x_test = out_SBM['x_test']  # The simulated time series for out-of-sample tests

# count the number of edges
nb_edges = np.count_nonzero(A)
print("nb of edges: {0} -- as a percentage (%): {1:.2f}".format(
      nb_edges, nb_edges * 100.0 / N**2))

# --------------------------------------------------------------------------- #
# Assuming CGP-SBM parameters, simulate data points
# use previous simulation as initial state
xO = x[:, -M:]

# Returns simulated CGP-SBM time series with x0 as initial state
x_p = fc_cgp.simulate_CGP(xO, P, A, N, M, K)

# visualise the time series
plt.plot(x.T)

###############################################################################
# ------------------- Run CGP-CCD for a specific lambda_1 ------------------- #
###############################################################################
isSimu = True  # we are using simulated data with a known adjacency matrix

# for A
lbd1 = 100  # weight of the L1 regularisation
eps_cvg = 1e-1  # convergence criterion
maxIt = 50  # max number of iteration for the optimisation

# Only estimate the adjacency matrix A
out_A = fc_ccd.compA(x, N, M, K, eps_cvg, maxIt, lbd1,
                     isSimu=isSimu, A=A, doLog=doLog,
                     device=device, ftype=ftype)
A_o = out_A['A_o']  # The estimated adjacency matrix
Err = out_A['Err']  # The L2 estimation error and evolution difference
F = out_A['F']  # The err and err^d error metrics used to select lambda_1

# For simulated environment, err_A has metrics measurgin the estimation error
# see fc_perf.perf_A() function for more details
err_A = out_A['err_A']

# for C
maxIt_c = 10
lbd1_c = 0.5e-1 * (K - M)
lbd2_c = 1e3 * (K - M)
eps_cvg_c = 1e-4

# estimate the adjacency matrix A and the coefficients C
out_CGP = fc_ccd.compCGP(x, N, M, K, eps_cvg, maxIt, lbd1,
                         eps_cvg_c, maxIt_c, lbd1_c, lbd2_c,
                         isSimu=isSimu, A=A, doLog=doLog,
                         device=device, ftype=ftype)
A_o = out_CGP['A_o']  # The estimated adjacency matrix
Err = out_CGP['Err']  # The L2 estimation error and evolution difference
F_A = out_CGP['F_A']  # The err and err^d error metrics used to select lambda_1
C = out_CGP['C']  # The estimated CGP polynom coefficients C
err_C = out_CGP['err_C']  # MSE obtained when fitting C

# For simulated environment, err_A has metrics measurgin the estimation error
# see fc_perf.perf_A() function for more details
err_A = out_CGP['err_A']


###############################################################################
# ---------------------- Automatic lambda_1 selection ----------------------- #
###############################################################################
# perform grid search for Lambda_1
# define grid
L_lbd1 = np.arange(1.0, 500.0, 5.0)

# measure execution time
start_time_T = timeit.default_timer()

# run block CCD to obtain the the adjacency matrix
out = fc_ccd.compA_lbd(x, N, M, K, eps_cvg, maxIt, L_lbd1,
                       isSimu=isSimu, A=A, th_zero=th_zero,
                       doLog=doLog, device=device, ftype=ftype)
A_L = out['A_L']  # List of estimated adjacency matrix
Err = out['Err']  # List of the L2 estimation error and evolution difference
MSE = Err[:, 0] / N / (K - M)  # only MSE-in
err_A = out['err_A']  # List of estimation error for A in simulated environment
F = out['F']  # List of the err and err^d error metrics used to select lambda_1
AIC = out['AIC']  # AIC criterion
BIC = out['BIC']  # BIC criterion

elapsed_T = timeit.default_timer() - start_time_T
print("----> Total time = {0}".format(elapsed_T))

# --------------------------------------------------------------------------- #
"""
observe err and err^d
plot all the err metrics on the same graph
right axis shows the error metrics err and err^d
if on simulations:
    - left axis shows the number of different edges as %
else:
    - left axis shows the sparsity level
"""
# plot err & err^d
fc_perf.plotCompMetrics_Err(L_lbd1, err_A[:, 0], F[:, 0], F[:, 1], N,
                            isSimu=isSimu)

# plot err, err^d, AIC & BIC
fc_perf.plotCompMetrics_Full(L_lbd1, err_A[:, 0],
                             F[:, 0], F[:, 1],
                             AIC, BIC,
                             N, isSimu=isSimu)

# select lambda_1 and A
if doBIC:
    # with err, err^d and BIC
    lbd1_d, idx_d = fc_perf.selectLambda_errBic(
        L_lbd1, F[:, 0], F[:, 1], BIC,
        A_L, N, MSE, MSE_A=err_A[:, 3],
        isSimu=isSimu, err_A=err_A)
else:
    # with err and err^d
    lbd1_d, idx_d = fc_perf.selectLambda(F[:, 0], F[:, 1],
                                         L_lbd1, A_L, N,
                                         err_A[:, 3], MSE,
                                         isSimu=isSimu, err_A=err_A)
    

# Plot the True adjacency matrix A and its estimation A_o
# and the matrix of their difference
fc_perf.plotMat_compdiff(-1.0 * np.abs(A), 'True Adjacency matrix',
                         -1.0 * np.abs(A_o), 'Estimated Adjacency matrix')

