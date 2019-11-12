#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:30:29 2019

@author: theophile
"""
import timeit
import logging
import numpy as np
import CGP_CCD_funcPerf as fc_perf


###############################################################################
# ----------------------------- error metrics ------------------------------- #
###############################################################################
def compErr(Rh, x, xh, M):
    """
    Compute the L2 norm of the prediction error

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    Rh: CGP coefficients stacks horizontally; dim(N x NM)
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    xh: matrix of the lagged time series stack vertically; dim(NM, K)
    """
    err = x[:, M:] - Rh.dot(xh)

    return np.sum(err**2)


def compFj(R1_j, x, xh_j, N, M, K):
    """
    Comp err and err^d for node j

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    R1_j: column j of the adjacency matrix; dim(N)
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    xh_j: matrix of the lagged time series of node j stack vertically; dim(1, K)
    """

    # select non-zero edges
    idx_nnZero_j = np.where(R1_j != 0)[0]

    xj = x[idx_nnZero_j, M:]
    Rj = R1_j[idx_nnZero_j]

    # comp forecasts
    xP = Rj * xh_j
    rss = (xj - xP)**2

    # comp err and err^d for node j
    err_j = np.mean(rss)
    err_d_j = np.sum(rss) / np.sum(np.abs(Rj))

    return err_j, err_d_j
    # return np.log(err_j), np.log(err_d_j)


def compF(R_1, x, N, M, K, L_j):
    """
    Comp the error metrics err and err^d
    used to select lambda_1 the LASSO coefficient

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    R_1: adjacency matrix; dim(N, N)
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    L_j: indices of the columns of R_1; dim(N)
    """

    err_j = np.zeros((N, 2))
    for j in L_j:
        xh_j = x[[j], M:]
        Rh_j = R_1[:, [j]]

        if np.sum(np.abs(Rh_j)) != 0:
            err_j[j, 0], err_j[j, 1] = compFj(
                Rh_j, x, xh_j, N, M, K)
        else:
            err_j[j, :] = 0.0

    return np.sum(err_j, axis=0)


###############################################################################
# -----------------------------Functions for CCD  --------------------------- #
###############################################################################
def softT(b1, b2):
    """
    Soft thresholding function:
    f(b1, b2) = sign(b1) max(abs(b1) - b2, 0)
    """
    return np.sign(b1) * np.maximum(np.abs(b1) - b2, 0)


def comp_xh(x, N, M, K):
    """
    Stack the lagged matrix of time series vertically

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    """
    # need to flip order since R1 is at the start of Rh and goes with x[t-1]
    xh = x[:, M - 1:-1]
    for m in np.arange(2, M + 1):
        xh = np.vstack([xh, x[:, M - m:-m]])

    # xh = x[:, :M][:, ::-1].T.reshape((N * M, 1))
    # for k in np.arange(M + 1, K):
    #     xh = np.hstack([xh, x[:, k - M:k][:, ::-1].T.reshape((N * M, 1))])


    return xh


###############################################################################
# ----------------------------- CCD update of R_1  -------------------------- #
###############################################################################
def compR1_denum(x, N, M, K):
    """
    Compute the denominator of the CCD update for R_1

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    """
    return np.array([np.sum(x[j, M - 1:K - 1]**2) for j in np.arange(N)])


def compCol_R1(SR, R1_j, x_j, denum_j, lbd1, N):
    """
    Compute CCD update for column j of R_1, denoted R1_j

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    SR: prediction error without the contribution of node j at lag 1
    R1_j: column j of R_1
    x_j: vector of the time series j; dim(K)
    denum_j: denominator of the CCD update equation for node j
    lbd1: LASSO coefficient lambda_1
    """
    # delete column j
    Rxij = SR + R1_j * x_j

    # comp numerator with soft thresholding
    num = np.sum(Rxij * x_j, axis=1, keepdims=True)

    # update column
    R1_j = softT(num, lbd1) / denum_j

    # update sum
    SR = Rxij - R1_j * x_j
    return R1_j, SR


def compR1(Rh, x, xh, lbd1, denum_R1, N, M, K, L_j):
    """
    Compute CCD update for R_1

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    Rh: CGP coefficients stacks horizontally; dim(N x NM)
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    xh: matrix of the lagged time series stack vertically; dim(NM, K)
    lbd1: LASSO coefficient lambda_1
    denum_R1: denumerator of the CCD update for columns of R_1; dim(N)
    L_j: indices of the columns of R_1
    """
    R1 = Rh[:, :N]
    SR = x[:, M:] - Rh.dot(xh)

    # shuffle the list of indices
    np.random.shuffle(L_j)
    for j in L_j:
        R1[:, [j]], SR = compCol_R1(SR, R1[:, [j]], x[[j], M - 1:K - 1],
                                    denum_R1[j], lbd1, N)

    return R1


###############################################################################
# ----------------------------- CCD update of Ri  --------------------------- #
###############################################################################
def compRi_denum(x, N, M, K, th_zero=1e-10):
    """
    Compute the matrices inverse used in the CCD matrix-update of Ri

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    """

    L_i = np.arange(1, M + 1)
    L_k = np.arange(M + 1, K)

    # compute the denominator
    xx_inv = []
    for i in L_i:
        xxk = x[:, [M - i]].dot(x[:, [M - i]].T)
        for k in L_k:
            xxk += x[:, [k - i]].dot(x[:, [k - i]].T)

        # add noise to zero elements of the diagonal to avoid singularity
        xxk_d = np.diag(xxk)
        xxk_d0 = np.copy(xxk_d) * 0.0
        xxk_d0[np.abs(xxk_d) <= th_zero] = th_zero
        xxk += np.diag(xxk_d0)

        # for Ri with i>1
        xx_inv.append(np.linalg.solve(xxk, np.eye(N)))

    return xx_inv


def compS(i, Rh, x, xh, N, M):
    """
    Compute the prediction error without lag i

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    i: time lag
    Rh: CGP coefficients stacks horizontally; dim(N x NM)
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    xh: matrix of the lagged time series stack vertically; dim(NM, K)
    """

    if N > 200:
        xh[N * (i - 1): N * i, :] = 0.0
        Sk = x[:, M:] - Rh.dot(xh)
    else:
        # On low environments we observes numerical instability
        # of putting values to zero, hence this more costly option
        if i - 1 < M - 1:
            Rh_i = np.hstack([Rh[:, :N * (i - 1)], Rh[:, N * i:]])
            xh_i = np.vstack([xh[:N * (i - 1), :], xh[N * i:, :]])
        else:
            Rh_i = Rh[:, :-N]
            xh_i = xh[:-N, :]

        Sk = x[:, M:] - Rh_i.dot(xh_i)

    return Sk


def compRi(i, Rh, x, xh, denum_Ri, N, M, K):
    """
    Compute CCD matrix-update for R_i the lag-i CGP matrix coefficient

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    Rh: CGP coefficients stacks horizontally; dim(N x NM)
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    xh: matrix of the lagged time series stack vertically; dim(NM, K)
    lbd1: LASSO coefficient lambda_1
    denum_Ri: denumerator of the CCD update R_i; dim(N, N)
    L_j: indices of the columns of R_1
    """
    Sk = compS(i, Rh, x, xh, N, M)
    Sx = Sk.dot(x[:, M - i:K - i].T)
    Ri = Sx.dot(denum_Ri)

    return Ri


###############################################################################
# ------------------------------ CCD update of C  --------------------------- #
###############################################################################
def softT_c(x, t):
    """
    ##########################################
    This version uses numpy instead of PyTorch
    ##########################################

    Soft thresholding function:
    f(b1, b2) = sign(b1) max(abs(b1) - b2, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)


def comp_pA(A, N, M):
    """
    Compute the matrix-power of the adjacency matrix A
    A, A^2, ... , A^M

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    A: The adjacency matrix
    """
    Ap = [np.eye(N)]
    Ap.append(A)
    for j in np.arange(2, M + 1):
        Ap.append(Ap[-1].dot(A))

    return Ap


def comp_AX(Ap, x, M):
    """
    Compute the matrix polynome variables
    Ax, A^2 x^2, ... , A^M x^M

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    A: The adjacency matrix
    """
    AX = []
    for i in np.arange(2, M + 1):
        for j in np.arange(i + 1):
            AX.append(Ap[j].dot(x[:, M - i:-i]))

    return AX


def compCij(C, C_L, ij, AX, Y, lbd1_c, lbd2_c):
    """
    #############################################
    we add subscript "wij" to vect/mat without "ij"
    #############################################

    Compute CCD update for coefficient C_ij

    C: current polynomial coefficients
    C_L: polynom's coefficients indices
    AX: polynome variables
    Y: output variables to regress

    lbd1_c: L1 coefficient lambda_1
    lbd2_c: L2 coefficient lambda_2
    """

    C_wij = np.copy(C)
    C_wij[ij] = 0
    W = np.array([C_wij[l] * AX[l] for l in np.arange(C_L)]).sum(axis=0)

    num = np.sum(np.diag(AX[ij].T.dot(Y - W)))
    denum = np.sum(np.diag(AX[ij].T.dot(AX[ij]))) + 2 * lbd2_c

    cij = softT_c(num, lbd1_c) / denum

    return cij


def MSE_c(C, C_L, AX, Y):
    """
    Compute MSE error between Y and the obtained polynom values

    C: current polynomial coefficients
    C_L: polynom's coefficients indices
    AX: polynome variables
    Y: output variables to regress
    """
    err = Y - np.array([C[l] * AX[l] for l in np.arange(C_L)]).sum(axis=0)
    return np.mean(err**2)


def CCD_C(A, x, N, M, eps_cvg_c, maxIt, lbd1_c, lbd2_c):
    """
    #############################################
    This function is not converted to PyTorch yet
    #############################################

    Compute the polynomial coefficients C
    Run a CCD update of each coefficient C_ij

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    A: The adjacency matrix
    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)

    eps_cvg_c: convergence threshold
    maxIt: maximum number of iterations allowed
    lbd1_c: L1 coefficient lambda_1
    lbd2_c: L2 coefficient lambda_2
    """

    # init C randomly
    C_L = np.sum(np.arange(2, M + 1) + 1)
    C = np.random.uniform(-0.1, 0.1, C_L)

    # compute the powers of A
    Aj = comp_pA(A, N, M)

    # compute the tensor AX
    AX = comp_AX(Aj, x, M)

    # compute Y
    Y = x[:, M:] - A.dot(x[:, M - 1:-1])

    # run CCD over the Cij
    it = 0
    diff = [1e10]
    MSE = []
    while (diff[-1] > eps_cvg_c) & (it < maxIt):
        it += 1
        C_T = np.copy(C)

        # loop over the Cij
        for ij in np.arange(C_L):
            C[ij] = compCij(C, C_L, ij, AX, Y, lbd1_c, lbd2_c)

        diff.append(np.sum((C_T - C)**2))
        MSE.append(MSE_c(C, C_L, AX, Y))

    return C, diff, MSE


###############################################################################
# --------------------------------- Block CCD ------------------------------- #
###############################################################################
def CCD_Ri(x_d, xh_d, denum_Ri, denum_R1, Rh_d, N, M, K,
           eps_cvg, maxIt, lbd1,
           doLog=False, device='cpu', ftype=np.float32):
    """
    Compute the block coordinate descent to obtain the CGP coefficients
    and recover the adjacency matrix with lag-1 coefficient

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    Rh_d: CGP coefficients stacks horizontally; dim(N x NM)
    x_d: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)
    xh_d: matrix of the lagged time series stack vertically; dim(NM, K)
    denum_Ri: denumerator of the CCD update R_i; dim(N, N)
    denum_R1: denumerator of the CCD update for columns of R_1; dim(N)

    eps_cvg: convergence threshold
    maxIt: maximum number of iterations allowed
    lbd1: LASSO coefficient lambda_1

    doLog: if True the output or written on the log file instead of printed on the consol
    device: for PyTorch, specifies the device on which the code should be run
    ftype: specifies the float type to use for the computations
    """
    # init metrics
    it = 0
    L_i = np.arange(2, M + 1)
    L_j = np.arange(N)

    Err = np.zeros(maxIt + 1, dtype=ftype)
    R_diff = np.zeros(maxIt + 1, dtype=ftype)
    Err_diff = np.zeros(maxIt + 1, dtype=ftype)

    Err[0] = 1.0e10
    R_diff[0] = 1.0e20
    Err_diff[0] = 1.0e20

    while (R_diff[it] > eps_cvg) & (it < maxIt) & (Err_diff[it] > eps_cvg):
        it += 1
        start_time_k = timeit.default_timer()
        Rh_d_t = np.copy(Rh_d)

        # compute CCD update on the columns of R1
        Rh_d[:, :N] = compR1(Rh_d, x_d, xh_d, lbd1, denum_R1, N, M, K, L_j)

        # perform CCD matrix-update for lag > 1
        for i in L_i:
            Rh_d[:, N * (i - 1):N * i] = compRi(i, Rh_d, x_d, xh_d,
                                                denum_Ri[i - 1], N, M, K)

        R_diff[it] = np.sum(np.abs(Rh_d_t - Rh_d))
        Err[it] = compErr(Rh_d, x_d, xh_d, M)
        Err_diff[it] = np.abs(Err[it] - Err[it - 1])

        elapsed_k = timeit.default_timer() - start_time_k
        if doLog:
            logging.info("R_diff = {0:.3f} -- Err = {1:.3f} -- Err_diff = {3:.3f} -- time = {2:.3f}".format(
                R_diff[it], Err[it], elapsed_k, Err_diff[it]))
        else:
            print("R_diff = {0:.3f} -- Err = {1:.3f} -- Err_diff = {3:.3f} -- time = {2:.3f}".format(
                R_diff[it], Err[it], elapsed_k, Err_diff[it]))

        # having second thought on using (Err[it] > Err[it - 1])....
        if (R_diff[it] > R_diff[it - 1]) | (Err[it] > Err[it - 1]):
            if doLog:
                logging.info("-----------------------------------> diff increased")
            else:
                print("-----------------------------------> diff increased")

            break

    if it == maxIt:
        if doLog:
            logging.info("-----------------------------------> max iteration reached")
        else:
            print("-----------------------------------> max iteration reached")

    elif R_diff[it] <= eps_cvg:
        if doLog:
            logging.info("-----------------------------------> R converged")
        else:
            print("-----------------------------------> R converged")

    elif Err_diff[it] <= eps_cvg:
        if doLog:
            logging.info("-----------------------------------> F converged")
        else:
            print("-----------------------------------> F converged")

    # run an extra step of the descent to obtain the adjacency matrix A
    Rh_d[:, :N] = compR1(Rh_d, x_d, xh_d, lbd1, denum_R1, N, M, K, L_j)

    # compute the prediciton error
    Err[-1] = compErr(Rh_d, x_d, xh_d, M)

    return Rh_d[:, :N], Rh_d, R_diff, Err, Err_diff, it


def CCD_initialisation(x, N, M, K,
                       device='cpu', ftype=np.float32, th_zero=1e-10):
    """
    Initialise the variables needed to run the block CCD

    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)

    device: for PyTorch, specifies the device on which the code should be run
    ftype: specifies the float type to use for the computations
    """
    # stack x
    xh = comp_xh(x, N, M, K)

    # put tensors on the device
    Rh_d = np.zeros((N, N * M), dtype=ftype)

    # comp denumenators for CCD updates
    denum_R1 = compR1_denum(x, N, M, K)
    denum_Ri = compRi_denum(x, N, M, K, th_zero=th_zero)

    # comp indices for loop
    L_j = np.arange(N)
    return xh, x, xh, Rh_d, denum_R1, denum_Ri, L_j


###############################################################################
# ------------------------------- Find Lambda_1 ----------------------------- #
###############################################################################
def compA_lbd(x, N, M, K, eps_cvg, maxIt, L_lbd1, th_zero=1e-10,
              isSimu=False, A=[],
              doLog=False, device='cpu', ftype=np.float32):
    """
    Run a grid search on the LASSO coefficient Lambda_1

    output:
    A_L: List of the estimated adjacency matrix
    Err: List of the L2 estimation error and evolution difference
    F: List of the err and err^d error metrics used to select lambda_1
    err_A: For simulated environment only, list of error metrics
           err_A has metrics measuring the estimation error
           see fc_perf.perf_A() function for more details

    inputs:
    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)

    eps_cvg: convergence threshold
    maxIt: maximum number of iterations allowed
    lbd1: LASSO coefficient lambda_1

    isSimu: True if running a simulation for which we know
            the true adjancency matrix A
    A: the true adjacency matrix used for the simulation
    in this case we will compute the error metrics with the estimation

    doLog: if True the output or written on the log file
           instead of printed on the consol
    device: for PyTorch, specifies the device on which the code should be run
    ftype: specifies the float type to use for the computations
    """

    # ----------------------------------------------------------------------- #
    xh, x_d, xh_d, Rh_d, denum_R1, denum_Ri, L_j = CCD_initialisation(
        x, N, M, K, device=device, ftype=ftype, th_zero=th_zero)

    # ----------------------------------------------------------------------- #
    N_lbd1 = len(L_lbd1)
    A_L = []

    # sum of the prediction error
    # and difference of the error between two iterations
    Err = np.zeros((N_lbd1, 2))
    # err and err^d metrics used to select lambda_1
    F = np.zeros((N_lbd1, 2))

    AIC = np.zeros((N_lbd1, 1))
    BIC = np.zeros((N_lbd1, 1))

    # For simulations we compute the adjacency matrix estimation error
    # MSE, diff_nb_edges, p_correct_edg, p_missed_edg
    err_A = np.zeros((N_lbd1, 4))

    for l in np.arange(N_lbd1):
        lbd1 = L_lbd1[l]

        if doLog:
            logging.info('-' * 80)
            logging.info('l = {0} -- lambda = {1}'.format(l, lbd1))
            logging.info('-' * 80)
        else:
            print('-' * 80)
            print('l = {0} -- lambda = {1}'.format(l, lbd1))
            print('-' * 80)

        # init R at zero
        # for some reason it is more efficient to start from 0 at every lbd in simu
        # however on real dataset using the previous result creates better results
        if isSimu:
            Rh_d = Rh_d * 0.0
        else:
            Rh = np.zeros((N, N * M))

        # ------------------------------------------------------------------- #
        # perform optimisation with CCD looping on the columns
        if doLog:
            logging.info("--------------------- Start Optimisation for Ri & A --------------------")
        start_time = timeit.default_timer()
        A_o, Rh_d, R_diff, Err_R, Err_diff_R, it = CCD_Ri(
            x_d, xh_d, denum_Ri, denum_R1, Rh_d, N, M, K,
            eps_cvg, maxIt, lbd1,
            doLog=doLog, device=device, ftype=ftype)

        elapsed = timeit.default_timer() - start_time
        if doLog:
            logging.info("----> time for Ri = {0}".format(elapsed))
        else:
            print("----> time for Ri = {0}".format(elapsed))

        # error metrics
        Err[l, 0], Err[l, 1] = Err_R[-1], Err_diff_R[-1]
        F[l, 0], F[l, 1] = compF(A_o, x_d, N, M, K, L_j)

        # save results on cpu memory
        A_L.append(A_o)
        n_A0 = np.count_nonzero(A_o)

        if isSimu:
            err_A[l, 0], err_A[l, 1], err_A[l, 2], err_A[l, 3] = fc_perf.perf_A(A, A_o)

        if doLog:
            logging.info('% of sparsity = {0:.2f}'.format(n_A0 * 100.0 / N**2))
        else:
            print('% of sparsity = {0:.2f}'.format(n_A0 * 100.0 / N**2))

        # ------------------------------------------------------------------- #
        # Compute AIC criteria: AIC = 2k + n ln(RSS)
        Nn = (K - M) * N
        AIC[l] = 2 * n_A0 + Nn * np.log(Err[l, 0])
        # Compute BIC cireteria: BIC = k ln(n) + n ln(RSS / n)
        BIC[l] = n_A0 * np.log(Nn) + Nn * np.log(Err[l, 0] / Nn)
        # ------------------------------------------------------------------- #

    out = {'A_L': A_L,
           'Err': Err,
           'F': F,
           'err_A': err_A,
           'AIC': AIC, 'BIC': BIC}
    return out


def compCGP(x, N, M, K,
            eps_cvg, maxIt, lbd1,
            eps_cvg_c, maxIt_c, lbd1_c, lbd2_c,
            isSimu=False, A=[],
            doLog=False, device='cpu', ftype=np.float32):
    """
    Compute the CGP coefficients:
        -> the Adjancency matrix and the coefficients C
    the value of lambda_1 is assumed known

    output:
    A_o: The estimated adjacency matrix
    Err: The L2 estimation error and evolution difference
    F: The err and err^d error metrics used to select lambda_1
    err_A: For simulated environment, err_A has metrics measuring the estimation error
           see fc_perf.perf_A() function for more details
    C: The estimated CGP polynom coefficients C
    err_C: MSE obtained when fitting C

    inputs:
    N: number of time series
    M: number of time lag in CGP
    K: number of time points

    x: matrix of input time series with each line corresponding
       to a different time series; dim(N, K)

    eps_cvg: convergence threshold
    maxIt: maximum number of iterations allowed
    lbd1: LASSO coefficient lambda_1

    isSimu: True if running a simulation for which we know
            the true adjancency matrix A
    A: the true adjacency matrix used for the simulation
    in this case we will compute the error metrics with the estimation

    doLog: if True the output or written on the log file
           instead of printed on the consol
    device: for PyTorch, specifies the device on which the code should be run
    ftype: specifies the float type to use for the computations
    """

    if doLog:
        logging.info("----------------------- Start Computation of A ----------------------")
    else:
        print("----------------------- Start Computation of A ----------------------")

    out_A = compA(x, N, M, K, eps_cvg, maxIt, lbd1,
                  isSimu=isSimu, A=A,
                  doLog=doLog, device=device, ftype=ftype)
    A_o = out_A['A_o']

    if doLog:
        logging.info("----------------------- Start Computation of C ----------------------")
    else:
        print("----------------------- Start Computation of C ----------------------")

    start_time = timeit.default_timer()

    C, diff_C, err_C = CCD_C(A_o, x, N, M,
                             eps_cvg_c, maxIt_c, lbd1_c, lbd2_c)

    elapsed = timeit.default_timer() - start_time
    if doLog:
        logging.info("----> time for C = {0}".format(elapsed))
    else:
        print("----> time for C = {0}".format(elapsed))

    out = {'A_o': A_o,
           'Err': out_A['Err'],
           'F_A': out_A['F'],
           'err_A': out_A['err_A'],
           'C': C,
           'err_C': err_C}
    return out
