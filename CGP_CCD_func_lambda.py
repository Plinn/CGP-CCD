#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:26:29 2019

@author: theophile

functions to perform the CGP-CCD computation
"""
import timeit
import numpy as np
import CGP_CCD_funcPerf as fc_perf

###############################################################################
# ------------------------ functions to select lambda_1 ---------------------- #
###############################################################################
def printErrLbd(ti, L_lbd1, A_L, N, MSE_A, MSE_X, idx_s, err_A, isSimu):
    fc_perf.idxPrintMax(ti, idx_s, L_lbd1, A_L, N)
    fc_perf.print_mse_idx(MSE_A, MSE_X, idx_s, isSimu)
    if isSimu:
        fc_perf.print_err_idxj(err_A, L_lbd1, idx_s)


def lbd_findMinErr(L_lbd1, x, A_L, N, MSE_A, MSE_X, isSimu, err_A):
    idx_min = np.where(x == x.min())[0]

    # if there are more than one minimum we take the average
    if len(idx_min) > 1:
        idx_min = np.mean(idx_min)

    idx_min = np.int(idx_min)

    # select lambda
    lbd_min = L_lbd1[idx_min]

    # print perf results
    print('-' * 80)
    printErrLbd('Selected lambda', L_lbd1, A_L, N, MSE_A, MSE_X,
                idx_min, err_A, isSimu)
    print('-' * 80)

    return lbd_min, idx_min


def selectLambda(F_err, F_errd, L_lbd1, A_L, N, MSE_A=[], MSE_X=[],
                 isSimu=False, err_A=[]):
    """
    Selects the LASSO coefficient lambda_1 with err and err^d
    from a grid search on the values of lambda_1

    F_err: list of err
    F_errd: list of err^d
    L_lbd1: list of the lambda_1 used for the grid search
    A_L: list of the obtained adjacency matrix
    N: number of time series

    isSimu: True if doing simulations where True A is known
    err_A: in case of simulation, matrix of estimations errors
    """
    N_lbd1 = len(L_lbd1)

    # check if err peaked
    print('-' * 80)
    idx_err = np.argsort(F_err)[-1]
    printErrLbd('Err', L_lbd1, A_L, N, MSE_A, MSE_X, idx_err, err_A, isSimu)
    err_peaked = (idx_err != 0) & (idx_err != (N_lbd1 - 1))

    # check if err^d peaked
    print('-' * 80)
    idx_errd = np.argsort(F_errd)[-1]
    printErrLbd('Err^d', L_lbd1, A_L, N, MSE_A, MSE_X, idx_errd, err_A, isSimu)
    errd_peaked = (idx_errd != 0) & (idx_errd != (N_lbd1 - 1))

    # compute mean
    print('-' * 80)
    if err_peaked & errd_peaked:
        print("Selected the mean")
        idx_s = np.int((idx_err + idx_errd) / 2.0)
    elif err_peaked:
        print('Only err peaked')
        idx_s = idx_err
    elif errd_peaked:
        print('Only err^d peaked')
        idx_s = idx_errd
    else:
        print("No peaks, take the mean")
        idx_s = np.int((idx_err + idx_errd) / 2.0)

    print('-' * 80)
    printErrLbd('Selected lambda', L_lbd1, A_L, N, MSE_A, MSE_X,
                idx_s, err_A, isSimu)
    print('-' * 80)

    return L_lbd1[idx_s], idx_s


def selectLambda_errBic(L_lbd1, F_err, F_errd, f_bic, A_L, N, MSE_X, MSE_A=[],
                        isSimu=False, err_A=[]):
    """
    Selects the LASSO coefficient lambda_1
    with err and err^d and BIC
    from a grid search on the values of lambda_1

    F_err: list of err
    F_errd: list of err^d
    BIC: BIC criteria value
    L_lbd1: list of the lambda_1 used for the grid search
    A_L: list of the obtained adjacency matrix
    N: number of time series

    isSimu: True if doing simulations where True A is known
    err_A: in case of simulation, matrix of estimations errors
    """
    N_lbd1 = len(L_lbd1)

    print('-' * 80)
    print('-' * 20 + '> With err, err^d and BIC combined <' + '-' * 20)
    print('-' * 80)

    # check if err peaked
    print('-' * 80)
    idx_err = np.argsort(F_err)[-1]
    printErrLbd('Err', L_lbd1, A_L, N, MSE_A, MSE_X,
                idx_err, err_A, isSimu)
    err_peaked = (idx_err != 0) & (idx_err != (N_lbd1 - 1))

    # check if err^d peaked
    print('-' * 80)
    idx_errd = np.argsort(F_errd)[-1]
    printErrLbd('Err^d', L_lbd1, A_L, N, MSE_A, MSE_X,
                idx_errd, err_A, isSimu)
    errd_peaked = (idx_errd != 0) & (idx_errd != (N_lbd1 - 1))

    # for BIC
    print('-' * 80)
    print('-' * 20 + '> BIC')
    lbd_bic, idx_bic = lbd_findMinErr(L_lbd1, f_bic, A_L,
                                      N, MSE_A, MSE_X, isSimu, err_A)
    bic_peaked = (idx_bic != 0) & (idx_bic != (N_lbd1 - 1))

    # compute mean
    print('-' * 80)
    if bic_peaked:
        if err_peaked & errd_peaked:
            print("Selected the mean")
            idx_s = np.int((idx_err + idx_errd + idx_bic) / 3.0)
        elif err_peaked:
            print('Only err and BIC peaked')
            idx_s = np.int((idx_err + idx_bic) / 2.0)
        elif errd_peaked:
            print('Only err^d and BIC peaked')
            idx_s = np.int((idx_errd + idx_bic) / 2.0)
        else:
            print("Only BIC peaked")
            idx_s = np.int(idx_bic)
    else:
        if err_peaked & errd_peaked:
            print("Only err and err^d peaked")
            idx_s = np.int((idx_err + idx_errd) / 2.0)
        elif err_peaked:
            print('Only err peaked')
            idx_s = np.int(idx_err)
        elif errd_peaked:
            print('Only err^d')
            idx_s = np.int(idx_errd)
        else:
            print("No peaks, take the mean")
            idx_s = np.int((idx_err + idx_errd + idx_bic) / 3.0)

    print('-' * 80)
    printErrLbd('Selected lambda', L_lbd1, A_L, N, MSE_A, MSE_X,
                idx_s, err_A, isSimu)
    print('-' * 80)
    return L_lbd1[idx_s], idx_s
