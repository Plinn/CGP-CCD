#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:48:29 2019

@author: theophile
"""
import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# --------------------- functions to visualise results ---------------------- #
###############################################################################
def idxPrintMax(name, idx, L_lbd1, A_L, N):
    """
    Function to print the selected lamnda
    and the sparsity of the corresponding andjacency matrix

    name: name of which lambda_1 it corresponds to
    idx: selected index
    L_lbd1: array of lambda_1 values used in grid search
    A_L: list of the obtained adjacency matrix
    N: number of time series
    """
    print(name + " --> lbd1 = {0}".format(L_lbd1[idx]))
    print('% of sparsity = {0:.2f}'.format(
        np.count_nonzero(A_L[idx]) * 100.0 / N ** 2))


def print_err_idxj(err, L_lbd, idx_j):
    """
    Function to print performance metrics for a selected lambda
    when in a simulated environmnent with True adjacency matrix known

    err: matrix of metrics measuring approximation erro between A_o and A
    idx_j: selected index
    L_lbd: array of lambda_1 values used in grid search
    """
    print("Best at lbd1 = {3:.2f} -- diff_nb_edges = {0} -- p_correct_edg = {1:.2f} -- p_missed_edg = {2:.2f}".format(
        err[idx_j, 0], err[idx_j, 1], err[idx_j, 2], L_lbd[idx_j]))


def print_mse_idx(MSE_A, MSE_X, idx, isSimu):
    """
    Function to print the MSE of the generated x 
    and for simulations, MSE of the estimated A
    """
    if isSimu:
        print("MSE(A) = {0:.2e} -- MSE(x) = {1:.2e}".format(
            MSE_A[idx], MSE_X[idx]))
    else:
        print("MSE(x) = {0:.2e}".format(MSE_X[idx]))


def rescale_F(f):
    """
    Rescale f to [0, 1]
    """
    return (f - f.min()) / (f.max() - f.min())


def plotCompMetrics_Err(L_lbd1, diff_e, f_err, f_errd, N,
                        isSimu=False, p_title=''):
    """
    compare True and estimated adjacency matrix
    plot all the err metrics on the same graph
    right axis shows the error metrics err and err^d
    if on simulations:
        - left axis shows the number of different edges as %
    else:
        - left axis shows the sparsity level

    L_lbd1: grid used for lambda_1
    diff_e: vector of number of different edges
    f_err: vector of err metric
    f_errd: vector of err^d metric
    N: number of time series
    """
    err = rescale_F(f_err)
    err_d = rescale_F(f_errd)

    f, ax = plt.subplots()
    ax.set_title(p_title)
    if isSimu:
        ax.plot(L_lbd1, diff_e * 100.0 / N**2, label='Edge diff')
        ax.plot(L_lbd1, np.zeros(len(L_lbd1)), color='r')
        ax.set_ylabel('Difference in number of edges over $N^2$ in %')
    else:
        ax.plot(L_lbd1, diff_e, label='% sparsity')
        ax.set_ylabel('% sparsity level')
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$\lambda_1$')
    ax.grid()
    ax2 = ax.twinx()
    ax2.plot(L_lbd1, err, color='C2', linestyle='-.', label=r'$ERR$')
    ax2.plot(L_lbd1, err_d, color='C3', linestyle='-.', label=r'$ERR^d$')
    ax2.legend(loc='lower right')
    ax2.set_ylabel('error')
    f.tight_layout()
    f.show()


def plotCompMetrics_Full(L_lbd1, diff_e, f_err, f_errd,
                         f_aic, f_bic,
                         N, isSimu=False, p_title=''):
    """
    compare True and estimated adjacency matrix
    plot all the err metrics on the same graph
    right axis shows the following rescaled error metrics:
        - err
        - err^d
        - MSE-in
        - MSE-out
        - AIC
        - BIC

    if on simulations:
        - left axis shows the number of different edges as %
    else:
        - left axis shows the sparsity level

    L_lbd1: grid used for lambda_1
    diff_e: vector of number of different edges
    f_err: vector of err metric
    f_errd: vector of err^d metric
    N: number of time series
    """
    err = rescale_F(f_err)
    err_d = rescale_F(f_errd)
    mse_in = rescale_F(f_mseIn)
    mse_out = rescale_F(f_mseOut)
    aic_r = rescale_F(f_aic)
    bic_r = rescale_F(f_bic)

    f, ax = plt.subplots()
    ax.set_title(p_title)
    if isSimu:
        ax.plot(L_lbd1, diff_e * 100.0 / N**2, label='Edge diff')
        ax.plot(L_lbd1, np.zeros(len(L_lbd1)), color='r')
        ax.set_ylabel('Difference in number of edges over $N^2$ in %')
    else:
        ax.plot(L_lbd1, diff_e, label='% sparsity')
        ax.set_ylabel('% sparsity level')
    ax.legend(loc='upper left')
    ax.set_xlabel(r'$\lambda_1$')
    ax.grid()
    ax2 = ax.twinx()
    ax2.plot(L_lbd1, err, color='C2', linestyle='-.', label=r'$ERR$')
    ax2.plot(L_lbd1, err_d, color='C3', linestyle='-.', label=r'$ERR^d$')
    ax2.plot(L_lbd1, aic_r, color='C6', linestyle=':', label="AIC")
    ax2.plot(L_lbd1, bic_r, color='C7', linestyle=':', label="BIC")
    ax2.legend(loc='lower right')
    ax2.set_ylabel('error')
    f.tight_layout()
    f.show()


def perf_A(A, A_o):
    """
    compute different metrics to assess the difference between A and A_o
    """
    A_1 = (A != 0) * 1
    A_o_1 = (A_o != 0) * 1
    A_diff_1 = A_1 - A_o_1
    nb_edge_A = np.count_nonzero(A_1)
    nb_edge_A_o = np.count_nonzero(A_o_1)
    diff_nb_edges = nb_edge_A - nb_edge_A_o

    # nb of correct edges
    if nb_edge_A > 0:
        p_correct_edg = (nb_edge_A - np.count_nonzero(A_diff_1 > 0)) *\
            100.0 / nb_edge_A
    else:
        p_correct_edg = -1.0

    # nb of wrong edges
    if nb_edge_A_o > 0:
        p_missed_edg = np.count_nonzero(A_diff_1 < 0) * 100.0 / nb_edge_A_o
    else:
        p_missed_edg = -1.0
    mse_A = np.mean((A - A_o)**2)

    return diff_nb_edges, p_correct_edg, p_missed_edg, mse_A


def plotMat_compdiff(matrix_1, t1, matrix_2, t2):
    """
    Plot matrix_1 and matrix_2 And
    the matrix of the difference: abs(matrix_1 - matrix_2)
    on the same subplot

    matrix_1: matrix to plot
    t1: title of matrix_1
    matrix_2: matrix to plot and compare to matrix_1
    t2: title of matrix_2
    """
    plt.figure()

    vmin = np.minimum(matrix_1.min(), matrix_2.min())
    vmax = np.maximum(matrix_1.max(), matrix_2.max())

    # plot two matrices on one subplot
    ax = plt.subplot(1, 3, 1)
    ax.imshow(matrix_1, interpolation='none', aspect='auto',
              cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(t1)
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(matrix_2, interpolation='none', aspect='auto',
               cmap='gray', vmin=vmin, vmax=vmax)
    ax2.set_title(t2)
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(-1.0 * np.abs(matrix_1 - matrix_2),
               interpolation='none', aspect='auto', cmap='gray',
               vmin=vmin, vmax=vmax)
    ax3.set_title('Difference')

