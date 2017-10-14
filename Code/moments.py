'''
------------------------------------------------------------------------
This module contains a function to compute moments from the SS of the
firm model.

* firm_moments()
------------------------------------------------------------------------
'''

import numpy as np
import numba

def firm_moments(delta, k_params, z_params, output_vars, output_dir,
                 print_moments=False):
    '''
    ------------------------------------------------------------------------
    Compute moments
    ------------------------------------------------------------------------
    '''

    # unpack tuples
    kgrid, sizek, dens, kstar = k_params
    Pi, zgrid, sizez = z_params
    optK, optI, op, e, eta, VF, PF, Gamma = output_vars

    kgrid_2d = np.tile(kgrid.reshape(1, sizek), (sizez, 1))
    # Aggregate Investment Rate
    agg_IK = (optI * Gamma).sum() / (kgrid_2d * Gamma).sum()

    # Aggregate Dividends/Earnings
    equity, div = find_equity_div(e, eta, PF, sizez, sizek)
    agg_DE = (div * Gamma).sum() / (op * Gamma).sum()

    # Aggregate New Equity/Investment
    # NOTE: with Gourio-Miao (AEJ: Macro, 2010) calibration this ratio
    # is 60% higher than what they report.  Already using gross
    # investment here, so not sure why the diff - other moments much closer
    agg_SI = (equity * Gamma).sum() / (optI * Gamma).sum()

    # Volatility of the Investment Rate
    # this is determined as the standard deviation in the investment rate
    # across the steady state distribution of firms
    mean_IK = ((optI / kgrid_2d) * Gamma).sum() / Gamma.sum()
    sd_IK = np.sqrt(((((optI / kgrid_2d) - mean_IK) ** 2) * Gamma).sum())

    # Volatility of Earnings/Capital
    mean_EK = ((op / kgrid_2d) * Gamma).sum() / Gamma.sum()
    sd_EK = np.sqrt(((((op / kgrid_2d) - mean_EK) ** 2) * Gamma).sum())

    # Autocorrelation of the Investment Rate
    # Autocorrelation of Earnings/Capital
    ac_IK, ac_EK = find_autocorr(op, optI, PF, Gamma, kgrid, Pi, sizez,
                                 sizek, mean_IK, mean_EK, sd_IK, sd_EK)

    if print_moments:
        print('The aggregate investment rate = ', agg_IK)
        print('The aggregate ratio of dividends to earnings = ', agg_DE)
        print('The aggregate ratio of equity to new investment = ', agg_SI)
        print('The volatility in the investment rate = ', sd_IK)
        print('The autocorrelation in the investment rate = ', ac_IK)
        print('The volatility of the earnings/capital ratio = ', sd_EK)
        print('The autocorrelation in the earnings/capital ratio = ', ac_EK)

    return agg_IK, agg_DE, agg_SI, sd_IK, ac_IK, sd_EK, ac_IK


@numba.jit
def find_equity_div(e, eta, PF, sizez, sizek):
    '''
    Determine equity issuance and dividend distributions
    '''
    div = np.empty((sizez, sizek))
    equity = np.empty((sizez, sizek))
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            div[i, j] = max(0, e[i, j, PF[i, j]])
            equity[i, j] = max(0, -1 * e[i, j, PF[i, j]])

    return equity, div


@numba.jit
def find_autocorr(op, optI, PF, Gamma, kgrid, Pi, sizez, sizek, mean_IK,
                  mean_EK, sd_IK, sd_EK):
    '''
    Compute autocovariances for endogenous variables
    '''
    cov_IK_IK = 0
    cov_EK_EK = 0
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for m in range(sizez):  # loop over z'
                cov_IK_IK = (cov_IK_IK + (((optI[i, j] / kgrid[j]) -
                                           mean_IK) *
                                          ((optI[m, PF[i, j]] /
                                           kgrid[PF[i, j]]) - mean_IK) *
                                          Gamma[i, j] * Pi[i, m]))
                cov_EK_EK = (cov_EK_EK + ((op[i, j] / kgrid[j] -
                                           mean_EK) *
                                          (op[m, PF[i, j]] /
                                           kgrid[PF[i, j]] - mean_EK) *
                                          Gamma[i, j] * Pi[i, m]))

    ac_IK = cov_IK_IK / (sd_IK ** 2)
    ac_EK = cov_EK_EK / (sd_EK ** 2)

    return ac_IK, ac_EK
