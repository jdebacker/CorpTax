'''
------------------------------------------------------------------------
This module contains a function to compute moments from the SS of the
firm model.

* firm_moments()
------------------------------------------------------------------------
'''

import numpy as np
import numba

import VFI
import SS



@numba.jit
def find_equity_div(e, eta, PF_k, PF_b, sizez, sizek, sizeb):
    '''
    Determine equity issuance and dividend distributions
    '''
    div = np.empty((sizez, sizek, sizeb))
    equity = np.empty((sizez, sizek, sizeb))
    is_constrained = np.empty((sizez, sizek, sizeb))
    is_using_equity = np.empty((sizez, sizek, sizeb))
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for m in range(sizeb):  # loop over b
                div[i, j, m] = max(0, e[i, j, PF_k[i, j, m], m, PF_b[i, j, m]])
                equity[i, j, m] = max(0, -1 * e[i, j, PF_k[i, j, m], m, PF_b[i, j, m]])
                # can't finance investment larger than current one without using external funds
                is_constrained[i, j, m] = e[i, j, min(sizek - 1, PF_k[i, j, m] + 1), m, PF_b[i, j, m]] < 0
                # is issuing equity (making sure equity > 0 not just do to discerete grid)
                is_using_equity[i, j, m] = (equity[i, j, m] > (-e[i, j, PF_k[i, j, m], m, PF_b[i, j, m]]
                                     + e[i, j, max(0, PF_k[i, j, m] - 1), m, PF_b[i, j, m]]))

    return equity, div, is_constrained, is_using_equity


@numba.jit
def find_autocorr(op, optI, optB, PF_k, PF_b, VF, Gamma, kgrid, Pi, sizez, sizek, sizeb, mean_IK,
                  mean_EK, mean_BV, sd_IK, sd_EK, sd_BV):
    '''
    Compute autocovariances for endogenous variables
    '''
    cov_IK_IK = 0
    cov_EK_EK = 0
    cov_BV_BV = 0
    cov_EK_BV = 0
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for m in range(sizeb):  # loop over b
                for ii in range(sizez):  # loop over z'
                    cov_IK_IK = (cov_IK_IK + (((optI[i, j, m] / kgrid[j]) -
                                               mean_IK) *
                                              ((optI[ii, PF_k[i, j, m], PF_b[i, j, m]] /
                                                kgrid[PF_k[i, j, m]]) - mean_IK) *
                                              Gamma[i, j, m] * Pi[i, ii]))
                    cov_EK_EK = (cov_EK_EK + (((op[i, j] / kgrid[j]) -
                                               mean_EK) *
                                              (op[ii, PF_k[i, j, m]] /
                                               kgrid[PF_k[i, j, m]] - mean_EK) *
                                              Gamma[i, j, m] * Pi[i, ii]))
                    cov_BV_BV = (cov_BV_BV + (((optB[i, j, m] / (VF[i,j,m] - optB[i,j,m])) -
                                               mean_BV) *
                                              ((optB[ii, PF_k[i, j, m], PF_b[i, j, m]] /
                                                (VF[ii, PF_k[i, j, m], PF_b[i, j, m]] + optB[i, j, m])) - mean_BV) *
                                              Gamma[i, j, m] * Pi[i, ii]))
                    cov_EK_BV = (cov_EK_BV + (((op[i, j] / kgrid[j]) -
                                               mean_EK) *
                                              ((optB[ii, PF_k[i, j, m], PF_b[i, j, m]] /
                                                (VF[ii, PF_k[i, j, m], PF_b[i, j, m]] + optB[i, j, m])) - mean_BV) *
                                              Gamma[i, j, m] * Pi[i, ii]))

    ac_IK = cov_IK_IK / (sd_IK ** 2)
    ac_EK = cov_EK_EK / (sd_EK ** 2)
    if sd_BV != 0:
        ac_BV = cov_BV_BV / (sd_BV ** 2)
        sc_EK_BV = cov_EK_BV / (sd_BV * sd_EK)  # serial corr of leverage and lagged profits
    else:
        ac_BV = 0
        sc_EK_BV = 0

    return ac_IK, ac_EK, ac_BV, sc_EK_BV


def firm_moments(w, r, delta, psi, h, k_params, z_params, b_params, tax_params, output_vars, output_dir,
                 print_moments=False):
    '''
    ------------------------------------------------------------------------
    Compute moments
    ------------------------------------------------------------------------
    '''

    # unpack tuples
    kgrid, sizek, dens, kstar = k_params
    Pi, zgrid, sizez = z_params
    bgrid, sizeb = b_params
    tau_l, tau_i, tau_d, tau_g, tau_c, f_e, f_b = tax_params
    optK, optI, optB, op, e, l_d, y, eta, VF, PF_k, PF_b, Gamma = output_vars

    k3grid = np.tile(np.reshape(kgrid, (1, sizek, 1)), (sizez, 1, sizeb))
    k2grid = np.tile(np.reshape(kgrid, (1, sizek)), (sizez, 1))
    op3 = np.tile(np.reshape(op, (sizez, sizek, 1)), (1, 1, sizeb))
    # Aggregate Investment Rate
    agg_IK = (optI * Gamma).sum() / (k3grid * Gamma).sum()

    # Aggregate Dividends/Earnings
    equity, div, is_constrained, is_using_equity =\
        find_equity_div(e, eta, PF_k, PF_b, sizez, sizek, sizeb)
    agg_DE = ((div * Gamma).sum() / (np.tile(np.reshape(op, (sizez, sizek, 1)),
                                             (1, 1, sizeb)) * Gamma).sum())

    # Aggregate New Equity/Investment
    mean_SI = ((equity / optI) * Gamma).sum() / Gamma.sum()
    agg_SI = (equity * Gamma).sum() / (optI * Gamma).sum()

    # Aggregate leverage ratio
    agg_BV = (optB * Gamma).sum() / ((VF + optB) * Gamma).sum()

    # Volatility of the Investment Rate
    # this is determined as the standard deviation in the investment rate
    # across the steady state distribution of firms
    mean_IK = ((optI / k3grid) * Gamma).sum() / Gamma.sum()
    sd_IK = np.sqrt(((((optI / k3grid) - mean_IK) ** 2) * Gamma).sum())

    # Volatility of Earnings/Capital
    mean_EK = ((op / k2grid) * Gamma.sum(axis=2)).sum() / Gamma.sum()
    sd_EK = np.sqrt(((((op / k2grid) - mean_EK) ** 2) * Gamma.sum(axis=2)).sum())

    # Volatility of leverage ratio
    mean_BV = ((optB / (VF + optB)) * Gamma).sum() / Gamma.sum()
    sd_BV = np.sqrt(((((optB / (VF + optB)) - mean_BV) ** 2) * Gamma).sum())

    # Autocorrelation of the Investment Rate, Earnings/Capital ratio,
    # leverage ratio, serial correlation between lagged profits and leverage
    # Autocorrelation of Earnings/Capital
    ac_IK, ac_EK, ac_BV, sc_EK_BV =\
        find_autocorr(op, optI, optB, PF_k, PF_b, VF, Gamma, kgrid, Pi,
                      sizez, sizek, sizeb, mean_IK, mean_EK, mean_BV,
                      sd_IK, sd_EK, sd_BV)

    # compute covariances
    cov_BV_EK = (((optB / (VF + optB)) - mean_BV) * ((op3 / k3grid) - mean_EK) * Gamma).sum()
    cov_SI_EK = (((equity / optI) - mean_SI) * ((op3 / k3grid) - mean_EK) * Gamma).sum()
    cov_BV_IK = (((optB / (VF + optB)) - mean_BV) * ((optI / k3grid) - mean_IK) * Gamma).sum()
    cov_IK_SI = (((optI / k3grid) - mean_IK) * ((equity / optI) - mean_SI) * Gamma).sum()
    cov_IK_EK = (((optI / k3grid) - mean_IK) * ((op3 / k3grid) - mean_EK) * Gamma).sum()
    corr_BV_EK = cov_BV_EK / (sd_BV * sd_EK)

    # put these cross-sectional moments in a dictionary
    cross_section_dict = {'agg_IK': agg_IK, 'agg_DE': agg_DE, 'agg_SI': agg_SI,
                          'agg_BV': agg_BV, 'mean_IK': mean_IK, 'sd_IK': sd_IK,
                          'mean_EK': mean_EK, 'sd_EK': sd_EK,
                          'mean_BV': mean_BV, 'sd_BV': sd_BV, 'ac_IK': ac_IK,
                          'ac_EK': ac_EK, 'ac_BV': ac_BV, 'sc_EK_BV': sc_EK_BV,
                          'cov_BV_EK': cov_BV_EK, 'cov_SI_EK': cov_SI_EK,
                          'cov_BV_IK': cov_BV_IK, 'cov_IK_SI': cov_IK_SI,
                          'cov_IK_EK': cov_IK_EK, 'corr_BV_EK': corr_BV_EK}

    # Macro aggregates:
    agg_B = (optB * Gamma).sum()
    agg_E = (op3 * Gamma).sum()
    agg_I = (optI * Gamma).sum()
    agg_K = (k3grid * Gamma).sum()
    agg_Y = (Gamma.sum(axis=2) * y).sum()
    agg_D = (div * Gamma).sum()
    agg_S = (equity * Gamma).sum()
    agg_L_d = (Gamma.sum(axis=2) * l_d).sum()
    agg_Psi = (Gamma * VFI.adj_costs(optK, k3grid, delta, psi)).sum()
    agg_C = agg_Y - agg_I - agg_Psi
    agg_L_s = SS.get_L_s(w, agg_C, h, tau_l)
    mean_Q = (VF * Gamma).sum() / Gamma.sum()
    AvgQ = (VF * Gamma).sum() / (k3grid * Gamma).sum()
    agg_IIT = ((tau_d * agg_D) + (tau_l * agg_L_s) + (tau_i * r * agg_B)
               - (tau_g * agg_S))
    agg_CIT = tau_c * (agg_E - (r * f_b * agg_B) - ((1 - f_e) * delta
                                                    * agg_K) - f_e * agg_I)
    total_taxes = agg_CIT + agg_IIT
    # put these aggregate moments in a dictionary
    macro_dict = {'agg_B': agg_B, 'agg_E': agg_E, 'agg_I': agg_I,
                  'agg_K': agg_K, 'agg_Y': agg_Y, 'agg_D': agg_D,
                  'agg_S': agg_S, 'agg_L_d': agg_L_d, 'agg_Psi': agg_Psi,
                  'agg_C': agg_C, 'agg_L_s': agg_L_s, 'mean_Q': mean_Q,
                  'AvgQ': AvgQ, 'total_taxes': total_taxes,
                  'agg_CIT': agg_CIT, 'w': w, 'r': r}

    # Financing regimes
    frac_equity = (is_using_equity * Gamma).sum() / Gamma.sum()
    frac_div = ((1 - is_constrained) * Gamma).sum() / Gamma.sum()
    frac_constrained = ((is_constrained - is_using_equity) * Gamma).sum() / Gamma.sum()
    frac_debt = ((optB > 0) * Gamma).sum() / Gamma.sum()
    frac_neg_debt = ((optB < 0) * Gamma).sum() / Gamma.sum()
    frac_equity_debt = (((equity > 0) & (optB > 0)) * Gamma).sum() / Gamma.sum()
    frac_div_debt = (((div > 0) & (optB > 0)) * Gamma).sum() / Gamma.sum()
    frac_constrained_debt = (((equity == 0) & (div == 0) & (optB > 0)) * Gamma).sum() / Gamma.sum()

    # moments by regime
    share_K_equity = ((is_using_equity * k3grid) * Gamma).sum() / agg_K
    share_K_div = (((1 - is_constrained) * k3grid) * Gamma).sum() / agg_K
    share_K_constrained = (((is_constrained - is_using_equity) * k3grid) * Gamma).sum() / agg_K
    share_I_equity = ((is_using_equity * optI) * Gamma).sum() / agg_I
    share_I_div = (((1 - is_constrained) * optI) * Gamma).sum() / agg_I
    share_I_constrained = (((is_constrained - is_using_equity) * optI) * Gamma).sum() / agg_I
    if agg_B != 0:
        share_B_equity = ((is_using_equity * optB) * Gamma).sum() / agg_B
        share_B_div = (((1 - is_constrained) * optB) * Gamma).sum() / agg_B
        share_B_constrained = (((is_constrained - is_using_equity) * optB) * Gamma).sum() / agg_B
    else:
        share_B_equity = 0
        share_B_div = 0
        share_B_constrained = 0
    IK_equity = ((is_using_equity * optI) * Gamma).sum() / ((is_using_equity * k3grid) * Gamma).sum()
    IK_div = (((1 - is_constrained) * optI) * Gamma).sum() / (((1 - is_constrained) * k3grid) * Gamma).sum()
    IK_constrained = (((is_constrained - is_using_equity) * optI) * Gamma).sum() / (((is_constrained - is_using_equity) * k3grid) * Gamma).sum()
    BV_equity = ((is_using_equity * optB) * Gamma).sum() / ((is_using_equity * (VF + optB)) * Gamma).sum()
    BV_div = (((1 - is_constrained) * optB) * Gamma).sum() / (((1 - is_constrained) * (VF + optB)) * Gamma).sum()
    BV_constrained = (((is_constrained - is_using_equity) * optB) * Gamma).sum() / (((is_constrained - is_using_equity) * (VF + optB)) * Gamma).sum()
    EK_equity = ((is_using_equity * op3) * Gamma).sum() / ((is_using_equity * k3grid) * Gamma).sum()
    EK_div = (((1 - is_constrained) * op3) * Gamma).sum() / (((1 - is_constrained) * k3grid) * Gamma).sum()
    EK_constrained = (((is_constrained - is_using_equity) * op3) * Gamma).sum() / (((is_constrained - is_using_equity) * k3grid) * Gamma).sum()
    AvgQ_equity = ((is_using_equity * VF) * Gamma).sum() / ((is_using_equity * k3grid) * Gamma).sum()
    AvgQ_div = (((1 - is_constrained) * VF) * Gamma).sum() / (((1 - is_constrained) * k3grid) * Gamma).sum()
    AvgQ_constrained = (((is_constrained - is_using_equity) * VF) * Gamma).sum() / (((is_constrained - is_using_equity) * k3grid) * Gamma).sum()
    # put these moments by regime type in a dictionary
    regimes_dict = {'frac_equity': frac_equity, 'frac_div': frac_div, 'frac_constrained': frac_constrained,
                    'frac_debt': frac_debt, 'frac_neg_debt': frac_neg_debt, 'frac_equity_debt': frac_equity_debt,
                    'frac_div_debt': frac_div_debt, 'frac_constrained_debt': frac_constrained_debt,
                    'share_K_equity': share_K_equity, 'share_K_div': share_K_div,
                    'share_K_constrained': share_K_constrained,
                    'share_I_equity': share_I_equity, 'share_I_div': share_I_div,
                    'share_I_constrained': share_I_constrained,
                    'share_B_equity': share_B_equity, 'share_B_div': share_B_div,
                    'share_B_constrained': share_B_constrained, 'IK_equity': IK_equity,
                    'IK_div': IK_div, 'IK_constrained': IK_constrained, 'EK_equity': EK_equity,
                    'EK_div': EK_div, 'EK_constrained': EK_constrained, 'BV_equity': BV_equity,
                    'BV_div': BV_div, 'BV_constrained': BV_constrained,
                    'AvgQ_equity': AvgQ_equity, 'AvgQ_div': AvgQ_div,
                    'AvgQ_constrained': AvgQ_constrained}

    if print_moments:
        print('The aggregate investment rate = ', agg_IK)
        print('The aggregate ratio of dividends to earnings = ', agg_DE)
        print('The aggregate ratio of equity to new investment = ', agg_SI)
        print('The volatility in the investment rate = ', sd_IK)
        print('The autocorrelation in the investment rate = ', ac_IK)
        print('The volatility of the earnings/capital ratio = ', sd_EK)
        print('The autocorrelation in the earnings/capital ratio = ', ac_EK)
        # print('The fraction of firms issuing equity is: ', frac_equity)
        # print('The fraction of firms who are financially constrained is: ', frac_constrained)
        # print('The fraction of firms distributing dividends is: ', frac_div)
        # print('Share of capital for equity regime: ', share_K_equity)
        # print('Share of capital for constrained regime: ', share_K_constrained)
        # print('Share of capital for dividend regime: ', share_K_div)
        # print('Share of investment for equity regime: ', share_I_equity)
        # print('Share of investment for constrained regime: ', share_I_constrained)
        # print('Share of investment for dividend regime: ', share_I_div)
        # print('Mean E/K for equity regime: ', EK_equity)
        # print('Mean E/K for constrained regime: ', EK_constrained)
        # print('Mean E/K for dividend regime: ', EK_div)
        # print('Mean I/K for equity regime: ', IK_equity)
        # print('Mean I/K for constrained regime: ', IK_constrained)
        # print('Mean I/K for dividend regime: ', IK_div)
        # print('Avg Q for equity regime: ', AvgQ_equity)
        # print('Avg Q for constrained regime: ', AvgQ_constrained)
        # print('Avg Q for dividend regime: ', AvgQ_div)
        print('The aggregate leverage ratio =  ', agg_BV)
        print('The fraction with positive debt = ', frac_debt)
        print('The fraction with negative debt = ', frac_neg_debt)

    model_moments = {'cross_section': cross_section_dict,
                     'macro': macro_dict, 'regimes': regimes_dict}

    return model_moments
