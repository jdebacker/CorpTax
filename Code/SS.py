'''
------------------------------------------------------------------------
This module contains functions used in the steady state solution to the
firm problem.  These functions include:

* find_SD()
* GE_loop()
* get_L_s()
------------------------------------------------------------------------
'''

# imports
import numba
import numpy as np
import VFI


@numba.jit
def find_SD(PF_k, PF_b, Pi, sizez, sizek, sizeb, Gamma_initial):
    '''
    ------------------------------------------------------------------------
    Compute the stationary distribution of firms over (z, k)
    ------------------------------------------------------------------------
    SDtol     = tolerance required for convergence of SD
    SDdist    = distance between last two distributions
    SDiter    = current iteration
    SDmaxiter = maximium iterations allowed to find stationary distribution
    Gamma     = stationary distribution
    HGamma    = operated on stationary distribution
    ------------------------------------------------------------------------
    '''
    Gamma = Gamma_initial
    SDtol = 1e-8#1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 2000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = np.zeros((sizez, sizek, sizeb))
        for i in range(sizez):  # z
            for j in range(sizek):  # k
                for m in range(sizeb):  # b
                    for ii in range(sizez):  # z'
                        HGamma[ii, PF_k[i, j, m], PF_b[i, j, m]] = \
                            (HGamma[ii, PF_k[i, j, m], PF_b[i, j, m]] +
                             Pi[i, ii] * Gamma[i, j, m])
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1

    # if SDiter < SDmaxiter:
    #     print('Stationary distribution converged after this many iterations: ',
    #           SDiter)
    # else:
    #     print('Stationary distribution did not converge')
    #
    # # Check if state space is binding
    # if (Gamma.sum(axis=0)).sum(axis=1)[-1] > 0.002:
    #     print('Stationary distribution is binding on k-grid.  Consider ' +
    #           'increasing the upper bound.')
    # if (Gamma.sum(axis=0)).sum(axis=0)[-1] > 0.01:
    #     print('Stationary distribution is binding on b-grid.  Consider ' +
    #           'increasing the upper bound.')
    # if (Gamma.sum(axis=0)).sum(axis=0)[0] > 0.01:
    #     print('Stationary distribution is binding on b-grid.  Consider ' +
    #           'decreasing the lower bound.')

    return Gamma


def Market_Clearing(w, args):
    (r, alpha_k, alpha_l, delta, psi, fixed_cost, betafirm, kgrid, zgrid,
     bgrid, Pi, eta0, eta1, eta2, s, sizek, sizez, sizeb, h, tax_params,
     VF_initial, Gamma_initial) = args
    # print('VF_initial = ', VF_initial[:4, 50:60])
    tau_l, tau_i, tau_d, tau_g, tau_c, f_e, f_b = tax_params
    op, e, l_d, y, eta, collateral_constraint =\
        VFI.get_firmobjects(r, w, zgrid, kgrid, bgrid, alpha_k, alpha_l,
                            delta, psi, fixed_cost, eta0, eta1, eta2, s,
                            sizez, sizek, sizeb, tax_params)
    VF, PF_k, PF_b, optK, optI, optB =\
        VFI.VFI(e, eta, collateral_constraint, betafirm, delta, kgrid,
                bgrid, Pi, sizez, sizek, sizeb, tax_params, VF_initial)
    Gamma = find_SD(PF_k, PF_b, Pi, sizez, sizek, sizeb, Gamma_initial)
    L_d = (Gamma.sum(axis=2) * l_d).sum()
    Y = (Gamma.sum(axis=2) * y).sum()
    I = (Gamma * optI).sum()
    k3grid = np.tile(np.reshape(kgrid, (1, sizek, 1)), (sizez, 1, sizeb))
    Psi = (Gamma * VFI.adj_costs(optK, k3grid, delta, psi, fixed_cost)).sum()
    C = Y - I - Psi
    # note that financial frictions not here- they aren't real costs,
    # rather they are costs paid by firms and recieved for financial
    # interemediaries and flow to households as income
    L_s = get_L_s(w, C, h, tau_l)
    # print('Labor demand and supply = ', L_d, L_s)
    MCdist = np.absolute(L_d - L_s)

    return MCdist, VF, Gamma


def get_L_s(w, C, h, tau_l):
    L_s = ((1 - tau_l) * w) / (h * C)

    return L_s


def golden_ratio_eqm(lb, ub, args, tolerance=1e-4):
    '''
    Use the golden section search method to find the GE
    '''
    (r, alpha_k, alpha_l, delta, psi, fixed_cost, betafirm, kgrid, zgrid,
     bgrid, Pi, eta0, eta1, eta2, s, sizek, sizez, sizeb, h, tax_params,
     VF_initial, Gamma_initial) = args
    golden_ratio = 2 / (np.sqrt(5) + 1)

    # Use the golden ratio to set the initial test points
    x1 = ub - golden_ratio * (ub - lb)
    x2 = lb + golden_ratio * (ub - lb)

    #  Evaluate the function at the test points
    f1, VF1, Gamma1 = Market_Clearing(x1, args)
    f2, VF2, Gamma2 = Market_Clearing(x2, args)

    iteration = 0
    while (np.absolute(ub - lb) > tolerance):
        iteration += 1
        # print('Iteration #', iteration)
        # print('f1 =', f1)
        # print('f2 =', f2)

        if (f2 > f1):
            # then the minimum is to the left of x2
            # let x2 be the new upper bound
            # let x1 be the new upper test point
            # print('f2 > f1')
            # Set the new upper bound
            ub = x2
            # print('New Upper Bound =', ub)
            # print('New Lower Bound =', lb)
            # Set the new upper test point
            # Use the special result of the golden ratio
            x2 = x1
            # print('New Upper Test Point = ', x2)
            f2 = f1

            # Set the new lower test point
            x1 = ub - golden_ratio * (ub - lb)
            # print('New Lower Test Point = ', x1)
            args = (r, alpha_k, alpha_l, delta, psi, fixed_cost,
                    betafirm, kgrid, zgrid, bgrid, Pi, eta0, eta1, eta2,
                    s, sizek, sizez, sizeb, h,  tax_params, VF1, Gamma1)
            f1, VF1, Gamma1 = Market_Clearing(x1, args)
        else:
            # print('f2 < f1')
            # the minimum is to the right of x1
            # let x1 be the new lower bound
            # let x2 be the new lower test point

            # Set the new lower bound
            lb = x1
            # print('New Upper Bound =', ub)
            # print('New Lower Bound =', lb)

            # Set the new lower test point
            x1 = x2
            # print('New Lower Test Point = ', x1)

            f1 = f2

            # Set the new upper test point
            x2 = lb + golden_ratio * (ub - lb)
            # print('New Upper Test Point = ', x2)
            args = (r, alpha_k, alpha_l, delta, psi, fixed_cost, betafirm,
                    kgrid, zgrid, bgrid, Pi, eta0, eta1, eta2, s, sizek,
                    sizez, sizeb, h, tax_params, VF2, Gamma2)
            f2, VF2, Gamma2 = Market_Clearing(x2, args)

    # Use the mid-point of the final interval as the estimate of the optimzer
    # print('', '\n')
    # print('Final Lower Bound =', lb, '\n')
    # print('Final Upper Bound =', ub, '\n')
    est_min = (lb + ub)/2
    # print('Estimated Minimizer =', est_min, '\n')
    # print('MC_dist =', f2, '\n')

    return est_min
