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
def find_SD(PF, Pi, sizez, sizek, Gamma_initial):
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
    SDtol = 1e-12
    SDdist = 7
    SDiter = 0
    SDmaxiter = 1000
    while SDdist > SDtol and SDmaxiter > SDiter:
        HGamma = np.zeros((sizez, sizek))
        for i in range(sizez):  # z
            for j in range(sizek):  # k
                for m in range(sizez):  # z'
                    HGamma[m, PF[i, j]] = \
                        HGamma[m, PF[i, j]] + Pi[i, m] * Gamma[i, j]
        SDdist = (np.absolute(HGamma - Gamma)).max()
        Gamma = HGamma
        SDiter += 1

    if SDiter < SDmaxiter:
        print('Stationary distribution converged after this many iterations: ',
              SDiter)
    else:
        print('Stationary distribution did not converge')

    # Check if state space is binding
    if Gamma.sum(axis=0)[-1] > 0.002:
        print('Stationary distribution is binding on k-grid.  Consider ' +
              'increasing the upper bound.')

    return Gamma


def Market_Clearing(w, args):
    (alpha_k, alpha_l, delta, psi, betafirm, kgrid, zgrid, Pi, eta0, eta1, sizek,
     sizez, h, tax_params, VF_initial, Gamma_initial) = args
    # print('VF_initial = ', VF_initial[:4, 50:60])
    op, e, l_d, y, eta = VFI.get_firmobjects(w, zgrid, kgrid, alpha_k, alpha_l,
                                             delta, psi, eta0, eta1,
                                             sizez, sizek, tax_params)
    VF, PF, optkgrid, optI = VFI.VFI(e, eta, betafirm, delta, kgrid, Pi, sizez, sizek,
                                 tax_params, VF_initial)
    Gamma = find_SD(PF, Pi, sizez, sizek, Gamma_initial)
    L_d = (Gamma * l_d).sum()
    Y = (Gamma * y).sum()
    I = (Gamma * optI).sum()
    Psi = (Gamma * VFI.adj_costs(optkgrid, kgrid, delta, psi)).sum()
    C = Y - I - Psi
    L_s = get_L_s(w, C, h)
    # print('Labor demand and supply = ', L_d, L_s)
    MCdist = np.absolute(L_d - L_s)

    return MCdist, VF, Gamma


def get_L_s(w, C, h):
    L_s = w / (h * C)

    return L_s


def golden_ratio_eqm(lb, ub, args, tolerance=1e-4):
    '''
    Use the golden section search method to find the GE
    '''
    (alpha_k, alpha_l, delta, psi, betafirm, kgrid, zgrid, Pi, eta0, eta1, sizek,
     sizez, h, tax_params, VF_initial, Gamma_initial) = args
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
            args = (alpha_k, alpha_l, delta, psi, betafirm, kgrid, zgrid, Pi,
                    eta0, eta1, sizek, sizez, h, tax_params, VF1, Gamma1)
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
            args = (alpha_k, alpha_l, delta, psi, betafirm, kgrid, zgrid, Pi,
                    eta0, eta1, sizek, sizez, h, tax_params, VF2, Gamma2)
            f2, VF2, Gamma2 = Market_Clearing(x2, args)

    # Use the mid-point of the final interval as the estimate of the optimzer
    # print('', '\n')
    # print('Final Lower Bound =', lb, '\n')
    # print('Final Upper Bound =', ub, '\n')
    est_min = (lb + ub)/2
    # print('Estimated Minimizer =', est_min, '\n')
    # print('MC_dist =', f2, '\n')

    return est_min
