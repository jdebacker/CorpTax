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
def find_SD(PF, Pi, sizez, sizek):
    '''
    ------------------------------------------------------------------------
    Compute the stationary distribution of firms over (k, z)
    ------------------------------------------------------------------------
    SDtol     = tolerance required for convergence of SD
    SDdist    = distance between last two distributions
    SDiter    = current iteration
    SDmaxiter = maximium iterations allowed to find stationary distribution
    Gamma     = stationary distribution
    HGamma    = operated on stationary distribution
    ------------------------------------------------------------------------
    '''
    Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
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


def GE_loop(w, *args):
    (alpha_k, alpha_l, delta, psi, betafirm, K, z, Pi, sizek, sizez, h,
     tax_params) = args
    op, e, l_d, y = VFI.get_firmobjects(w, z, K, alpha_k, alpha_l,
                                        delta, psi, sizez, sizek,
                                        tax_params)
    VF, PF, optK, optI = VFI.VFI(e, betafirm, delta, K, Pi, sizez, sizek,
                                 tax_params)
    Gamma = find_SD(PF, Pi, sizez, sizek)
    L_d = (Gamma * l_d).sum()
    Y = (Gamma * y).sum()
    I = (Gamma * optI).sum()
    Psi = (Gamma * VFI.adj_costs(optK, K, delta, psi)).sum()
    C = Y - I - Psi
    L_s = get_L_s(w, C, h)
    print('Labor demand and supply = ', L_d, L_s)
    MCdist = L_d - L_s

    return MCdist


def get_L_s(w, C, h):
    L_s = w / (h * C)

    return L_s
