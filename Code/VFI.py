'''
------------------------------------------------------------------------
This module contains functions used to solve and indiviudal firm's
problem via value function iteration.

* create_Vmat()
* adj_costs()
* get_firmobjects()
* VFI()
------------------------------------------------------------------------
'''

# imports
import numba
import numpy as np
import time


@numba.jit
def create_EV(Pi, V, sizez, sizek, sizeb):
    '''
    Compute expectation of continuation value function.

    Args:
        Pi: 2D array, transition probabilities for exogenous state var
        V: 3D array, value function

    Returns:
        EV: 3D array, expected value function (conditional on z)
    '''
    EV = np.zeros_like(V)
    # start = time.time()
    for i in range(sizez):  # loop over z
        for jj in range(sizek):  # loop over k'
            for mm in range(sizeb):  # loop over b'
                for ii in range(sizez):  # loop over z'
                    EV[i, jj, mm] = EV[i, jj, mm] + Pi[i, ii] * V[ii, jj, mm]
    # end = time.time()
    # print('EV loop takes ', end-start, ' seconds to complete.')

    return EV



@numba.jit
def create_Vmat(EV, e, eta, betafirm, Pi, sizez, sizek, sizeb, tax_params):
    '''
    ------------------------------------------------------------------------
    This function loops over the state and control variables, operating on the
    value function to update with the last iteration's value function
    ------------------------------------------------------------------------
    INPUTS:
    EV       = (sizez, sizek) matrix, expected value function (expectations
               over z')
    e        = (sizek, sizek) matrix, cash flow values for each possible
               combination of capital stock today (state) and choice of capital
               stock tomorrow (control)
    betafirm = scalar in [0, 1], the discount factor of the firm
    Pi       = (sizez, sizez) matrix, transition probabilities between points
               in the productivity state space
    sizez    = integer, number of grid points for firm productivity shocks
               state space
    sizek    = integer, the number of grid points in capital space
    Vmat     = (sizek, sizek) matrix, matrix with values of firm at each
               combination of state (k) and control (k')

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION: None

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Vmat
    ------------------------------------------------------------------------
    '''
    tau_l, tau_i, tau_d, tau_g, tau_c, f_e, f_b = tax_params
    # initialize Vmat array
    # start = time.time()
    Vmat = np.empty((sizez, sizek, sizek, sizeb, sizeb))
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for jj in range(sizek):  # loop over k'
                for m in range(sizeb):  # loop over b
                    for mm in range(sizeb):  # loop over b'
                        Vmat[i, j, jj, m, mm] = ((((1 - tau_d) / (1 - tau_g)) *
                                 e[i, j, jj, m, mm]) * (e[i, j, jj, m, mm] >= 0) +
                                 ((e[i, j, jj, m, mm] + eta[i, j, jj, m, mm]) *
                                  (e[i, j, jj, m, mm] < 0)) + betafirm * EV[i, jj, mm])
    # end = time.time()
    # print('Vmat loop takes ', end-start, ' seconds to complete.')

    return Vmat


@numba.jit
def adj_costs(kprime, k, delta, psi, fixed_cost):
    '''
    -------------------------------------------------------------------------
    Compute adjustment costs
    -------------------------------------------------------------------------
    c   = (sizek, sizek) array, adjustment costs for each combination of
          combination of capital stock today (state), and choice of capital
          stock tomorrow (control)
    -------------------------------------------------------------------------
    '''
    c = ((psi / 2) * (((kprime - ((1 - delta) * k)) ** 2) / k) +
         fixed_cost * k * (kprime != ((1 - delta) * k)))

    return c


@numba.jit
def get_firmobjects(r, w, zgrid, kgrid, bgrid, alpha_k, alpha_l, delta,
                    psi, fixed_cost, eta0, eta1, eta2, s, sizez, sizek,
                    sizeb, tax_params):
    '''
    -------------------------------------------------------------------------
    Generating possible cash flow levels
    -------------------------------------------------------------------------
    op  = (sizez, sizek) matrix, operating profits for each point in capital
          stock and productivity shock grid spaces
    l_d = (sizez, sizek) matrix, firm labor demand for each point in capital
          stock and productivity shock grid spaces
    y   = (sizez, sizek) matrix, firm output for each point in capital
          stock and productivity shock grid spaces
    e   = (sizez, sizek, sizek) array, cash flow values for each possible
          combination of current productivity shock (state), capital stock
          today (state), and choice of capital stock tomorrow (control)
    -------------------------------------------------------------------------
    '''
    tau_l, tau_i, tau_d, tau_g, tau_c, f_e, f_b = tax_params
    # Initialize arrays
    op = np.empty((sizez, sizek))
    l_d = np.empty((sizez, sizek))
    y = np.empty((sizez, sizek))
    e = np.empty((sizez, sizek, sizek, sizeb, sizeb))
    collateral_constraint = np.empty((sizez, sizek, sizek, sizeb, sizeb))
    # start = time.time()
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            op[i, j] = ((1 - alpha_l) * ((alpha_l / w) **
                                         (alpha_l / (1 - alpha_l))) *
                        ((zgrid[i] * (kgrid[j] ** alpha_k)) **
                         (1 / (1 - alpha_l))))
            l_d[i, j] = (((alpha_l / w) ** (1 / (1 - alpha_l))) *
                         (zgrid[i] ** (1 / (1 - alpha_l))) *
                         (kgrid[j] ** (alpha_k / (1 - alpha_l))))
            y[i, j] = zgrid[i] * (kgrid[j] ** alpha_k) * (l_d[i, j] ** alpha_l)
            for m in range(sizeb):  # loop over b
                for jj in range(sizek):  # loop over k'
                    for mm in range(sizeb):  # loop over b'
                        e[i, j, jj, m, mm] =\
                            (((1 - tau_c) * op[i, j]) +
                             (delta * (1 - f_e) * tau_c * kgrid[j]) +
                             (f_e * tau_c * (kgrid[jj] > ((1 - delta) *
                                                          kgrid[j])) *
                              (kgrid[jj] - ((1 - delta) * kgrid[j]))) -
                             (kgrid[jj] - ((1 - delta) * kgrid[j])) -
                             adj_costs(kgrid[jj], kgrid[j], delta, psi,
                                       fixed_cost) + bgrid[mm] -
                                       ((1 + r) * bgrid[m]) +
                             (r * tau_c * bgrid[m]) -
                             (r * tau_c * (1 - f_b) * bgrid[m] *
                              (bgrid[m] > 0)))
                        collateral_constraint[i, j, jj, m, mm] =\
                            (((1 + r) * bgrid[mm] - (tau_c * r * bgrid[mm])
                              + (r * tau_c * (1 - f_b) * bgrid[mm] *
                                 (bgrid[mm] > 0))) >
                             (((1 - tau_c) * op[0, jj]) +
                              ((1 - f_e) * tau_c * delta * kgrid[jj]) +
                              s * kgrid[jj]))
    eta = (-1 * eta0 + eta1 * e - eta2 * (e ** 2)) * (e < 0)
    # end = time.time()
    # print('Firm objects loop takes ', end-start, ' seconds to complete.')

    return op, e, l_d, y, eta, collateral_constraint


def VFI(e, eta, collateral_constraint, betafirm, delta, kgrid, bgrid, Pi, sizez, sizek, sizeb,
        tax_params, VF_initial):
    '''
    ------------------------------------------------------------------------
    Value Function Iteration
    ------------------------------------------------------------------------
    VFtol     = scalar, tolerance required for value function to converge
    VFdist    = scalar, distance between last two value functions
    VFmaxiter = integer, maximum number of iterations for value function
    VFiter    = integer, current iteration number
    Vmat      = (sizez, sizek, sizek) array, array with values of firm at each
                combination of state (z, k) and control (k')
    Vstore    = (sizez, sizek, VFmaxiter) array, value function at each
                iteration of VFI
    V & TV    = (sizez, sizek) matrix, store the value function at each
                iteration (V being the most current value and TV the one prior)
    EV        = (sizez, sizek) matrix, expected value function (expectations
                over z')
    PF        = (sizez, sizek) matrix, indicies of choices (k') for all states
                (z, k)
    VF        = (sizez, sizek) matrix, matrix of value functions for each
                possible value of the state variables (k)
    ------------------------------------------------------------------------
    '''
    tau_l, tau_i, tau_d, tau_g, tau_c, f_e, f_b = tax_params
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = VF_initial
    #Vstore = np.empty((sizez, sizek, sizeb, VFmaxiter))  # initialize Vstore array
    VFiter = 1
    # while VFdist > VFtol and VFiter < VFmaxiter:
    #     TV = V
    #     EV = create_EV(Pi, V, sizez, sizek, sizeb)  # expected VF (expectation over z')
    #     Vmat = create_Vmat(EV, e, eta, betafirm, Pi, sizez, sizek, sizeb,
    #                        tax_params) + (collateral_constraint * -1000000000)
    #
    #     #Vstore[:, :, :, VFiter] = V.reshape(sizez, sizek, sizeb)  # store value function
    #     # at each iteration for graphing later
    #     # apply max operator to Vmat (to get V(z,k,b))
    #     V = (Vmat.max(axis=4)).max(axis=2)
    #     PF_k = np.argmax(Vmat.max(axis=4), axis=2)
    #     PF_b = np.argmax(Vmat.max(axis=2), axis=3)
    #     VFdist = (np.absolute(V - TV)).max()  # check distance between value
    #     # function for this iteration and value function from past iteration
    #     # print('VF iteration: ', VFiter)
    #     VFiter += 1

    VFflag = 0
    PF_k_old = np.zeros_like(V)
    PF_b_old = np.zeros_like(V)
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        EV = create_EV(Pi, V, sizez, sizek, sizeb)  # expected VF (expectation over z')
        Vmat = create_Vmat(EV, e, eta, betafirm, Pi, sizez, sizek, sizeb,
                           tax_params) + (collateral_constraint * -1000000000)

        if VFiter%10 == 0:

            V = (Vmat.max(axis=4)).max(axis=2)
            PF_k = np.argmax(Vmat.max(axis=4), axis=2)
            PF_b = np.argmax(Vmat.max(axis=2), axis=3)
            if (np.absolute(PF_k-PF_k_old).max() == 0) & (np.absolute(PF_b-PF_b_old).max() == 0):
                VFiter = VFmaxiter
                VFflag = 1
            else:
                PF_k_old = PF_k
                PF_b_old = PF_b
        V = (Vmat.max(axis=4)).max(axis=2)
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        # print('VF iteration: ', VFiter)
        VFiter += 1

    if VFiter < VFmaxiter:
        print('Value function converged after this many iterations:', VFiter)
    elif VFflag == 0:
        print('Value function did not converge')

    VF = V  # solution to the functional equation

    '''
    ------------------------------------------------------------------------
    Find optimal capital and investment policy functions
    ------------------------------------------------------------------------
    optK = (sizez, sizek) vector, optimal choice of k' for each (z, k)
    optI = (sizez, sizek) vector, optimal choice of investment for each (z, k)
    ------------------------------------------------------------------------
    '''
    optK = kgrid[PF_k]
    k3grid = np.tile(np.reshape(kgrid, (1, sizek, 1)), (sizez, 1, sizeb))
    optI = optK - (1 - delta) * k3grid
    optB = bgrid[PF_b]

    return VF, PF_k, PF_b, optK, optI, optB
