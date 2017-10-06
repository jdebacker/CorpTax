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


@numba.jit
def create_Vmat(EV, e, eta, betafirm, Pi, sizez, sizek, Vmat, tax_params):
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
    tau_i, tau_d, tau_g, tau_c = tax_params
    for i in range(sizez):  # loop over z
        for j in range(sizek):  # loop over k
            for m in range(sizek):  # loop over k'
                Vmat[i, j, m] = (((1 - tau_d) / (1 - tau_g)) *
                                 (e[i, j, m] - eta[i, j, m]) +
                                 betafirm * EV[i, m])

    return Vmat


@numba.jit
def adj_costs(kprime, k, delta, psi):
    '''
    -------------------------------------------------------------------------
    Compute adjustment costs
    -------------------------------------------------------------------------
    c   = (sizek, sizek) array, adjustment costs for each combination of
          combination of capital stock today (state), and choice of capital
          stock tomorrow (control)
    -------------------------------------------------------------------------
    '''
    c = (psi / 2) * (((kprime - ((1 - delta) * k)) ** 2) / k)

    return c


@numba.jit
def get_firmobjects(w, z, K, alpha_k, alpha_l, delta, psi, eta0, eta1,
                    sizez, sizek, tax_params):
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
    tau_i, tau_d, tau_g, tau_c = tax_params
    # Initialize arrays
    op = np.empty((sizez, sizek))
    l_d = np.empty((sizez, sizek))
    y = np.empty((sizez, sizek))
    e = np.empty((sizez, sizek, sizek))
    for i in range(sizez):
        for j in range(sizek):
            op[i, j] = ((1 - alpha_l) * ((alpha_l / w) **
                                         (alpha_l / (1 - alpha_l))) *
                        ((z[i] * (K[j] ** alpha_k)) **
                         (1 / (1 - alpha_l))))
            l_d[i, j] = (((alpha_l / w) ** (1 / (1 - alpha_l))) *
                         (z[i] ** (1 / (1 - alpha_l))) *
                         (K[j] ** (alpha_k / (1 - alpha_l))))
            y[i, j] = z[i] * (K[j] ** alpha_k) * (l_d[i, j] ** alpha_l)
            for m in range(sizek):
                e[i, j, m] = ((1 - tau_c) * op[i, j] + (delta * tau_c
                                                        * K[j]) - K[m]
                                                        + ((1 - delta)
                                                           * K[j]) -
                              adj_costs(K[m], K[j], delta, psi))

    eta = (eta0 + eta1 * e) * (e < 0)

    return op, e, l_d, y, eta


def VFI(e, eta, betafirm, delta, K, Pi, sizez, sizek, tax_params, VF_initial):
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
    tau_i, tau_d, tau_g, tau_c = tax_params
    VFtol = 1e-6
    VFdist = 7.0
    VFmaxiter = 3000
    V = VF_initial
    Vmat = np.empty((sizez, sizek, sizek))  # initialize Vmat matrix
    Vstore = np.empty((sizez, sizek, VFmaxiter))  # initialize Vstore array
    VFiter = 1
    while VFdist > VFtol and VFiter < VFmaxiter:
        TV = V
        EV = np.dot(Pi, V)  # expected VF (expectation over z')
        Vmat = create_Vmat(EV, e, eta, betafirm, Pi, sizez, sizek, Vmat, tax_params)

        Vstore[:, :, VFiter] = V.reshape(sizez, sizek)  # store value function
        # at each iteration for graphing later
        V = Vmat.max(axis=2)  # apply max operator to Vmat (to get V(k))
        PF = np.argmax(Vmat, axis=2)
        VFdist = (np.absolute(V - TV)).max()  # check distance between value
        # function for this iteration and value function from past iteration
        # print('VF iteration: ', VFiter)
        VFiter += 1

    # if VFiter < VFmaxiter:
    #     print('Value function converged after this many iterations:', VFiter)
    # else:
    #     print('Value function did not converge')

    VF = V  # solution to the functional equation

    '''
    ------------------------------------------------------------------------
    Find optimal capital and investment policy functions
    ------------------------------------------------------------------------
    optK = (sizez, sizek) vector, optimal choice of k' for each (z, k)
    optI = (sizez, sizek) vector, optimal choice of investment for each (z, k)
    ------------------------------------------------------------------------
    '''
    optK = K[PF]
    optI = optK - (1 - delta) * K

    return VF, PF, optK, optI
