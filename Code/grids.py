'''
------------------------------------------------------------------------
This module contains functions used discretize the grid used for the
solution of the firm's problem

* discrete_z()
* discrete_k()
------------------------------------------------------------------------
'''

# import modules
import numpy as np
import ar1_approx as ar1


def discrete_z(rho, mu, sigma_eps, zgrid_params):
    '''
    -------------------------------------------------------------------------
    Discretizing state space for productivity shocks
    -------------------------------------------------------------------------
    sigma_z   = scalar, standard deviation of ln(z)
    num_sigma = scalar, number of standard deviations around mean to include in
                grid space for z
    step      = scalar, distance between grid points in the productivity state
                space
    Pi        = (sizez, sizez) matrix, transition probabilities between points in
                the productivity state space
    z         = (sizez,) vector, grid points in the productivity state space
    -------------------------------------------------------------------------
    '''
    # We will use the Rouwenhorst (1995) method to approximate a continuous
    # distribution of shocks to the AR1 process with a Markov process.
    num_sigma, sizez = zgrid_params
    sigma_z = sigma_eps / ((1 - rho ** 2) ** (1 / 2))
    step = (num_sigma * sigma_z) / (sizez / 2)
    Pi, z = ar1.rouwen(rho, mu, step, sizez)
    # wgt = 0.5 + rho / 4
    # baseSigma = wgt * sigma_eps + (1 - wgt) * sigma_z
    # z, Pi = ar1.tauchenhussey(sizez, mu, rho, sigma_eps, baseSigma)
    Pi = np.transpose(Pi)  # make so rows are where start, columns where go
    z = np.exp(z)  # because the AR(1) process was for the log of productivity

    return Pi, z


def discrete_k(w, firm_params, kgrid_params, zgrid, sizez):
    '''
    -------------------------------------------------------------------------
    Discretizing state space for capital
    -------------------------------------------------------------------------
    dens   = integer, density of the grid: number of grid points between k and
             (1 - delta) * k
    kstar  = scalar, capital stock choose w/o adjustment costs and mean
             productivity shock
    kbar   = scalar, maximum capital stock the firm would ever accumulate
    ub_k   = scalar, upper bound of capital stock space
    lb_k   = scalar, lower bound of capital stock space
    krat   = scalar, difference between upper and lower bound in log points
    numb   = integer, the number of steps between the upper and lower bounds for
             the capital stock. The number of grid points is dens*numb.
    kgrid      = (sizek,) vector, grid points in the capital state space, from high
             to low
    kvec  = (sizek,) vector, capital grid points
    sizek = integer, the number of grid points in capital space
    -------------------------------------------------------------------------
    '''
    dens, lb_k = kgrid_params
    betafirm, delta, alpha_k, alpha_l = firm_params

    # put in bounds here for the capital stock space
    kstar = ((((1 / betafirm - 1 + delta) * ((alpha_l / w) **
                                             (alpha_l / (alpha_l - 1)))) /
             (alpha_k * (zgrid[(sizez - 1) // 2] ** (1 / (1 - alpha_l))))) **
             ((1 - alpha_l) / (alpha_k + alpha_l - 1)))
    kbar = 12  # kstar * 500
    ub_k = kbar
    krat = np.log(lb_k / ub_k)
    numb = np.ceil(krat / np.log(1 - delta))
    kvec = np.empty(int(numb * dens))
    for j in range(int(numb * dens)):
        kvec[j] = ub_k * (1 - delta) ** (j / dens)
    kgrid = kvec[::-1]
    sizek = kgrid.shape[0]

    return kgrid, sizek, kstar

def discrete_b(lb_b, ub_b, sizeb):
    '''
    This function creates the grid space for corporate debt.

    Args:
        lb_b: scalar, lower bound of debt (can be negative)
        ub_b: scalar, upper bound of debt
        sizeb: interger, number of grid points for debt

    Returns:
        bgrid: vector with possible value of debt
    '''
    zgrid = np.linspace(lb_b, ub_b, num=sizeb)

    return zgrid
