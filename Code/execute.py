# TODO: (0) check equations.  Also, can I replicate GM (2010)??(1) add costly equity, (2) add debt, (3) add more tax params - tax deprec rate, interest deduct, etc (as outline in OG-USA guide),
# (4) think more about tables and figures, (5) update figures scripot (6) add tables script,
# (7) write moments script to generate moments (maybe do before tables), (8) estimation script
#(8) work on passing past solution of VFI so have good starting values

'''
------------------------------------------------------------------------
Solves the dynamic programming problem of the firm with:
- Quadratic adjustment costs
- TFP/Profit shocks
- General Equilibrium
- Taxes
- No external finance

This py-file calls the following other file(s):

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            graphs/
------------------------------------------------------------------------
'''

# Import packages
import scipy.optimize as opt
import os
import time
import numpy as np

import grids
import VFI
import SS
import plots
import moments

# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


'''
------------------------------------------------------------------------
Specify Parameters
------------------------------------------------------------------------
beta      = scalar in (0, 1), rate of time preference
alpha_k   = scalar in [0, 1], exponent on capital in firm production function
alpha_l   = scalar in [0, 1], exponent on labor in firm production function
delta     = scalar in [0, 1], depreciation rate on capital
psi       = scalar, coefficient in quadratic adjustment costs for capital
w         = scalar, exogenous wage rate
r         = scalar, risk free interest rate, in eqm, r = (1 / beta) - 1
beta_firm = scalar in [0, 1], the discount factor of the firm
sigma_eps = scalar > 0, standard deviation of profit/productivity shocks to
            AR(1) process for firm productivity
mu        = scalar, unconditional mean of productivity shocks
rho       = scalar in [0, 1], persistence of productivity shocks
sizez     = integer, number of grid points for firm productivity shocks state
            space
------------------------------------------------------------------------
'''
# Household parameters
beta = 0.96
h = 6.616

# Firm parameters
alpha_k = 0.29715
alpha_l = 0.65
delta = 0.154
psi = 1.08
mu = 0
rho = 0.7605
sigma_eps = 0.213

# financial frictions
eta0 = 0.04  # fixed cost to issuing equity
eta1 = 0.02  # linear cost to issuing equity

# taxes
tau_i = 0.25
tau_d = 0.25
tau_g = 0.20
tau_c = 0.34
tax_params = (tau_i, tau_d, tau_g, tau_c)

# state space parameters
sizez = 9
num_sigma = 3  # number of std dev of profit shock to include in grid
dens = 5
lb_k = 0.001
zgrid_params = (num_sigma, sizez)
kgrid_params = (dens, lb_k)


# Gourio and Miao (AEJ: Macro, 2010) calibration
beta = 0.971
alpha_k = 0.311
alpha_l = 0.65
delta = 0.095
psi = 1.08
w = 1.3
r = ((1 / beta) - 1)
betafirm = (1 / (1 + r))
sigma_eps = 0.211
mu = 0
rho = 0.767
sizez = 9
h = 6.616

# Whited calibration
# alpha_k = 0.7
# alpha_l = 0.0
# delta = 0.15
# psi = 0.01
# w = 1.0
# r = 0.04
# betafirm = (1 / (1 + r))
# sigma_eps = 0.15
# mu = 0
# rho = 0.7
# sizez = 9

# Factor prices
w0 = 1.3  # initial guess as wage rate
r = ((1 / beta) - 1)
betafirm = (1 / (1 + (r * ((1 - tau_i) / (1 - tau_g)))))

firm_params = (betafirm, delta, alpha_k, alpha_l)

# get the grids for K and z
Pi, z = grids.discrete_z(rho, mu, sigma_eps, zgrid_params)
K, sizek, kstar = grids.discrete_k(w0, firm_params, kgrid_params, z, sizez)


'''
------------------------------------------------------------------------
Solve for general equilibrium
------------------------------------------------------------------------
'''
start_time = time.clock()
VF_initial = np.zeros((sizez, sizek))  # initial guess at Value Function
# initial guess at stationary distribution
Gamma_initial = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
results = opt.bisect(SS.GE_loop, 0.1, 2, args=(alpha_k, alpha_l, delta, psi,
                                               betafirm, K, z, Pi, eta0,
                                               eta1, sizek, sizez, h,
                                               tax_params, VF_initial,
                                               Gamma_initial),
                     xtol=1e-4, full_output=True)
print('SS results: ', results)
w = results[0]
GE_time = time.clock() - start_time
print('Solving the GE model took ', GE_time, ' seconds to solve')
print('SS wage rate = ', w)

'''
------------------------------------------------------------------------
Find model outputs given eq'm wage rate
------------------------------------------------------------------------
'''
op, e, l_d, y, eta = VFI.get_firmobjects(w, z, K, alpha_k, alpha_l, delta, psi,
                                         eta0, eta1, sizez, sizek, tax_params)
VF, PF, optK, optI = VFI.VFI(e, eta, betafirm, delta, K, Pi,
                             sizez, sizek, tax_params, VF_initial)
Gamma = SS.find_SD(PF, Pi, sizez, sizek, Gamma_initial)
print('Sum of Gamma = ', Gamma.sum())

'''
------------------------------------------------------------------------
Plot results
------------------------------------------------------------------------
'''
k_params = (K, sizek, dens, kstar)
z_params = (Pi, z, sizez)
output_vars = (optK, optI, op, e, VF, PF, Gamma)
plots.firm_plots(delta, k_params, z_params, output_vars, output_dir)

'''
------------------------------------------------------------------------
Print moments
------------------------------------------------------------------------
'''
agg_IK, agg_DE, agg_SI, sd_IK, ac_IK, sd_EK, ac_IK =\
    moments.firm_moments(delta, k_params, z_params, output_vars,
                         output_dir, print_moments=True)
