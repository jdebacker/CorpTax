# TODO: (0) Can I replicate GM (2010)??(1) add costly equity, (2) add debt, (3) add more tax params - tax deprec rate, interest deduct, etc (as outline in OG-USA guide),
# (4) think more about tables and figures, (5) update figures script (6) add tables script,
# (8) estimation script
#(9) Try to solve for stationary distribution by finding eigenvector

'''
------------------------------------------------------------------------
This script kicks of runs of the model of firm dynamics used run
counterfactual policy simulations.

This py-file calls the following other file(s):

This py-file creates the following other file(s):
    (make sure that an OUTPUT folder exists)
            graphs/
------------------------------------------------------------------------
'''

# Import packages
import os
import time
from execute import solve_GE, solve_PE

# Create directory to save files
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'OUTPUT'
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
hh_params_baseline = {'beta': 0.96, 'h': 6.616}
hh_params_GM_AEJ_2010 = {'beta': 0.971, 'h': 6.616}


# Gourio and Miao (AEJ: Macro, 2010) calibration
firm_params_GM_AEJ_2010 = {'alpha_k': 0.311, 'alpha_l': 0.65, 'delta': 0.095,
                      'psi': 1.08, 'mu': 0, 'rho': 0.767,
                      'sigma_eps': 0.211}

# Whited calibration
firm_params_whited = {'alpha_k': 0.7, 'alpha_l': 0.0, 'delta': 0.15,
                      'psi': 0.01, 'mu': 0.0, 'rho': 0.7, 'sigma_eps': 0.15}

# Baseline firm parameters
firm_params_baseline = {'alpha_k': 0.29715, 'alpha_l': 0.65, 'delta': 0.154,
                      'psi': 1.08, 'mu': 0, 'rho': 0.765, 'sigma_eps': 0.213}

# financial frictions
# equity flotation costs
# eta0 = 0.0#0.04  # fixed cost to issuing equity
# eta1 = 0.0#0.02  # linear cost to issuing equity
# eta2 = 0.00  # quadratic costs to issuing equity
# debt costs
# fire sale price of capital
# theta = 0.5
# instead of cost of adjusting debt (as in Gamba and Triantis (2008)),
# how about a cost of debt function that is a reduced form way to capture
# endogenous default
# this function will be increasing as leverage ratios increase and will
# asymptote as infinity when it would be optimal to debt - i.e., debt
# exceeds the expected present value of the firm, beta*EV(z,k,b)
# questions: 1) should it increase at all if can cover debt costs in fire sale?
# 2) EV is endgodenous to the cost of debt - so need additional fixed point
# loop - as would with model with endogenous interest rates...
# FOR NOW, NO COSTS
fin_frictions = {'eta0': 0.0, 'eta1': 0.02, 'eta2': 0.0, 'theta': 0.3}
# fin_frictions = {'eta0': 0.0, 'eta1': 0.0, 'eta2': 0.0, 'theta': 0.5}


# taxes
tax_params = {'tau_l': 0.25, 'tau_i': 0.25, 'tau_d': 0.20, 'tau_g': 0.20, 'tau_c': 0.35,
              'f_e': 0.0, 'f_b': 1.0}
tax_params_elas = {'tau_l': 0.25, 'tau_i': 0.25, 'tau_d': 0.20, 'tau_g': 0.20, 'tau_c': 0.34,
              'f_e': 0.0, 'f_b': 1.0}
# tax_params_CIT = {'tau_l': 0.25, 'tau_i': 0.25, 'tau_d': 0.20, 'tau_g': 0.20, 'tau_c': 0.25,
#                   'f_e': 0.0, 'f_b': 1.0}
# tax_params_CFT = {'tau_l': 0.25, 'tau_i': 0.25, 'tau_d': 0.20, 'tau_g': 0.20, 'tau_c': 0.25,
#                   'f_e': 1.0, 'f_b': 0.0}

# with values from B-Tax/Tax-Calc
# Interest income, taxable, mtr =  0.3466729052394317
# wage income, mtr =  0.24972454804159058 0.3609129610839168
# {'tau_nc': 0.3208582284636031, 'tau_div': 0.19136683768259502,
#  'tau_int': 0.3466729052394317, 'tau_scg': 0.3200815758636131,
#  'tau_lcg': 0.22529460597677733, 'tau_td': 0.24553735804206817, 'tau_h': 0.1712837798257659}
# tax rates on interest, dividend, and capital gains income:  0.3466729052394317 0.19136683768259502 0.0202124374724
# tax on cap gains w/o deferral:  0.12262889814384441
# tax on cap gains w/o defferal and excluding death:  0.23137527951668754

# tax_params = {'tau_l': 0.36, 'tau_i': 0.34, 'tau_d': 0.19, 'tau_g': 0.19, 'tau_c': 0.35,
#               'f_e': 0.0, 'f_b': 1.0}
# tax_params_elas = {'tau_l': 0.36, 'tau_i': 0.34, 'tau_d': 0.19, 'tau_g': 0.19, 'tau_c': 0.34,
#               'f_e': 0.0, 'f_b': 1.0}
# tax_params_CIT = {'tau_l': 0.36, 'tau_i': 0.34, 'tau_d': 0.19, 'tau_g': 0.19, 'tau_c': 0.25,
#                   'f_e': 0.0, 'f_b': 1.0}
# tax_params_CFT = {'tau_l': 0.36, 'tau_i': 0.34, 'tau_d': 0.19, 'tau_g': 0.19, 'tau_c': 0.25,
#                   'f_e': 1.0, 'f_b': 0.0}

'''
Set state space parameters and then compute grids.  Then put grids in
dictionary to pass to funtions that solve models.
'''
# state space parameters
grid_params = {'sizez': 9, 'num_sigma': 3, 'dens_k': 1, 'lb_k': 0.0001,
               'sizeb': 33, 'lb_b': 0.0, 'ub_b': 0.0}

# Factor prices
w0 = 1.23  # initial guess at wage rate (or exogenous wage rate in PE model)

#solve GE
solve_GE(w0, tax_params, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
         fin_frictions, grid_params, output_dir, guid='baseline',
         plot_results=False)
# solve_GE(w0, tax_params_CIT, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
#          fin_frictions, grid_params, output_dir, guid='CIT',
#          plot_results=False)
# solve_GE(w0, tax_params_CFT, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
#          fin_frictions, grid_params, output_dir, guid='CFT',
#          plot_results=False)
solve_GE(w0, tax_params_elas, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
         fin_frictions, grid_params, output_dir, guid='elas',
         plot_results=False)

#solve PE
start = time.time()
solve_PE(w0, tax_params, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
         fin_frictions, grid_params, output_dir, guid='baseline')
end = time.time()
# print('Solving the PE model took ', end-start, ' seconds.')
# solve_PE(w0, tax_params_CIT, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
#          fin_frictions, grid_params, output_dir, guid='CIT')
# solve_PE(w0, tax_params_CFT, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
#          fin_frictions, grid_params, output_dir, guid='CFT')
solve_PE(w0, tax_params_elas, hh_params_GM_AEJ_2010, firm_params_GM_AEJ_2010,
         fin_frictions, grid_params, output_dir, guid='elas')
