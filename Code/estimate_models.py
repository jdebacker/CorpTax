import numpy as np
import scipy.optimize as opt
import smm
import os

# Create directory to save files
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'OUTPUT'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Call estimator
theta0 = [1.08, 0.031]  # intial guesses at params
data_moments = np.array([0.095, 0.137, 0.130, 0.156, 0.596, 0.623, 0.791])
Q = data_moments.shape[0]
W = np.eye(Q)  # smm.optimal_weight(data)
# Household parameters
hh_params_baseline = {'beta': 0.96, 'h': 6.616}
# Baseline firm parameters
firm_params_baseline = {'alpha_k': 0.29715, 'alpha_l': 0.65, 'delta': 0.095,
                        'psi': 1.08, 'fixed_cost': 0.031, 'mu': 0,
                        'rho': 0.765, 'sigma_eps': 0.213}
fin_frictions = {'eta0': 0.0, 'eta1': 0.02, 'eta2': 0.0, 'theta': 0.3}
# taxes
tax_params = {'tau_l': 0.25, 'tau_i': 0.25, 'tau_d': 0.20, 'tau_g': 0.20,
              'tau_c': 0.35, 'f_e': 0.0, 'f_b': 1.0}
# state space parameters
grid_params = {'sizez': 9, 'num_sigma': 3, 'dens_k': 1, 'lb_k': 0.0001,
               'sizeb': 1, 'lb_b': 0.0, 'ub_b': 0.0}

# Factor prices
w0 = 1.23  # initial guess at wage rate (or exogenous wage rate in PE model)

# theta_hat = opt.basinhopping(
#     smm.objective_func, theta0, niter=100, T=1.0, stepsize=0.5,
#     minimizer_kwargs={"args": (data_moments, W, w0, tax_params,
#                                hh_params_baseline, firm_params_baseline,
#                                fin_frictions, grid_params, output_dir,
#                                'est_quad_fixed', False)}, interval=50)
# print('Estimation results: ', theta_hat)

theta_hat = theta0
# compute standard errors
args = (w0, tax_params,
                   hh_params_baseline, firm_params_baseline, fin_frictions, grid_params,
                   output_dir, 'est_quad_fixed', False)
std_errors = smm.std_errors(theta_hat, W, args)

# save results to pickle
