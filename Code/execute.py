
'''
------------------------------------------------------------------------
This script contains runner functions that find the partial or general
equilibrium solution to the model of firm dynamics.

This py-file calls the following other file(s):

This py-file creates the following other file(s):

------------------------------------------------------------------------
'''

# Import packages
import time
import numpy as np
import pickle
import os
import pathlib

import VFI
import SS
import plots
import grids
import moments



def solve_GE(w0, tax_params, hh_params, firm_params, fin_frictions, grid_params,
             output_dir, guid='baseline', plot_results=False):

    # set directory specific to this model run
    output_path = os.path.join(output_dir, 'GE', guid)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # upack dictionaries
    tau_i = tax_params['tau_i']
    tau_d = tax_params['tau_d']
    tau_g = tax_params['tau_g']
    tau_c = tax_params['tau_c']
    f_e = tax_params['f_e']
    f_b = tax_params['f_b']
    tax_tuple = (tau_i, tau_d, tau_g, tau_c, f_e, f_b)

    alpha_k = firm_params['alpha_k']
    alpha_l = firm_params['alpha_l']
    delta = firm_params['delta']
    psi = firm_params['psi']
    mu = firm_params['mu']
    rho = firm_params['rho']
    sigma_eps = firm_params['sigma_eps']

    eta0 = fin_frictions['eta0']
    eta1 = fin_frictions['eta1']
    eta2 = fin_frictions['eta2']
    theta = fin_frictions['theta']

    sizez = grid_params['sizez']
    num_sigma = grid_params['num_sigma']
    dens_k = grid_params['dens_k']
    lb_k = grid_params['lb_k']
    sizeb = grid_params['sizeb']
    lb_b = grid_params['lb_b']
    ub_b = grid_params['ub_b']

    beta = hh_params['beta']
    h = hh_params['h']

    # compute equilibrium interest rate
    r = ((1 / beta) - 1) / (1 - tau_i)
    # compute the firm's discount factor
    betafirm = (1 / (1 + (r * ((1 - tau_i) / (1 - tau_g)))))

    '''
    ------------------------------------------------------------------------
    Compute grids for z, k, b
    ------------------------------------------------------------------------
    '''
    firm_params_k = (betafirm, delta, alpha_k, alpha_l)
    Pi, zgrid = grids.discrete_z(rho, mu, sigma_eps, num_sigma, sizez)
    kgrid, sizek, kstar, ub_k = grids.discrete_k(w0, firm_params_k,
                                                 zgrid, sizez, dens_k, lb_k)
    bgrid = grids.discrete_b(lb_b, ub_b, sizeb, w0, firm_params_k, zgrid,
                             tau_c, theta, ub_k)
    # grid_params = (zgrid, sizez, Pi, kgrid, sizek, kstar, bgrid, sizeb)

    '''
    ------------------------------------------------------------------------
    Solve for general equilibrium
    ------------------------------------------------------------------------
    '''
    VF_initial = np.zeros((sizez, sizek, sizeb))  # initial guess at Value Function
    # initial guess at stationary distribution
    Gamma_initial = np.ones((sizez, sizek, sizeb)) * (1 / (sizek * sizez *
                                                           sizeb))
    gr_args = (r, alpha_k, alpha_l, delta, psi, betafirm, kgrid, zgrid, bgrid,
               Pi, eta0, eta1, eta2, theta, sizek, sizez, sizeb, h, tax_tuple,
               VF_initial, Gamma_initial)
    start_time = time.time()
    w = SS.golden_ratio_eqm(0.8, 1.6, gr_args, tolerance=1e-4)
    end_time = time.time()
    print('Solving the GE model took ', end_time - start_time, ' seconds to solve')
    print('SS wage rate: ', w)

    '''
    ------------------------------------------------------------------------
    Find model outputs given eq'm wage rate
    ------------------------------------------------------------------------
    '''
    op, e, l_d, y, eta, collateral_constraint =\
        VFI.get_firmobjects(r, w, zgrid, kgrid, bgrid, alpha_k, alpha_l,
                            delta, psi, eta0, eta1, eta2, theta, sizez,
                            sizek, sizeb, tax_tuple)
    VF, PF_k, PF_b, optK, optI, optB =\
        VFI.VFI(e, eta, collateral_constraint, betafirm, delta, kgrid,
                bgrid, Pi, sizez, sizek, sizeb, tax_tuple, VF_initial)
    Gamma = SS.find_SD(PF_k, PF_b, Pi, sizez, sizek, sizeb, Gamma_initial)

    '''
    ------------------------------------------------------------------------
    Compute model moments
    ------------------------------------------------------------------------
    '''
    output_vars = (optK, optI, optB, op, e, l_d, y, eta, VF, PF_k, PF_b, Gamma)
    k_params = (kgrid, sizek, dens_k, kstar)
    z_params = (Pi, zgrid, sizez)
    b_params = (bgrid, sizeb)
    model_moments =\
        moments.firm_moments(w, r, delta, psi, h, k_params, z_params,
                             b_params, tax_tuple, output_vars,
                             output_dir, print_moments=True)

    if plot_results:
        '''
        ------------------------------------------------------------------------
        Plot results
        ------------------------------------------------------------------------
        '''
        # plots.firm_plots(delta, k_params, z_params, output_vars, output_dir)

    # create dictionaries of output, params, grids, moments
    # output_dict = {'optK': optK, 'optI': optI, 'optB': optB, 'op': op,
    #                'e': e, 'eta': eta, 'VF': VF, 'PF_k': PF_k,
    #                'PF_b': PF_b, 'Gamma': Gamma}
    output_dict = {'optK': optK, 'optI': optI, 'optB': optB, 'VF': VF,
                   'PF_k': PF_k, 'PF_b': PF_b, 'Gamma': Gamma}
    param_dict = {'w': w, 'r': r, 'tax_params': tax_params,
                  'hh_params': hh_params, 'firm_params': firm_params,
                  'fin_frictions': fin_frictions,
                  'grid_params': grid_params}
    grid_dict = {'zgrid': zgrid, 'sizez': sizez, 'Pi': Pi, 'kgrid': kgrid,
                 'sizek': sizek, 'kstar': kstar, 'bgrid': bgrid,
                 'sizeb': sizeb}
    model_out_dict = {'params': param_dict, 'grid': grid_dict,
                      'moments': model_moments, 'output': output_dict}
    # Save pickle of model output
    pkl_path = os.path.join(output_path, 'model_output.pkl')
    pickle.dump(model_out_dict, open(pkl_path, 'wb'))


def solve_PE(w0, tax_params, hh_params, firm_params, fin_frictions, grid_params,
             output_dir, guid='baseline', plot_results=False):
    '''
    ------------------------------------------------------------------------
    Solve partial equilibrium model
    ------------------------------------------------------------------------
    '''
    # set directory specific to this model run
    output_path = os.path.join(output_dir, 'PE', guid)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    # upack dictionaries
    tau_i = tax_params['tau_i']
    tau_d = tax_params['tau_d']
    tau_g = tax_params['tau_g']
    tau_c = tax_params['tau_c']
    f_e = tax_params['f_e']
    f_b = tax_params['f_b']
    tax_tuple = (tau_i, tau_d, tau_g, tau_c, f_e, f_b)

    alpha_k = firm_params['alpha_k']
    alpha_l = firm_params['alpha_l']
    delta = firm_params['delta']
    psi = firm_params['psi']
    mu = firm_params['mu']
    rho = firm_params['rho']
    sigma_eps = firm_params['sigma_eps']

    eta0 = fin_frictions['eta0']
    eta1 = fin_frictions['eta1']
    eta2 = fin_frictions['eta2']
    theta = fin_frictions['theta']

    sizez = grid_params['sizez']
    num_sigma = grid_params['num_sigma']
    dens_k = grid_params['dens_k']
    lb_k = grid_params['lb_k']
    sizeb = grid_params['sizeb']
    lb_b = grid_params['lb_b']
    ub_b = grid_params['ub_b']

    beta = hh_params['beta']
    h = hh_params['h']

    # compute equilibrium interest rate
    r = ((1 / beta) - 1) / (1 - tau_i)
    # compute the firm's discount factor
    betafirm = (1 / (1 + (r * ((1 - tau_i) / (1 - tau_g)))))

    '''
    ------------------------------------------------------------------------
    Compute grids for z, k, b
    ------------------------------------------------------------------------
    '''
    firm_params_k = (betafirm, delta, alpha_k, alpha_l)
    Pi, zgrid = grids.discrete_z(rho, mu, sigma_eps, num_sigma, sizez)
    kgrid, sizek, kstar, ub_k = grids.discrete_k(w0, firm_params_k,
                                                 zgrid, sizez, dens_k, lb_k)
    bgrid = grids.discrete_b(lb_b, ub_b, sizeb, w0, firm_params_k, zgrid,
                             tau_c, theta, ub_k)
    # grid_params = (zgrid, sizez, Pi, kgrid, sizek, kstar, bgrid, sizeb)

    '''
    ------------------------------------------------------------------------
    Solve for partial equilibrium
    ------------------------------------------------------------------------
    '''
    VF_initial = np.zeros((sizez, sizek, sizeb))  # initial guess at Value Function
    # initial guess at stationary distribution
    Gamma_initial = np.ones((sizez, sizek, sizeb)) * (1 / (sizek * sizez *
                                                           sizeb))
    op, e, l_d, y, eta, collateral_constraint =\
        VFI.get_firmobjects(r, w0, zgrid, kgrid, bgrid, alpha_k, alpha_l,
                            delta, psi, eta0, eta1, eta2, theta, sizez,
                            sizek, sizeb, tax_tuple)
    VF, PF_k, PF_b, optK, optI, optB =\
        VFI.VFI(e, eta, collateral_constraint, betafirm, delta, kgrid,
                bgrid, Pi, sizez, sizek, sizeb, tax_tuple, VF_initial)
    # print('Policy funtion, debt: ', PF_b[0, int(np.ceil(sizek/2)), :])
    # print('Policy funtion, debt: ', PF_b[5, int(np.ceil(sizek/2)), :])
    # print('Policy funtion, debt: ', PF_b[-1, int(np.ceil(sizek/2)), :])
    #
    # print('Policy funtion, debt: ', PF_b[0, -3, :])
    # print('Policy funtion, debt: ', PF_b[5, -3, :])
    # print('Policy funtion, debt: ', PF_b[-1, -3, :])
    #
    # print('Policy funtion, debt: ', PF_b[0, 3, :])
    # print('Policy funtion, debt: ', PF_b[5, 3, :])
    # print('Policy funtion, debt: ', PF_b[-1, 3, :])
    #
    # print('bgrid = ', bgrid)
    # print('Policy funtion, investment: ', optI[0, :, :])
    # print('Policy funtion, investment: ', optI[5, :, :])
    # print('Policy funtion, investment: ', optI[-1, :, :])
    # quit()

    #
    # print('VF: ', VF[0, -3, :])
    # print('VF: ', VF[5, -3, :])
    # print('VF: ', VF[-1, -3, :])
    #
    # print('Collateral Constraint = ', collateral_constraint[-1, -3, :, -1, :])
    # print('Collateral Constraint = ', collateral_constraint[-1, -3, -3, -1, :])
    #

    Gamma = SS.find_SD(PF_k, PF_b, Pi, sizez, sizek, sizeb, Gamma_initial)

    # print('SD over z and b: ', Gamma.sum(axis=1))

    '''
    ------------------------------------------------------------------------
    Compute model moments
    ------------------------------------------------------------------------
    '''
    output_vars = (optK, optI, optB, op, e, l_d, y, eta, VF, PF_k, PF_b, Gamma)
    k_params = (kgrid, sizek, dens_k, kstar)
    z_params = (Pi, zgrid, sizez)
    b_params = (bgrid, sizeb)
    model_moments =\
        moments.firm_moments(w0, r, delta, psi, h, k_params, z_params,
                             b_params, tax_tuple, output_vars,
                             output_dir, print_moments=True)

    if plot_results:
        '''
        ------------------------------------------------------------------------
        Plot results
        ------------------------------------------------------------------------
        '''
        # plots.firm_plots(delta, k_params, z_params, output_vars, output_dir)

    # create dictionaries of output, params, grids, moments
    # output_dict = {'optK': optK, 'optI': optI, 'optB': optB, 'op': op,
    #                'e': e, 'eta': eta, 'VF': VF, 'PF_k': PF_k,
    #                'PF_b': PF_b, 'Gamma': Gamma}
    output_dict = {'optK': optK, 'optI': optI, 'optB': optB, 'VF': VF,
                   'PF_k': PF_k, 'PF_b': PF_b, 'Gamma': Gamma}
    param_dict = {'w': w0, 'r': r, 'tax_params': tax_params,
                  'hh_params': hh_params, 'firm_params': firm_params,
                  'fin_frictions': fin_frictions,
                  'grid_params': grid_params}
    grid_dict = {'zgrid': zgrid, 'sizez': sizez, 'Pi': Pi, 'kgrid': kgrid,
                 'sizek': sizek, 'kstar': kstar, 'bgrid': bgrid,
                 'sizeb': sizeb}
    model_out_dict = {'params': param_dict, 'grid': grid_dict,
                      'moments': model_moments, 'output': output_dict}
    # Save pickle of model output
    pkl_path = os.path.join(output_path, 'model_output.pkl')
    pickle.dump(model_out_dict, open(pkl_path, 'wb'))
