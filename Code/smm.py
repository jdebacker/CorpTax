import numpy as np
import numdifftools as nd
from execute import solve_GE, solve_PE


def objective_func(params_to_estimate, data_moments, W, w0, tax_params,
                   hh_params, firm_params, fin_frictions, grid_params,
                   output_dir, guid, plot_results):
    '''
    Compute the value of the statistical objective function for SMM
    '''
    model_moments = moment_func(params_to_estimate, w0, tax_params,
                       hh_params, firm_params, fin_frictions, grid_params,
                       output_dir, guid, plot_results)
    distance = np.dot(
        np.dot((model_moments - data_moments), W),
               (model_moments - data_moments))
    return distance


def optimal_weight(data):
    '''
    Calculate optimal weight matrix.  This will use the VCV of the
    moments found by bootstrapping the data.

    Gourieroux, Monfort, and Renault, (1993), Journal of Applied Econometrics
    '''
    simM = 1000  # number of iterations for bootstrapping
    numObs = data.shape[0]
    numMoments = Q  # need to input something telling number and which moments to compute
    bootM = np.empty([simM, numMoments])  # moments at each iteration
    for i in range(simM):
        for j in range(numObs):
            random_draw = np.ceil(np.random.uniform * numObs)
            data_boot[j, :] = data[random_draw, :]

        # Calculate moments from one random sample of size of original sample
        # Maybe call a function to do this - which can vary depending on moments used in estimation

        bootM[i, :] = []

    boot_mean = np.mean(bootM, axis=0)
    vcv = (np.dot(np.transpose(bootM - boot_mean),
                  np.transpose(bootM - boot_mean)) / simM)
    W = np.lingalg.inv((1 + (1 / simM)) * vcv)

    return W


def std_errors(theta_hat, W, args):
    '''
    Compute standard errors around parameter estimates.
    '''
    # find gradient vector evaluated at estimated values
    deriv_moments = nd.Jacobian(moment_func, *args)(theta_hat)
    # compute VCV matrix with vector of derivatives
    vcv = np.lingalg.inv(np.dot(np.dot(np.transpose(deriv_moments), W),
                                deriv_moments))
    # find standard errors are the sqrt of the diagonal VCV elements
    std_errors = np.diag(vcv) ** (1 / 2)

    return std_errors


def moment_func(params_to_estimate, args):
    '''
    Compute moments given parameters to estimate
    '''
    (w0, tax_params, hh_params, firm_params, fin_frictions, grid_params,
     output_dir, guid, plot_results) = args
    # put parameters to be estimated in dictionaries to pass
    psi = params_to_estimate[0]
    fixed_cost = params_to_estimate[1]
    firm_params['psi'] = psi
    firm_params['fixed_cost'] = fixed_cost

    # model_moments = solve_GE(w0, tax_params, hh_params,
    #                          firm_params, fin_frictions,
    #                          grid_params, output_dir, guid, plot_results)
    model_moments = solve_PE(w0, tax_params, hh_params,
                             firm_params, fin_frictions,
                             grid_params, output_dir, guid, plot_results)

    return model_moments
