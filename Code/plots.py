'''
------------------------------------------------------------------------
This module contains a function to plot results of the firm model

* firm_plots()
------------------------------------------------------------------------
'''

# imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import numpy as np


def firm_plots(delta, k_params, z_params, output_vars, output_dir):
    '''
    ------------------------------------------------------------------------
    Plot Results
    ------------------------------------------------------------------------
    '''

    # unpack tuples
    K, sizek, dens, kstar = k_params
    Pi, z, sizez = z_params
    optK, optI, op, e, eta, VF, PF, Gamma = output_vars

    # Plot value function
    # plt.figure()
    fig, ax = plt.subplots()
    ax.plot(K, VF[0, :], 'k--', label='z = ' + str(z[0]))
    ax.plot(K, VF[(sizez - 1) // 2, :], 'k:', label='z = ' + str(z[(sizez - 1)
                                                                      // 2]))
    ax.plot(K, VF[-1, :], 'k', label='z = ' + str(z[-1]))
    # Now add the legend with some customizations.
    legend = ax.legend(loc='lower right', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.xlabel('Size of Capital Stock')
    plt.ylabel('Value Function')
    plt.title('Value Function - stochastic firm w/ adjustment costs')
    output_path = os.path.join(output_dir, 'V_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


    # Plot optimal capital stock rule as a function of firm size
    # plt.figure()
    fig, ax = plt.subplots()
    ax.plot(K, optK[0, :], 'k--', label='z = ' + str(z[0]))
    ax.plot(K, optK[(sizez - 1) // 2, :], 'k:', label='z = ' + str(z[(sizez - 1)
                                                                        // 2]))
    ax.plot(K, optK[-1, :], 'k', label='z = ' + str(z[-1]))
    ax.plot(K, K, 'k:', label='45 degree line')
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper left', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.xlabel('Size of Capital Stock')
    plt.ylabel('Optimal Choice of Capital Next Period')
    plt.title('Policy Function, Next Period Capital - stochastic firm w/ ' +
              'adjustment costs')
    output_path = os.path.join(output_dir, 'Kprime_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


    # Plot operating profits as a function of firm size
    # plt.figure()
    fig, ax = plt.subplots()
    ax.plot(K, op[0, :], 'k--', label='z = ' + str(z[0]))
    ax.plot(K, op[(sizez - 1) // 2, :], 'k:', label='z = ' + str(z[(sizez - 1)
                                                                    // 2]))
    ax.plot(K, op[-1, :], 'k', label='z = ' + str(z[-1]))
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper left', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.xlabel('Size of Capital Stock')
    plt.ylabel('Operating Profits')
    plt.title('Operating Profits as a Function of Firm Size')
    output_path = os.path.join(output_dir, 'Profits_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


    # Plot investment rule as a function of firm size
    # plt.figure()
    fig, ax = plt.subplots()
    ax.plot(K, optI[(sizez - 1) // 2, :]/K, 'k--', label='Investment rate')
    ax.plot(K, np.ones(sizek)*delta, 'k:', label='Depreciation rate')
    # Now add the legend with some customizations.
    legend = ax.legend(loc='upper left', shadow=True)
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.xlabel('Size of Capital Stock')
    plt.ylabel('Optimal Investment Rate')
    plt.title('Policy Function, Investment - stochastic firm w/ adjustment ' +
              'costs')
    output_path = os.path.join(output_dir, 'invest_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    # Plot investment rule as a function of productivity
    # plt.figure()
    fig, ax = plt.subplots()
    ind = np.argmin(np.absolute(K - kstar))  # find where kstar is in grid
    ax.plot(z, optI[:, ind - dens * 5] / K[ind - dens * 5], 'k', label='k = ' +
            str(K[ind - dens * 5]))
    ax.plot(z, optI[:, ind] / K[ind], 'k:', label='k = ' + str(K[ind]))
    ax.plot(z, optI[:, ind + dens * 5] / K[ind + dens * 5], 'k--', label='k = '
            + str(K[ind + dens * 5]))
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width
    plt.xlabel('Productivity')
    plt.ylabel('Optimal Investment Rate')
    plt.title('Policy Function, Investment - stochastic firm w/ adjustment ' +
              'costs')
    output_path = os.path.join(output_dir, 'invest_z_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    # Plot the stationary distribution
    fig, ax = plt.subplots()
    ax.plot(K, Gamma.sum(axis=0))
    plt.xlabel('Size of Capital Stock')
    plt.ylabel('Density')
    plt.title('Stationary Distribution over Capital')
    output_path = os.path.join(output_dir, 'SD_k_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    # Plot the stationary distribution
    fig, ax = plt.subplots()
    ax.plot(np.log(z), Gamma.sum(axis=1))
    plt.xlabel('Productivity')
    plt.ylabel('Density')
    plt.title('Stationary Distribution over Productivity')
    output_path = os.path.join(output_dir, 'SD_z_firm7')
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    # Stationary distribution in 3D
    zmat, kmat = np.meshgrid(K, np.log(z))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kmat, zmat, Gamma, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.view_init(elev=20., azim=20)  # to rotate plot for better view
    ax.set_xlabel(r'Log Productivity')
    ax.set_ylabel(r'Capital Stock')
    ax.set_zlabel(r'Density')
    output_path = os.path.join(output_dir, 'SD_3D_firm7')
    plt.savefig(output_path)
    # plt.show()
    plt.close()
