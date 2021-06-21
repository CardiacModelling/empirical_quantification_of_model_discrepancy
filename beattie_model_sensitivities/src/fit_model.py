#!/usr/bin/env python3

import os
import math
import pints
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
import scipy.integrate as integrate
import symengine as se
import matplotlib.pyplot as plt
import matplotlib
import argparse

from   settings import Params
from   sensitivity_equations import GetSensitivityEquations, CreateSymbols

class PintsWrapper(pints.ForwardModelS1):

    def __init__(self, settings, args, times_to_use):
        par = Params()
        self.times_to_use = times_to_use
        self.starting_parameters = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]
        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(settings)

        # Choose starting parameters (from J Physiol paper)
        para = [2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524]

        # Create symbols for symbolic functions
        p, y, v = CreateSymbols(par)

        # Define system equations and initial conditions
        k1 = p[0] * se.exp(p[1] * v)
        k2 = p[2] * se.exp(-p[3] * v)
        k3 = p[4] * se.exp(p[5] * v)
        k4 = p[6] * se.exp(-p[7] * v)

        # Write in matrix form taking y = ([C], [O], [I])^T

        A = se.Matrix([[-k1 - k3 - k4, k2 -  k4, -k4], [k1, -k2 - k3, k4], [-k1, k3 - k1, -k2 - k4 - k1]])
        B = se.Matrix([k4, 0, k1])

        rhs = np.array(A * y + B)

        self.funcs = GetSensitivityEquations(par, p, y, v, A, B, para, times_to_use, sine_wave=args.sine_wave)

    def n_parameters(self):
        return len(self.starting_parameters)

    def simulate(self, parameters, times):
        ret = self.funcs.SimulateForwardModel(parameters)
        # print(ret.shape)
        return ret

    def simulateS1(self, parameters, times):
        return self.funcs.SimulateForwardModelSensitivites(parameters, data), self.times_to_use, 1



class Boundaries(pints.Boundaries):
    def check(self, parameters):
        '''Check that each rate constant lies in the range 1.67E-5 < A*exp(B*V) < 1E3
        '''

        for i in range(0, 4):
            alpha = parameters[2*i]
            beta  = parameters[2*i + 1]

            vals = [0,0]
            vals[0] = alpha * np.exp(beta * -90 * 1E-3)
            vals[1] = alpha * np.exp(beta *  50 * 1E-3)

            for val in vals:
                if val < 1.67E-5 or val > 1E3:
                    return False
        # Check maximal conductance
        if parameters[8] > 0 and parameters[8] < 2:
            return True
        else:
            return False

    def n_parameters(self):
        return 9

def extract_times(lst, time_ranges, step):
    """
    Take values from a list, lst which are indexes between the upper and lower
    bounds provided by time_ranges. Each element of time_ranges specifies an
    upper and lower bound.

    Returns a 2d numpy array containing all of the relevant data points
    """
    ret_lst = []
    for time_range in time_ranges:
        ret_lst.extend(lst[time_range[0]:time_range[1]:step].tolist())
    return np.array(ret_lst)

def main():
    #constants
    indices_to_use = [[1,2499], [2549,2999], [3049,4999], [5049,14999], [15049,19999], [20049,29999], [30049,64999], [65049,69999], [70049,-1]]
    starting_parameters = [3.87068845e-04, 5.88028759e-02, 6.46971727e-05, 4.87408447e-02, 8.03073893e-02, 7.36295506e-03, 5.32908518e-03, 3.32254316e-02, 6.56614672e-02]

    plt.rcParams['axes.axisbelow'] = True

    # Check input arguments
    parser = argparse.ArgumentParser(
        description='Plot sensitivities of the Beattie model')
    parser.add_argument("data_file_path", help="path to csv data for the model to be fit to")
    parser.add_argument("-s", "--sine_wave", action='store_true', help="whether or not to use sine wave protocol",
        default=False)
    parser.add_argument("-p", "--plot", action='store_true', help="whether to plot figures or just save",
        default=False)
    parser.add_argument("--dpi", type=int, default=100, help="what DPI to use for figures")
    parser.add_argument("-o", "--output", type=str, default="output", help="The directory to output figures and data to")
    args = parser.parse_args()

    data  = pd.read_csv(args.data_file_path, delim_whitespace=True)

    print("outputting to {}".format(args.output))

    # Create output directory
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if not os.path.exists(args.data_file_path):
        print("Input file not provided. Doing nothing.")
        return

    par = Params()

    skip = int(par.timestep/0.1)

    dat = extract_times(data.values, indices_to_use, skip)
    times=dat[:,0]
    values=dat[:,1]

    model = PintsWrapper(par, args, times)

    current = model.simulate(starting_parameters, times)
    problem = pints.SingleOutputProblem(model, times, values)
    error = pints.SumOfSquaresError(problem)
    boundaries  = Boundaries()
    x0 = np.array([0.1]*9)
    # found_parameters, found_value = pints.optimise(error, starting_parameters, boundaries=boundaries)
    # found_parameters = np.array([2.26E-04, 0.0699, 3.45E-05, 0.05462, 0.0873, 8.92E-03, 5.150E-3, 0.03158, 0.1524])
    # print("finished! found parameters : {} ".format(found_parameters, found_value))
    found_parameters = np.array([1.87451202e-03, 1.36254787e-02, 1.68324276e-05, 8.77532812e-02, 5.67114947e-02, 2.66069061e-02, 1.21159939e-03, 7.96959925e-03, 5.49219181e-02])
    # found_value=100

    # Find error sensitivities
    funcs = model.funcs
    current, sens = funcs.SimulateForwardModelSensitivities(found_parameters)
    sens = sens * found_parameters[:, None]

    plt.plot(times, current)
    for vec in sens:
        plt.plot(times, vec)

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.output, "maximal_likelihood_sensitivities"))

    # Compute the Fischer information matrix
    FIM = sens @ sens.T
    cov = FIM**-1
    eigvals = np.linalg.eigvals(FIM)
    for i in range(0, par.n_params):
        for j in range(i+1, par.n_params):
            parameters_to_view = np.array([i,j])
            sub_cov = cov[parameters_to_view[:,None], parameters_to_view]
            eigen_val, eigen_vec = np.linalg.eigh(sub_cov)
            eigen_val=eigen_val.real
            if eigen_val[0] > 0 and eigen_val[1] > 0:
                print("COV_{},{} : well defined".format(i, j))
                cov_ellipse(sub_cov, i, j)
            else:
                print("COV_{},{} : negative eigenvalue".format(i,j))


    print('Eigenvalues of FIM:\n{}'.format(eigvals))
    print("Covariance matrix is: \n{}".format(cov))

def cov_ellipse(cov, i, j, q=None, nsig=1, **kwargs):
    """
    Parameters
    ----------
    copied from stackoverflow


    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * scipy.stats.norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = scipy.stats.chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    print("width, height, rotation = {}, {}, {}".format(width, height,rotation))

    fig = plt.figure(0)
    e = matplotlib.patches.Ellipse([0,0], width[0], height[0], rotation)
    ax = fig.add_subplot(111, aspect='equal')
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_facecolor([0.25, 0.25, 0])
    ax.set_xlim(width[0])
    ax.set_ylim(height[0])

    if args.plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.output, "covariance_plot_{}_{}".format(i,j)))


if __name__ == "__main__":
    main()
    print("done")