#! /usr/bin/env python3

from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.KempModel import KempModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel

import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import multiprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--model", default='Beattie')
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--prefix", default='synthetic')
    parser.add_argument('-P', '--protocols', nargs='+', default=None)
    parser.add_argument('-p', '--plot', action='store_true', default=False)
    parser.add_argument('--noise', default=0.05, type=float)
    parser.add_argument('--E_rev', '-e', default=None)
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--repeats', '-r', default=10, type=int)

    parser.add_argument('--sampling_period', default=0.1, type=float)

    global args
    args = parser.parse_args()

    E_rev = common.calculate_reversal_potential()\
        if args.E_rev is None\
        else args.E_rev

    global output_dir
    output_dir = common.setup_output_directory(args.output, 'synthetic_data_%s' % args.model)

    global sigma
    sigma = args.noise

    global model_class
    model_class = common.get_model_class(args.model)

    global parameters
    if args.parameters is not None:
        param_labels = model_class().get_parameter_labels()
        parameters = pd.read_csv(args.parameters)[param_labels].values[0, :]
    else:
        parameters = model_class().get_default_parameters()

    protocols = common.get_protocol_list()\
        if args.protocols is None\
        else args.protocols

    tasks = [(p, args.repeats, model_class, E_rev, sigma, output_dir) for p in
             protocols]
    with multiprocessing.Pool(args.cpus) as pool:
        pool.starmap(generate_data, tasks)


def generate_data(protocol, no_repeats, model_class, E_rev, sigma, output_dir,
                  sampling_period=None, figsize=None, noise=None, prefix=None,
                  plot=None, parameters=None):

    if sampling_period is None:
        sampling_period = args.sampling_period
    if figsize is None:
        figsize = (12, 10)
    if noise is None:
        noise = args.noise
    if prefix is None:
        prefix = args.prefix
    if plot is None:
        plot = args.plot

    prot_func, _times, desc = common.get_ramp_protocol_from_csv(protocol)
    print('generating data')

    no_samples = int((_times[-1] - _times[0]) / sampling_period) + 1

    times = np.linspace(_times[0], (no_samples - 1) * sampling_period,
                        no_samples)

    times_df = pd.DataFrame(times.T, columns=('time',))
    times_df.to_csv(os.path.join(output_dir, f"{prefix}-{protocol}-times.csv"))
    model = model_class(voltage=prot_func, times=times, E_rev=E_rev,
                        parameters=parameters, protocol_description=desc)

    mean = model.make_forward_solver_current(njitted=True)()

    if not np.all(np.isfinite(mean)):
        print('inf times', times[np.argwhere(~np.isfinite(mean))])
        raise Exception()

    for repeat in range(no_repeats):
        data = np.random.normal(mean, sigma, times.shape)

        # Output data
        out_fname = os.path.join(output_dir, f"{prefix}-{protocol}-{repeat}.csv")
        pd.DataFrame(data.T, columns=('current',)).to_csv(out_fname)

        if plot:
            fig = plt.figure(figsize=(14, 12))
            axs = fig.subplots(3)
            axs[0].plot(times, mean, label='mean')
            axs[0].plot(times, data, label='data', color='grey', alpha=0.5)
            axs[1].plot(times, model.GetStateVariables())
            axs[1].legend(model.state_labels)
            axs[0].legend()
            axs[1].set_xlabel('time / ms')
            axs[0].set_ylabel('current / nA')
            axs[2].plot(times, [model.voltage(t) for t in times], label='voltage / mV')
            fig.savefig(os.path.join(output_dir, f"plot-{protocol}plot-{repeat}.png"))
            plt.close(fig)


if __name__ == "__main__":
    main()
