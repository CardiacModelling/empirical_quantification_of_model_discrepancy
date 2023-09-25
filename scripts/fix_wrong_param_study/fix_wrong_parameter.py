#!/usr/bin/env python3

# Synthetic data study where we fix the maximal conductance parameter to the
# wrong value and fit the model. Make this parameter pertubation larger and larger
# and see what thisdoes to parameter spread, predictions, etc...

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pints
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel

from threadpoolctl import threadpool_limits

import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')


T = 298
K_in = 120
K_out = 5

global E_rev
E_rev = common.calculate_reversal_potential(T=298, K_in=120, K_out=5)

pool_kws = {'maxtasksperchild': 1}


def main():
    parser = common.get_parser(
        data_reqd=False, description="Fit a given well to the data from each\
        of the protocols. Output the resulting parameters to a file for later use")

    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    # Number of repeats for each optimisation
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--method', default='CMAES', type=str)
    parser.add_argument('--figsize', '-f', nargs=2, type=int)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--protocols', default=common.get_protocol_list(), nargs='+')
    parser.add_argument('--noise', default=0.05, type=float)
    # Number of repeated samples of synthetic data
    parser.add_argument('--no_repeats', default=100, type=int)
    parser.add_argument('--no_parameter_steps', default=25, type=int)
    parser.add_argument('--fix_params', default=[], type=int, nargs='+')
    parser.add_argument('--sampling_period', default=0.1, type=float)

    global args
    args = parser.parse_args()

    if args.method == 'CMAES':
        args.method = pints.CMAES
    if args.method == 'NelderMead':
        args.method = pints.NelderMead

    global output_dir
    output_dir = common.setup_output_directory(args.output,
                                               "fix_wrong_parameter")

    # Set up logging
    logging.basicConfig(filename=os.path.join(output_dir,
                                              'log-fix_wrong_parameter'),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s\
                        %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    global model_class
    model_class = common.get_model_class(args.model)

    global true_params
    if args.use_parameter_file:
        true_params = pd.read_csv(args.use_parameter_file, header=None).values[0,:].astype(np.float64)
    else:
        true_params = model_class().get_default_parameters()

    param_labels = model_class().get_parameter_labels()

    global data_sets
    data_sets = generate_synthetic_data_sets(args.protocols,
                                             args.no_repeats,
                                             parameters=true_params,
                                             noise=args.noise,
                                             output_dir=output_dir)

    if not args.fix_params:
        args.fix_params = [len(param_labels) - 1]

    # Use multiprocessing to compute process multiple synthetic datasets in parallel
    tasks = []
    for fix_param in args.fix_params:
        sub_dir = os.path.join(output_dir, 'fitting', f"param_{fix_param}")
        if not os.path.exists(sub_dir):
            try:
                os.makedirs(sub_dir)
            except FileExistsError:
                pass

        for i, data in enumerate(data_sets[0]):
            for protocol in args.protocols:
                # model and dataset index
                tasks.append([args.model, i, fix_param, protocol])
    print(tasks, len(tasks))

    pool_size = min(args.cores, len(tasks))

    # Setup fitting dir
    fitting_dir = os.path.join(output_dir, 'fitting')
    if not os.path.exists(fitting_dir):
        try:
            os.makedirs(sub_dir)
        except FileExistsError:
            pass

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(fit_func, tasks)

    res = np.array(res)
    scores = res[:, :, -1]
    fitted_params = res[:, :, :res.shape[-1]]

    rows = []

    for task in tasks:
        for i in range(fitted_params.shape[1]):
            rows.append(task)

    res_df = pd.DataFrame(rows, columns=('model_class', 'dataset_index', 'fix_param', 'protocol'))

    for i, param_label in enumerate(model_class().get_parameter_labels()):
        res_df[param_label] = fitted_params[:, :, i].flatten(order='C')

    res_df['score'] = scores.flatten(order='C')

    res_df['well'] = res_df['dataset_index']

    res_df.to_csv(os.path.join(output_dir, 'results_df.csv'))

    datasets_df = []
    for protocol_index, protocol in enumerate(args.protocols):
        for i in range(args.no_repeats):
            datasets_df.append([protocol, protocol_index, i])

    datasets_df = pd.DataFrame(datasets_df, columns=('protocol', 'protocol_index', 'repeat'))

    predictions_df = compute_predictions_df(res_df, model_class, data_sets,
                                            datasets_df, args,
                                            output_dir=output_dir,
                                            true_params=true_params)
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'))


def fit_func(model_class_name, dataset_index, fix_param, protocol):
    sub_dir = os.path.join(output_dir, 'fitting', f"param_{fix_param}",
                           f"{protocol}", f"{dataset_index}")

    np.random.seed()

    if not os.path.exists(sub_dir):
        try:
            os.makedirs(sub_dir)
        except FileExistsError:
            pass

    protocol_index = args.protocols.index(protocol)
    times, data = data_sets[protocol_index][int(dataset_index)]

    voltage_func, _, protocol_desc = common.get_ramp_protocol_from_csv(protocol)

    model_class = common.get_model_class(model_class_name)
    default_fixed_param_val = true_params[fix_param]

    param_val_multipliers = np.concatenate([np.linspace(0, 2,
                                                        int(args.no_parameter_steps/2)
                                                        + 1),
                                            np.linspace(0,
                                                        -2,
                                                        int(args.no_parameter_steps/2)
                                                        + 1)
                                            ])
    param_val_multipliers = np.unique(param_val_multipliers)

    param_vals = 2**param_val_multipliers * default_fixed_param_val

    mm = model_class(voltage=voltage_func,
                     protocol_description=protocol_desc,
                     times=times,
                     E_rev=E_rev)

    params = true_params.copy()
    solver = mm.make_forward_solver_current()

    def score_func(parameters):
        pred = solver(parameters)
        return np.sum((pred - data)**2)

    fit_fig = plt.figure(figsize=args.figsize)
    fit_ax = fit_fig.gca()

    res = []
    for i, fix_param_val in enumerate(param_vals):
        default_guess = true_params.copy()
        default_guess[fix_param] = fix_param_val

        assert(len(params == len(true_params)))
        params[fix_param] = fix_param_val

        # Ensure that we use a good first guess
        pre_score1 = score_func(params)
        pre_score2 = score_func(default_guess)

        if pre_score2 < pre_score1:
            params = default_guess.copy()

        fitting_output_dir = os.path.join(sub_dir, f"{fix_param_val:.4e}")
        randomise_initial_guess = False
        repeats = 25 if i == 0 else args.repeats

        try:
            params, score, fitting_df = common.fit_model(mm, data,
                                                         fix_parameters=[fix_param],
                                                         randomise_initial_guess=randomise_initial_guess,
                                                         repeats=repeats,
                                                         max_iterations=args.max_iterations,
                                                         starting_parameters=params,
                                                         solver=solver,
                                                         method=args.method,
                                                         output_dir=fitting_output_dir,
                                                         return_fitting_df=True,
                                                         no_conductance_boundary=True,
                                                         threshold=1e-8
                                                         )

        except ValueError as e:
            logging.warning(str(e))
            score = np.inf

        if score > min(pre_score1, pre_score2):
            logging.warning("Fitting resulting in worse score than default/previous parameters."
                            + f"Refitting with random initial guesses\n ({score}"
                            + f" vs {min(pre_score1, pre_score2)})")

            params, score, fitting_df = common.fit_model(mm, data, fix_parameters=[fix_param],
                                                         repeats=repeats,
                                                         randomise_initial_guess=randomise_initial_guess,
                                                         max_iterations=args.max_iterations,
                                                         starting_parameters=default_guess,
                                                         solver=solver,
                                                         method=args.method,
                                                         output_dir=fitting_output_dir,
                                                         return_fitting_df=True,
                                                         no_conductance_boundary=True,
                                                         threshold=1e-8)

            append_df = pd.DataFrame([[*true_params.copy(), pre_score2]],
                                     columns=[*mm.get_parameter_labels(),
                                              'score'])
            fitting_df = pd.concat([fitting_df, append_df], ignore_index=True)

        score = np.sqrt(score / len(times))
        params[fix_param] = fix_param_val

        fit_ax.plot(times, data, color='grey')
        fit_ax.plot(times, solver(params), label='fitted_params')
        fit_ax.plot(times, solver(true_params), label='true_params')
        fit_ax.legend()

        if os.path.exists(sub_dir):
            fit_fig.savefig(os.path.join(fitting_output_dir, f"fit_{protocol}_{fix_param_val:.4e}_"
                                         + f"{dataset_index}.png"))
            plt.close(fit_fig)
            fitting_df.to_csv(os.path.join(fitting_output_dir, f"fit_{protocol}_{fix_param_val:.4e}_"
                                           + f"{dataset_index}.csv"))

        fit_ax.cla()

        res.append(np.append(params.copy(), score))

    res = np.vstack(res)
    return res


def generate_synthetic_data_sets(protocols, n_repeats, parameters=None,
                                 noise=None,
                                 output_dir=None):

    if not noise:
        noise = args.noise

    list_of_data_sets = []
    for protocol in protocols:
        prot, _times, desc = common.get_ramp_protocol_from_csv(protocol)

        no_samples = int((_times[-1] - _times[0]) / args.sampling_period) + 1
        times = np.linspace(_times[0], (no_samples - 1) * args.sampling_period,
                            no_samples)

        model_class = common.get_model_class(args.model)
        model = model_class(voltage=prot, times=times, E_rev=E_rev,
                            parameters=parameters, protocol_description=desc)

        mean = model.make_forward_solver_current()()

        data_sets = [(times, np.random.normal(mean, noise, times.shape)) for i in
                     range(n_repeats)]

        for _, current in data_sets:
            print('RMSE at true data is ',
                  np.sqrt(np.mean((mean - current)**2)))

        if output_dir:
            fig = plt.figure(figsize=args.figsize)
            ax = fig.gca()
            for i, data_set in enumerate(data_sets):
                ax.plot(times, data_set[1], color='grey')
                ax.plot(times, mean)
                ax.set_xlabel('time / ms')
                ax.set_ylabel('current / nA')
                fig.savefig(os.path.join(output_dir, f"synthetic_data_{protocol}_{i}.png"), dpi=300)
                ax.cla()

                data_df = pd.DataFrame(np.vstack(data_set).T, columns=['time / ms', 'current / nA'])
                data_df.to_csv(os.path.join(output_dir, f"synthetic_data_{protocol}_{i}.csv"))

            plt.close(fig)

        list_of_data_sets.append(data_sets)
    return list_of_data_sets


def compute_predictions_df(params_df, model_class, datasets, datasets_df, args,
                           label='predictions', output_dir=None,
                           fix_params=None, true_params=None):

    if fix_params is None:
        fix_params = args.fix_params

    if output_dir:
        predictions_dir = os.path.join(output_dir, label)

        if not os.path.exists(predictions_dir):
            try:
                os.makedirs(predictions_dir)
            except FileExistsError:
                pass

        trace_fig = plt.figure(figsize=args.figsize)
        trace_axs = trace_fig.subplots(2)
        all_models_fig = plt.figure(figsize=args.figsize)
        all_models_axs = all_models_fig.subplots(2)

    predictions_df = []
    protocols_list = datasets_df['protocol'].unique()

    param_labels = model_class().get_parameter_labels()

    for fix_param in fix_params:
        fixed_param_label = param_labels[fix_param]

        for sim_protocol in protocols_list:
            prot_func, _, desc = common.get_ramp_protocol_from_csv(sim_protocol)
            protocol_index = datasets_df[datasets_df.protocol == sim_protocol]['protocol_index'].values[0]
            times = datasets[protocol_index][0][0]

            model = model_class(prot_func,
                                times=times,
                                E_rev=E_rev,
                                protocol_description=desc)

            voltages = np.array([prot_func(t) for t in times])

            if output_dir:
                colours = sns.color_palette('husl', len(params_df['protocol'].unique()))

            solver = model.make_forward_solver_current(njitted=True)
            default_prediction = solver(true_params)

            for well in params_df['well'].unique():
                full_data = datasets[protocol_index][int(well)][1]
                data = full_data

                for i, protocol_fitted in enumerate(params_df['protocol'].unique()):

                    if output_dir:
                        all_models_axs[1].plot(times, voltages,
                                               label=protocol_fitted, color=colours[i])

                    for val in params_df[fixed_param_label].unique():
                        df = params_df[(params_df.well == well) &
                                       (params_df.protocol == protocol_fitted) &
                                       (params_df[fixed_param_label] == val) &
                                       (params_df.fix_param == fix_param)]

                        if df.empty:
                            continue

                        assert(len(df.index) == 1)
                        params = df[param_labels].values.astype(np.float64).flatten()

                        if output_dir:
                            sub_dir = os.path.join(predictions_dir, f"param_{fix_param}", f"{well}_{sim_protocol}_predictions")
                            if not os.path.exists(sub_dir):
                                try:
                                    os.makedirs(sub_dir)
                                except FileExistsError:
                                    pass

                        prediction = solver(params)

                        score = np.sqrt(np.mean((data - prediction)**2))

                        RMSE_DGP = np.sqrt(np.mean((prediction - default_prediction)**2))

                        predictions_df.append((well, fix_param, protocol_fitted, sim_protocol,
                                               score, RMSE_DGP, *params))

                        if not np.all(np.isfinite(prediction)):
                            logging.warning(f"running {sim_protocol} with parameters\
                            from {protocol_fitted} gave non-finite values")
                        elif output_dir:
                            # Output trace
                            trace_axs[0].plot(times, prediction, label='prediction')
                            trace_axs[0].plot(times, solver(true_params),
                                              label='true DGP')

                            trace_axs[1].set_xlabel("time / ms")
                            trace_axs[0].set_ylabel("current / nA")
                            trace_axs[0].plot(times, data, label='data', alpha=0.25, color='grey')
                            trace_axs[0].legend()
                            trace_axs[1].plot(times, voltages)
                            trace_axs[1].set_ylabel('voltage / mV')
                            fname = f"fitted_to_{protocol_fitted}_{val:.4f}.png" if protocol_fitted != sim_protocol else f"fit_{val:.4f}.png"
                            trace_fig.savefig(os.path.join(sub_dir, fname))

                            trace_axs[0].cla()
                            trace_axs[0].plot(times,
                                              prediction - default_prediction)
                            trace_axs[0].set_yscale('log')
                            fname = f"fitted_to_{protocol_fitted}_{val:.4f}_error.png" if protocol_fitted != sim_protocol else f"fit_{val:.4f}_error.png"
                            trace_fig.savefig(os.path.join(sub_dir, fname))

                            for ax in trace_axs:
                                ax.cla()

                        if output_dir:
                            all_models_axs[0].plot(times, prediction, label=protocol_fitted,
                                                   color=colours[i])

            if output_dir:
                all_models_fig.savefig(os.path.join(sub_dir, "all_predictions.png"))

                for ax in all_models_axs:
                    ax.cla()

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well', 'fixed_param', 'fitting_protocol',
                                                                      'validation_protocol',
                                                                      'score', 'RMSE_DGP'] + param_labels)
    predictions_df['RMSE'] = predictions_df['score']
    print(predictions_df)

    if output_dir:
        plt.close(trace_fig)
        plt.close(all_models_fig)

    return predictions_df


if __name__ == "__main__":
    multiprocessing.freeze_support()
    with threadpool_limits(limits=1, user_api='blas'):
        main()
