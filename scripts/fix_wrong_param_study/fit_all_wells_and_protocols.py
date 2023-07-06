#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel

import matplotlib
matplotlib.use('agg')

import os
import pandas as pd
import numpy as np

pool_kws = {'maxtasksperchild': 1}


def fit_func(protocol, well, model_class, default_parameters=None, E_rev=None,
             randomise_initial_guess=True, prefix="", repeats=1):
    this_output_dir = os.path.join(output_dir, f"{prefix}{protocol}_{well}")

    res_df = common.fit_well_data(model_class, well, protocol,
                                  args.data_directory, args.max_iterations,
                                  output_dir=this_output_dir,
                                  default_parameters=default_parameters,
                                  removal_duration=args.removal_duration,
                                  repeats=repeats,
                                  infer_E_rev=infer_E_rev,
                                  experiment_name=args.experiment_name,
                                  E_rev=E_rev,
                                  randomise_initial_guess=randomise_initial_guess,
                                  solver_type=args.solver_type,
                                  threshold=1e-8,
                                  )

    res_df['well'] = well
    res_df['protocol'] = protocol

    print(res_df)
    return res_df


def main():
    Erev = common.calculate_reversal_potential()

    parser = common.get_parser(
        data_reqd=True, description="Fit a given well to the data from each\
        of the protocols. Output the resulting parameters to a file for later use")

    parser.add_argument('--max_iterations', '-i', type=int, default=100000)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--dont_randomise_initial_guess', action='store_true')
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--cores', '-c', default=1, type=int)
    parser.add_argument('--model', '-m', default='Beattie', type=str)
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, type=int)
    parser.add_argument('--chain_length', '-l', default=500, type=int)
    parser.add_argument('--figsize', '-f', help='mcmc chains to run', type=int)
    parser.add_argument('--use_parameter_file')
    parser.add_argument('--solver_type')
    parser.add_argument('--selection_file')
    parser.add_argument('--ignore_protocols', nargs='+', default=[])
    parser.add_argument('--ignore_wells', nargs='+', default=[])

    global args
    args = parser.parse_args()

    global output_dir
    output_dir = args.output

    global experiment_name
    experiment_name = args.experiment_name

    if args.selection_file:
        with open(args.selection_file) as fin:
            selected_wells = fin.read().splitlines()
    else:
        selected_wells = None

    output_dir = common.setup_output_directory(args.output, f"fitting_{args.removal_duration:.2f}_removed_{args.model}")

    global model_class
    model_class = common.get_model_class(args.model)

    if len(args.wells) == 0:
        args.wells = common.get_all_wells_in_directory(args.data_directory, experiment_name)

    if len(args.protocols) == 0:
        protocols = common.get_protocol_list()
    else:
        protocols = args.protocols

    if args.selection_file:
        args.wells = [well for well in args.wells if well in selected_wells]

    print(args.wells, protocols)

    if args.use_parameter_file:
        # Here we can use previous results to refit. Just use the best
        # parameters for each validation protocol as the initial guess
        best_params_df = pd.read_csv(args.use_parameter_file)
        if 'validation_protocol' in best_params_df:
            protocol_label = 'validation_protocol'
            sweep_label = 'prediction_sweep'
        else:
            protocol_label = 'protocol'
            sweep_label = 'sweep'

        best_params_df = get_best_params(best_params_df,
                                         protocol_label=protocol_label,
                                         sweep_label=sweep_label)
        assert(args.dont_randomise_initial_guess)

    else:
        best_params_df = None

    tasks = []
    protocols_list = []

    regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z|0-9]*).csv$")
    param_labels = model_class().get_parameter_labels()
    for f in filter(regex.match, os.listdir(args.data_directory)):
        groups = re.search(regex, f).groups()
        protocol = groups[0]
        well = groups[1]
        if protocol not in protocols or well not in args.wells:
            continue
        if protocol in args.ignore_protocols or well in args.ignore_wells:
            continue

        if best_params_df is not None:
            parameter_row = best_params_df[(best_params_df.well.astype(str) == str(well))
                                           & (best_params_df[protocol_label] == protocol)].head(1)
            starting_parameters = parameter_row[param_labels].values.flatten().astype(np.float64)
        else:
            starting_parameters = None

        for i in range(args.repeats):
            prefix = f"rep_{i}_"
            tasks.append([protocol, well, model_class, starting_parameters,
                          Erev, not args.dont_randomise_initial_guess, prefix])

        protocols_list.append(protocol)

    print(f"fitting tasks are {tasks}")

    assert len(tasks) > 0, "no valid protocol/well combinations provided"

    protocols_list = np.unique(protocols_list)

    pool_size = min(args.cores, len(tasks))

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(fit_func, tasks)

    print(res)
    fitting_df = pd.concat(res, ignore_index=True)

    print("=============\nfinished fitting first round\n=============")

    # wells_rep = [task[1] for task in tasks]
    # protocols_rep = [task[0] for task in tasks]

    fitting_df.to_csv(os.path.join(output_dir, f"fitting.csv"))

    params_df = get_best_params(fitting_df)
    params_df.to_csv(os.path.join(output_dir, f"best_fitting.csv"))

    print(params_df)
    predictions_df = compute_predictions_df(params_df, output_dir,
                                            f"predictions", args=args,
                                            model_class=model_class)

    # Plot predictions
    predictions_df.to_csv(os.path.join(output_dir, f"predictions_df.csv"))

    # Select best parameters for each protocol
    best_params_df_rows = []
    print(predictions_df)
    for well in predictions_df.well.unique():
        for validation_protocol in predictions_df['validation_protocol'].unique():
            sub_df = predictions_df[(predictions_df.validation_protocol ==
                                     validation_protocol) & (predictions_df.well == well)]

            best_param_row = sub_df[sub_df.score == sub_df['score'].min()].head(1).copy()
            best_params_df_rows.append(best_param_row)

    best_params_df = pd.concat(best_params_df_rows, ignore_index=True)
    print(best_params_df)


def compute_predictions_df(params_df, output_dir, label='predictions',
                           model_class=None, fix_EKr=None,
                           args=None,
                           protocol_label='protocol',
                           sweep_label='sweep'):

    param_labels = model_class().get_parameter_labels()
    params_df = get_best_params(params_df, protocol_label=protocol_label,
                                sweep_label=sweep_label)
    predictions_dir = os.path.join(output_dir, label)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    predictions_df = []
    protocols_list = list(params_df['protocol'].unique())
    if 'longap' not in protocols_list:
        protocols_list.append('longap')

    trace_fig = plt.figure(figsize=args.figsize)
    trace_axs = trace_fig.subplots(2)

    all_models_fig = plt.figure(figsize=args.figsize)
    all_models_axs = all_models_fig.subplots(2)

    for sim_protocol in protocols_list:
        prot_func, times, desc = common.get_ramp_protocol_from_csv(sim_protocol)

        full_times = pd.read_csv(os.path.join(args.data_directory,
                                              f"{args.experiment_name}-{sim_protocol}-times.csv"))['time'].values.flatten()

        voltages = np.array([prot_func(t) for t in full_times])

        colours = sns.color_palette('husl', len(params_df['protocol'].unique()))

        for well in params_df['well'].unique():
            try:
                data = common.get_data(well, sim_protocol,
                                       args.data_directory,
                                       experiment_name=args.experiment_name
                                       )

            except (FileNotFoundError, StopIteration) as exc:
                print(str(exc))
                continue

            model = model_class(prot_func,
                                times=full_times,
                                E_rev=common.calculate_reversal_potential(),
                                protocol_description=desc)

            # Probably not worth compiling solver
            solver = model.make_forward_solver_of_type(args.solver_type, njitted=False)

            for i, protocol_fitted in enumerate(params_df['protocol'].unique()):
                print(protocol_fitted)
                # Get parameters
                df = params_df[params_df.well == well].copy()
                df = df[df.protocol == protocol_fitted]
                if df.empty:
                    print(protocol_fitted, well)
                    print(params_df)
                    continue

                print(df)

                params = df.iloc[0][param_labels].values\
                                                    .astype(np.float64)\
                                                    .flatten()
                try:
                    fitting_data = pd.read_csv(
                        os.path.join(args.data_directory,
                                        f"{args.experiment_name}-{protocol_fitted}-{well}.csv"))
                except FileNotFoundError as e:
                    print(str(e))
                    continue

                # fitting_current = fitting_data['current'].values.flatten()
                # fitting_times = fitting_data['time'].values.flatten()

                subdir_name = f"{well}_{sim_protocol}_predictions"
                sub_dir = os.path.join(predictions_dir, subdir_name)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)

                full_prediction = solver(params)
                prediction = full_prediction

                score = np.sqrt(np.mean((data - prediction)**2))
                predictions_df.append((well, protocol_fitted, sim_protocol,
                                       score, *params))

                if not np.all(np.isfinite(prediction)):
                    logging.warning(f"running {sim_protocol} with parameters\
                    from {protocol_fitted} gave non-finite values")
                else:
                    # Output trace
                    trace_axs[0].plot(full_times, full_prediction, label='prediction')

                    trace_axs[1].set_xlabel("time / ms")
                    trace_axs[0].set_ylabel("current / nA")
                    trace_axs[0].plot(times, data, label='data', alpha=0.25, color='grey')
                    trace_axs[0].legend()
                    trace_axs[1].plot(full_times, voltages)
                    trace_axs[1].set_ylabel('voltage / mV')
                    fname = f"fitted_to_{protocol_fitted}.png" if protocol_fitted != sim_protocol else "fit.png"

                    handles, labels = trace_axs[1].get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())

                    trace_fig.savefig(os.path.join(sub_dir, fname))

                    for ax in trace_axs:
                        ax.cla()

                    all_models_axs[0].plot(full_times, full_prediction,
                                            label=f"{protocol_fitted}",
                                            color=colours[i])

            all_models_axs[1].set_xlabel("time / ms")
            all_models_axs[0].set_ylabel("current / nA")
            all_models_axs[0].plot(times, data, color='grey', alpha=0.5, label='data')
            # all_models_axs[0].legend()
            all_models_axs[0].set_title(f"{well} {sim_protocol} fits comparison")
            all_models_axs[0].set_ylabel("Current / nA")

            all_models_axs[1].plot(full_times, voltages)
            all_models_axs[1].set_ylabel('voltage / mV')

            all_models_fig.savefig(os.path.join(sub_dir, "all_fits.png"))

            for ax in all_models_axs:
                ax.cla()

    predictions_df = pd.DataFrame(np.array(predictions_df), columns=['well',
                                                                     'fitting_protocol',
                                                                     'validation_protocol',
                                                                     'score'] +
                                  param_labels)
    predictions_df['RMSE'] = predictions_df['score']
    # predictions_df['sweep'] = predictions_df.fitting_sweep

    plt.close(all_models_fig)
    plt.close(trace_fig)

    return predictions_df


def get_best_params(fitting_df, protocol_label='protocol', sweep_label='sweep'):
    best_params = []

    # Ensure score is a float - may be read from csv file
    fitting_df['score'] = fitting_df['score'].astype(np.float64)
    fitting_df = fitting_df[np.isfinite(fitting_df['score'])].copy()

    if sweep_label not in fitting_df.columns and sweep_label != 'sweep':
        raise Exception(f"{sweep_label} is not a column in fitting_df")
    else:
        fitting_df['sweep'] = -1

    for protocol in fitting_df[protocol_label].unique():
        for well in fitting_df['well'].unique():
            for sweep in fitting_df[sweep_label].unique():
                sub_df = fitting_df[(fitting_df['well'] == well)
                                    & (fitting_df[protocol_label] == protocol)].copy()
                sub_df = sub_df[sub_df.sweep == sweep]
                sub_df = sub_df.dropna()
                # Get index of min score
                if len(sub_df.index) == 0:
                    continue
                best_params.append(sub_df[sub_df.score == sub_df.score.min()].head(1).copy())

    if not best_params:
        raise Exception()

    return pd.concat(best_params, ignore_index=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
