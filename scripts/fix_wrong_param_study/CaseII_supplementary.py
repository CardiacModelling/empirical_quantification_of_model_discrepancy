#!/usr/bin/env python3

import multiprocessing
import regex as re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from MarkovModels import common
from MarkovModels.BeattieModel import BeattieModel
from MarkovModels.ClosedOpenModel import ClosedOpenModel
from MarkovModels.KempModel import KempModel
import argparse
import seaborn as sns
import os
import string
import re
import scipy
import math

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from matplotlib.gridspec import GridSpec

from matplotlib.patches import ConnectionPatch, Rectangle

import matplotlib.lines as mlines

from matplotlib import rc

from generate_synthetic_data import generate_data

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=1000, facecolor=[0]*4)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)

plt.rcParams['legend.framealpha'] = 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir')
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--experiment_name', default='newtonrun4', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', nargs=2, default=[4.685, 2.5])
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+',
                        default=['longap'])

    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument("-m", "--model_class", default='Beattie')
    parser.add_argument('--true_param_file')
    parser.add_argument('--fixed_param', default='Gkr')
    parser.add_argument('--prediction_protocol', default='longap')
    parser.add_argument('--sampling_period', default=0.1, type=float)
    parser.add_argument('--no_data_repeats', default=10, type=int)
    parser.add_argument('--noise', default=0.03)

    parser.add_argument("--vlim", nargs=2, type=float)

    global linestyles
    linestyles = [(0, ()),
                  (0, (1, 2)),
                  (0, (1, 1)),
                  (0, (5, 5)),
                  (0, (3, 5, 1, 5)),
                  (0, (3, 5, 1, 5, 1, 5))]

    global args
    args = parser.parse_args()

    model_class = common.get_model_class(args.model_class)

    if args.true_param_file:
        assert(False)
    else:
        parameter_labels = model_class().get_parameter_labels()

    global true_parameters
    true_parameters = model_class().get_default_parameters()

    global output_dir
    output_dir = common.setup_output_directory(args.output_dir, "CaseI_supp")

    global fig
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = create_axes(fig)

    results_df = pd.read_csv(os.path.join(args.results_dir, 'results_df.csv'))

    results_df = results_df[~results_df.protocol.isin(args.ignore_protocols)]

    global palette
    palette = sns.color_palette('husl', len(results_df.protocol.unique()))

    global protocols
    protocols = sorted(results_df.protocol.unique())

    global relabel_dict
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols if p not in args.ignore_protocols and p != 'longap'])}

    relabel_dict['longap'] = '$d_0$'

    results_df.replace({'protocol': relabel_dict}, inplace=True)

    results_df = results_df.sort_values('protocol')

    protocols = results_df.protocol.unique()

    print(results_df)
    for param_label in parameter_labels:
        results_df[param_label] = results_df[param_label].astype(np.float64)

    prediction_df = pd.read_csv(os.path.join(args.results_dir, 'predictions.csv'))
    prediction_df.replace({'fitting_protocol': relabel_dict,
                           'validation_protocol': relabel_dict},
                          inplace=True)

    make_table(results_df)
    # do_prediction_error_plot(ax, prediction_df, results_df)

    fig.savefig(os.path.join(output_dir, f"FigS1.{args.file_format}"))


def make_table(fitting_df):

    vals = sorted(fitting_df[args.fixed_param].unique())
    # vstep = int((len(vals) - 1) / 4)
    # vals = vals[::vstep]

    df_rows = []
    parameter_labels = BeattieModel().get_parameter_labels()
    prot_func, times, desc = common.get_ramp_protocol_from_csv('longap')
    full_times = pd.read_csv(os.path.join(output_dir,
                                          f'synthetic-longap-times.csv'))['time'].values.astype(np.float64)
    model = BeattieModel(prot_func,
                         times=full_times,
                         E_rev=common.calculate_reversal_potential(298, 120, 5),
                         protocol_description=desc)

    prediction_solver = model.make_forward_solver_current(njitted=False)

    default_prediction = prediction_solver()

    for val in vals:
        for well in fitting_df.well.unique():
            predictions = []
            new_rows = []
            for protocol in fitting_df.protocol.unique():
                # print(protocol)
                sub_df = fitting_df[
                    (fitting_df.well == well) & \
                    (fitting_df.protocol == protocol) & \
                    (fitting_df[args.fixed_param] == val)
                ]

                len(sub_df.index == 1)
                row = sub_df.head(1)
                params = row[parameter_labels].values.flatten().astype(np.float64)

                # Predict longap protocol
                prediction = prediction_solver(params)

                predictions.append(prediction)
                # Compute RMSE
                current = pd.read_csv(os.path.join(output_dir,
                                                   f'synthetic-longap-{well}.csv'))['current'].values.astype(np.float64)
                RMSE = np.sqrt(np.mean((prediction - current)**2))
                RMSE_DGP = np.sqrt(np.mean((prediction - default_prediction)**2))

                new_row = pd.DataFrame([[well, args.fixed_param, protocol,
                                         'longap', RMSE, RMSE_DGP, RMSE,
                                         *params]],
                                       columns=( 'well',
                                                 'fixed_param', 'fitting_protocol',
                                                 'validation_protocol', 'score',
                                                 'RMSE_DGP', 'RMSE', *parameter_labels
                                                ))
                new_rows.append(new_row)

            predictions = np.array(predictions)

            # extreme predictions for each timepoint
            max_predict = predictions.max(axis=0)
            min_predict = predictions.min(axis=0)

            average_interval_width = np.mean(max_predict - min_predict)

            points_in_interval_DGP_noise = np.mean(
                (current <= max_predict + 2*args.noise)
                & (current >= min_predict - 2*args.noise)
            )

            points_in_interval_DGP = np.mean(
                (default_prediction <= max_predict)
                & (default_prediction >= min_predict)
            )



            new_rows = pd.concat(new_rows, ignore_index=True)
            new_rows['average_interval_width'] = average_interval_width
            new_rows['points_in_interval_DGP'] = points_in_interval_DGP
            new_rows['points_in_interval_DGP_noise'] = points_in_interval_DGP

            df_rows.append(new_rows)

    # First table. Amount of points inside interval and interval width
    df = pd.concat(df_rows)
    default_params = BeattieModel().get_default_parameters()
    df[r'$\lambda$'] = df[args.fixed_param].astype(np.float64) / default_params[-1]

    print(df)

    df.replace({'fitting_protocol': relabel_dict}, inplace=True)
    orig_df = df.copy()

    print(orig_df['fitting_protocol'].unique())

    print(df.columns)

    print(df[r'$\lambda$'])

    df = df[['average_interval_width', 'points_in_interval_DGP', r'$\lambda$',
             'well', 'points_in_interval_DGP_noise', 'RMSE', 'midpoint RMSE']]

    pd.options.display.float_format = '{:.1E}'.format

    agg_dict = {}
    agg_dict['average_interval_width'] = ['mean', 'std']
    agg_dict['points_in_interval_DGP'] = ['mean', 'std']
    agg_dict['midpoint RMSE'] = ['mean', 'std']

    df = df.groupby([r'$\lambda$'], as_index=True).agg(agg_dict)
    print(df)

    s = df.style.format("{:.1E}".format)
    s = s.format_index("{:.2f}".format)

    ltx_output = s.to_latex(sparse_columns=True, multicol_align="c" )
    print(ltx_output)
    output_fname = os.path.join(output_dir, 'CaseII_table.tex')
    with open(output_fname, 'w') as fout:
        fout.write(ltx_output)

    # Second table: variability in parameter estimates
    df = orig_df.copy()
    vstep = int((len(vals) - 1) / 4)
    vals = df[r'$\lambda$'].unique()[::vstep]

    df = df[df[r'$\lambda$'].isin(vals)]

    # One row per each parameter
    df = pd.melt(df, id_vars=['fitting_protocol', 'well', r'$\lambda$'],
                 value_vars=parameter_labels[:-1])

    variables_relabel_dict = {
        'p1': '$p_1$',
        'p2': '$p_2$',
        'p3': '$p_3$',
        'p4': '$p_4$',
        'p5': '$p_5$',
        'p6': '$p_6$',
        'p7': '$p_7$',
        'p8': '$p_8$',
                              }
    df.replace({'variable': variables_relabel_dict}, inplace=True)

    print(df)

    def mean_std_func(x):
        x = x.astype(np.float64)
        return f"{x.mean():.1E}" r'\(\pm\)' f"{x.std():.0E}"

    agg_dict = {prot: mean_std_func for prot in orig_df.fitting_protocol.unique()}

    df = df.pivot_table(index=[r'$\lambda$', 'variable'],
                        columns='fitting_protocol', values='value',
                        aggfunc=mean_std_func)
    # df = df.groupby([r'$\lambda$'], as_index=True).agg(agg_dict)

    print(df)

    s = df.style
    # s = s.format_index(["{:.2f}".format])
    ltx_output = s.to_latex(sparse_columns=True, multicol_align="c" )
    print(ltx_output)

    output_fname = os.path.join(output_dir, 'CaseI_parameter_table.tex')
    with open(output_fname, 'w') as fout:
        fout.write(ltx_output)

    s.to_excel(os.path.join(output_dir, 'CaseII_parameter_table.xlsx'))


def do_prediction_error_plot(ax, prediction_df, fitting_df):

    vals = sorted(prediction_df[args.fixed_param].unique())
    # vstep = int((len(vals) - 1) / 4)
    # vals = vals[::vstep]

    parameter_labels = BeattieModel().get_parameter_labels()
    prot_func, times, desc = common.get_ramp_protocol_from_csv('longap')
    full_times = pd.read_csv(os.path.join(output_dir,
                                          f'synthetic-longap-times.csv'))['time'].values.astype(np.float64)
    model = BeattieModel(prot_func,
                         times=full_times,
                         E_rev = common.calculate_reversal_potential(298, 120, 5),
                         protocol_description=desc)

    prediction_solver = model.make_forward_solver_current(njitted=False)

    df_rows = []
    for val in vals:
        for protocol in fitting_df.protocol.unique():
            for well in fitting_df.well.unique():
                sub_df = fitting_df[
                    (fitting_df.well == well) & \
                    (fitting_df.protocol == protocol) & \
                    (fitting_df[args.fixed_param] == val)
                ]

                len(sub_df.index == 1)
                row = sub_df.head(1)
                params = row[parameter_labels].values.flatten().astype(np.float64)

                print(params)

                # Predict longap protocol
                prediction = prediction_solver(params)
                # Compute RMSE
                current = pd.read_csv(os.path.join(output_dir,
                                                   f'synthetic-longap-{well}.csv'))['current'].values.astype(np.float64)
                RMSE = np.sqrt(np.mean((prediction - current)**2))
                RMSE_DGP = np.sqrt(np.mean((prediction - prediction_solver())**2))

                new_row = pd.DataFrame([[well, args.fixed_param, protocol,
                                         'longap', RMSE, RMSE_DGP, RMSE, *params]],
                                       columns=(
                                           'well', 'fixed_param', 'fitting_protocol',
                                           'validation_protocol',
                                           'score', 'RMSE_DGP', 'RMSE', *parameter_labels
                                       ))
                new_row.replace({'fitting_protocol': relabel_dict}, inplace=True)

            df_rows.append(new_row)

    prediction_df = pd.concat((prediction_df, *df_rows))
    prediction_df = prediction_df[prediction_df.validation_protocol=='longap']


    default_params = BeattieModel().get_default_parameters()

    prediction_df[r'$\lambda$'] = prediction_df[args.fixed_param].astype(np.float64) / default_params[-1]
    # df_to_plot = prediction_df.groupby([args.fixed_param, 'fitting_protocol',
    #                                     'validation_protocol']).mean().reset_index()

    markers = ['1', '2', '3', '4', '+', 'x']
    # markers = [markers[i] for i in range(len(results_df.protocol.unique()))]
    # colours = [palette[i] for i in range(len(results_df.protocol.unique()))]

    print(prediction_df[r'$\lambda$'])

    sns.lineplot(
        data=prediction_df,
        x=r'$\lambda$',
        y='RMSE',
        hue='fitting_protocol',
        style='fitting_protocol',
        palette=palette,
        dashes=[ls[1] for ls in linestyles],
        ax=ax,
        linewidth=.5,
        errorbar=('ci', 95)
    )

    ax.set_ylabel(r'RMSE$\big(\mathbf{y}(\mathbf \theta; d_0), \mathbf z(d)\big)$ (nA)')

    ax.axvline(1, linestyle='--', lw=.5, color='grey', alpha=.5)

    # ax.set_xscale('log')

    leg = ax.legend()
    for line in leg.get_lines():
        line.set_linewidth(1)



def create_axes(fig):
    ax = fig.subplots()
    return ax

def generate_longap_data():
    generate_data('longap', args.no_data_repeats, BeattieModel,
                  common.calculate_reversal_potential(T=298, K_in=120, K_out=5),
                  args.noise,
                  output_dir,
                  noise=args.noise, figsize=args.figsize,
                  sampling_period=args.sampling_period, plot=True,
                  prefix='synthetic')


if __name__ == "__main__":
    main()
