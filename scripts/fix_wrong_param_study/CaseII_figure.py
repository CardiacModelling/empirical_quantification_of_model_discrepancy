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
from MarkovModels.WangModel import WangModel
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

from fit_all_wells_and_protocols import compute_predictions_df


rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=1000, facecolor=[0]*4)
rc('axes', facecolor=[0]*4)
rc('savefig', facecolor=[0]*4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('results_dirs', nargs=2)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--wells', '-w', type=str, default=[], nargs='+')
    parser.add_argument('--removal_duration', '-r', default=5, type=float)
    parser.add_argument('--experiment_name', default='synthetic', type=str)
    parser.add_argument('--no_chains', '-N', default=0, help='mcmc chains to run', type=int)
    parser.add_argument('--no_data_repeats', default=10, type=int)
    parser.add_argument('--noise', default=0.03)
    parser.add_argument('--sampling_period', default=0.1, type=float)
    parser.add_argument('--chain_length', '-l', default=500, help='mcmc chains to run', type=int)
    parser.add_argument('--figsize', '-f', type=int, nargs=2, default=[4.685, 3.25])
    parser.add_argument('--use_parameter_file')
    parser.add_argument('-i', '--ignore_protocols', nargs='+',
                        default=['longap'])

    parser.add_argument('-o', '--output_dir')
    parser.add_argument("-F", "--file_format", default='pdf')
    parser.add_argument("-m", "--model_class", default='Beattie')
    parser.add_argument('--prediction_protocol', default='longap')
    parser.add_argument('--solver_type')

    parser.add_argument('--predictions_df')

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
    args.data_directory = args.data_dir

    model_class = common.get_model_class(args.model_class)

    global true_parameters
    true_parameters = model_class().get_default_parameters()

    global output_dir
    output_dir = common.setup_output_directory(args.output_dir, "CaseII_main")

    global fig
    fig = plt.figure(figsize=args.figsize)
    axes = create_axes(fig)

    global protocols
    protocols = sorted(pd.read_csv(os.path.join(args.results_dirs[0], 'fitting.csv')).protocol.unique())

    global relabel_dict
    relabel_dict = {p: f"$d_{i+1}$" for i, p in enumerate([p for p in protocols if p not in args.ignore_protocols and p not in args.prediction_protocol])}
    relabel_dict['longap'] = '$d_0$'

    print("protocols protocols:", relabel_dict)

    results_dfs = []
    print(args.results_dirs)
    for results_dir in args.results_dirs:
        results_df = pd.read_csv(os.path.join(results_dir, 'fitting.csv'))
        results_df['protocol'] = results_df.protocol
        results_df = results_df[~results_df.protocol.isin(args.ignore_protocols)]
        results_df.replace({'protocol': relabel_dict}, inplace=True)
        results_df = results_df.sort_values('protocol')

        results_df = get_best_params(results_df)

        if 'g_Kr' in results_df:
            results_df['Gkr'] = results_df['g_Kr']

        results_dfs.append(results_df)

    global palette
    palette = sns.color_palette('husl', len(results_df.protocol.unique()))

    # plot scatter_plots
    scatter_plots(axes, results_dfs)
    # fig.tight_layout()

    # Plot heatmaps
    models = ['Beattie', 'Wang']
    prediction_dfs = []
    for model, results_dir, results_df in zip(models, args.results_dirs, results_dfs):

        model_class = common.get_model_class(model)
        parameter_labels = model_class().get_parameter_labels()

        if args.predictions_df:
            prediction_df = pd.read_csv(args.predictions_df)

        else:
            try:
                prediction_df = pd.read_csv(os.path.join(results_dir,
                                                        'predictions_df.csv'))
            except FileNotFoundError:
                inv_map = {v: k for k, v in relabel_dict.items()}
                results_df_original_names = results_df.replace(inv_map)
                print(results_df_original_names, model_class)
                prediction_df = compute_predictions_df(results_df_original_names,
                                                    output_dir,
                                                    model_class=common.get_model_class(model),
                                                    args=args)
                prediction_df.to_csv(os.path.join(output_dir,
                                                "predictions_df.csv"))

                prot_func, times, desc = common.get_ramp_protocol_from_csv('longap')
                full_times = pd.read_csv(os.path.join(args.data_dir,
                                                    f'synthetic-longap-times.csv'))['time'].values.astype(np.float64)
                model = model_class(prot_func,
                                    times=full_times,
                                    protocol_description=desc)

                prediction_solver = model.make_forward_solver_current(njitted=False)

                new_rows = []
                for protocol in results_df.protocol.unique():
                    for well in results_df.well.unique():
                        sub_df = results_df[
                            (results_df.well == well) & \
                            (results_df.protocol == protocol)
                        ]

                        len(sub_df.index)
                        row = sub_df.head(1)
                        params = row[parameter_labels].values.flatten().astype(np.float64)

                        print(params)

                        # Predict longap protocol
                        prediction = prediction_solver(params)
                        # Compute RMSE
                        current = pd.read_csv(os.path.join(args.data_dir,
                                                        f'synthetic-longap-{well}.csv'))['current'].values.astype(np.float64)
                        RMSE = np.sqrt(np.mean((prediction - current)**2))
                        RMSE_DGP = np.sqrt(np.mean((prediction - prediction_solver())**2))

                        new_row = pd.DataFrame([[well, protocol,
                                                'longap', RMSE, RMSE_DGP, RMSE, *params]],
                                            columns=(
                                                'well', 'fitting_protocol',
                                                'validation_protocol',
                                                'score', 'RMSE_DGP', 'RMSE', *parameter_labels
                                            ))
                        new_rows.append(new_row)

            prediction_df = pd.concat([*new_rows, prediction_df],
                                    ignore_index=True)

            prediction_df.replace({'fitting_protocol': relabel_dict,
                                'validation_protocol': relabel_dict}, inplace=True)

        keep_rows = ~prediction_df.validation_protocol.isin(args.ignore_protocols) &\
            prediction_df.fitting_protocol.isin(relabel_dict.values())

        prediction_df = prediction_df[keep_rows]
        prediction_df['model'] = model_class().get_model_name()
        prediction_dfs.append(prediction_df)

    plot_heatmaps(axes, prediction_dfs)

    current = pd.read_csv(os.path.join(args.data_dir,
                                               f"synthetic-{args.prediction_protocol}-1.csv"))['current'].values.flatten().astype(np.float64)
    times = pd.read_csv(os.path.join(args.data_dir,
                                            f"synthetic-{args.prediction_protocol}-times.csv"))['time'].values.flatten().astype(np.float64)
    do_prediction_plots(axes, results_dfs, args.prediction_protocol, current, times)

    make_table(results_dfs, args.prediction_protocol)

    fig.savefig(os.path.join(output_dir, f"Fig7.{args.file_format}"))


def make_table(dfs, protocol):

    prot_func, times, desc = common.get_ramp_protocol_from_csv('longap')
    full_times = pd.read_csv(os.path.join(args.data_dir,
                                          f'synthetic-longap-times.csv'))['time'].values.astype(np.float64)
    dgp_model = WangModel(prot_func,
                          times=full_times,
                          E_rev=common.calculate_reversal_potential(),
                          protocol_description=desc)

    beattie_model = BeattieModel(prot_func,
                                 times=full_times,
                                 E_rev=common.calculate_reversal_potential(),
                                 protocol_description=desc)

    dgp_solver = dgp_model.make_forward_solver_current(njitted=True)
    beattie_solver = beattie_model.make_forward_solver_current(njitted=True)

    default_prediction = dgp_solver()

    combined_df_rows = []

    for df, model_class_name, solver in zip(dfs, ['Beattie', 'Wang'],
                                            (beattie_solver, dgp_solver,)):

        df_rows = []
        model_class = common.get_model_class(model_class_name)

        parameter_labels = model_class().get_parameter_labels()

        print(model_class, parameter_labels)

        for well in df.well.unique():
            predictions = []
            new_rows = []

            for protocol in df.protocol.unique():
                # print(protocol)
                sub_df = df[
                    (df.well == well) & \
                    (df.protocol == protocol)
                ]

                len(sub_df.index == 1)
                row = sub_df.head(1)
                params = row[parameter_labels].values.flatten().astype(np.float64)

                # Predict longap protocol
                prediction = solver(params)

                predictions.append(prediction)
                # Compute RMSE
                current = pd.read_csv(os.path.join(args.data_dir,
                                                f'synthetic-longap-{well}.csv'))['current'].values.astype(np.float64)
                RMSE = np.sqrt(np.mean((prediction - current)**2))
                RMSE_DGP = np.sqrt(np.mean((prediction - default_prediction)**2))

                new_row = pd.DataFrame([[well, model_class_name, protocol,
                                        'longap', RMSE, RMSE_DGP, RMSE,
                                        *params]],
                                    columns=('well', 'model_class',
                                                'fitting_protocol',
                                                'validation_protocol', 'score',
                                                'RMSE_DGP', 'RMSE', *parameter_labels
                                                ))
                new_rows.append(new_row)

            predictions = np.array(predictions)

            # extreme predictions for each timepoint
            max_predict = predictions.max(axis=0)
            min_predict = predictions.min(axis=0)

            midpoint_prediction = (max_predict + min_predict) / 2

            midpoint_prediction_RMSE = np.sqrt(
                np.mean((midpoint_prediction - current)**2))

            average_interval_width = np.mean(max_predict - min_predict)
            points_in_interval_DGP = np.mean(
                (default_prediction <= max_predict)
                & (default_prediction >= min_predict)
            )

            new_rows = pd.concat(new_rows, ignore_index=True)
            new_rows['average_interval_width'] = average_interval_width
            new_rows['points_in_interval_DGP'] = points_in_interval_DGP
            new_rows['points_in_interval_DGP_noise'] = points_in_interval_DGP
            new_rows['midpoint RMSE'] = midpoint_prediction_RMSE

            combined_df_rows.append(new_rows)
            df_rows.append(new_rows)

        # First table. Amount of points inside interval and interval width
        df = pd.concat(df_rows, ignore_index=True)

        df.replace({'fitting_protocol': relabel_dict}, inplace=True)
        orig_df = df.copy()

        # Second table: variability in parameter estimates
        df = orig_df.copy()

        for lab in parameter_labels:
            df[lab] = df[lab].astype(np.float64)

        # One row per each parameter
        df = pd.melt(df, id_vars=['fitting_protocol', 'well'],
                     value_vars=parameter_labels)

        variables_relabel_dict = {
            'p1': '$p_1$',
            'p2': '$p_2$',
            'p3': '$p_3$',
            'p4': '$p_4$',
            'p5': '$p_5$',
            'p6': '$p_6$',
            'p7': '$p_7$',
            'p8': '$p_8$',
            'k_f_a': '$k_f$',
            'b_a1_a': '$q_7$',
            'b_a1_b': '$q_8$',
            'a_1_a': '$q_1$',
            'a_1_b': '$q_2$',
            'b_1_a': '$q_9$',
            'b_1_b': '$q_{10}$',
            'a_a0_a': '$q_3$',
            'a_a0_b': '$q_4$',
            'k_b_a': '$k_b$',
            'a_a1_a': '$q_5$',
            'a_a1_b': '$q_6$',
            'b_a0_a': '$q_{11}$',
            'b_a0_b': '$q_{12}$',
            'g_Kr': '$g$',
            'Gkr': '$g$'
        }
        df.replace({'variable': variables_relabel_dict}, inplace=True)

        def mean_std_func(x):
            x = x.astype(np.float64)
            return f"{x.mean():.1E}" r'\(\pm\)' f"{x.std():.0E}"

        agg_dict = {prot: mean_std_func for prot in orig_df.fitting_protocol.unique()}

        df = df.pivot_table(index=['variable'],
                            columns='fitting_protocol', values='value',
                            aggfunc=mean_std_func)

        s = df.style
        ltx_output = s.to_latex(sparse_columns=True, multicol_align="c" )

        output_fname = os.path.join(output_dir, f"CaseII_table2_{model_class_name}.tex")
        with open(output_fname, 'w') as fout:
            fout.write(ltx_output)
        s.to_excel(os.path.join(output_dir, f"CaseII_parameter_table_{model_class_name}.xlsx"))

    combined_df = pd.concat(combined_df_rows, ignore_index=True)
    combined_df = combined_df[['model_class', 'average_interval_width',
                               'points_in_interval_DGP', 'well', 'points_in_interval_DGP_noise', 'midpoint RMSE',
                               'RMSE']].set_index('model_class')

    agg_dict = {}
    agg_dict['average_interval_width'] = mean_std_func
    agg_dict['points_in_interval_DGP'] = mean_std_func
    agg_dict['midpoint RMSE'] = mean_std_func
    combined_df = combined_df.groupby(['model_class'], as_index=True).agg(agg_dict)
    # combined_df.to_csv(os.path.join(output_dir, f"{model_class_name}_table.csv"))

    s = combined_df.style

    ltx_output = s.to_latex(sparse_columns=True, multicol_align="c" )
    output_fname = os.path.join(output_dir, "CaseII_table1.tex")
    with open(output_fname, 'w') as fout:
        fout.write(ltx_output)


def do_prediction_plots(axes, results_dfs, prediction_protocol, current, times):

    voltage_func, times, protocol_desc = common.get_ramp_protocol_from_csv(prediction_protocol)

    voltages = np.array([voltage_func(t) for t in times])
    spike_times, _ = common.detect_spikes(times, voltages, window_size=0)
    indices = None

    colno = 1
    prediction_axes = [axes[i] for i in range(len(axes)) if (i % 3) == colno
                       and i > 2]

    for ax in prediction_axes[2:]:
        ax.set_visible(False)

    training_protocols = sorted(results_dfs[0].protocol.unique())

    unmap_dict = {v: k for k, v in relabel_dict.items()}

    # filter out ignored protocols
    training_protocols = [p for p in training_protocols if unmap_dict[p] not in
                          args.ignore_protocols]

    model_class = common.get_model_class(args.model_class)

    model_names = ['Beattie', 'Wang']
    ymin, ymax = [np.inf, -np.inf]

    print(training_protocols)

    for i, results_df in enumerate(results_dfs):
        # plot data
        ax = prediction_axes[i]
        ax.plot(times, current, color='grey', alpha=.5, lw=0.3)

        model_class = common.get_model_class(model_names[i])
        parameter_labels = model_class().get_parameter_labels()

        model = model_class(voltage_func, times,
                            protocol_description=protocol_desc)
        solver = model.make_forward_solver_current()

        predictions = []
        for training_protocol in sorted(training_protocols):
            print(training_protocol)

            results_df['score'] = results_df['score'].astype(np.float64)
            print(results_df[results_df.protocol == training_protocol])

            row = results_df[(results_df.protocol == training_protocol)
                             & (results_df.well.astype(int) == 0)].sort_values('score')
            parameters = row[parameter_labels].head(1).values.flatten().astype(np.float64)
            print('parameters are', parameters)

            prediction = solver(parameters)
            predictions.append(prediction)
            # ax.plot(prediction, times, linewidth=0.1)

            ymin = min(ymin, prediction.min())
            ymax = max(ymax, prediction.max())

        predictions = np.array(predictions)

        max_pred = predictions.max(axis=0)
        min_pred = predictions.min(axis=0)
        ax.plot(times, max_pred, color='red',
                linewidth=.15, )
        ax.plot(times, min_pred, color='red',
                linewidth=.15, )

        ax.fill_between(times, min_pred, max_pred, color='orange', alpha=0,
                        linewidth=0, rasterized=False)
        axins = inset_axes(ax, width='50%', height='45%', loc='lower center')

        # axins.axis('off')
        axins.set_xticks([])
        axins.set_yticks([])

        axins.fill_between(times, min_pred, max_pred, color='orange', alpha=.2,
                           linewidth=0, rasterized=False)

        for j in range(predictions.shape[0]):
            linestyle = linestyles[j]
            prediction = predictions[j, :]
            axins.plot(times, prediction, ls=linestyle,
                       linewidth=0.5, color=palette[j],
                       )

        # axins.plot(times, current, color='grey', alpha=.2,
                   # linewidth=0)

        axins.set_xlim([5000, 6000])
        axins.set_ylim(-0.5, 2)

        mark_inset(ax, axins, edgecolor="black", fc="none", loc1=1, loc2=2,
                   linewidth=.3, alpha=.8)

    for i, ax in enumerate(prediction_axes):
        ax.set_xticks([0, 8000])
        ax.set_xticklabels(['0s', '8s'], rotation='horizontal')

        ax.set_yticks([-15, 0, 10])
        yticks = ax.get_yticks()

        ylabs = [str(l) + '' for l in yticks]

        ax.set_yticklabels(ylabs, rotation='horizontal')
        ax.set_ylim([-15, 10])

        # remove spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        box = prediction_axes[i].get_position()
        # box.x0 += 0.05
        box.x1 += 0.05
        ax.set_position(box)

        ax.set_rasterization_zorder(10)

        ax.set_ylabel(r'$I_\textrm{Kr}$ (nA)')


    # Plot voltage
    axes[colno].plot(times[::50], [voltage_func(t) for t in times][::50], color='black',
                     linewidth=.5)

    axes[colno].set_ylabel(r'$V$ (mV)')

    # axes[colno].yaxis.tick_right()
    axes[colno].spines.right.set_visible(False)
    axes[colno].spines.top.set_visible(False)

    prediction_axes[-1].set_xlabel(r'$t$ (s)')

    prediction_axes[-1].sharex(axes[colno])

    axes[colno].set_yticks([-100, 40])
    # axes[colno].set_yticklabels(['-100mV', '+40mV'])

    ax = axes[colno]
    box = ax.get_position()
    # box.x0 += 0.05
    box.x1 += 0.05
    ax.set_position(box)

    axes[colno].set_xticklabels([])
    axes[colno + 3].set_xticklabels([])
    labels = ['0', '7.5']
    axes[colno + 6].set_xticks([0, 7500])
    axes[colno + 6].set_xticklabels(labels)

    for ax in prediction_axes:
        ax.set_xlim([0, 9000])

    axes[1].set_xlim([0, 9000])


def plot_heatmaps(axes, prediction_dfs):

    colno = 2
    # Drop parameter sets fitted to 'longap', for example
    # Get central column
    heatmap_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno]
    prediction_axes = [axes[i] for i in range(len(axes)) if i > 2 and (i % 3) == colno - 1]

    for ax in heatmap_axes[2:]:
        ax.set_visible(False)

    cmap = sns.cm.mako_r

    joint_df = pd.concat(prediction_dfs, ignore_index=True).copy()

    joint_df['RMSE'] = joint_df['RMSE'].astype(np.float64)
    averaged_df = joint_df.groupby(['fitting_protocol', 'validation_protocol',
                                    'model'])['RMSE'].mean().reset_index()

    if args.vlim is None:
        vmin, vmax = averaged_df['RMSE'].min(), averaged_df['RMSE'].max()
    else:
        vmin, vmax = args.vlim

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    for i, model in enumerate(sorted(averaged_df.model.unique())):
        ax = heatmap_axes[i]
        sub_df = averaged_df[averaged_df.model == model].copy()

        # Ignore training to validation protocols, labels 'V' and '$d^*$', have
        # been used for validation protocols
        sub_df = sub_df[~sub_df.fitting_protocol.isin(['V', '$d_0$'])]

        pivot_df = sub_df.pivot(columns='fitting_protocol',
                                index='validation_protocol', values='RMSE')

        hm = sns.heatmap(pivot_df, ax=ax, square=True, cbar=False, norm=norm,
                         cmap=cmap)

        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)

        # Add arrow from heatmap to prediction plot
        ax2 = prediction_axes[i]
        xyA = [7750, 5]
        xyB = [-.1, 0.5]
        con = ConnectionPatch(
            xyA=xyB, coordsA=ax.transData,
            xyB=xyA, coordsB=ax2.transData,
            arrowstyle="->", shrinkB=5)

        # Add yellow highlight to first row
        autoAxis = ax.axis()
        rec = Rectangle(
            (autoAxis[0] - 0.05, autoAxis[3] - 0.05),
            (autoAxis[1] - autoAxis[0] + 0.1),
            1.1,
            fill=False,
            color='yellow',
            lw=.75
            )

        if i == 0:
            fig.add_artist(con)
            rec = ax.add_patch(rec)
            rec.set_clip_on(False)

        if i != 0:
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.set_xlabel('training', labelpad=0)
            ax.set_ylabel('validation')
            ax.xaxis.tick_top()
            ax.yaxis.tick_right()

    cbar_kws = {'orientation': 'horizontal',
                'fraction': 1,
                'aspect': 10,
                }

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = axes[colno]
    # cax.axis('off')
    cax.set_xticks([])
    cax.set_yticks([])

    for side in ['left', 'top', 'right', 'bottom']:
        cax.spines[side].set_visible(False)

    im = cax.imshow([[vmin, vmax]], cmap=cmap, norm=norm)
    im.set_visible(False)
    cax.plot([0], [0])

    cbar = plt.colorbar(ax=cax,
                        norm=norm, cmap=cmap, label='', **cbar_kws,
                        mappable=im, ticks=matplotlib.ticker.LogLocator(base=10))

    cax = cbar.ax
    # Move cbar up
    box = cax.get_position()
    box.y1 += 0.06
    box.y0 += 0.06
    cax.set_position(box)

    cax.xaxis.set_label_position('top')
    cax.set_xlabel(r'$\log_{10}$ RMSE')


def create_axes(fig):

    ncols = 4
    nrows = 3

    global gs

    gs = GridSpec(nrows, ncols, height_ratios=[0.2, 1, 1],
                  width_ratios=[.1, 1, 1, .8],
                  wspace=.65,
                  right=.95,
                  left=.1,
                  hspace=.4,
                  bottom=0.15,
                  top=.85,
                  figure=fig)

    bottom_axes = [fig.add_subplot(gs[2, i]) for i in range(ncols) if i % 4 != 0]

    axes = []
    for i in range(2):
        cells = [gs[i, j + 1] for j in range(ncols - 1)]

        for j, cell in enumerate(cells):
            # if j != 2:
            #     sharex = bottom_axes[j]
            # else:
            # sharex = None
            axes.append(fig.add_subplot(cell))

    axes = axes + list(bottom_axes)

    for ax in axes:
        ax.set_rasterization_zorder(2)

    axes[3].set_title(r'\textbf{a}', loc='left')
    axes[1].set_title(r'\textbf{b}', loc='left')
    axes[5].set_title(r'\textbf{d}', x=-0.2, y=1.01)
    axes[4].set_title(r'\textbf{c}', loc='left')

    # move entire first row up
    for i, ax in enumerate(axes[:3]):
        box = ax.get_position()
        box.y0 += .075
        box.y1 += .075
        ax.set_position(box)

    box = axes[0].get_position()
    box.x0 -= 0.1
    box.x1 -= 0.1
    axes[0].set_position(box)

    number_line_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
    ]

    for ax in number_line_axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        pos1 = ax.get_position()
        # pos1.y1 -= .05
        # pos1.y0 -= .05
        pos1.x0 -= 0.1
        pos1.x1 -= 0.1
        ax.set_position(pos1)

        for side in ['right', 'top', 'bottom', 'left']:
            ax.spines[side].set_visible(False)

    for ax, model in zip(number_line_axes, ['Beattie model', 'Wang model']):
        ax.text(0.5, 0, model, rotation=90)

    # pos1 = number_line_axes.get_position()
    # pos1.y1 -= .05
    # pos1.y0 += .05
    # pos1.x0 -= 0.05
    # pos1.x1 -= 0.05
    # number_line_axes.set_position(pos1)

    return axes


def scatter_plots(axes, results_dfs, params=['p1', 'p2'], col=0):
    scatter_axes = [ax for i, ax in enumerate(axes) if (i % 3) == col and i > 2]

    for ax in scatter_axes[2:]:
        ax.set_visible(False)

    # assert(len(scatter_axes) == 5)
    gkrs = pd.concat(results_dfs)['Gkr'].values.flatten()
    ylims = [0.95 * min(gkrs), 1.05 * max(gkrs)]
    models = ['Beattie', 'Wang']
    for i, _ in enumerate(results_dfs):
        results_dfs[i]['model'] = models[i]

    print(palette)

    markers = ['1', '2', '3', '4', '+', 'x']
    # markers = [markers[i] for i in range(len(results_dfs[0].protocol.unique()))]
    # colours = [palette[i] for i in range(len(results_dfs[0].protocol.unique()))]

    for i, results_df in enumerate(results_dfs):
        ax = scatter_axes[i]

        ax.axhline(BeattieModel().get_default_parameters()[-1], ls='--',
                   color='grey', alpha=.9, lw=.5)
        sns.scatterplot(ax=ax, data=results_df, y=r'Gkr', x='protocol',
                        palette=palette, hue='protocol', style='protocol',
                        legend=False, size=2, linewidth=0)
        ax.set_ylim(ylims)
        ax.set_ylabel(r'$g$', rotation=0)
        ax.set_xlabel(r'protocol', rotation=0)

    for i, ax in enumerate(scatter_axes[:2]):
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xlabel('')

    # Put legend on the top left axis
    ax = axes[0]
    legend_kws = {'loc': 10,
                  'frameon': False,
                  'bbox_to_anchor': [0, 0, 1, 1],
                  'ncol': 2,
                  'fontsize': 8
                  }

    ticks = scatter_axes[0].get_xticks()
    tick_labels = scatter_axes[0].get_xticklabels()
    scatter_axes[0].set_xticklabels([])
    scatter_axes[1].set_xticks(ticks)
    scatter_axes[1].set_xticklabels(tick_labels)
    scatter_axes[1].set_xlabel('protocol')

    handles = [mlines.Line2D(xdata=[1], ydata=[1], color=color, marker=marker,
                             linestyle=linestyles[i], markersize=5,
                             label=label, linewidth=.3) for i, (label, marker,
                                                                 color) in enumerate(zip(protocols, markers,
                                                                                         palette))]

    handles, labels = list(handles), list(results_dfs[0]['protocol'].unique())
    ax.legend(labels=labels, handles=handles, **legend_kws)
    ax.axis('off')

    for ax in scatter_axes:
        pos = ax.get_position()
        pos.x0 -= .05
        pos.x1 -= .05
        ax.set_position(pos)


def get_best_params(fitting_df, protocol_label='protocol'):
    best_params = []

    fitting_df['score'] = fitting_df['score'].astype(np.float64)
    fitting_df = fitting_df[np.isfinite(fitting_df['score'])].copy()

    for protocol in fitting_df[protocol_label].unique():
        for well in fitting_df['well'].unique():
            sub_df = fitting_df[(fitting_df['well'] == well)
                                & (fitting_df[protocol_label] == protocol)].copy()

            # Get index of min score
            if len(sub_df.index) == 0:
                continue
            best_params.append(sub_df[sub_df.score == sub_df.score.min()].head(1).copy())

    return pd.concat(best_params, ignore_index=True)

if __name__ == "__main__":
    main()
