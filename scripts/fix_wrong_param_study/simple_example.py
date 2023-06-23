#!/usr/bin/env python3

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import scipy
import pints

from MarkovModels import common
from matplotlib.gridspec import GridSpec

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
rc('figure', dpi=500)
rc('axes', facecolor=[0]*4)

plt.rcParams['legend.title_fontsize'] = '0'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['legend.markerscale'] = 1

def create_axes(fig):
    gs = GridSpec(5, 4, figure=fig,
                  height_ratios=[.5, .5, 1, 1, 1],
                  wspace=0.35,
                  hspace=.25,
                  bottom=0.025,
                  top=0.97)

    # Setup plots of observation times
    observation_time_axes = [[
         fig.add_subplot(gs[0, 0]),
         fig.add_subplot(gs[0, 1]),
         fig.add_subplot(gs[1, 0]),
         fig.add_subplot(gs[1, 1]),
    ], [
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
    ]]

    for ax in (observation_time_axes[0][0],
               observation_time_axes[0][2],
               ):
        ax.set_ylabel('$y$', rotation=0)

    for ax in (observation_time_axes[0][1],
               observation_time_axes[0][3],
               observation_time_axes[1][0],
               observation_time_axes[1][1],
               observation_time_axes[1][2],
               observation_time_axes[1][3]):
        ax.set_yticks([])

    for ax in (observation_time_axes[0][0],
               observation_time_axes[0][1],
               observation_time_axes[1][0],
               observation_time_axes[1][1]):
        ax.set_xticks([])
    for ax in (observation_time_axes[0][2], observation_time_axes[0][3],
               observation_time_axes[1][2], observation_time_axes[1][3]):
        ax.set_xlabel(r'$t$')

    prediction_plot_axs = [fig.add_subplot(gs[4, 0:2]),
                           fig.add_subplot(gs[4, 2:4])]

    mcmc_axs = [fig.add_subplot(gs[3, 0:2]),
                fig.add_subplot(gs[3, 2:4])]

    scatter_axs = [fig.add_subplot(gs[2, 0:2]),
                   fig.add_subplot(gs[2, 2:4])
                   ]

    for ax in scatter_axs + mcmc_axs + prediction_plot_axs:
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

    scatter_axs[1].set_ylabel('')
    scatter_axs[1].set_yticks([])

    scatter_axs[0].set_xlabel('')
    scatter_axs[0].set_xticks([])
    scatter_axs[1].set_xlabel('')
    scatter_axs[1].set_xticks([])

    prediction_plot_axs[1].set_ylabel('')
    prediction_plot_axs[1].set_yticks([])

    mcmc_axs[1].set_ylabel('')
    mcmc_axs[1].set_yticks([])

    return observation_time_axes, scatter_axs, mcmc_axs, prediction_plot_axs


def true_dgp(theta, T):
    theta1, theta2 = theta
    T = np.array(T)

    trajectory = np.exp(-T / theta1) + np.exp(-T / theta2)
    return trajectory


def discrepant_forward_model(theta, T):
    T = np.array(T)
    return theta[1] * np.exp(-T / theta[0])


def generate_data_set(T, theta=[10, 1], sigma=0.01):
    true_trajectory = true_dgp(theta, T)
    obs = np.random.normal(true_trajectory, sigma, T.shape)
    data = np.vstack((T, obs,)).T

    return data


def fit_model(dataset, T, ax=None, label='', dataset_index=0):

    observed_dataset = np.vstack(list({tuple(row) for row in dataset if row[0]
                                       in T}))
    observed_dataset = observed_dataset[observed_dataset[:, 0].argsort()]

    def min_func(theta):
        return np.sum((observed_dataset[:, 1] - discrepant_forward_model(theta,
                                                                         T))**2)
    x0 = [1, 1]

    bounds = [[0, 1e5], [0, 1e5]]

    n_repeats = args.optim_repeats
    result = None
    for i in range(n_repeats):
        x0 = [1, 1]
        if x0[1] > x0[0]:
            x0 = x0[[1, 0]]
        # use scipy optimise
        new_result = scipy.optimize.dual_annealing(min_func, x0=x0,
                                                   bounds=bounds)
        if result:
            if new_result.fun < result.fun:
                result = new_result
        else:
            result = new_result

    if ax:
        all_T = np.linspace(0, max(*T, 1.2), 100)
        ax.plot(all_T, discrepant_forward_model(result.x, all_T), '--',
                lw=.5, color='grey', label='fitted_model', alpha=.8)
        ax.plot(all_T, true_dgp(true_theta, all_T), label='true DGP', lw=.5,
                color='black')

        ax.set_xlim(0, 1.3)
        ax.set_ylim(0, 2.25)

        if len(T) < 15:
            ax.scatter(*observed_dataset.T, color=palette[dataset_index],
                       marker='x', s=12.5, zorder=2, lw=.6)
        else:
            ax.plot(*observed_dataset.T, lw=0.5,
                    color=palette[dataset_index], alpha=.75)
            ax.axvspan(observed_dataset[0, 0], observed_dataset[-1, 0],
                       alpha=.15, color=palette[dataset_index])

    return result.x


def main():

    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-o', '--output')
    argument_parser.add_argument('--figsize', default=[4.685, 7.67], type=int,
                                 nargs=2)
    argument_parser.add_argument('--no_datasets', default=10, type=int)
    argument_parser.add_argument('--sigma', default=0.01, type=float)
    argument_parser.add_argument('--file_format', default='pdf')
    argument_parser.add_argument('--no_chains', default=1, type=int)
    argument_parser.add_argument('--chain_length', default=10000, type=int)
    argument_parser.add_argument('--burn_in', default=0, type=int)
    argument_parser.add_argument('--sampling_frequency', default=10, type=int)
    argument_parser.add_argument('--optim_repeats', type=int, default=15)
    argument_parser.add_argument('--results_dir')

    global args
    args = argument_parser.parse_args()
    global output_dir
    output_dir = common.setup_output_directory(args.output, subdir_name='simple_example')

    global true_theta
    true_theta = np.array([1, 0.1])

    global palette
    palette = sns.color_palette()

    global markers
    markers = ['1', '2', '3', '4', '+', 'x', '.']

    print(palette)

    fig = plt.figure(figsize=args.figsize)
    observation_axes, scatter_axes, mcmc_axes, prediction_axes = create_axes(fig)

    generate_data_and_fit(observation_axes[0], scatter_axes[0], mcmc_axes[0],
                          prediction_axes[0], sampling_frequency=10,
                          sigma=args.sigma)
    generate_data_and_fit(observation_axes[1], scatter_axes[1], mcmc_axes[1],
                          prediction_axes[1], sampling_frequency=100,
                          sigma=args.sigma, dash=True)

    for ax in scatter_axes + mcmc_axes:
        ax.set_ylim([.95, 2.25])
        ax.set_xlim([.15, 1.2])

    page_ax = fig.subplots()
    page_ax.set_xticks([])
    page_ax.set_yticks([])
    page_ax.set_position([0, 0, 1, 1])
    page_ax.axvline(0.5, ls='--', lw=.5, color='black')
    page_ax.axis('off')

    fig.savefig(os.path.join(output_dir, f"Fig1.{args.file_format}"))


def generate_data_and_fit(observation_axes, scatter_ax, mcmc_ax, prediction_ax,
                          sampling_frequency, sigma, dash=False):
    # Generate data sets
    N_datasets = args.no_datasets

    T1 = np.linspace(0, 0.1, sampling_frequency + 1)
    T2 = np.linspace(0, 1, sampling_frequency + 1)
    T3 = np.linspace(.2, 1.2, sampling_frequency + 1)
    T4 = np.linspace(0.5, 1, sampling_frequency + 1)

    Ts = [T1, T2, T3, T4]

    all_T = np.unique(sorted(np.concatenate(Ts)))

    datasets = [generate_data_set(all_T, true_theta, sigma=sigma) for i in
                range(N_datasets)]

    def fit_datasets_using_times(datasets, T, ax, label='', design_index=0):
        thetas = []
        for i, dataset in enumerate(datasets):
            if i != 0:
                ax = None
            theta_1, theta_2 = fit_model(dataset, T, ax=ax,
                                         label=f"{label}_{i}",
                                         dataset_index=design_index)
            thetas.append([theta_1, theta_2])
        return np.vstack(thetas)

    print(datasets)

    if not args.results_dir:
        estimates = [fit_datasets_using_times(datasets, T,
                                              observation_axes[i],
                                              label=f"{i}",
                                              design_index=i) for i, T in enumerate((Ts))]
        all_label = r"$T_{\textrm{all}}'$" if sampling_frequency > 10 \
            else r"$T_{\textrm{all}}$"
        estimates.append(fit_datasets_using_times(datasets, all_T, None, all_label))
    text_x, text_y = (.75, 1.5)
    if dash:
        observation_axes[0].set_title(r"n = 101",  x=1)
    else:
        observation_axes[0].set_title(r"n = 11", x=1)

    observation_axes[0].text(text_x, text_y, r"$T_1'$" if dash else r'$T_1$')
    observation_axes[1].text(text_x, text_y, r"$T_2'$" if dash else r'$ T_2$')
    observation_axes[2].text(text_x, text_y, r"$T_3'$" if dash else r'$T_3$')
    observation_axes[3].text(text_x, text_y, r"$T_4'$" if dash else r'$T_4$')

    rows = []
    if dash:
        T_labels = [r"$T_1'$", r"$T_2'$", r"$T_3'$", r"$T_4'$",  r"$T_{\textrm{all}}'$"]
    else:
        T_labels = ['$T_1$', '$T_2$', '$T_3$', '$T_4$', r'$T_{\textrm{all}}$']

    if not args.results_dir:
        for x, T in zip(estimates, T_labels):
            row = pd.DataFrame(x, columns=[r'$\hat\theta_1$', r'$\hat\theta_2$'])
            row['time_range'] = T
            row['dataset_index'] = list(range(row.values.shape[0]))
            rows.append(row)

            estimates_df = pd.concat(rows, ignore_index=True)
            estimates_df.to_csv(os.path.join(output_dir, f"fitting_results_{sampling_frequency}.csv"))

    else:
        fitting_fname = os.path.join(args.results_dir, f"fitting_results_{sampling_frequency}.csv")
        estimates_df = pd.read_csv(fitting_fname)
    make_scatter_plots(estimates_df, scatter_ax, legend=True)
    make_prediction_plots(estimates_df, datasets, prediction_ax)

    # Now use PINTS MCMC on the same problem
    Ts.append(np.array(all_T))

    do_mcmc(datasets, Ts, mcmc_ax, sampling_frequency, estimates_df)

    offset = -0.05

    if sampling_frequency <= 10:
        observation_axes[0].set_title(r'\textbf a', loc='left', x=offset)
        scatter_ax.set_title(r'\textbf c', loc='left', x=offset*2)
        prediction_ax.set_title(r'\textbf g', loc='left', x=offset)
        mcmc_ax.set_title(r'\textbf e', loc='left', x=offset)

    else:
        observation_axes[1].set_title(r'\textbf b', loc='left', x=1-offset)
        scatter_ax.set_title(r'\textbf d', loc='left', x=1-offset*2)
        prediction_ax.set_title(r'\textbf h', loc='left', x=1-offset)
        mcmc_ax.set_title(r'\textbf f', loc='left', x=1-offset)


def do_mcmc(datasets, observation_times, mcmc_ax, sampling_frequency,
            fitting_df):
    # Use uninformative prior

    if not args.results_dir:
        prior = pints.UniformLogPrior([0, 0], [1e1, 1e1])
        class pints_log_likelihood(pints.LogPDF):
            def __init__(self, observation_times, data, sigma2):
                self.observation_times = observation_times
                self.data = data
                self.sigma2 = sigma2

            def __call__(self, p):
                # Likelihood function

                observed_dataset = np.vstack(list({tuple(row) for row in self.data if row[0]
                                                in self.observation_times}))

                observed_dataset = observed_dataset[observed_dataset[:, 0].argsort()][:, 1]

                error = discrepant_forward_model(p, self.observation_times) - observed_dataset
                SSE = np.sum(error**2)

                n = len(self.observation_times)

                ll = -n * 0.5 * np.log(2 * np.pi * self.sigma2) - SSE / (2 * self.sigma2)
                return ll

            def n_parameters(self):
                return len(true_theta)

        starting_parameters = prior.sample(n=args.no_chains)
        # starting_parameters = np.tile(true_theta, reps=[args.no_chains])

        data_set = datasets[0]

        dfs = []
        samples_list = []
        for i, observation_times in enumerate(observation_times):
            print('performing mcmc on dataset %d' % i)
            print(data_set)
            posterior = pints.LogPosterior(pints_log_likelihood(observation_times,
                                                                data_set, args.sigma**2), prior)
            mcmc = pints.MCMCController(posterior, args.no_chains,
                                        starting_parameters,
                                        method=pints.MetropolisRandomWalkMCMC)

            mcmc.set_max_iterations(args.chain_length)
            samples = mcmc.run()[:, args.burn_in:, :]
            samples_list.append(samples)

            np.save(os.path.join(output_dir, f"mcmc_chains_{i}_{sampling_frequency}.npy"),
                    samples)

            sub_df = pd.DataFrame(samples.reshape([-1, 2]),
                                  columns=[r'$\theta_1$',
                                           r'$\theta_2$'])

            sub_df['observation times'] = r'$T_{' f"{i+1}" r'}$' if i < 4 else r'$T_{\textrm{all}}$'
            dfs.append(sub_df)

    else:

        dfs = []
        samples_list = []
        for i, observation_times in enumerate(observation_times):

            mcmc_fname = os.path.join(args.results_dir, f"mcmc_chains_{i}_{sampling_frequency}.npy")
            samples = np.load(mcmc_fname)
            samples_list.append(samples)

            sub_df = pd.DataFrame(samples.reshape([-1, 2]), columns=[r'$\theta_1$',
                                                                     r'$\theta_2$'])

            sub_df['observation times'] = r'$T_{' f"{i+1}" r'}$' if i < 4 else r'$T_{\textrm{all}}$'
            dfs.append(sub_df)

    df = pd.concat(dfs, ignore_index=True)

    plot_mcmc_kde(mcmc_ax, samples_list, df, fitting_df, palette,
                  sampling_frequency)


def plot_mcmc_kde(mcmc_ax, samples_list, df, fitting_df, palette, sampling_frequency):
    sns.kdeplot(data=df, x=r'$\theta_1$', y=r'$\theta_2$', palette=palette,
                hue='observation times', levels=[.01, 0.99999], ax=mcmc_ax,
                fill=True, legend=False)

    # for i, ts in enumerate(df['observation times'].unique()):
    #     row = df[(fitting_df['observation times'] == ts) &
    #              (fitting_df['dataset_index' == 0])].head()
    #     x = row[df.columns[0]]
    #     y = row[df.columns[1]]
    #     mcmc_ax.scatter(x=[x], y=[y], color=palette[i], marker='+', lw=.3, s=5,
    #                     alpha=0.4)

    text_locs = [
        [.25, 2.25],
        [.35, 1.6],
        [.75, 1.3],
        [1.1, 1.2],
        [0.7, 1.85],
    ]

    for i, d in enumerate(fitting_df['observation times'].unique()):
        print(fitting_df)
        row = fitting_df[(fitting_df['time_range'] == d) & \
                         (fitting_df['dataset_index'].astype(int) == 0)].head()

        print(row)

        x, y = row[[r"$\hat\theta_1$", r"$\hat\theta_2$"]].values.flatten()

        mcmc_ax.annotate(d, xy=(x, y), xytext=text_locs[i],
                         textcoords='data',
                         ha='center', va='bottom',
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                         color='black'))

    mcmc_ax.set_xlabel(r'$\theta_1$')
    mcmc_ax.set_ylabel(r'$\theta_2$')

    for i, time_range in enumerate(df['observation times'].unique()):
        samples = samples_list[i]
        fig, _ = pints.plot.trace(samples, parameter_names=[r'$\theta_1$', r'$\theta_2$]'])
        fig.savefig(os.path.join(output_dir, f"mcmc_trace_{time_range}_{sampling_frequency}.pdf"))
        plt.close(fig)


def make_scatter_plots(df, ax, label='', legend=False):
    df['observation times'] = df['time_range']

    g = sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1],
                        palette=palette, style='observation times', s=12.5,
                        hue='observation times', ax=ax, edgecolor=None)

    sns.move_legend(g, "best", title='', markerfirst=False)


def make_prediction_plots(estimates, datasets, ax):
    # Use only first dataset
    df = estimates[estimates.dataset_index == 0]

    linestyles = [(0, ()),
      (0, (1, 2)),
      (0, (1, 1)),
      (0, (5, 5)),
      (0, (3, 5, 1, 5)),
      (0, (3, 5, 1, 5, 1, 5))]

    predictions = []
    T = np.linspace(0, 2, 100)

    for time_range in df.time_range.unique():
        params = df[df.time_range == time_range][[r'$\hat\theta_1$', r'$\hat\theta_2$']].values[0, :].astype(np.float64)
        prediction = discrepant_forward_model(params, T)
        predictions.append(prediction)

        # ax.plot(T, prediction, '--', color='red', lw=.5)

    predictions = np.vstack(predictions)
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # blue = colors[0]
    ax.plot(T, true_dgp(true_theta, T), label='true DGP', lw=.75, color='black')
    for prediction in predictions:
        ax.plot(T, prediction, '--', color='red', lw=.5)
    max_predict = np.max(predictions, axis=0)
    min_predict = np.min(predictions, axis=0)

    # ax.plot(T, min_predict, '--', color='red')
    # ax.plot(T, max_predict, '--', color='red')

    ax.fill_between(T, min_predict, max_predict, color='grey', alpha=0.1)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$y$', rotation=0)

    # fig.savefig(os.path.join(output_dir, 'prediction_plot'))


if __name__ == '__main__':
    main()
