#!/usr/bin/env python3

from MarkovModels import common
import os
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

from .fit_all_wells_and_protocols import compute_predictions_df, get_best_params

protocol_chrono_order = ['staircaseramp1',
                         'sis',
                         'rtovmaxdiff',
                         'rvotmaxdiff',
                         'spacefill10',
                         'spacefill19',
                         'spacefill26',
                         'longap',
                         'hhbrute3gstep',
                         'hhsobol3step',
                         'wangbrute3gstep',
                         'wangsobol3step',
                         'staircaseramp2']

def main():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="CSV file listing model errors for each cell and protocol")
    parser.add_argument("--output_dir", "-o", help="Directory to output plots to.\
    By default a new directory will be generated", default=None)
    parser.add_argument("--normalise_diagonal", action="store_true")
    parser.add_argument("-s", "--sort", action='store_true')
    parser.add_argument("--sort_wells", action='store_true')
    parser.add_argument("--vmax", "-m", default=None, type=float)

    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    df = df.drop_duplicates(subset=['well', 'fitting_protocol',
                                    'validation_protocol'],
                            keep='first')

    if args.sort:

        protocols = df['fitting_protocol'].unique()

        # Rank protocols
        def score(protocol):
            return df[df.fitting_protocol == protocol]['RMSE'].sum()
        scores = [score(protocol) for protocol in protocols]

        order = protocols[np.argsort(scores)]
        score_df = pd.DataFrame(np.column_stack((protocols, scores)), columns=('protocol', 'score'))
        score_df['protocol'] = pd.Categorical(score_df['protocol'], order)

        # Change order of protocols
        df['fitting_protocol'] = pd.Categorical(df['fitting_protocol'],
                                                categories=order)
        df['validation_protocol'] = pd.Categorical(df['validation_protocol'],
                                                   categories=order)
        df['protocol'] = df['fitting_protocol']

    order = protocol_chrono_order

    fig = plt.figure(figsize=(14, 10))

    output_dir = common.setup_output_directory(args.output_dir)
    protocol_list = df['fitting_protocol'].unique()

    params_df = get_best_params(df, protocol_label='fitting_protocol')

    df_adjusted_kinetics = compute_predictions_df(params_df, output_dir, label='predictions',
                                                  adjust_kinetic_parameters=True)

    # Share cbar limits across wells
    vmax = max(np.max(params_df['RMSE'].values), np.max(df_adjusted_kinetics['RMSE'].values))
    vmin = min(np.min(params_df['RMSE'].values), np.min(df_adjusted_kinetics['RMSE'].values))

    # Iterate over wells for heatmap
    for well in df['well'].unique():
        ax = fig.subplots()
        sub_df = df[df.well == well].copy()
        adjusted_sub_df = df_adjusted_kinetics[df_adjusted_kinetics.well == well].copy()

        pivot_df = sub_df.pivot(index='fitting_protocol',
                                columns='validation_protocol', values='RMSE')

        adjusted_pivot_df = adjusted_sub_df.pivot(index='fitting_protocol',
                                                  columns='validation_protocol',
                                                  values='RMSE')
        # pivot_df = pivot_df[np.isfinite(pivot_df)]

        cmap = sns.cm.rocket_r

        for i, df in enumerate([pivot_df, adjusted_pivot_df]):
            sns.heatmap(df, ax=ax, cbar_kws={'label': 'RMSE'}, vmin=None,
                        vmax=vmax, vmin=vmin, cmap=cmap)

            ax.set_title(f"well {well}")
            ax.set_ylabel("Fitting protocol")
            ax.set_xlabel("Validation protocol")

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     f"{well}_fit_predict_heatmap_method_{i}.png"))
            ax.cla()


if __name__ == "__main__":
    main()
