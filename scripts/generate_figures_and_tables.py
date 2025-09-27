import argparse
import csv
import os
import shlex

import numpy as np

from ste import empirical_performance
from ste.empirical_performance import get_minimax_ss, get_minimean_flat_ss, get_minimean_ss, get_minimixed_flat_ss
from ste.optimization import get_optimal_stopping_strategy, plot_stopping_strategy_state_graphs
from ste.utils.data import get_datasets_with_names
from ste.utils.figures import create_subplot_grid, label_subplots, plot_stopping_strategies_as_envelopes, save_drawing
from ste.utils.misc import get_output_path, unzip


def generate_figure_1(output_dir):
    oss = get_optimal_stopping_strategy(N=3, alpha=0.1)
    fig = plot_stopping_strategy_state_graphs(oss, font_size=24)
    output_path = f"{output_dir}/Figure 1"
    save_drawing(fig, output_path)


def generate_figure_2(output_dir):
    adrs = [1e-2, 1e-4, 0]
    ss_getters = get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss
    sses = [[get_ss(adr, 101, None) for adr in adrs] for get_ss in ss_getters]

    fig, axs = create_subplot_grid(len(ss_getters), n_rows=1, tight_layout=False, figsize=(10, 3))
    fig.subplots_adjust(hspace=10)

    for i_ss_kind, ss_getter in enumerate(ss_getters):
        ax = axs[0, i_ss_kind]
        sses = [ss_getter(adr, 101, None) for adr in adrs]
        plot_stopping_strategies_as_envelopes(sses, ax, [f"ADR = {adr}" for adr in adrs])
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_xlim((0, 101))
        ax.set_ylim((0, 52))
        ax.grid(which="major")

    label_subplots(axs, from_top=0, from_left=-0.205, fontsize=14, bbox=None)

    output_path = f"{output_dir}/Figure 2"
    save_drawing(fig, output_path)


def generate_table_2(output_dir):
    output_path = f"{output_dir}/Table 2.csv"
    with open(output_path, "wt", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Name", "N", "p", "C", "%pos"])

        named_datasets = get_datasets_with_names()
        for name, dataset in named_datasets.items():
            X, y = dataset.load_raw()
            n_obs = len(y)
            n_features = X.shape[1]
            classes, class_counts = np.unique(y, return_counts=True)
            n_classes = len(classes)
            largest_class = classes[np.argmax(class_counts)]
            proportion_of_positive_obs = np.sum(y == largest_class) / n_obs

            writer.writerow([name, n_obs, n_features, n_classes, f"{100*proportion_of_positive_obs:.1f}%"])


def generate_figure_3(output_dir):
    output_sub_dir = f"{output_dir}/Figure 3"
    os.mkdir(output_sub_dir)

    output_path_1 = f"{output_sub_dir}/Page 1"
    empirical_performance.main(
        shlex.split(
            f'ss-comparison -n 101 -f 30 --combine-plots --dataset-names "Ground Cover" "Income" "Diabetes" "Skin" -o {shlex.quote(output_path_1)}'
        )
    )

    output_path_2 = f"{output_sub_dir}/Page 2"
    empirical_performance.main(
        shlex.split(
            f'ss-comparison -n 101 -f 30 --combine-plots --dataset-names "Sepsis" "Dota2" "Hospitalization" "Shuttle" -o {shlex.quote(output_path_2)}'
        )
    )

def generate_figure_4(output_dir):
    output_path = f"{output_dir}/Figure 4"
    
    empirical_performance.main(
        shlex.split(
            f'tree-distribution -n 101 -f 30 -o {shlex.quote(output_path)}'
        )
    )


def generate_table_3(output_dir):
    output_path = f"{output_dir}/Table 3.csv"
    with open(output_path, "wt", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Dataset Name", "Disagreement Rate", "Expected Runtime", "Base Error Rate", "Error Rate"])

        dataset_names, datasets = unzip(get_datasets_with_names().items())

        # np.ndarray: 5D array of estimated metrics, with axes corresponding to:
        # 0. Forest (length = n_forests)
        # 1. Dataset (length = len(datasets))
        # 2. Stopping strategy (length = len(stopping_strategy_getters))
        # 3. Allowable disagreement rate (length = len(adrs))
        # 4. Metric kind: disagreement rate, expected runtime, error rate, and base error rate (length = 4).
        metrics = empirical_performance.get_metrics(
            n_forests=30,
            n_trees=101,
            datasets=datasets,
            adrs=[1e-3],
            stopping_strategy_getters=[get_minimean_ss]
        )

        mean_metrics = metrics.mean(axis=0)

        for i_dataset, dataset_name in enumerate(dataset_names):
            disagreement_rate, expected_runtime, error_rate, base_error_rate = mean_metrics[i_dataset, 0, 0]
            writer.writerow([
                dataset_name,
                f"{100*disagreement_rate:.2f}%",
                f"{100*expected_runtime/101:.2f}%",
                f"{100*error_rate:.2f}%",
                f"{100*base_error_rate:.2f}%",
            ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir or get_output_path("all_figs_and_tables", file_name_suffix="")
    os.mkdir(output_dir)

    generate_figure_1(output_dir)
    generate_figure_2(output_dir)
    generate_table_2(output_dir)
    generate_figure_3(output_dir)
    generate_figure_4(output_dir)
    generate_table_3(output_dir)


if __name__ == "__main__":
    main()
