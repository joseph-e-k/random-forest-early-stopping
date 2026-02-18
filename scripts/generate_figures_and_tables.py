import argparse
import csv
import os
import shlex

import numpy as np

from scripts import draw_supp_fig_timings, measure_tree_certainties
from ste import empirical_performance
from ste.empirical_performance import get_minimax_ss, get_minimean_flat_ss, get_minimean_ss, get_minimixed_flat_ss
from ste.optimization import get_optimal_stopping_strategy, plot_stopping_strategy_state_graphs
from ste.utils.data import get_datasets_with_names
from ste.utils.figures import create_subplot_grid, label_subplots, plot_stopping_strategies_as_envelopes, save_drawing
from ste.utils.logging import configure_logging
from ste.utils.misc import get_output_path, unzip


def generate_figure_1(output_dir, n_trees, n_forests):
    oss = get_optimal_stopping_strategy(N=3, alpha=0.1)
    fig = plot_stopping_strategy_state_graphs(oss, font_size=24)
    output_path = f"{output_dir}/Figure 1"
    save_drawing(fig, output_path)


def generate_figure_2(output_dir, n_trees, n_forests):
    adrs = [1e-2, 1e-4, 0]
    ss_getters = get_minimax_ss, get_minimean_flat_ss, get_minimixed_flat_ss
    sses = [[get_ss(adr, n_trees, None) for adr in adrs] for get_ss in ss_getters]

    fig, axs = create_subplot_grid(len(ss_getters), n_rows=1, tight_layout=False, figsize=(10, 3))
    fig.subplots_adjust(hspace=10)

    for i_ss_kind, ss_getter in enumerate(ss_getters):
        ax = axs[0, i_ss_kind]
        sses = [ss_getter(adr, n_trees, None) for adr in adrs]
        plot_stopping_strategies_as_envelopes(sses, ax, [f"ADR = {adr}" for adr in adrs])
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_xlim((0, n_trees))
        ax.set_ylim((0, int(n_trees / 2) + 2))
        ax.grid(which="major")

    label_subplots(axs, from_top=0, from_left=-0.205, fontsize=14, bbox=None)

    output_path = f"{output_dir}/Figure 2"
    save_drawing(fig, output_path)


def generate_table_2(output_dir, n_trees, n_forests):
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


def generate_figure_3(output_dir, n_trees, n_forests):
    output_sub_dir = f"{output_dir}/Figure 3"
    os.mkdir(output_sub_dir)

    output_path_1 = f"{output_sub_dir}/Page 1"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} --combine-plots --dataset-names "Ground Cover" "Income" "Diabetes" "Skin" -o {shlex.quote(output_path_1)}'
        )
    )

    output_path_2 = f"{output_sub_dir}/Page 2"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} --combine-plots --dataset-names "Sepsis" "Dota2" "Hospitalization" "Shuttle" -o {shlex.quote(output_path_2)}'
        )
    )


def generate_figure_4(output_dir, n_trees, n_forests):
    output_path = f"{output_dir}/Figure 4"
    
    empirical_performance.main(
        shlex.split(
            f'er-rt-comparison -N {n_trees} -f {n_forests} -o {shlex.quote(output_path)}'
        )
    )


def generate_figure_5(output_dir, n_trees, n_forests):
    output_path = f"{output_dir}/Figure 5"
    
    empirical_performance.main(
        shlex.split(
            f'tree-distribution -N {n_trees} -f {n_forests} -o {shlex.quote(output_path)}'
        )
    )


def generate_table_3(output_dir, n_trees, n_forests):
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
            n_forests=n_forests,
            n_trees=n_trees,
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
                f"{100*expected_runtime/n_trees:.2f}%",
                f"{100*error_rate:.2f}%",
                f"{100*base_error_rate:.2f}%",
            ])


def generate_figure_supp_1(output_dir, n_trees, n_forests):
    output_path = f"{output_dir}/Figure 1 (Supplementary)"
    draw_supp_fig_timings.main(output_path)


def generate_figure_supp_2(output_dir, n_trees, n_forests):
    output_sub_dir = f"{output_dir}/Figure 2 (Supplementary)"
    os.mkdir(output_sub_dir)

    output_path_1 = f"{output_sub_dir}/Page 1"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} -b --combine-plots --dataset-names "Higgs" "eye_movements" "jannis" "KDDCup09_upselling" -o {shlex.quote(output_path_1)}'
        )
    )

    output_path_2 = f"{output_sub_dir}/Page 2"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} -b --combine-plots --dataset-names "MagicTelescope" "bank-marketing" "phoneme" "MiniBooNE" -o {shlex.quote(output_path_2)}'
        )
    )

    output_path_3 = f"{output_sub_dir}/Page 3"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} -b --combine-plots --dataset-names "covertype" "pol" "house_16H" "kdd_ipums_la_97-small" -o {shlex.quote(output_path_3)}'
        )
    )

    output_path_4 = f"{output_sub_dir}/Page 4"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} -b --combine-plots --dataset-names "credit" "california" "wine" "electricity" -o {shlex.quote(output_path_4)}'
        )
    )

    output_path_5 = f"{output_sub_dir}/Page 5"
    empirical_performance.main(
        shlex.split(
            f'detailed-comparison -N {n_trees} -f {n_forests} -b --combine-plots --dataset-names "rl" "road-safety" "compass" -o {shlex.quote(output_path_5)}'
        )
    )


def generate_table_supp_1(output_dir, n_trees, n_forests):
    output_path = f"{output_dir}/Table 1 (Supplementary).csv"

    base_datasets_by_name = get_datasets_with_names()
    more_datasets_by_name = get_datasets_with_names(full_benchmark=True)

    datasets = list(base_datasets_by_name.values()) + list(more_datasets_by_name.values())
    dataset_names = list(base_datasets_by_name.keys()) + list(more_datasets_by_name.keys())

    metrics = measure_tree_certainties.compute_metrics_for_each_dataset(
        n_forests=n_forests,
        n_trees=n_trees,
        eval_proportion=0.1,
        datasets=datasets
    )

    with open(output_path, "wt", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Dataset name", "Tree prediction equal to 0 or 1", "Ensemble classification unaffected by rounding"])

        for i_dataset, dataset_name in enumerate(dataset_names):
            certainty_proportion, indifference_proportion = metrics[i_dataset, :]
            writer.writerow([
                dataset_name,
                f"{100*certainty_proportion:.2f}%",
                f"{100*indifference_proportion:.2f}%",
            ])


def parse_args(argv=None):
    tasks_by_name = {
        "fig1": generate_figure_1,
        "fig2": generate_figure_2,
        "fig3": generate_figure_3,
        "fig4": generate_figure_4,
        "fig5": generate_figure_5,
        "table2": generate_table_2,
        "table3": generate_table_3,
        "fig1s": generate_figure_supp_1,
        "fig2s": generate_figure_supp_2,
        "table1s": generate_table_supp_1,
    }
    parser = argparse.ArgumentParser(
        description="Script to reproduce numeric figures and tables in the main body and supplementary material of the paper. " \
        "Note that the timing figure in the supplementary material (fig1s) will fail to generate if you don't have a Gurobi license, and will take days even if you do; you may wish to exclude it using the command-line options."
    )
    parser.add_argument(
        "--output-dir", "-o", nargs="?",
        help="Directory in which to save generated files. Defaults to <project root>/results/all_figs_and_tables_<timestamp>."
    )
    parser.add_argument(
        "--n-trees", "-N", type=int, default=101,
        help="Size of ensembles to use in empirical evaluations. Defaults to 101."
    )
    parser.add_argument(
        "--n-forests", "-f", type=int, default=30,
        help="Number of ensembles to average over in empirical evaluations. Defaults to 30."
    )
    parser.add_argument(
        "--include", nargs="*", default=tasks_by_name.keys(), choices=tasks_by_name.keys(),
        help=f"List of figures and tables to generate. Names with 's' on the end are those in the Supplementary Materials. Defaults to everything."
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[], choices=tasks_by_name.keys(),
        help="List of figures and tables NOT to generate."
    )

    namespace = parser.parse_args(argv)
    namespace.tasks = []

    for name in set(namespace.include) - set(namespace.exclude):
        namespace.tasks.append(tasks_by_name[name])
    
    return namespace


def main(argv=None):
    configure_logging()
    args = parse_args(argv)
    output_dir = args.output_dir or get_output_path("all_figs_and_tables", file_name_suffix="")
    os.makedirs(output_dir)

    for task in args.tasks:
        task(output_dir, args.n_trees, args.n_forests)


if __name__ == "__main__":
    main()
