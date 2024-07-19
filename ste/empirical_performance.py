import argparse
import functools
import operator
import random
from typing import Callable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ste.Forest import Forest
from ste.ForestWithEnvelope import ForestWithEnvelope
from ste.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from ste.figure_utils import create_subplot_grid
from ste.logging_utils import configure_logging, get_module_logger
from ste.multiprocessing_utils import parallelize
from ste.optimization import get_optimal_stopping_strategy
from ste.utils import Dataset, load_datasets, get_output_path, memoize, split_dataset


_logger = get_module_logger()


def plot_smopdises(n_trees, datasets):
    raise NotImplementedError()

    fig, axs = create_subplot_grid(len(datasets), n_rows=2)
    n_rows = axs.shape[0]

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        distribution = estimate_smopdis(dataset, n_trees=n_trees)

        ax = axs[i_dataset // n_rows, i_dataset % n_rows]
        ax.bar(np.arange(n_trees + 1), distribution, width=1)
        ax.title.set_text(f"{dataset_name}")
        ax.set_xlim((0, n_trees))
        ax.set_yticks([])

    return fig


def _get_stopping_strategies(n_trees, smopdis_estimate, aer, stopping_strategy_getters):
    stopping_strategies = np.empty(
        shape=(
            len(stopping_strategy_getters),
            n_trees + 1,
            n_trees + 1
        )
    )

    task_outcomes = parallelize(
        operator.call,
        argses_to_iter=[
            (functools.partial(
                ss_getter,
                n_trees=n_trees,
                allowable_error=aer,
                smopdis_estimate=smopdis_estimate
            ),)
            for ss_getter in stopping_strategy_getters
        ],
        job_name="get_ss"
    )

    for outcome in task_outcomes:
        stopping_strategies[outcome.index] = outcome.result

    return stopping_strategies


def _analyse_stopping_strategy_if_relevant(i_ss_kind, n_positive_trees, n_trees, smopdis_estimate, stopping_strategies):
    if smopdis_estimate[n_positive_trees] == 0:
        return None

    ss = stopping_strategies[i_ss_kind]
    forest = Forest(n_trees, n_positive_trees)
    fwss = ForestWithGivenStoppingStrategy(forest, ss)

    return fwss.analyse()


def _analyse_stopping_strategies(stopping_strategies, smopdis_estimate):
    n_ss_kinds, n_trees_plus_one, _ = stopping_strategies.shape
    n_trees = n_trees_plus_one - 1

    runtimes = np.zeros(n_ss_kinds)
    error_rates = np.zeros_like(runtimes)

    tasks = parallelize(
        functools.partial(
            _analyse_stopping_strategy_if_relevant,
            n_trees=n_trees,
            smopdis_estimate=smopdis_estimate,
            stopping_strategies=stopping_strategies
        ),
        argses_to_combine=(
            range(n_ss_kinds),
            range(n_trees + 1),
        ),
        job_name="analyse"
    )

    for task in tasks:
        if task.result is None:
            continue
        i_ss_kind, n_positive_trees = task.args_or_kwargs
        prob_n_positive_trees = smopdis_estimate[n_positive_trees] / smopdis_estimate.sum()
        runtimes[i_ss_kind] += task.result.expected_runtime * prob_n_positive_trees
        error_rates[i_ss_kind] += task.result.prob_error * prob_n_positive_trees

    return error_rates, runtimes


type StoppingStrategyGetter = Callable[[int, float, Dataset], np.ndarray]


def train_forest(n_trees, training_data) -> RandomForestClassifier:
    rf_classifier = RandomForestClassifier(n_estimators=n_trees)
    rf_classifier.fit(*training_data)
    return rf_classifier


def estimate_smopdis(rf_classifier: RandomForestClassifier, calibration_data: Dataset):
    X, y = calibration_data
    n_trees = len(rf_classifier.estimators_)
    tree_predictions = np.array(np.vstack([tree.predict(X) for tree in rf_classifier.estimators_]), dtype=int)
    return np.bincount(np.sum(tree_predictions, axis=0), minlength=n_trees + 1)


def get_error_rates_and_runtimes_once(_, data: Dataset, aer: float, n_trees: int, stopping_strategy_getters: list[StoppingStrategyGetter], data_partition_ratios):
    training_data, calibration_data_for_evaluation, calibration_data_for_bayesian_ss, testing_data = split_dataset(data, data_partition_ratios)

    forest = train_forest(n_trees, training_data)
    smopdis_estimate_for_evaluation = estimate_smopdis(forest, calibration_data_for_evaluation)
    smopdis_estimate_for_bayesian_ss = estimate_smopdis(forest, calibration_data_for_bayesian_ss)

    stopping_strategies = _get_stopping_strategies(n_trees, smopdis_estimate_for_bayesian_ss, aer, stopping_strategy_getters)

    return _analyse_stopping_strategies(stopping_strategies, smopdis_estimate_for_evaluation)


def get_error_rates_and_runtimes(n_forests, n_trees, datasets, aers, stopping_strategy_getters, data_partition_ratios=(0.6, 0.1, 0.1, 0.2)):
    error_rates = np.zeros((n_forests, len(datasets), len(aers), len(stopping_strategy_getters)))
    runtimes = np.zeros_like(error_rates)
    
    tasks = parallelize(
        function=functools.partial(
            get_error_rates_and_runtimes_once,
            n_trees=n_trees,
            data_partition_ratios=data_partition_ratios,
            stopping_strategy_getters=stopping_strategy_getters
        ),
        argses_to_combine=[
            range(n_forests),
            datasets,
            aers
        ]
    )

    for task in tasks:
        task_error_rates, task_runtimes = task.result
        error_rates[task.index] = task_error_rates
        runtimes[task.index] = task_runtimes

    return error_rates, runtimes


def show_error_rates_and_runtimes(n_trees, error_rates, runtimes, dataset_names, allowable_error_rates, analysis_names):
    assert error_rates.shape == runtimes.shape

    n_datasets, n_aers, n_analyses = error_rates.shape

    assert len(dataset_names) == n_datasets
    assert len(allowable_error_rates) == n_aers
    assert len(analysis_names) == n_analyses

    fig, axs = plt.subplots(2, n_analyses, tight_layout=True, figsize=(15, 5))
    fig.suptitle(f"Error rates and runtimes for early-stopping random forests with {n_trees} trees", fontsize=16)
    fig.patch.set_visible(False)

    for i_analysis in range(n_analyses):
        name = analysis_names[i_analysis]
        table_specs = zip(
            [f"Expected Runtime ({name})", f"Error Rate ({name})"],
            axs[:, i_analysis],
            [runtimes[:, :, i_analysis], error_rates[:, :, i_analysis]],
            ["{:.2f}", "{:.2e}"]
        )

        for (title, ax, table_content, cell_format) in table_specs:
            ax.axis("off")
            ax.axis("tight")
            ax.title.set_text(title)

            ax.table(
                cellText=[
                    [cell_format.format(table_content[i_dataset, i_aer]) for (i_aer, aer) in enumerate(allowable_error_rates)]
                    for i_dataset in range(n_datasets)
                ],
                rowLabels=list(dataset_names),
                colLabels=[f"{aer:.0e}" for aer in allowable_error_rates],
                loc="center"
            )

    plt.show()


def get_and_show_error_rates_and_runtimes(n_trees, datasets, allowable_error_rates, ss_getters_by_name):
    ss_names, ss_getters = zip(*ss_getters_by_name.items())

    error_rates, runtimes = get_error_rates_and_runtimes(
        2, n_trees, datasets.values(), allowable_error_rates, ss_getters
    )

    mean_error_rates = error_rates.mean(axis=0)
    mean_runtimes = runtimes.mean(axis=0)

    show_error_rates_and_runtimes(
        n_trees,
        mean_error_rates,
        mean_runtimes,
        datasets.keys(),
        allowable_error_rates,
        ss_names or [func.__name__ for func in ss_getters]
    )


@memoize(args_to_ignore=["smopdis_estimate"])
def get_greedy_ss(n_trees: int, allowable_error: float, smopdis_estimate: np.ndarray) -> np.ndarray:
    fwe = ForestWithEnvelope.create_greedy(n_trees, n_trees, allowable_error)
    return fwe.get_prob_stop()


@memoize(args_to_ignore=["smopdis_estimate"])
def get_minimax_ss(n_trees: int, allowable_error: float, smopdis_estimate: np.ndarray) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, allowable_error)


@memoize()
def get_bayesian_ss(n_trees: int, allowable_error: float, smopdis_estimate: np.ndarray) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, allowable_error, smopdis_estimate, error_minimax=False, runtime_minimax=False)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    ss_comparison_subparser = subparsers.add_parser("ss-comparison")
    ss_comparison_subparser.set_defaults(action_name="empirical_comparison")
    ss_comparison_subparser.add_argument("--n-trees", "--number-of-trees", "-n", type=int, default=101)
    ss_comparison_subparser.add_argument("--alphas", "--aers", "-a", type=float, nargs="+", default=[10 ** -3, 10 ** -6, 0.0])
    ss_comparison_subparser.add_argument("--output-path", "-o", type=str, default=None)
    ss_comparison_subparser.add_argument("--random-seed", "-s", type=int, default=1234)

    tree_distribution_subparser = subparsers.add_parser("tree-distribution")
    tree_distribution_subparser.set_defaults(action_name="positive_tree_distribution")
    tree_distribution_subparser.add_argument("--n-trees", "--number-of-trees", "-n", type=int, default=100)
    tree_distribution_subparser.add_argument("--output-path", "-o", type=str, default=None)
    tree_distribution_subparser.add_argument("--random-seed", "-s", type=int, default=1234)

    return parser.parse_args()


def main():
    args = parse_args()

    configure_logging()
    random.seed(args.random_seed)
    pd.options.mode.chained_assignment = None

    datasets = load_datasets()

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        if args.action_name == "empirical_comparison":
            get_and_show_error_rates_and_runtimes(
                args.n_trees,
                datasets,
                args.alphas,
                {
                    "Greedy": get_greedy_ss,
                    "Minimax": get_minimax_ss,
                    "Bayesian": get_bayesian_ss
                }
            )
        elif args.action_name == "positive_tree_distribution":
            plot_smopdises(args.n_trees, datasets)

    output_path = args.output_path or get_output_path(f"{args.action_name}_{args.n_trees}_trees")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
