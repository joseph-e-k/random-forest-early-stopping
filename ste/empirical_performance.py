import argparse
import functools
import operator
import random
import warnings
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ste.Forest import Forest
from ste.ForestWithEnvelope import ForestWithEnvelope
from ste.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from ste.utils.figures import create_subplot_grid, plot_functions
from ste.utils.logging import configure_logging, get_module_logger
from ste.utils.multiprocessing import parallelize
from ste.optimization import get_optimal_stopping_strategy
from ste.utils.caching import memoize
from ste.utils.data import Dataset, load_datasets, split_dataset
from ste.utils.misc import get_output_path


def _split_and_train_and_estimate_smopdis(dataset, n_trees, eval_proportion):
    training_data, evaluation_data = split_dataset(dataset, (1-eval_proportion, eval_proportion))
    forest = RandomForestClassifier(n_trees)
    forest.fit(*training_data)
    return estimate_smopdis(forest, evaluation_data)


def plot_smopdises(n_trees: int, datasets: Sequence[Dataset], dataset_names: Sequence[str], eval_proportion: float = 0.2, n_forests: int = 30):
    fig, axs = create_subplot_grid(len(datasets), n_rows=2)
    try:
        n_columns = axs.shape[1]
    except IndexError:
        n_columns = 1

    smopdis_estimates = np.zeros((n_forests, len(datasets), n_trees + 1))

    tasks = parallelize(
        functools.partial(
            _split_and_train_and_estimate_smopdis,
            n_trees=n_trees,
            eval_proportion=eval_proportion
        ),
        reps=n_forests,
        argses_to_combine=[
            datasets
        ]
    )

    for task in tasks:
        smopdis_estimates[task.index] = task.result

    mean_smopdises = smopdis_estimates.mean(axis=0)

    for i_dataset, dataset_name in enumerate(dataset_names):
        if (len(axs.shape) == 1):
            ax = axs[i_dataset]
        else:
            ax = axs[i_dataset // n_columns, i_dataset % n_columns]
        ax.bar(np.arange(n_trees + 1), mean_smopdises[i_dataset], width=1)
        ax.title.set_text(f"{dataset_name}")
        ax.set_xlim((0, n_trees))
        ax.set_yticks([])

    return fig


def _get_stopping_strategies(n_trees, smopdis_estimate, adr, stopping_strategy_getters):
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
                adr=adr,
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
    disagreement_rates = np.zeros_like(runtimes)

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
        disagreement_rates[i_ss_kind] += task.result.prob_disagreement * prob_n_positive_trees

    return disagreement_rates, runtimes


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


def get_disagreement_rates_and_runtimes_once(data: Dataset, adr: float, n_trees: int, stopping_strategy_getters: list[StoppingStrategyGetter], data_partition_ratios):
    training_data, calibration_data_for_evaluation, calibration_data_for_bayesian_ss, testing_data = split_dataset(data, data_partition_ratios)

    forest = train_forest(n_trees, training_data)
    smopdis_estimate_for_evaluation = estimate_smopdis(forest, calibration_data_for_evaluation)
    smopdis_estimate_for_bayesian_ss = estimate_smopdis(forest, calibration_data_for_bayesian_ss)

    stopping_strategies = _get_stopping_strategies(n_trees, smopdis_estimate_for_bayesian_ss, adr, stopping_strategy_getters)

    return _analyse_stopping_strategies(stopping_strategies, smopdis_estimate_for_evaluation)


@memoize()
def get_disagreement_rates_and_runtimes(n_forests, n_trees, datasets, adrs, stopping_strategy_getters, data_partition_ratios=(0.6, 0.1, 0.1, 0.2)):
    disagreement_rates = np.zeros((n_forests, len(datasets), len(adrs), len(stopping_strategy_getters)))
    runtimes = np.zeros_like(disagreement_rates)
    
    tasks = parallelize(
        function=functools.partial(
            get_disagreement_rates_and_runtimes_once,
            n_trees=n_trees,
            data_partition_ratios=data_partition_ratios,
            stopping_strategy_getters=stopping_strategy_getters
        ),
        reps=n_forests,
        argses_to_combine=[
            datasets,
            adrs
        ]
    )

    for task in tasks:
        task_disagreement_rates, task_runtimes = task.result
        disagreement_rates[task.index] = task_disagreement_rates
        runtimes[task.index] = task_runtimes

    return disagreement_rates, runtimes


def _getitem(sequence, index):
    return sequence[index]


def show_disagreement_rates_and_runtimes(n_trees, disagreement_rates, runtimes, dataset_names, allowable_disagreement_rates, ss_names):
    assert disagreement_rates.shape == runtimes.shape

    n_datasets, n_adrs, n_ss_kinds = disagreement_rates.shape

    assert len(dataset_names) == n_datasets
    assert len(allowable_disagreement_rates) == n_adrs
    assert len(ss_names) == n_ss_kinds

    fig, axs = create_subplot_grid(2 * n_datasets, n_rows=n_datasets, tight_layout=False, figsize=(10, 15))
    fig.suptitle(f"Empirical performance of early-stopping random forests with {n_trees} trees", fontsize=16)

    metric_names = ["Disagreement Rate", "Expected Runtime"]
    metric_maxima = [np.max(disagreement_rates), np.max(runtimes)]

    for i_dataset, dataset_name in enumerate(dataset_names):
        for i_metric, metric in enumerate([disagreement_rates, runtimes]):
            ax = axs[i_dataset, i_metric]

            metric_value_getters = [
                functools.partial(_getitem, metric[i_dataset, :, i_ss])
                for i_ss in range(n_ss_kinds)
            ]
            
            plot_functions(
                ax=ax,
                x_axis_arg_name="index",
                functions=metric_value_getters,
                function_kwargs=dict(
                    index=range(n_adrs)[::-1],
                ),
                concurrently=False,
                labels=ss_names,
                x_axis_values_transform=lambda i_adrs: [allowable_disagreement_rates[i_adr] for i_adr in i_adrs],
                plot_kwargs=dict(marker="o")
            )

            if i_metric == 0:
                ax.set_yscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
                ax.plot([0, 1], [0, 1], color="black", label="ADR", linestyle='dashed')
                ax.legend()
            else:
                ax.set_yticks(list(range(0, int(metric_maxima[i_metric]), 5)), minor=True)
                ax.grid(visible=True, axis='y', which='both')

            ax.set_ylim((0, metric_maxima[i_metric]))
            ax.set_xscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
            ax.set_xlim((min(allowable_disagreement_rates), max(allowable_disagreement_rates)))
            ax.set_xlabel("allowable disagreement rate")
            ax.set_title(f"{metric_names[i_metric]} ({dataset_names[i_dataset]})")
    
    plt.show()


def get_and_show_disagreement_rates_and_runtimes(n_forests, n_trees, datasets, dataset_names, allowable_disagreement_rates, ss_getters_by_name):
    ss_names, ss_getters = zip(*ss_getters_by_name.items())

    disagreement_rates, runtimes = get_disagreement_rates_and_runtimes(
        n_forests, n_trees, datasets, allowable_disagreement_rates, ss_getters
    )

    mean_disagreement_rates = disagreement_rates.mean(axis=0)
    mean_runtimes = runtimes.mean(axis=0)

    show_disagreement_rates_and_runtimes(
        n_trees,
        mean_disagreement_rates,
        mean_runtimes,
        dataset_names,
        allowable_disagreement_rates,
        ss_names or [func.__name__ for func in ss_getters]
    )


@memoize(args_to_ignore=["smopdis_estimate"])
def get_greedy_ss(n_trees: int, adr: float, smopdis_estimate: np.ndarray) -> np.ndarray:
    fwe = ForestWithEnvelope.create_greedy(n_trees, n_trees, adr)
    return fwe.get_prob_stop()


@memoize(args_to_ignore=["smopdis_estimate"])
def get_minimax_ss(n_trees: int, adr: float, smopdis_estimate: np.ndarray) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr)


def get_bayesian_ss(n_trees: int, adr: float, smopdis_estimate: np.ndarray) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr, smopdis_estimate, disagreement_minimax=False, runtime_minimax=False)


DEFAULT_ADRS = tuple(10 ** -(i/2) for i in range(2, 13)) + (0,)

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    ss_comparison_subparser = subparsers.add_parser("ss-comparison")
    ss_comparison_subparser.set_defaults(action_name="empirical_comparison")
    ss_comparison_subparser.add_argument("--n-trees", "--number-of-trees", "-n", type=int, default=101)
    ss_comparison_subparser.add_argument("--alphas", "--adrs", "-a", type=float, nargs="+", default=DEFAULT_ADRS)
    ss_comparison_subparser.add_argument("--output-path", "-o", type=str, default=None)
    ss_comparison_subparser.add_argument("--random-seed", "-s", type=int, default=1234)
    ss_comparison_subparser.add_argument("--n-forests", "--number-of-forests", "-f", type=int, default=30)

    tree_distribution_subparser = subparsers.add_parser("tree-distribution")
    tree_distribution_subparser.set_defaults(action_name="smopdis")
    tree_distribution_subparser.add_argument("--n-trees", "--number-of-trees", "-n", type=int, default=100)
    tree_distribution_subparser.add_argument("--output-path", "-o", type=str, default=None)
    tree_distribution_subparser.add_argument("--random-seed", "-s", type=int, default=1234)
    tree_distribution_subparser.add_argument("--n-forests", "--number-of-forests", "-f", type=int, default=30)

    return parser.parse_args()


def main():
    args = parse_args()

    configure_logging()
    random.seed(args.random_seed)
    pd.options.mode.chained_assignment = None

    dataset_names, datasets = load_datasets()

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        if args.action_name == "empirical_comparison":
            get_and_show_disagreement_rates_and_runtimes(
                args.n_forests,
                args.n_trees,
                datasets,
                dataset_names,
                args.alphas,
                {
                    "Greedy": get_greedy_ss,
                    "Minimax": get_minimax_ss,
                    "Bayesian": get_bayesian_ss
                }
            )
        elif args.action_name == "smopdis":
            plot_smopdises(args.n_trees, datasets, dataset_names, n_forests=args.n_forests)

    output_path = args.output_path or get_output_path(f"{args.action_name}_{args.n_forests}_forests_of_{args.n_trees}_trees")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
