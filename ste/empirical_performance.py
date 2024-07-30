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
from ste.utils.multiprocessing import parallelize_to_array
from ste.optimization import get_optimal_stopping_strategy
from ste.utils.caching import memoize
from ste.utils.data import Dataset, load_datasets, split_dataset
from ste.utils.misc import get_output_path


_logger = get_module_logger()


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

    smopdis_estimates = parallelize_to_array(
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


def _get_stopping_strategies(n_trees, smopdis_estimate, adrs, stopping_strategy_getters):
    return parallelize_to_array(
        operator.call,
        argses_to_combine=[
            [
                functools.partial(
                    ss_getter,
                    n_trees=n_trees,
                    smopdis_estimate=smopdis_estimate
                ) for ss_getter in stopping_strategy_getters
            ],
            adrs
        ],
        job_name="get_ss"
    )


def _analyse_stopping_strategy_if_relevant(i_ss_kind, i_adr, true_class, n_positive_trees, n_trees, smopdis_estimates, stopping_strategies):
    base_ensemble_result = (n_positive_trees > (n_trees // 2))
    is_base_ensemble_correct = (base_ensemble_result == true_class)
    
    out = np.zeros(shape=4, dtype=float)
    out[-1] = 1 - is_base_ensemble_correct

    if smopdis_estimates[true_class, n_positive_trees] == 0:
        return out

    ss = stopping_strategies[i_ss_kind, i_adr]
    forest = Forest(n_trees, n_positive_trees)
    fwss = ForestWithGivenStoppingStrategy(forest, ss)
    analysis = fwss.analyse()

    if is_base_ensemble_correct:
        prob_error = analysis.prob_disagreement
    else:
        prob_error = 1 - analysis.prob_disagreement

    out[:-1] = [analysis.prob_disagreement, analysis.expected_runtime, prob_error]
    return out


def _analyse_stopping_strategies(stopping_strategies, smopdis_estimates):
    n_ss_kinds, n_adrs, n_trees_plus_one, _ = stopping_strategies.shape
    n_trees = n_trees_plus_one - 1

    metrics = parallelize_to_array(
        functools.partial(
            _analyse_stopping_strategy_if_relevant,
            n_trees=n_trees,
            smopdis_estimates=smopdis_estimates,
            stopping_strategies=stopping_strategies
        ),
        argses_to_combine=(
            range(n_ss_kinds),
            range(n_adrs),
            [0, 1],
            range(n_trees + 1),
        ),
        job_name="analyse"
    )

    weighted_metrics = metrics * smopdis_estimates[:, :, np.newaxis]
    weighted_metrics /= smopdis_estimates.sum()
    averaged_metrics = weighted_metrics.sum(axis=(2, 3))

    return averaged_metrics


type StoppingStrategyGetter = Callable[[float, np.ndarray, int], np.ndarray]


def train_forest(n_trees, training_data) -> RandomForestClassifier:
    rf_classifier = RandomForestClassifier(n_estimators=n_trees)
    rf_classifier.fit(*training_data)
    return rf_classifier


def estimate_smopdis(rf_classifier: RandomForestClassifier, dataset: Dataset):
    X, y = dataset
    n_trees = len(rf_classifier.estimators_)
    tree_predictions = np.array(np.vstack([tree.predict(X) for tree in rf_classifier.estimators_]), dtype=int)
    return np.bincount(np.sum(tree_predictions, axis=0), minlength=n_trees + 1)


def estimate_conditional_smopdises(rf_classifier: RandomForestClassifier, dataset: Dataset):
    X, y = dataset

    return np.stack([
        estimate_smopdis(rf_classifier, (X[y == 0], y[y == 0])),
        estimate_smopdis(rf_classifier, (X[y == 1], y[y == 1])),
    ])


def get_metrics_once(data: Dataset, adrs: Sequence[float], n_trees: int, stopping_strategy_getters: list[StoppingStrategyGetter], data_partition_ratios):
    training_data, calibration_data, evaluation_data = split_dataset(data, data_partition_ratios)

    forest = train_forest(n_trees, training_data)
    smopdis_estimates_for_evaluation = estimate_conditional_smopdises(forest, evaluation_data)
    smopdis_estimate_for_bayesian_ss = estimate_smopdis(forest, calibration_data)

    stopping_strategies = _get_stopping_strategies(n_trees, smopdis_estimate_for_bayesian_ss, adrs, stopping_strategy_getters)

    return _analyse_stopping_strategies(stopping_strategies, smopdis_estimates_for_evaluation)


@memoize()
def get_metrics(n_forests, n_trees, datasets, adrs, stopping_strategy_getters, data_partition_ratios=(0.7, 0.1, 0.2)):
    metrics = parallelize_to_array(
        function=functools.partial(
            get_metrics_once,
            n_trees=n_trees,
            data_partition_ratios=data_partition_ratios,
            stopping_strategy_getters=stopping_strategy_getters,
            adrs=adrs
        ),
        reps=n_forests,
        argses_to_combine=[
            datasets
        ]
    )

    return metrics


def _getitem(sequence, index):
    return sequence[index]


def show_metrics(n_trees, metrics, dataset_names, allowable_disagreement_rates, ss_names):
    base_error_rates = metrics[..., -1]

    metrics = metrics[..., :-1]
    n_datasets, n_ss_kinds, n_adrs,  n_metrics = metrics.shape

    assert len(dataset_names) == n_datasets
    assert len(ss_names) == n_ss_kinds
    assert len(allowable_disagreement_rates) == n_adrs
    assert n_metrics == 3

    for i_dataset in range(n_datasets):
        assert np.all(np.isclose(base_error_rates[i_dataset, :, :], base_error_rates[i_dataset, 0, 0]))
    
    base_error_rates = base_error_rates[:, 0, 0]

    fig, axs = create_subplot_grid(n_metrics * n_datasets, n_rows=n_datasets, tight_layout=False, figsize=(10, 15))
    fig.suptitle(f"Empirical performance of early-stopping random forests with {n_trees} trees", fontsize=16)

    metric_names = ["Disagreement Rate", "Expected Runtime", "Error Rate"]
    metric_maxima = np.max(metrics, axis=(0, 1, 2))

    for i_dataset, dataset_name in enumerate(dataset_names):
        for i_metric in range(n_metrics):
            metric = metrics[:, :, :, i_metric]
            ax = axs[i_dataset, i_metric]

            metric_value_getters = [
                functools.partial(_getitem, metric[i_dataset, i_ss])
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
            elif i_metric == 1:
                ax.set_yticks(list(range(0, int(metric_maxima[i_metric]), 5)), minor=True)
                ax.grid(visible=True, axis='y', which='both')
            else:
                ax.set_yscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
                ax.axhline(y=base_error_rates[i_dataset], color="black", label="Base", linestyle='dashed')
                _logger.info(f"Base error rate for {dataset_names[i_dataset]}: {base_error_rates[i_dataset]}")
                ax.legend()

            ax.set_ylim((0, metric_maxima[i_metric]))
            ax.set_xscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
            ax.set_xlim((min(allowable_disagreement_rates), max(allowable_disagreement_rates)))
            ax.set_xlabel("allowable disagreement rate")
            ax.set_title(f"{metric_names[i_metric]} ({dataset_names[i_dataset]})")
    
    plt.show()


def get_and_show_disagreement_rates_and_runtimes(n_forests, n_trees, datasets, dataset_names, allowable_disagreement_rates, ss_getters_by_name):
    ss_names, ss_getters = zip(*ss_getters_by_name.items())

    metrics = get_metrics(
        n_forests, n_trees, datasets, allowable_disagreement_rates, ss_getters
    )

    mean_metrics = metrics.mean(axis=0)

    show_metrics(
        n_trees,
        mean_metrics,
        dataset_names,
        allowable_disagreement_rates,
        ss_names or [func.__name__ for func in ss_getters]
    )


@memoize(args_to_ignore=["smopdis_estimate"])
def get_greedy_ss(adr: float, smopdis_estimate: np.ndarray, n_trees: int) -> np.ndarray:
    fwe = ForestWithEnvelope.create_greedy(n_trees, n_trees, adr)
    return fwe.get_prob_stop()


@memoize(args_to_ignore=["smopdis_estimate"])
def get_minimax_ss(adr: float, smopdis_estimate: np.ndarray, n_trees: int) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr)


def get_bayesian_ss(adr: float, smopdis_estimate: np.ndarray, n_trees: int) -> np.ndarray:
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
