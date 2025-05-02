import argparse
import functools
import operator
import os
import random
import warnings
from typing import Callable, Sequence

from matplotlib import ticker
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ste.EnsembleVote import EnsembleVote
from ste.EnsembleVoteWithStoppingStrategy import EnsembleVoteWithGivenStoppingStrategy
from ste.utils.figures import (
    DISTINCT_DASH_STYLES, MARKERS, create_independent_plots_grid, create_subplot_grid, enforce_character_limit, plot_functions, save_drawing
)
from ste.utils.logging import configure_logging, get_module_logger
from ste.utils.multiprocessing import parallelize_to_array
from ste.optimization import get_optimal_stopping_strategy
from ste.utils.caching import memoize
from ste.utils.data import Dataset, get_names_and_datasets, split_dataset
from ste.utils.misc import get_output_path, swap_indices_of_axis


_logger = get_module_logger()


class ReproducibilityError(Exception):
    pass


@memoize()
def _split_and_train_and_estimate_smopdises(dataset, n_trees, eval_proportion):
    training_data, evaluation_data = split_dataset(dataset, (1-eval_proportion, eval_proportion))
    forest = RandomForestClassifier(n_trees)
    forest.fit(*training_data)
    return estimate_conditional_smopdises(forest, evaluation_data)


def draw_smopdises(n_trees: int, datasets: Sequence[Dataset], dataset_names: Sequence[str], eval_proportion: float = 0.2, n_forests: int = 30):
    n_datasets = len(datasets)
    if n_datasets % 2 == 0:
        n_columns = 2
    elif n_datasets % 3 == 0:
        n_columns = 3
    else:
        n_columns = 1
    fig, axs = create_independent_plots_grid(n_datasets, n_columns=n_columns, figsize=(5, 3))

    n_actual_positive = [int(np.sum(y)) for (X, y) in datasets]
    n_actual_negative = [int(np.sum(1 - y)) for (X, y) in datasets]
    n_obs = np.array([n_actual_negative, n_actual_positive]).T
    n_eval_obs = np.asarray(n_obs * eval_proportion, dtype=int)

    smopdis_estimates = parallelize_to_array(
        functools.partial(
            _split_and_train_and_estimate_smopdises,
            n_trees=n_trees,
            eval_proportion=eval_proportion
        ),
        reps=n_forests,
        argses_to_combine=[
            datasets
        ]
    )

    mean_smopdises = smopdis_estimates.mean(axis=0) / n_eval_obs[:, :, np.newaxis]

    for i_dataset, (dataset_name, ax) in enumerate(zip(dataset_names, axs.flat)):
        for (actual_class, color, label) in zip((0, 1), ["orange", "blue"], ["False", "True"]):
            ax.bar(np.arange(n_trees + 1), mean_smopdises[i_dataset, actual_class], width=1, color=color, alpha=0.5, label=label)

        ax.title.set_text(f"{dataset_name}")
        ax.set_xlim((-0.5, n_trees + 0.5))
        ax.set_ylim((0, np.max(mean_smopdises[i_dataset]) * 1.05))
        for side in ["top", "left", "right"]:
            ax.spines[side].set_visible(False)

    return fig


def _get_stopping_strategies(n_trees, smopdis_estimated_normally, smopdis_estimated_badly, smopdis_estimated_perfectly, adrs, stopping_strategy_getters):
    return parallelize_to_array(
        operator.call,
        argses_to_combine=[
            [
                functools.partial(
                    ss_getter,
                    n_trees=n_trees,
                    smopdis_estimated_normally=smopdis_estimated_normally,
                    smopdis_estimated_badly=smopdis_estimated_badly,
                    smopdis_estimated_perfectly=smopdis_estimated_perfectly
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
    ensemble_vote = EnsembleVote(n_trees, n_positive_trees)
    evwss = EnsembleVoteWithGivenStoppingStrategy(ensemble_vote, ss)
    analysis = evwss.analyse()

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
    smopdis_estimated_normally = estimate_smopdis(forest, calibration_data)
    smopdis_estimated_perfectly = estimate_smopdis(forest, evaluation_data)
    smopdis_estimated_badly = estimate_smopdis(forest, training_data)

    stopping_strategies = _get_stopping_strategies(n_trees, smopdis_estimated_normally, smopdis_estimated_badly, smopdis_estimated_perfectly, adrs, stopping_strategy_getters)

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


def draw_metrics(n_trees, metrics, dataset_names, allowable_disagreement_rates, ss_names, combine_plots=False):
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

    if combine_plots:
        fig, axs = create_subplot_grid(n_metrics * n_datasets, n_rows=n_datasets, tight_layout=False, figsize=(10, 4 * n_datasets))
        fig.subplots_adjust(hspace=10)
    else:
        fig, axs = create_independent_plots_grid(n_metrics * n_datasets, n_rows=n_datasets, figsize=(4, 4))

    metrics = swap_indices_of_axis(metrics, 1, 2, axis=3)

    metric_names = ["Disagreement Rate", "Error Rate",  "Expected Runtime"]
    metric_maxima = np.max(metrics, axis=(0, 1, 2))

    for i_dataset, dataset_name in enumerate(dataset_names):
        for i_metric in range(n_metrics):
            metric = metrics[:, :, :, i_metric]
            metric_name = metric_names[i_metric]
            ax = axs[i_dataset, i_metric]

            if metric_name == "Disagreement Rate":
                ax.set_yscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
                ax.set_ylim((0, 1))
                ax.plot([0, 1], [0, 1], color="black", label="ADR", linestyle='dashed')
                ax.legend()
            elif metric_name == "Error Rate":
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(functools.partial(enforce_character_limit, max_characters=5)))
                ax.axhline(y=base_error_rates[i_dataset], color="black", label="Base", linestyle='dashed')
                ax.legend()
            elif metric_name == "Expected Runtime":
                ax.set_yticks(list(range(0, int(metric_maxima[i_metric]), 5)), minor=True)
                ax.set_ylim((0, metric_maxima[i_metric]))
            else:
                raise ValueError(f"Unrecognized metric name: {metric_name!r}")
            
            ax.grid(visible=True, axis='y', which='both')
            ax.set_xscale("symlog", linthresh=min(set(allowable_disagreement_rates) - {0}), linscale=0.5)
            ax.set_xlim((min(allowable_disagreement_rates), max(allowable_disagreement_rates)))

            metric_value_getters = [
                functools.partial(_getitem, metric[i_dataset, i_ss])
                for i_ss in range(n_ss_kinds)
            ]
            
            lines = plot_functions(
                ax=ax,
                x_axis_arg_name="index",
                functions=metric_value_getters,
                function_kwargs=dict(
                    index=range(n_adrs)[::-1],
                ),
                concurrently=False,
                labels=ss_names,
                x_axis_values_transform=lambda i_adrs: [allowable_disagreement_rates[i_adr] for i_adr in i_adrs],
                plot_kwargses=[dict(marker=marker, fillstyle="none") for marker in MARKERS]
            )

            for (line, dash_pattern) in zip(lines, DISTINCT_DASH_STYLES):
                line.set_dashes(dash_pattern)

            ax.legend()
            ax.set_title(f"{metric_names[i_metric]} ({dataset_names[i_dataset]})")
            ax.set_xlabel("Allowable disagreement rate")
            ax.figure.canvas.draw()

            xtick_offset = matplotlib.transforms.ScaledTranslation(-0.1, 0, ax.figure.dpi_scale_trans)
            for tick in ax.xaxis.get_major_ticks():
                if tick.get_loc() >= ax.get_xlim()[1]:
                    tick.label1.set_transform(tick.label1.get_transform() + xtick_offset)
    
    return fig


def get_and_draw_disagreement_rates_and_runtimes(n_forests, n_trees, datasets, dataset_names, allowable_disagreement_rates, ss_getters_by_name, combine_plots=False):
    ss_names, ss_getters = zip(*ss_getters_by_name.items())

    metrics = get_metrics(
        n_forests, n_trees, datasets, allowable_disagreement_rates, ss_getters
    )

    mean_metrics = metrics.mean(axis=0)

    return draw_metrics(
        n_trees,
        mean_metrics,
        dataset_names,
        allowable_disagreement_rates,
        ss_names or [func.__name__ for func in ss_getters],
        combine_plots
    )


@memoize(args_to_ignore=["smopdis_estimated_normally", "smopdis_estimated_badly", "smopdis_estimated_perfectly"])
def get_minimax_ss(adr: float, smopdis_estimated_normally: np.ndarray, smopdis_estimated_badly: np.ndarray, smopdis_estimated_perfectly: np.ndarray, n_trees: int) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr)


def get_bayesian_ss(adr: float, smopdis_estimated_normally: np.ndarray, smopdis_estimated_badly: np.ndarray, smopdis_estimated_perfectly: np.ndarray, n_trees: int) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr, smopdis_estimated_normally, disagreement_minimax=False, runtime_minimax=False)


def get_bayesian_bad_ss(adr: float, smopdis_estimated_normally: np.ndarray, smopdis_estimated_badly: np.ndarray, smopdis_estimated_perfectly: np.ndarray, n_trees: int) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr, smopdis_estimated_badly, disagreement_minimax=False, runtime_minimax=False)


def get_bayesian_perfect_ss(adr: float, smopdis_estimated_normally: np.ndarray, smopdis_estimated_badly: np.ndarray, smopdis_estimated_perfectly: np.ndarray, n_trees: int) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr, smopdis_estimated_perfectly, disagreement_minimax=False, runtime_minimax=False)


@memoize(args_to_ignore=["smopdis_estimated_normally", "smopdis_estimated_badly", "smopdis_estimated_perfectly"])
def get_bayesian_flat_ss(adr: float, smopdis_estimated_normally: np.ndarray, smopdis_estimated_badly: np.ndarray, smopdis_estimated_perfectly: np.ndarray, n_trees: int) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, adr, np.ones(shape=(n_trees + 1)), disagreement_minimax=False, runtime_minimax=False)


DEFAULT_ADRS = (0, 10**-4, 10**-3.5, 10**-3, 10**-2.5, 10**-2, 10**-1.5, 5*10**-2, 10**-1)

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
    ss_comparison_subparser.add_argument("--combine-plots", "-c", action="store_true")
    ss_comparison_subparser.add_argument("--benchmark", "-b", action="store_true")

    tree_distribution_subparser = subparsers.add_parser("tree-distribution")
    tree_distribution_subparser.set_defaults(action_name="smopdis")
    tree_distribution_subparser.add_argument("--n-trees", "--number-of-trees", "-n", type=int, default=100)
    tree_distribution_subparser.add_argument("--output-path", "-o", type=str, default=None)
    tree_distribution_subparser.add_argument("--random-seed", "-s", type=int, default=1234)
    tree_distribution_subparser.add_argument("--n-forests", "--number-of-forests", "-f", type=int, default=30)
    tree_distribution_subparser.add_argument("--benchmark", "-b", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    configure_logging()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    if not os.getenv("PYTHONHASHSEED"):
        raise ReproducibilityError("Please set the PYTHONHASHSEED environment variable for reproducible results")

    pd.options.mode.chained_assignment = None

    dataset_names, datasets = get_names_and_datasets(full_benchmark=args.benchmark)

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        if args.action_name == "empirical_comparison":
            drawing = get_and_draw_disagreement_rates_and_runtimes(
                args.n_forests,
                args.n_trees,
                datasets,
                dataset_names,
                args.alphas,
                {
                    "Minimax": get_minimax_ss,
                    "Minimean (Cal)": get_bayesian_ss,
                    "Minimean (Test)": get_bayesian_perfect_ss,
                    "Minimean (Train)": get_bayesian_bad_ss,
                    "Minimean (Flat)": get_bayesian_flat_ss
                },
                args.combine_plots
            )
        elif args.action_name == "smopdis":
            drawing = draw_smopdises(args.n_trees, datasets, dataset_names, n_forests=args.n_forests)
            
    output_path = args.output_path or get_output_path(f"{args.action_name}{'_b' if args.benchmark else ''}_{args.n_forests}_forests_of_{args.n_trees}_trees")
    save_drawing(drawing, output_path)


if __name__ == "__main__":
    main()
