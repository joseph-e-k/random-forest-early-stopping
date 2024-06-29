import argparse
import functools
import random
import warnings
from collections import Counter

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
from ste.utils import DATASETS, covariates_response_split, get_output_path, memoize


_logger = get_module_logger()


def to_binary_classifications(classifications):
    classes = set(classifications)
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Cannot make binary classification from fewer than 2 classes")

    positive_classes = np.random.choice(np.array(list(classes)), size=n_classes // 2, replace=False)

    return np.isin(classifications, positive_classes)


def coerce_nonnumeric_columns_to_numeric(df: pd.DataFrame):
    object_columns = df.select_dtypes(["object"]).columns
    df[object_columns] = df[object_columns].astype("category")
    category_columns = df.select_dtypes(["category"]).columns
    df[category_columns] = df[category_columns].apply(lambda x: x.cat.codes)
    return df


def _estimate_positive_tree_distribution_single_forest(dataset: pd.DataFrame, *, n_trees=100, test_proportion=0.2, response_column=-1):
    # Processing: get covariates and responses, convert responses to binary classes, and split into train and test sets
    X, y = covariates_response_split(dataset, response_column)
    X = coerce_nonnumeric_columns_to_numeric(X)
    y = to_binary_classifications(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion)

    # Train a random forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=n_trees)
    rf_classifier.fit(X_train, y_train)

    X_test_pos = X_test[y_test]
    X_test_neg = X_test[np.logical_not(y_test)]

    # Count number of "positive" trees for each testing observation, and add them up
    tree_predictions = np.array(np.vstack([tree.predict(X_test) for tree in rf_classifier.estimators_]), dtype=int)
    tree_predictions_for_pos = np.asarray(np.vstack([tree.predict(X_test_pos) for tree in rf_classifier.estimators_]), dtype=int)
    tree_predictions_for_neg = np.asarray(np.vstack([tree.predict(X_test_neg) for tree in rf_classifier.estimators_]), dtype=int)

    return np.stack([
        np.bincount(np.sum(tree_predictions, axis=0), minlength=n_trees + 1),
        np.bincount(np.sum(tree_predictions_for_pos, axis=0), minlength=n_trees + 1),
        np.bincount(np.sum(tree_predictions_for_neg, axis=0), minlength=n_trees + 1)
    ])


@memoize()
def estimate_positive_tree_distribution(dataset: pd.DataFrame, *, n_trees=100, n_forests=100, test_proportion=0.2, response_column=-1):
    _logger.info(f"Estimating positive tree distribution with {n_forests} forests of {n_trees} trees")
    estimates = np.empty(shape=(n_forests, 3, n_trees + 1))

    task_outcomes = parallelize(
        functools.partial(
            _estimate_positive_tree_distribution_single_forest,
            dataset=dataset,
            n_trees=n_trees,
            test_proportion=test_proportion,
            response_column=response_column
        ),
        argses_to_iter=[()] * n_forests
    )

    for outcome in task_outcomes:
        estimates[outcome.index, :, :] = outcome.result

    return np.mean(estimates, axis=0)


def plot_n_positive_distributions(n_trees, datasets):
    fig, axs = create_subplot_grid(len(datasets), n_rows=2)
    n_rows = axs.shape[0]

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        distribution = estimate_positive_tree_distribution(dataset, n_trees=n_trees)[0]

        ax = axs[i_dataset // n_rows, i_dataset % n_rows]
        ax.bar(np.arange(n_trees + 1), distribution, width=1)
        ax.title.set_text(f"{dataset_name}")
        ax.set_xlim((0, n_trees))
        ax.set_yticks([])

    return fig


def _apply_ss_getter(dataset, aer, ss_getter, n_trees):
    return ss_getter(n_trees, aer, dataset)


def _get_stopping_strategies(n_trees, datasets, aers, stopping_strategy_getters):
    stopping_strategies = np.empty(
        shape=(
            len(datasets),
            len(aers),
            len(stopping_strategy_getters),
            n_trees + 1,
            n_trees + 1
        )
    )

    task_outcomes = parallelize(
        functools.partial(_apply_ss_getter, n_trees=n_trees),
        argses_to_combine=(datasets, aers, stopping_strategy_getters)
    )

    for outcome in task_outcomes:
        stopping_strategies[outcome.index] = outcome.result

    return stopping_strategies


def _analyse_stopping_strategy_if_relevant(i_dataset, i_aer, i_ss_kind, n_positive_trees, n_trees, positive_tree_counters, stopping_strategies):
    if positive_tree_counters[i_dataset][n_positive_trees] == 0:
        return None

    ss = stopping_strategies[i_dataset, i_aer, i_ss_kind]
    forest = Forest(n_trees, n_positive_trees)
    fwss = ForestWithGivenStoppingStrategy(forest, ss)

    return fwss.analyse()


def get_error_rates_and_runtimes(n_trees, datasets, aers, stopping_strategy_getters):
    n_datasets = len(datasets)
    n_aers = len(aers)
    n_ss_kinds = len(stopping_strategy_getters)

    positive_tree_counters = [None] * n_datasets
    runtimes = np.zeros((n_datasets, n_aers, n_ss_kinds))
    error_rates = np.zeros_like(runtimes)

    stopping_strategies = _get_stopping_strategies(n_trees, datasets.values(), aers, stopping_strategy_getters)

    for i_dataset, dataset in enumerate(datasets.values()):
        positive_tree_distribution = estimate_positive_tree_distribution(dataset, n_trees=n_trees)[0]
        positive_tree_counters[i_dataset] = Counter({
            int(index): int(value)
            for (index, value) in enumerate(positive_tree_distribution)
        })

    task_outcomes = parallelize(
        functools.partial(
            _analyse_stopping_strategy_if_relevant,
            n_trees=n_trees,
            positive_tree_counters=positive_tree_counters,
            stopping_strategies=stopping_strategies
        ),
        argses_to_combine=(
            range(n_datasets),
            range(n_aers),
            range(n_ss_kinds),
            range(n_trees + 1),
        )
    )

    for outcome in task_outcomes:
        if outcome.result is None:
            continue
        _, _, _, n_positive_trees, *_ = outcome.args_or_kwargs
        i_dataset, i_aer, i_ss_kind, _ = outcome.index
        weights = positive_tree_counters[i_dataset]
        weight = weights[n_positive_trees]
        runtimes[i_dataset, i_aer, i_ss_kind] += outcome.result.expected_runtime * weight / weights.total()
        error_rates[i_dataset, i_aer, i_ss_kind] += outcome.result.prob_error * weight / weights.total()

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


def get_and_show_error_rates_and_runtimes(n_trees, datasets, allowable_error_rates, analysers_by_name):
    analyser_names, analysers = zip(*analysers_by_name.items())
    

    error_rates, runtimes = get_error_rates_and_runtimes(
        n_trees, datasets, allowable_error_rates, analysers
    )
    show_error_rates_and_runtimes(
        n_trees,
        error_rates,
        runtimes,
        datasets.keys(),
        allowable_error_rates,
        analyser_names or [func.__name__ for func in analysers]
    )


@memoize(args_to_ignore=["dataset"])
def get_greedy_ss(n_trees: int, allowable_error: float, dataset: pd.DataFrame) -> np.ndarray:
    fwe = ForestWithEnvelope.create_greedy(n_trees, n_trees, allowable_error)
    return fwe.get_prob_stop()


@memoize(args_to_ignore=["dataset"])
def get_minimax_ss(n_trees: int, allowable_error: float, dataset: pd.DataFrame) -> np.ndarray:
    return get_optimal_stopping_strategy(n_trees, allowable_error)


@memoize()
def get_bayesian_ss(n_trees: int, allowable_error: float, dataset: pd.DataFrame) -> np.ndarray:
    bs_dataset = dataset.sample(frac=1, replace=True)
    freqs_n_plus = estimate_positive_tree_distribution(bs_dataset, n_trees=n_trees)[0]
    return get_optimal_stopping_strategy(n_trees, allowable_error, freqs_n_plus, error_minimax=False, runtime_minimax=False)


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

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        if args.action_name == "empirical_comparison":
            get_and_show_error_rates_and_runtimes(
                args.n_trees,
                DATASETS,
                args.alphas,
                {
                    "Greedy": get_greedy_ss,
                    "Minimax": get_minimax_ss,
                    "Bayesian": get_bayesian_ss
                }
            )
        elif args.action_name == "positive_tree_distribution":
            plot_n_positive_distributions(args.n_trees, DATASETS)

    output_path = args.output_path or get_output_path(f"{args.action_name}_{args.n_trees}_trees")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
