import argparse
import os
import random
import warnings
from collections import Counter
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ste.Forest import Forest
from ste.ForestWithEnvelope import ForestWithEnvelope
from ste.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy, ForestAnalysis
from ste.figure_utils import create_subplot_grid
from ste.multiprocessing_utils import parallelize
from ste.optimization import get_optimal_stopping_strategy
from ste.utils import covariates_response_split, memoize


DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "../data")
RESULTS_DIRECTORY = os.path.join(os.path.dirname(__file__), "../results")
DATASETS = {
    "Banknotes": pd.read_csv(os.path.join(DATA_DIRECTORY, "data_banknote_authentication.txt")),
    "Heart Attacks": pd.read_csv(os.path.join(DATA_DIRECTORY, "heart_attack.csv")),
    "Salaries": pd.read_csv(os.path.join(DATA_DIRECTORY, "adult.data")),
    "Dry Beans": pd.read_excel(os.path.join(DATA_DIRECTORY, "dry_beans.xlsx"))
}


@memoize(args_to_ignore=["_"])
def analyse_greedy_fwe_or_get_cached(n_total, n_positive, allowable_error, _) -> ForestAnalysis:
    fwe = ForestWithEnvelope.create_greedy(n_total, n_positive, allowable_error)
    return fwe.analyse()


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


@memoize()
def estimate_positive_tree_distribution(dataset: pd.DataFrame, n_trees=100, test_proportion=0.2, *, response_column=-1):
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

    return (
        len(y),
        sum(y),
        np.bincount(np.sum(tree_predictions, axis=0), minlength=n_trees + 1),
        np.bincount(np.sum(tree_predictions_for_pos, axis=0), minlength=n_trees + 1),
        np.bincount(np.sum(tree_predictions_for_neg, axis=0), minlength=n_trees + 1)
    )


def plot_n_positive_distributions(n_trees, datasets):
    fig, axs = create_subplot_grid(len(datasets), n_rows=2)
    n_rows = axs.shape[0]

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        n_observations, n_positive_observations, *distributions = estimate_positive_tree_distribution(dataset, n_trees)
        distribution_total, distribution_for_pos, distribution_for_neg = distributions

        ax = axs[i_dataset // n_rows, i_dataset % n_rows]
        ax.bar(np.arange(n_trees + 1), distribution_total, width=1)
        ax.title.set_text(f"{dataset_name}")
        ax.set_xlim((0, n_trees))
        ax.set_yticks([])

    return fig


def get_error_rates_and_runtimes(n_trees, datasets, aers, analysers):
    positive_tree_counters = [None] * len(datasets)
    runtimes = np.zeros((len(datasets), len(aers), len(analysers)))
    error_rates = np.zeros_like(runtimes)

    for i_dataset, dataset in enumerate(datasets.values()):
        _, _, positive_tree_distribution, _, _ = estimate_positive_tree_distribution(dataset, n_trees=n_trees)
        positive_tree_counters[i_dataset] = Counter({
            int(index): int(value)
            for (index, value) in enumerate(positive_tree_distribution)
        })

    task_outcomes = parallelize(
        _do_analysis_if_relevant,
        product(
            enumerate(datasets),
            range(n_trees + 1),
            enumerate(aers),
            enumerate(analysers),
        ),
        fixed_args=(n_trees, positive_tree_counters),
        verbose=True
    )

    for (args, success, result, duration) in task_outcomes:
        if not success:
            raise result
        
        if result is None:
            continue

        (i_dataset, dataset), n_positive_trees, (i_aer, aer), (i_analyser, analyser), n_trees, positive_tree_counters = args
        weights = positive_tree_counters[i_dataset]
        weight = weights[n_positive_trees]
        runtimes[i_dataset, i_aer, i_analyser] += result.expected_runtime * weight / weights.total()
        error_rates[i_dataset, i_aer, i_analyser] += result.prob_error * weight / weights.total()

    return error_rates, runtimes


def _do_analysis_if_relevant(*args):
    (i_dataset, dataset), n_positive_trees, (i_aer, aer), (i_analyser, analyser), n_trees, positive_tree_counters = args

    weights = positive_tree_counters[i_dataset]
        
    weight = weights[n_positive_trees]
    if weight == 0:
        return None

    return analyser(n_trees, n_positive_trees, aer, weights)


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
            [2, 6]
        )

        for (title, ax, table_content, precision) in table_specs:
            ax.axis("off")
            ax.axis("tight")
            ax.title.set_text(title)

            cell_format = f"{{:.{precision}f}}"

            ax.table(
                cellText=[
                    [cell_format.format(table_content[i_dataset, i_aer]) for (i_aer, aer) in enumerate(allowable_error_rates)]
                    for i_dataset in range(n_datasets)
                ],
                rowLabels=list(dataset_names),
                colLabels=[str(aer) for aer in allowable_error_rates],
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


@memoize(args_to_ignore=["_"])
def analyse_minimax_fwss_or_get_cached(n_total, n_positive, allowable_error, _):
    optimal_stopping_strategy = get_optimal_stopping_strategy(n_total, allowable_error)
    fwss = ForestWithGivenStoppingStrategy(Forest(n_total, n_positive), optimal_stopping_strategy)
    return fwss.analyse()


@memoize()
def analyse_bayesian_fwss_or_get_cached(n_total, n_positive, allowable_error, weights):
    if (n_total + 1) in weights:
        print(f"{weights=}")

    freqs_n_plus = np.zeros(n_total + 1)
    for key, value in weights.items():
        freqs_n_plus[key] = value
    optimal_stopping_strategy = get_optimal_stopping_strategy(n_total, allowable_error, freqs_n_plus, error_minimax=False, runtime_minimax=False)
    fwss = ForestWithGivenStoppingStrategy(Forest(n_total, n_positive), optimal_stopping_strategy)
    return fwss.analyse()



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

    random.seed(args.random_seed)
    pd.options.mode.chained_assignment = None

    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        if args.action_name == "empirical_comparison":
            get_and_show_error_rates_and_runtimes(
                args.n_trees,
                DATASETS,
                args.alphas,
                {
                    "Greedy": analyse_greedy_fwe_or_get_cached,
                    "Minimax": analyse_minimax_fwss_or_get_cached,
                    "Bayesian": analyse_bayesian_fwss_or_get_cached
                }
            )
        elif args.action_name == "positive_tree_distribution":
            plot_n_positive_distributions(args.n_trees, DATASETS)

    timestamp = datetime.utcnow().isoformat().replace(":", "_").replace(".", "_")
    output_path = args.output_path or os.path.join(RESULTS_DIRECTORY, f"{args.action_name}_{args.n_trees}_trees_{timestamp}")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
