from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from predestined_k_approach.eb_experiments import analyse_fwe_or_get_cached, cache
from predestined_k_approach.optimization import get_optimal_stopping_strategy
from predestined_k_approach.utils import covariates_response_split, timed


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


@timed
@cache.memoize()
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
    tree_predictions = np.column_stack([tree.predict(X_test) for tree in rf_classifier.estimators_])
    tree_predictions_for_pos = np.column_stack([tree.predict(X_test_pos) for tree in rf_classifier.estimators_])
    tree_predictions_for_neg = np.column_stack([tree.predict(X_test_neg) for tree in rf_classifier.estimators_])
    return (
        len(y),
        sum(y),
        np.sum(tree_predictions, axis=1),
        np.sum(tree_predictions_for_pos, axis=1),
        np.sum(tree_predictions_for_neg, axis=1)
    )


def plot_n_positive_distributions(n_trees, datasets, nrows=None, ncols=None):
    nrows = nrows or 1
    ncols = ncols or len(datasets) // nrows
    if ncols * nrows != len(datasets):
        raise ValueError("Datasets do not divide evenly into given number of rows")

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, tight_layout=True)

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        n_observations, n_positive_observations, *distributions = estimate_positive_tree_distribution(dataset, n_trees)
        distribution_total, distribution_for_pos, distribution_for_neg = distributions

        ax = axs[i_dataset // nrows, i_dataset % nrows]
        ax.hist(distribution_total)
        ax.title.set_text(f"{dataset_name}")
        ax.set_xlim((0, n_trees))
        ax.set_yticks([])

    return fig


def get_error_rates_and_runtimes(n_trees, datasets, allowable_error_rates, analysis_getters):
    runtimes = np.zeros((len(datasets), len(allowable_error_rates), len(analysis_getters)))
    error_rates = np.zeros_like(runtimes)

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        _, _, positive_tree_distribution, _, _ = estimate_positive_tree_distribution(dataset, n_trees=n_trees)
        weights = Counter(positive_tree_distribution)

        for n_positive_trees in range(n_trees + 1):
            weight = weights[n_positive_trees]
            if weight == 0:
                continue

            for i_aer, allowable_error_rate in enumerate(allowable_error_rates):
                for i_analysis, get_analysis in enumerate(analysis_getters):
                    analysis = get_analysis(n_trees, n_positive_trees, allowable_error_rate)
                    runtimes[i_dataset, i_aer, i_analysis] += analysis.expected_runtime * weight / weights.total()
                    error_rates[i_dataset, i_aer, i_analysis] += analysis.prob_error * weight / weights.total()

    return error_rates, runtimes


def show_error_rates_and_runtimes(error_rates, runtimes, dataset_names, allowable_error_rates, analysis_names):
    assert error_rates.shape == runtimes.shape

    n_datasets, n_aers, n_analyses = error_rates.shape

    assert len(dataset_names) == n_datasets
    assert len(allowable_error_rates) == n_aers
    assert len(analysis_names) == n_analyses

    fig, axs = plt.subplots(2, n_analyses, tight_layout=True)
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


def get_and_show_error_rates_and_runtimes(n_trees, datasets, allowable_error_rates, analysis_getters):
    error_rates, runtimes = get_error_rates_and_runtimes(
        n_trees, datasets, allowable_error_rates, analysis_getters
    )
    show_error_rates_and_runtimes(
        error_rates,
        runtimes,
        datasets.keys(),
        allowable_error_rates,
        [func.__name__ for func in analysis_getters]
    )


@timed
@cache.memoize()
def analyse_optimal_fwss_or_get_cached(n_total, n_positive, allowable_error):
    optimal_stopping_strategy = get_optimal_stopping_strategy(n_total=n_total, allowable_error=allowable_error, precise=True)
    fwss = ForestWithGivenStoppingStrategy(Forest(n_total, n_positive), optimal_stopping_strategy)
    return fwss.analyse()


def main():
    n_trees = 201
    datasets = {
        "Banknotes": pd.read_csv(r"../data/data_banknote_authentication.txt"),
        "Heart Attacks": pd.read_csv(r"../data/heart_attack.csv"),
        "Salaries": pd.read_csv(r"../data/adult.data"),
        "Dry Beans": pd.read_excel(r"../data/dry_beans.xlsx")
    }

    get_and_show_error_rates_and_runtimes(n_trees, datasets, [10 ** -3, 10 ** -6, 0.0], [
        analyse_fwe_or_get_cached,
        analyse_optimal_fwss_or_get_cached
    ])


if __name__ == "__main__":
    main()
