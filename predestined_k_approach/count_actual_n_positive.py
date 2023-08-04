from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import covariates_response_split


def to_binary_classifications(classifications):
    classes = set(classifications)
    n_classes = len(classes)

    if n_classes < 2:
        raise ValueError("Cannot make binary classification from fewer than 2 classes")

    positive_classes = np.random.choice(np.array(list(classes)), size=n_classes // 2, replace=False)

    return np.isin(classifications, positive_classes)


def estimate_positive_tree_distribution(dataset, n_trees=100, test_proportion=0.2, *, response_column=-1):
    # Processing: get covariates and responses, convert responses to binary classes, and split into train and test sets
    X, y = covariates_response_split(dataset, response_column)
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
        np.sum(tree_predictions, axis=1),
        np.sum(tree_predictions_for_pos, axis=1),
        np.sum(tree_predictions_for_neg, axis=1)
    )


if __name__ == "__main__":
    n_trees = 1001
    datasets = {
        "Banknotes": np.loadtxt(fname=r"..\data\data_banknote_authentication.txt", delimiter=","),
        "Breast Cancer": datasets.load_breast_cancer(),
        "Iris": datasets.load_iris(),
        "Wine": datasets.load_wine(),
        "Digits": datasets.load_digits()
    }

    allowable_error_rates = [0, 0.0001, 0.001, 0.01, 0.05]
    envelopes = {
        aer: get_envelope_by_eb_greedily(n_trees, aer)
        for aer in allowable_error_rates
    }

    runtimes = {
        (i_dataset, aer): 0
        for (i_dataset, aer) in product(range(len(datasets)), allowable_error_rates)
    }

    error_rates = {
        (i_dataset, aer): 0
        for (i_dataset, aer) in product(range(len(datasets)), allowable_error_rates)
    }

    fig, axs = plt.subplots(2, 1, tight_layout=True)

    cached_analyses = {}

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        positive_tree_distribution, _, _ = estimate_positive_tree_distribution(dataset, n_trees=n_trees)
        weights = Counter(positive_tree_distribution)

        for n_positive_trees in range(n_trees + 1):
            weight = weights[n_positive_trees]
            if weight == 0:
                continue

            for allowable_error_rate in allowable_error_rates:
                analysis_key = n_trees, n_positive_trees, allowable_error_rate
                try:
                    analysis = cached_analyses[analysis_key]
                except KeyError:
                    fwe = ForestWithEnvelope.create(n_trees, n_positive_trees, envelopes[allowable_error_rate])
                    analysis = fwe.analyse()
                    cached_analyses[analysis_key] = analysis

                runtimes[i_dataset, allowable_error_rate] += analysis.expected_runtime * weight / weights.total()
                error_rates[i_dataset, allowable_error_rate] += analysis.prob_error * weight / weights.total()

    fig.patch.set_visible(False)

    table_specs = zip(
        ["Expected Runtime", "Error Rate"],
        axs,
        [runtimes, error_rates],
        [2, 6]
    )

    for (title, ax, table_content, precision) in table_specs:
        ax.axis("off")
        ax.axis("tight")
        ax.title.set_text(title)

        cell_format = f"{{:.{precision}f}}"

        ax.table(
            cellText=[
                [cell_format.format(table_content[i_dataset, aer]) for aer in allowable_error_rates]
                for i_dataset in range(len(datasets))
            ],
            rowLabels=list(datasets.keys()),
            colLabels=[str(aer) for aer in allowable_error_rates],
            loc="center"
        )

    plt.show()
