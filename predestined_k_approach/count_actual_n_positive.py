from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from predestined_k_approach.eb_experiments import analyse_fwe_or_get_cached
from predestined_k_approach.utils import covariates_response_split


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


def show_n_positive_distributions(n_trees, datasets):
    fig, axs = plt.subplots(nrows=1, ncols=len(datasets), tight_layout=True)

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        n_observations, n_positive_observations, *distributions = estimate_positive_tree_distribution(dataset, n_trees)
        distribution_total, distribution_for_pos, distribution_for_neg = distributions

        ax = axs[i_dataset]
        ax.hist((distribution_for_pos, distribution_for_neg), stacked=True)
        ax.title.set_text(f"{dataset_name} ({n_positive_observations} / {n_observations})")
        ax.set_xlim((0, n_trees))

    plt.show()


def show_error_rates_and_runtimes(n_trees, datasets, allowable_error_rates):
    runtimes = {
        (i_dataset, aer): 0
        for (i_dataset, aer) in product(range(len(datasets)), allowable_error_rates)
    }

    error_rates = {
        (i_dataset, aer): 0
        for (i_dataset, aer) in product(range(len(datasets)), allowable_error_rates)
    }

    fig, axs = plt.subplots(2, 1, tight_layout=True)

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        _, _, positive_tree_distribution, _, _ = estimate_positive_tree_distribution(dataset, n_trees=n_trees)
        weights = Counter(positive_tree_distribution)

        for n_positive_trees in range(n_trees + 1):
            weight = weights[n_positive_trees]
            if weight == 0:
                continue

            for allowable_error_rate in allowable_error_rates:
                analysis = analyse_fwe_or_get_cached(n_trees, n_positive_trees, allowable_error_rate)
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


def main():
    n_trees = 1001
    datasets = {
        "Banknotes": pd.read_csv(r"..\data\data_banknote_authentication.txt"),
        "Heart Attacks": pd.read_csv(r"..\data\heart_attack.csv"),
        "Salaries": pd.read_csv(r"..\data\adult.data"),
        "Dry Beans": pd.read_excel(r"..\data\dry_beans.xlsx")
    }

    show_n_positive_distributions(n_trees, datasets)


if __name__ == "__main__":
    main()
