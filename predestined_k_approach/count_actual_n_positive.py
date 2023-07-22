import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

    # Count number of "positive" trees for each testing observation, and add them up
    tree_predictions = np.column_stack([tree.predict(X_test) for tree in rf_classifier.estimators_])
    return np.sum(tree_predictions, axis=1)


if __name__ == "__main__":
    datasets = {
        "Banknotes": np.loadtxt(fname=r"..\data\data_banknote_authentication.txt", delimiter=","),
        "Breast Cancer": datasets.load_breast_cancer(),
        "Iris": datasets.load_iris(),
        "Wine": datasets.load_wine(),
        "Digits": datasets.load_digits()
    }

    n_forests_per_dataset = 3

    fig, axs = plt.subplots(len(datasets), n_forests_per_dataset, tight_layout=True)

    for i_dataset, (dataset_name, dataset) in enumerate(datasets.items()):
        for i_forest in range(n_forests_per_dataset):
            distribution = estimate_positive_tree_distribution(dataset)
            axs[i_dataset, i_forest].hist(distribution)
            axs[i_dataset, i_forest].title.set_text(f"{dataset_name} ({i_forest + 1})")

    plt.show()
