import functools
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ste.utils.caching import memoize
from ste.utils.data import get_datasets_with_names, split_dataset
from ste.utils.logging import configure_logging
from ste.utils.multiprocessing import parallelize_to_array


def split_and_train_and_run_rf(n_trees, eval_proportion, dataset):
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        training_data, evaluation_data = split_dataset(dataset, (1-eval_proportion, eval_proportion))
        forest = RandomForestClassifier(n_trees)
        forest.fit(*training_data)
        X_eval, y_eval = evaluation_data
        return np.array([tree.predict_proba(X_eval) for tree in forest.estimators_])


def compute_metrics_for_dataset(n_trees, eval_proportion, dataset):
    probs = split_and_train_and_run_rf(n_trees, eval_proportion, dataset)
    certainty_proportion = (np.asarray(probs, dtype=int) == probs).mean()
    is_indifferent_to_rounding = (np.round(np.mean(probs)) == np.round(np.mean(np.round(probs))))
    return np.array([certainty_proportion, is_indifferent_to_rounding])


@memoize()
def compute_metrics_for_each_dataset(n_forests, n_trees, eval_proportion, datasets):
    return parallelize_to_array(
        functools.partial(compute_metrics_for_dataset, n_trees, eval_proportion),
        argses_to_combine=[datasets],
        reps=n_forests
    ).mean(0)


def main():
    configure_logging()

    datasets_by_name = get_datasets_with_names()
    datasets = list(datasets_by_name.values())

    eval_proportion = 0.1
    n_trees = 101
    n_forests = 30

    print(compute_metrics_for_each_dataset(n_forests, n_trees, eval_proportion, datasets))


if __name__ == "__main__":
    main()
