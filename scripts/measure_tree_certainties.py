import functools
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ste.empirical_performance import main
from ste.utils.caching import memoize
from ste.utils.data import get_datasets_with_names, split_dataset
from ste.utils.multiprocessing import parallelize_to_array


def measure_tree_certainties_for_dataset(n_trees, eval_proportion, dataset):
    with warnings.catch_warnings(category=UserWarning, action="ignore"):
        training_data, evaluation_data = split_dataset(dataset, (1-eval_proportion, eval_proportion))
        forest = RandomForestClassifier(n_trees)
        forest.fit(*training_data)
        X_eval, y_eval = evaluation_data
        probs = np.array([tree.predict_proba(X_eval) for tree in forest.estimators_])
        return (np.asarray(probs, dtype=int) == probs).mean()


@memoize()
def measure_tree_certainties_for_each_dataset(n_forests, n_trees, eval_proportion, datasets):
    return parallelize_to_array(
        functools.partial(measure_tree_certainties_for_dataset, n_trees, eval_proportion),
        argses_to_combine=[datasets],
        reps=n_forests
    ).mean(axis=0)


def main():
    datasets_by_name = get_datasets_with_names()
    datasets = list(datasets_by_name.values())

    eval_proportion = 0.1
    n_trees = 5   
    n_forests = 3

    print(measure_tree_certainties_for_each_dataset(n_forests, n_trees, eval_proportion, datasets))


if __name__ == "__main__":
    main()
