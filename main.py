import dataclasses
import math
import random

import numpy as np
from numpy import exp
from scipy.special import gammaln
from functools import lru_cache


@lru_cache(1024)
def log_factorial(x):
    return gammaln(x+1)


def probability_of_n_unseen_positive(n_seen, n_seen_positive, n_unseen, n_unseen_positive):
    return exp(
        log_factorial(n_unseen)
        + log_factorial(n_seen + 1)
        + log_factorial(n_seen_positive + n_unseen_positive)
        + log_factorial(n_seen + n_unseen - n_seen_positive - n_unseen_positive)
        - log_factorial(n_unseen_positive)
        - log_factorial(n_unseen - n_unseen_positive)
        - log_factorial(n_seen_positive)
        - log_factorial(n_seen - n_seen_positive)
        - log_factorial(n_seen + n_unseen + 1)
    )


def posterior_probability_of_positive_classification(n_seen, n_seen_positive, n_unseen):
    return 1 - sum(
        probability_of_n_unseen_positive(n_seen, n_seen_positive, n_unseen, n_unseen_positive)
        for n_unseen_positive in range(math.floor((n_seen + n_unseen) / 2 - n_seen_positive) + 1)
    )


def get_early_stopping_classification(n_seen, n_seen_positive, n_unseen, threshold_p):
    p_positive_classification = posterior_probability_of_positive_classification(
        n_seen,
        n_seen_positive,
        n_unseen
    )

    if p_positive_classification > threshold_p:
        return True

    if p_positive_classification < 1 - threshold_p:
        return False

    return None


@dataclasses.dataclass(frozen=True)
class SingleSimulationOutcome:
    n_trees: int
    early_stopping_threshold: float
    p_positive_tree: float
    early_stopping_classification: bool
    early_stopping_index: int
    complete_forest_classification: bool

    @property
    def classifications_match(self):
        return self.early_stopping_classification == self.complete_forest_classification


def simulate_observation(n_trees, early_stopping_threshold, p_positive_tree=None):
    if p_positive_tree is None:
        p_positive_tree = random.random()

    n_positive_trees = 0
    early_stopping_classification = None
    early_stopping_index = None

    for i_tree in range(n_trees):
        if early_stopping_classification is None:
            early_stopping_classification = get_early_stopping_classification(
                i_tree,
                n_positive_trees,
                n_trees - i_tree,
                early_stopping_threshold
            )

            if early_stopping_classification is not None:
                early_stopping_index = i_tree

        if random.random() <= p_positive_tree:
            n_positive_trees += 1

    complete_forest_classification = n_positive_trees > n_trees / 2

    if early_stopping_classification is None:
        early_stopping_classification = complete_forest_classification
        early_stopping_index = n_trees

    return SingleSimulationOutcome(
        n_trees,
        early_stopping_threshold,
        p_positive_tree,
        early_stopping_classification,
        early_stopping_index,
        complete_forest_classification
    )


if __name__ == "__main__":
    random.seed("2023-05-06T16:39")

    n_simulations = 1000
    n_trees = 999

    for early_stopping_threshold in np.linspace(0.8, 0.99, num=20):
        print(f"Threshold: {early_stopping_threshold}")

        simulation_outcomes = [
            simulate_observation(n_trees, early_stopping_threshold=early_stopping_threshold)
            for i_simulation in range(n_simulations)
        ]

        n_matching_outcomes = sum(so.classifications_match for so in simulation_outcomes)
        mean_early_stopping_index = np.mean([so.early_stopping_index for so in simulation_outcomes])

        print(f" Matching outcomes: {n_matching_outcomes} / {n_simulations}")
        print(f" Mean early-stopping length: {mean_early_stopping_index} / {n_trees}")
