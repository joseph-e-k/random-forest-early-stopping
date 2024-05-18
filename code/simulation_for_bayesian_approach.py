import csv
import datetime
import dataclasses
import math
import os.path
import random
from typing import Callable

import numpy as np
from numpy import exp
from scipy.special import gammaln
from functools import lru_cache

from friendly_distributions import Multinomial, Uniform

OUTPUT_DIRECTORY_FORMAT = "./simulation_outcomes/{}"
DETAILED_OUTPUT_PATH_FORMAT = os.path.join(OUTPUT_DIRECTORY_FORMAT, "details.csv")
SUMMARY_OUTPUT_PATH_FORMAT = os.path.join(OUTPUT_DIRECTORY_FORMAT, "summary.txt")


@lru_cache(1024)
def log_factorial(x):
    return gammaln(x+1)


def probability_of_n_unseen_positives(n_seen, n_seen_positive, n_unseen, n_unseen_positive, prior_alpha, prior_beta):
    return exp(
        log_factorial(n_unseen)
        + log_factorial(prior_alpha + prior_beta + n_seen - 1)
        + log_factorial(prior_alpha + n_seen_positive + n_unseen_positive - 1)
        + log_factorial(prior_beta + n_seen + n_unseen - n_seen_positive - n_unseen_positive - 1)
        - log_factorial(n_unseen_positive)
        - log_factorial(n_unseen - n_unseen_positive)
        - log_factorial(prior_alpha + n_seen_positive - 1)
        - log_factorial(prior_beta + n_seen - n_seen_positive - 1)
        - log_factorial(prior_alpha + prior_beta + n_seen + n_unseen - 1)
    )


def posterior_probability_of_positive_classification(n_seen, n_seen_positive, n_unseen, prior_alpha, prior_beta):
    return 1 - sum(
        probability_of_n_unseen_positives(n_seen, n_seen_positive, n_unseen, n_unseen_positive, prior_alpha, prior_beta)
        for n_unseen_positive in range(math.floor((n_seen + n_unseen) / 2 - n_seen_positive) + 1)
    )


def get_early_classification_and_credence(n_seen, n_seen_positive, n_unseen, threshold_p, prior_alpha, prior_beta):
    p_positive_classification = posterior_probability_of_positive_classification(
        n_seen,
        n_seen_positive,
        n_unseen,
        prior_alpha,
        prior_beta
    )

    if p_positive_classification > threshold_p:
        return True, p_positive_classification

    if p_positive_classification < 1 - threshold_p:
        return False, 1 - p_positive_classification

    return None, None


@dataclasses.dataclass(frozen=True)
class SingleSimulationOutcome:
    prior_alpha: float
    prior_beta: float
    n_trees: int
    early_stopping_credence_threshold: float
    p_positive_tree: float
    early_stopping_classification: bool
    early_stopping_index: int
    early_stopping_credence: float
    complete_forest_classification: bool

    @property
    def classifications_match(self):
        return self.early_stopping_classification == self.complete_forest_classification


def simulate_observation(n_trees, early_stopping_credence_threshold, p_positive_tree=None, prior_alpha=1, prior_beta=1):
    if p_positive_tree is None:
        p_positive_tree = random.random()

    n_positive_trees = 0
    early_stopping_classification = None
    early_stopping_credence = None
    early_stopping_index = None

    for i_tree in range(n_trees):
        if early_stopping_classification is None:
            early_stopping_classification, early_stopping_credence = get_early_classification_and_credence(
                i_tree,
                n_positive_trees,
                n_trees - i_tree,
                early_stopping_credence_threshold,
                prior_alpha,
                prior_beta
            )

            if early_stopping_classification is not None:
                early_stopping_index = i_tree

        if random.random() <= p_positive_tree:
            n_positive_trees += 1

    complete_forest_classification = n_positive_trees > n_trees / 2

    if early_stopping_classification is None:
        early_stopping_classification = complete_forest_classification
        early_stopping_index = n_trees
        early_stopping_credence = 1

    return SingleSimulationOutcome(
        prior_alpha,
        prior_beta,
        n_trees,
        early_stopping_credence_threshold,
        p_positive_tree,
        early_stopping_classification,
        early_stopping_index,
        early_stopping_credence,
        complete_forest_classification
    )


def simulate_observations(
        n_simulations=1000,
        n_trees=999,
        prior_alpha: int | float = 1,
        prior_beta: int | float = 1,
        distrib_p_positive_tree: Callable[[], float] = Uniform(0, 1),
        random_seed=None
):
    timestamp = datetime.datetime.now().isoformat().replace(":", "_").replace(".", "_")
    output_directory_path = OUTPUT_DIRECTORY_FORMAT.format(timestamp)
    os.makedirs(output_directory_path)

    with open(DETAILED_OUTPUT_PATH_FORMAT.format(timestamp), "w", newline="") as detailed_output_file,\
         open(SUMMARY_OUTPUT_PATH_FORMAT.format(timestamp), "w") as summary_output_file:

        if random_seed is None:
            random_seed = timestamp
        random.seed(random_seed)

        outcome_writer = csv.writer(detailed_output_file)
        outcome_writer.writerow(field.name for field in dataclasses.fields(SingleSimulationOutcome))

        def log_summary(message):
            print(message)
            print(message, file=summary_output_file)

        log_summary(f"Seed: {random_seed}")

        log_summary(f"Prior distribution of p_positive_tree: Beta({prior_alpha}, {prior_beta})")
        log_summary(f"Actual distribution of p_positive_tree: {distrib_p_positive_tree}")

        for early_stopping_credence_threshold in np.linspace(0.9, 0.99, num=10):
            log_summary(f"Threshold: {early_stopping_credence_threshold}")

            simulation_outcomes = [
                simulate_observation(
                    n_trees,
                    early_stopping_credence_threshold,
                    distrib_p_positive_tree(),
                    prior_alpha,
                    prior_beta
                )
                for i_simulation in range(n_simulations)
            ]

            n_matching_outcomes = sum(so.classifications_match for so in simulation_outcomes)
            mean_early_stopping_index = np.mean([so.early_stopping_index for so in simulation_outcomes])
            mean_early_stopping_credence = np.mean([so.early_stopping_credence for so in simulation_outcomes])

            log_summary(f" Matching outcomes: {n_matching_outcomes} / {n_simulations}")
            log_summary(f" Mean early-stopping index: {mean_early_stopping_index} / {n_trees}")
            log_summary(f" Mean early-stopping credence: {mean_early_stopping_credence}")

            outcome_writer.writerows(dataclasses.astuple(so) for so in simulation_outcomes)


if __name__ == "__main__":
    heavy_fringe_distribution = Multinomial([0.1, 0.9])
    heavy_center_distribution = Multinomial([0.499, 0.501])

    simulate_observations()
    simulate_observations(distrib_p_positive_tree=heavy_fringe_distribution)
    simulate_observations(distrib_p_positive_tree=heavy_fringe_distribution, prior_alpha=0.1, prior_beta=0.1)
    simulate_observations(distrib_p_positive_tree=heavy_center_distribution)
    simulate_observations(
        distrib_p_positive_tree=heavy_center_distribution,
        prior_alpha=10000,
        prior_beta=10000,
        n_simulations=1
    )
