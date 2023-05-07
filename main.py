import csv
import dataclasses
import math
import random
import shlex
import string
import subprocess

import numpy as np
from numpy import exp
from scipy.special import gammaln
from functools import lru_cache


SIMULATION_OUTCOMES_CSV_OUTPUT_PATH_FORMAT = "./simulation_outcomes_commit_{}_seed_{}.csv"


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


if __name__ == "__main__":
    random_seed = "".join(random.choices(string.ascii_lowercase + string.ascii_uppercase + string.digits, k=8))
    random.seed(random_seed)

    print(f"Seed: {random_seed}")

    n_simulations = 1000
    n_trees = 999
    prior_alpha = 1000
    prior_beta = 1000
    p_positive_tree = 0.5

    all_simulation_outcomes = []

    print(f"Prior distribution of p_positive_tree: Beta({prior_alpha}, {prior_beta})")
    print(f"Actual p_positive_tree: {p_positive_tree}")

    for early_stopping_credence_threshold in np.linspace(0.9, 0.99, num=10):
        print(f"Threshold: {early_stopping_credence_threshold}")

        simulation_outcomes_for_threshold = [
            simulate_observation(
                n_trees,
                early_stopping_credence_threshold,
                p_positive_tree,
                prior_alpha,
                prior_beta
            )
            for i_simulation in range(n_simulations)
        ]

        n_matching_outcomes = sum(so.classifications_match for so in simulation_outcomes_for_threshold)
        mean_early_stopping_index = np.mean([so.early_stopping_index for so in simulation_outcomes_for_threshold])
        mean_early_stopping_credence = np.mean([so.early_stopping_credence for so in simulation_outcomes_for_threshold])

        print(f" Matching outcomes: {n_matching_outcomes} / {n_simulations}")
        print(f" Mean early-stopping index: {mean_early_stopping_index} / {n_trees}")
        print(f" Mean early-stopping credence: {mean_early_stopping_credence}")

        all_simulation_outcomes.extend(simulation_outcomes_for_threshold)

    git_commit_hash = subprocess.run(
        shlex.split("git rev-parse --short HEAD"),
        stdout=subprocess.PIPE
    ).stdout.decode().strip()

    csv_output_path = SIMULATION_OUTCOMES_CSV_OUTPUT_PATH_FORMAT.format(git_commit_hash, random_seed)

    with open(csv_output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(field.name for field in dataclasses.fields(SingleSimulationOutcome))
        writer.writerows(dataclasses.astuple(so) for so in all_simulation_outcomes)
