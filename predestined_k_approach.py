import numpy as np


def get_negative_observation_probability(n_total, n_total_positive, n_observed, n_observed_positive):
    n_remaining = n_total - n_observed
    n_remaining_positive = n_total_positive - n_observed_positive

    return (n_remaining - n_remaining_positive) / n_remaining


def get_positive_observation_probability(n_total, n_total_positive, n_observed, n_observed_positive):
    n_remaining = n_total - n_observed
    n_remaining_positive = n_total_positive - n_observed_positive

    return n_remaining_positive / n_remaining


def get_node_probability(n_total, n_total_positive, n_observed, n_observed_positive, prev_layer_probabilities):
    probability = prev_layer_probabilities[n_observed_positive] * get_negative_observation_probability(
        n_total, n_total_positive, n_observed - 1, n_observed_positive
    )

    if n_observed_positive > 0:
        probability += prev_layer_probabilities[n_observed_positive - 1] * get_positive_observation_probability(
            n_total, n_total_positive, n_observed - 1, n_observed_positive - 1
        )

    return probability


def get_node_probabilities(n_total, n_total_positive, limits=None):
    n_steps = n_total + 1
    n_values = n_total_positive + 1

    if limits is None:
        limits = [(-np.inf, np.inf)] * n_steps

    probabilities = [[0] * n_values for _ in range(n_steps)]
    probabilities[0][0] = 1

    for i_step in range(1, n_steps):
        lower_limit, upper_limit = limits[i_step]

        prev_layer_probabilities = probabilities[i_step - 1]

        for i_value in range(n_values):
            if i_value < lower_limit or upper_limit < i_value:
                probability = 0
            else:
                probability = get_node_probability(n_total, n_total_positive, i_step, i_value, prev_layer_probabilities)
            probabilities[i_step][i_value] = probability

    return probabilities


if __name__ == "__main__":
    print(get_node_probabilities(10, 5))
