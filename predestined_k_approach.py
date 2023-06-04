import numpy as np


def get_negative_observation_probability(n_total, n_total_positive, n_observed, n_observed_positive):
    n_remaining = n_total - n_observed
    n_remaining_positive = n_total_positive - n_observed_positive

    return (n_remaining - n_remaining_positive) / n_remaining


def get_positive_observation_probability(n_total, n_total_positive, n_observed, n_observed_positive):
    n_remaining = n_total - n_observed
    n_remaining_positive = n_total_positive - n_observed_positive

    return n_remaining_positive / n_remaining


def get_probabilities(n_total, n_total_positive, limits=None):
    n_steps = n_total + 1
    n_values = n_total_positive + 1

    if limits is None:
        limits = [(-np.inf, np.inf)] * n_steps

    probability = [[0] * n_values for _ in range(n_steps)]
    probability[0][0] = 1

    for i_step in range(1, n_steps):
        lower_limit, upper_limit = limits[i_step]

        if 0 < lower_limit:
            probability[i_step][0] = 0
        else:
            probability[i_step][0] = probability[i_step-1][0] * get_negative_observation_probability(
                n_total,
                n_total_positive,
                i_step - 1,
                0
            )

        for i_value in range(1, n_values):
            if i_value < lower_limit or upper_limit < i_value:
                probability[i_step][i_value] = 0
            else:
                probability[i_step][i_value] = (
                    probability[i_step-1][i_value] * get_negative_observation_probability(
                        n_total, n_total_positive, i_step - 1, i_value
                    )
                    + probability[i_step-1][i_value-1] * get_positive_observation_probability(
                        n_total, n_total_positive, i_step - 1, i_value - 1
                    )
                )

    return probability


if __name__ == "__main__":
    print(get_probabilities(10, 5))
