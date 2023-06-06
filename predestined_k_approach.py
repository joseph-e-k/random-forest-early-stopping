import dataclasses

import numpy as np


@dataclasses.dataclass
class Forest:
    n_total: int
    n_total_positive: int

    def get_negative_observation_probability(self, n_observed, n_observed_positive):
        n_remaining = self.n_total - n_observed
        n_remaining_positive = self.n_total_positive - n_observed_positive

        return (n_remaining - n_remaining_positive) / n_remaining

    def get_positive_observation_probability(self, n_observed, n_observed_positive):
        n_remaining = self.n_total - n_observed
        n_remaining_positive = self.n_total_positive - n_observed_positive

        return n_remaining_positive / n_remaining

    def get_node_probability(self, n_observed, n_observed_positive, prev_layer_probabilities):
        probability = prev_layer_probabilities[n_observed_positive] * self.get_negative_observation_probability(
            n_observed - 1, n_observed_positive
        )

        if n_observed_positive > 0:
            probability += prev_layer_probabilities[n_observed_positive - 1] * self.get_positive_observation_probability(
                n_observed - 1, n_observed_positive - 1
            )

        return probability

    def get_node_probabilities(self, limits=None):
        n_steps = self.n_total + 1
        n_values = self.n_total_positive + 1

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
                    probability = self.get_node_probability(i_step, i_value, prev_layer_probabilities)
                probabilities[i_step][i_value] = probability

        return probabilities


if __name__ == "__main__":
    print(Forest(10, 5).get_node_probabilities())
