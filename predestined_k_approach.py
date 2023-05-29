import dataclasses
from pprint import pprint

import numpy as np


@dataclasses.dataclass(frozen=True)
class Node:
    n_positive_so_far: int
    n_positive_remaining: int
    n_negative_remaining: int
    log_probability: float

    @property
    def n_total_remaining(self):
        return self.n_positive_remaining + self.n_negative_remaining

    @property
    def log_p_negative(self):
        return np.log(self.n_negative_remaining) - np.log(self.n_total_remaining)

    @property
    def log_p_positive(self):
        return np.log(self.n_positive_remaining) - np.log(self.n_total_remaining)

    def observe_negative(self):
        return Node(
            self.n_positive_so_far,
            self.n_positive_remaining,
            self.n_negative_remaining - 1,
            self.log_probability + self.log_p_negative
        )

    def observe_positive(self):
        return Node(
            self.n_positive_so_far + 1,
            self.n_positive_remaining - 1,
            self.n_negative_remaining,
            self.log_probability + self.log_p_positive
        )


def get_n_nodes(n_total, i_layer):
    if i_layer <= n_total / 2:
        return i_layer + 1
    return n_total - i_layer + 1


def get_probability_network(n_total, n_positive, n_layers=None):
    if n_layers is None:
        n_layers = n_total + 1

    n_negative = n_total - n_positive
    layers: list[list[Node]] = [
        [None] * get_n_nodes(n_total, i_layer)
        for i_layer in range(n_layers)
    ]
    layers[0][0] = Node(0, n_positive, n_negative, np.log(1))

    for i_layer, layer in enumerate(layers[1:], start=1):
        last_layer = layers[i_layer-1]

        layer[0] = last_layer[0].observe_negative()
        layer[-1] = last_layer[-1].observe_positive()

        for i in range(2, len(layer)-1):
            log_probability = np.log(
                np.exp(last_layer[i-1].observe_positive().log_probability)
                + np.exp(last_layer[i].observe_negative().log_probability)
            )

            layer[i] = dataclasses.replace(
                last_layer[i-1].observe_positive(),
                log_probability=log_probability
            )

    return layers


if __name__ == "__main__":
    pprint(
        get_probability_network(2, 1)
    )
