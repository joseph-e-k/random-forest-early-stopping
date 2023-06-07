import math
import sys
from pprint import pprint

from predestined_k_approach.Forest import ForestWithEnvelope, Forest


def get_optimal_envelope(n_total, error_rate):
    n_majority = math.ceil(n_total / 2 + 1 / 2)
    n_minority = math.ceil(n_total / 2 - 1 / 2)

    upper_boundary = Forest(n_total, n_minority).get_optimal_upper_boundary(error_rate)
    lower_boundary = Forest(n_total, n_majority).get_optimal_lower_boundary(error_rate)

    return tuple(zip(lower_boundary, upper_boundary))


if __name__ == "__main__":
    forest = Forest(100, 51)
    forest_with_envelope = ForestWithEnvelope(forest, get_optimal_envelope(100, 0.05))

    pprint(forest_with_envelope.analyse())
