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
    n_total = 101
    envelope = get_optimal_envelope(n_total, 0.05)

    for n_positive in range(0, n_total + 1):
        forest_with_envelope = ForestWithEnvelope.create(n_total, n_positive, envelope)
        analysis = forest_with_envelope.analyse()
        print(f"{n_positive} positive / {n_total}: "
              f"error rate = {analysis.prob_error}, expected runtime = {analysis.expected_runtime}")
