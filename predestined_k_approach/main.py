import math
import sys

from predestined_k_approach.Forest import ForestWithEnvelope, Forest


def get_optimal_envelope(n_total, error_rate):
    n_majority = math.ceil(n_total / 2 + 1 / 2)
    n_minority = math.ceil(n_total / 2 - 1 / 2)

    upper_boundary = Forest(n_total, n_minority).get_optimal_upper_boundary(error_rate)
    lower_boundary = Forest(n_total, n_majority).get_optimal_lower_boundary(error_rate)

    return list(zip(lower_boundary, upper_boundary))


def describe_envelope(envelope):
    lower_boundary = [bounds[0] for bounds in envelope]

    shifts = []

    for i in range(1, len(lower_boundary)):
        if lower_boundary[i] > lower_boundary[i-1]:
            shifts.append((i, lower_boundary[i]))

    return ", ".join(f"< {value} / {index}" for (index, value) in shifts)


def main():
    n_total = 1001
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2 * n_total))
    envelope = get_optimal_envelope(n_total, 0.05)

    print(f"Envelope: stop if {describe_envelope(envelope)}")

    for n_positive in range(0, n_total + 1):
        forest_with_envelope = ForestWithEnvelope.create(n_total, n_positive, envelope)
        analysis = forest_with_envelope.analyse()
        print(f"{n_positive} positive / {n_total}: "
              f"error rate = {analysis.prob_error}, expected runtime = {analysis.expected_runtime}")


if __name__ == "__main__":
    main()
