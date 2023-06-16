import math
import sys

from predestined_k_approach.Forest import ForestWithEnvelope, Forest, ErrorBudgetMetric


def get_greedy_envelope(n_total, metric):
    n_majority = math.ceil(n_total / 2 + 1 / 2)
    n_minority = math.ceil(n_total / 2 - 1 / 2)

    upper_boundary = Forest(n_total, n_minority).get_greedy_upper_boundary(metric)
    lower_boundary = Forest(n_total, n_majority).get_greedy_lower_boundary(metric)

    return list(zip(lower_boundary, upper_boundary))


def get_error_budget_envelope(n_total, allowable_error):
    return get_greedy_envelope(n_total, ErrorBudgetMetric(allowable_error))


def describe_envelope(envelope):
    lower_boundary = [bounds[0] for bounds in envelope]

    shifts = []

    for i in range(1, len(lower_boundary)):
        if lower_boundary[i] > lower_boundary[i-1]:
            shifts.append((i, lower_boundary[i]))

    return ", ".join(f"< {value} / {index}" for (index, value) in shifts)


def main():
    n_total = 101
    allowable_error = 0.05
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2 * n_total))
    envelope = get_error_budget_envelope(n_total, 0.05)

    print(f"Base score: {1 / n_total}")
    print(f"Envelope: stop if {describe_envelope(envelope)}")

    for n_positive in range(0, n_total + 1):
        forest_with_envelope = ForestWithEnvelope.create(n_total, n_positive, envelope)
        analysis = forest_with_envelope.analyse()
        print(f"{n_positive} positive / {n_total}: "
              f"error rate = {analysis.prob_error}, "
              f"expected runtime = {analysis.expected_runtime}, "
              f"score = {forest_with_envelope.get_score(allowable_error)}")


if __name__ == "__main__":
    main()
