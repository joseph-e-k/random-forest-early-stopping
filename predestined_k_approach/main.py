import itertools
import math

from predestined_k_approach.Forest import ForestWithEnvelope, Forest, ErrorBudgetMetric
from predestined_k_approach.utils import TimerContext


def get_greedy_envelope(n_total, metric):
    n_majority = math.ceil(n_total / 2 + 1 / 2)
    n_minority = math.ceil(n_total / 2 - 1 / 2)

    upper_boundary = Forest(n_total, n_minority).get_greedy_upper_boundary(metric)
    lower_boundary = Forest(n_total, n_majority).get_greedy_lower_boundary(metric)

    return list(zip(lower_boundary, upper_boundary))


def get_error_budget_envelope(n_total, allowable_error):
    return get_greedy_envelope(n_total, ErrorBudgetMetric(allowable_error))


def get_score_envelope(n_total, allowable_error):
    forest_with_envelope = ForestWithEnvelope.create(n_total, math.ceil((n_total + 1) / 2))

    for step in range(1, forest_with_envelope.n_steps):
        old_envelope = list(forest_with_envelope.envelope)
        prev_lower_bound, prev_upper_bound = old_envelope[step - 1]
        new_envelope_prefix = old_envelope[:step] + [(prev_lower_bound + 1, prev_upper_bound)]
        new_envelope = forest_with_envelope.forest.fill_envelope(new_envelope_prefix)

        old_score = forest_with_envelope.get_score(allowable_error)
        forest_with_envelope.update_envelope_suffix(new_envelope[step:])
        new_score = forest_with_envelope.get_score(allowable_error)

        if new_score <= old_score:
            forest_with_envelope.update_envelope_suffix(old_envelope[step:])

    return forest_with_envelope.envelope


def describe_envelope(envelope):
    lower_boundary = [bounds[0] for bounds in envelope]

    shifts = []

    for i in range(1, len(lower_boundary)):
        if lower_boundary[i] > lower_boundary[i-1]:
            shifts.append((i, lower_boundary[i]))

    return ", ".join(f"< {value} / {index}" for (index, value) in shifts)


def envelope_to_lower_bound_selected_indices(envelope):
    increments = []

    for i in range(1, len(envelope)):
        if envelope[i][0] > envelope[i-1][0]:
            increments.append(i)

    return increments


def selected_indices_to_symmetric_envelope(n_total, indices):
    lower_boundary = [0]

    for index in indices:
        last_bound = lower_boundary[-1]
        lower_boundary += [last_bound] * (index - len(lower_boundary)) + [last_bound + 1]

    return Forest(n_total, 0).fill_boundary_to_envelope(lower_boundary, is_upper=False, symmetrical=True)


def powerset(iterable, max_size=None):
    items = list(iterable)

    if max_size is None:
        max_size = len(items)

    return itertools.chain.from_iterable(
        itertools.combinations(items, n) for n in range(max_size + 1)
    )


def find_max_score_envelope(n_total, allowable_error):
    indices = list(range(1, n_total - 1))
    forests = [
        Forest(n_total, n_good)
        for n_good in [math.ceil((n_total + 1) / 2), math.ceil((n_total - 1) / 2)]
    ]

    envelopes = (
        selected_indices_to_symmetric_envelope(n_total, selected_indices)
        for selected_indices in powerset(indices, max_size=math.ceil(len(indices) / 2))
    )

    return max(
        envelopes,
        key=lambda envelope: min(
            ForestWithEnvelope(forest, envelope).get_score(allowable_error)
            for forest in forests
        )
    )


def main():
    n_total = 19
    allowable_error = 0.05
    with TimerContext("Greedy"):
        greedy_score_envelope = get_score_envelope(n_total, 0.05)
    with TimerContext("Exponential"):
        max_score_envelope = find_max_score_envelope(n_total, 0.05)

    print(f"Greedy: {envelope_to_lower_bound_selected_indices(greedy_score_envelope)}")
    print(f"Exponential: {envelope_to_lower_bound_selected_indices(max_score_envelope)}")

    forest = Forest(n_total, math.ceil(n_total / 2))
    print(f"Greedy min score: {ForestWithEnvelope(forest, greedy_score_envelope).get_score(allowable_error)}")
    print(f"Exponential min score: {ForestWithEnvelope(forest, max_score_envelope).get_score(allowable_error)}")

    # envelope = get_error_budget_envelope(n_total, 0.0005)

    # print(f"Base score: {1 / n_total}")
    # print(f"Envelope: stop if {describe_envelope(envelope)}")
    #
    # for n_positive in range(0, n_total + 1):
    #     forest_with_envelope = ForestWithEnvelope.create(n_total, n_positive, envelope)
    #     analysis = forest_with_envelope.analyse()
    #     score = forest_with_envelope.get_score(allowable_error)
    #     print(f"{n_positive} positive / {n_total}: "
    #           f"error rate = {analysis.prob_error}, "
    #           f"expected runtime = {analysis.expected_runtime}, "
    #           f"score = {score}, "
    #           f"score ratio = {score * n_total}")


if __name__ == "__main__":
    main()
