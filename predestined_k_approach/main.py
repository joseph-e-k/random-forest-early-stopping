from predestined_k_approach.optimization import get_envelope_greedy_eb
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope


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

    # with TimerContext("Greedy"):
    #     greedy_score_envelope = get_score_envelope(n_total, 0.05)
    # with TimerContext("Exponential"):
    #     max_score_envelope = find_max_score_envelope(n_total, 0.05)
    #
    # print(f"Greedy: {envelope_to_lower_bound_selected_indices(greedy_score_envelope)}")
    # print(f"Exponential: {envelope_to_lower_bound_selected_indices(max_score_envelope)}")
    #
    # forest = Forest(n_total, math.ceil(n_total / 2))
    # print(f"Greedy min score: {ForestWithEnvelope(forest, greedy_score_envelope).get_score(allowable_error)}")
    # print(f"Exponential min score: {ForestWithEnvelope(forest, max_score_envelope).get_score(allowable_error)}")

    envelope = get_envelope_greedy_eb(n_total, allowable_error)

    print(f"Envelope: stop if {describe_envelope(envelope)}")

    for n_positive in range(0, n_total + 1):
        forest_with_envelope = ForestWithEnvelope.create(n_total, n_positive, envelope)
        analysis = forest_with_envelope.analyse()
        score = forest_with_envelope.get_score(allowable_error)
        print(f"{n_positive} positive / {n_total}: "
              f"error rate = {analysis.prob_error}, "
              f"expected runtime = {analysis.expected_runtime}, "
              f"score = {score}, "
              f"score ratio = {score * n_total}")


if __name__ == "__main__":
    main()
