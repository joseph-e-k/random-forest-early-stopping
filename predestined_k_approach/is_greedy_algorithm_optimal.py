import math

from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.envelopes import increments_to_symmetric_envelope
from predestined_k_approach.utils import iter_unique_combinations


def main():
    n_total = 101
    n_good = math.ceil((n_total + 1) / 2)
    bare_forest = Forest(n_total, n_good)

    for i, increments in enumerate(iter_unique_combinations(range(1, n_total - 1), 2)):
        if i % 50 == 0:
            print(i)

        custom_envelope = increments_to_symmetric_envelope(n_total, increments)
        forest_with_custom_envelope = ForestWithEnvelope(bare_forest, custom_envelope)
        custom_envelope_analysis = forest_with_custom_envelope.analyse()

        forest_with_greedy_envelope = ForestWithEnvelope.create_greedy(
            n_total,
            n_good,
            custom_envelope_analysis.prob_error
        )
        greedy_envelope_analysis = forest_with_greedy_envelope.analyse()

        if greedy_envelope_analysis.expected_runtime > custom_envelope_analysis.expected_runtime:
            print(increments)
            break


if __name__ == "__main__":
    main()
