from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.main import describe_envelope
from predestined_k_approach.optimization import selected_indices_to_symmetric_envelope, get_envelope_greedy_eb


def main():
    bare_forest = Forest(101, 51)
    custom_envelope = selected_indices_to_symmetric_envelope(bare_forest.n_total, [5, 6, 10, 13])
    forest_with_custom_envelope = ForestWithEnvelope(bare_forest, custom_envelope)
    print(f"Custom envelope: {describe_envelope(custom_envelope)}")
    print(custom_envelope_analysis := forest_with_custom_envelope.analyse())

    greedy_envelope = get_envelope_greedy_eb(bare_forest.n_total, custom_envelope_analysis.prob_error)
    forest_with_greedy_envelope = ForestWithEnvelope(bare_forest, greedy_envelope)
    print(f"Greedy envelope: {describe_envelope(greedy_envelope)}")
    print(forest_with_greedy_envelope.analyse())


if __name__ == "__main__":
    main()
