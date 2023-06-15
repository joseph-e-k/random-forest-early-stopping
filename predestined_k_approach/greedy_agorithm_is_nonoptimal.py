from predestined_k_approach.main import *


def main():
    bare_forest = Forest(100, 51)
    custom_envelope = bare_forest.partial_lower_boundary_to_envelope([0] * 5 + [1, 2])
    forest_with_custom_envelope = ForestWithEnvelope(bare_forest, custom_envelope)
    print(f"Custom envelope: {describe_envelope(custom_envelope)}")
    print(custom_envelope_analysis := forest_with_custom_envelope.analyse())

    greedy_lower_boundary = bare_forest.get_greedy_lower_boundary(custom_envelope_analysis.prob_error)
    greedy_envelope = bare_forest.partial_lower_boundary_to_envelope(greedy_lower_boundary)
    forest_with_greedy_envelope = ForestWithEnvelope(bare_forest, greedy_envelope)
    print(f"Greedy envelope: {describe_envelope(greedy_envelope)}")
    print(forest_with_greedy_envelope.analyse())


if __name__ == "__main__":
    main()
