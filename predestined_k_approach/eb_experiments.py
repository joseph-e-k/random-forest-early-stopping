from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.envelopes import describe_envelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import TimerContext


def main():
    n_total = 1001
    allowable_error = 0.001

    print(f"Allowable error rate: {allowable_error}")

    with TimerContext("active envelope"):
        active_envelope = get_envelope_by_eb_greedily(n_total, allowable_error)

    with TimerContext("dummy envelope"):
        dummy_envelope = get_envelope_by_eb_greedily(n_total, 0)

    print(f"Active envelope: stop if {describe_envelope(active_envelope)}")

    for n_positive in range(int((n_total - 1) / 2), 0, -1):
        forest = Forest(n_total, n_positive)

        active_fwe = ForestWithEnvelope(forest, active_envelope)
        active_fwe_analysis = active_fwe.analyse()

        dummy_fwe = ForestWithEnvelope(forest, dummy_envelope)
        dummy_fwe_analysis = dummy_fwe.analyse()

        dummy_expected_runtime = dummy_fwe_analysis.expected_runtime * (1 - 2 * allowable_error)

        print(f"{n_positive} positive / {n_total}: "
              f"active error rate = {active_fwe_analysis.prob_error}, "
              f"active expected runtime = {active_fwe_analysis.expected_runtime}, "
              f"dummy expected runtime = {dummy_expected_runtime} "
              f"({'dummy' if dummy_expected_runtime < active_fwe_analysis.expected_runtime else 'active'} wins)")


if __name__ == "__main__":
    main()
