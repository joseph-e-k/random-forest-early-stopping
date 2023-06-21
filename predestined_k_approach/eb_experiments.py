from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.envelopes import describe_envelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import TimerContext


def main():
    n_total = 1001
    allowable_error = 0.05

    with TimerContext("envelope"):
        envelope = get_envelope_by_eb_greedily(n_total, allowable_error)

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
