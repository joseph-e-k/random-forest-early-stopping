from matplotlib import pyplot as plt

from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily


def plot_runtime_and_error(n_total, allowable_error, ax_runtime, ax_error):
    greedy_envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
    null_envelope = get_envelope_by_eb_greedily(n_total, 0)

    values_of_n_positive = list(range(n_total + 1))

    greedy_runtimes = []
    guess_runtimes = []

    greedy_errors = []
    guess_errors = [allowable_error] * len(values_of_n_positive)

    for n_positive in values_of_n_positive:
        forest = Forest(n_total, n_positive)

        greedy_fwe = ForestWithEnvelope(forest, greedy_envelope)
        greedy_fwe_analysis = greedy_fwe.analyse()

        null_fwe = ForestWithEnvelope(forest, null_envelope)
        null_fwe_analysis = null_fwe.analyse()

        greedy_runtimes.append(greedy_fwe_analysis.expected_runtime)
        guess_runtimes.append(null_fwe_analysis.expected_runtime * (1 - 2 * allowable_error))

        greedy_errors.append(greedy_fwe_analysis.prob_error)

    ax_runtime.plot(values_of_n_positive, greedy_runtimes, label="Greedy")
    ax_runtime.plot(values_of_n_positive, guess_runtimes, label="Guess")
    ax_runtime.title.set_text(f"Runtime ({n_total} trees, α={allowable_error})")
    ax_runtime.legend()

    ax_error.plot(values_of_n_positive, greedy_errors, label="Greedy")
    ax_error.plot(values_of_n_positive, guess_errors, label="Guess")
    ax_error.title.set_text(f"Error rate ({n_total} trees, α={allowable_error})")
    ax_error.legend()


def main():
    allowable_error_rates = [0.05, 0.01, 0.001]

    fig, axs = plt.subplots(2, len(allowable_error_rates), tight_layout=True)

    for i, allowable_error in enumerate(allowable_error_rates):
        plot_runtime_and_error(101, allowable_error, axs[0, i], axs[1, i])

    plt.show()


if __name__ == "__main__":
    main()
