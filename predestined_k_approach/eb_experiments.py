from matplotlib import pyplot as plt

from predestined_k_approach.Forest import Forest
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily


def plot_runtime_for_different_sized_forests(values_of_n_total, allowable_error, ax):
    for n_total in values_of_n_total:
        envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
        runtimes = []
        min_n_positive = 0
        max_n_positive = n_total // 2
        values_of_n_positive = list(range(min_n_positive, max_n_positive + 1))

        for n_positive in values_of_n_positive:
            forest = Forest(n_total, n_positive)
            fwe = ForestWithEnvelope(forest, envelope)
            runtimes.append(fwe.analyse().expected_runtime)

        ax.plot(
            [n_positive / n_total for n_positive in values_of_n_positive],
            [runtime / n_total for runtime in runtimes],
            label=f"{n_total} trees"
        )

    ax.title.set_text(f"Runtime (α={allowable_error})")
    ax.legend()


def plot_runtime_for_allowable_error_rates(n_total, allowable_error_rates, ax):
    runtimes = {
        aer: []
        for aer in allowable_error_rates
    }

    envelopes = {
        aer: get_envelope_by_eb_greedily(n_total, aer)
        for aer in allowable_error_rates
    }

    values_of_n_positive = list(range(n_total // 2))

    for n_positive in values_of_n_positive:
        forest = Forest(n_total, n_positive)

        for aer in allowable_error_rates:
            fwe = ForestWithEnvelope(forest, envelopes[aer])
            runtimes[aer].append(fwe.analyse().expected_runtime)

    for aer in allowable_error_rates:
        ax.plot(values_of_n_positive, runtimes[aer], label=f"α={aer}")

    ax.title.set_text(f"Runtime ({n_total} trees, various α)")
    ax.legend()


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
    ax = plt.subplot()
    plot_runtime_for_different_sized_forests(
        values_of_n_total=[200 * i + 1 for i in range(1, 6)],
        allowable_error=10**-4,
        ax=ax
    )
    plt.show()


if __name__ == "__main__":
    main()
