import dataclasses
import os
import random

import numpy as np
from matplotlib import pyplot as plt, colors
from diskcache import Cache
from scipy.special import logsumexp
from scipy.stats import nhypergeom, binom, bernoulli

from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope, ForestAnalysis
from predestined_k_approach.envelopes import get_null_envelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import plot_function_many_curves, plot_function, timed, TimerContext, \
    plot_functions, is_mean_surprising, is_proportion_surprising

cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))


@cache.memoize()
@timed
def analyse_fwe_or_get_cached(n_total, n_positive, allowable_error) -> ForestAnalysis:
    envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
    fwe = ForestWithEnvelope.create(n_total, n_positive, envelope)
    return fwe.analyse()


def get_expected_runtime(n_total, n_positive, allowable_error):
    if allowable_error == 0:
        n_negative_seen_before_stopping = np.ceil((n_total + 1) / 2)
        rv_n_positive_seen_before_stopping = nhypergeom(n_total, n_positive, np.ceil((n_total + 1) / 2))
        return n_negative_seen_before_stopping + rv_n_positive_seen_before_stopping.mean()

    return analyse_fwe_or_get_cached(n_total, n_positive, allowable_error).expected_runtime


def get_prob_error(n_total, n_positive, allowable_error):
    return analyse_fwe_or_get_cached(n_total, n_positive, allowable_error).prob_error


def get_expected_run_proportion_with_interpolation(n_total, prop_positive, allowable_error):
    f_positive = n_total * prop_positive
    n_positive_lower = int(f_positive)

    if n_positive_lower == f_positive:
        runtime = get_expected_runtime(n_total, n_positive_lower, allowable_error)
    else:
        n_positive_upper = n_positive_lower + 1
        lower_weight = n_positive_upper - f_positive
        upper_weight = 1 - lower_weight

        lower_runtime = get_expected_runtime(n_total, n_positive_lower, allowable_error)
        upper_runtime = get_expected_runtime(n_total, n_positive_upper, allowable_error)

        runtime = lower_runtime * lower_weight + upper_runtime * upper_weight

    return runtime / n_total


def get_expected_run_proportion_without_interpolation(n_total, prop_positive, allowable_error):
    return get_expected_runtime(n_total, int(n_total * prop_positive), allowable_error) / n_total


def get_lower_envelope_at_proportion(n_total, proportional_step, allowable_error):
    envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
    step = int(proportional_step * n_total)
    return envelope[step][0] / step


@dataclasses.dataclass(frozen=True)
class ForestSimulationResults:
    n_trees: int
    n_positive_trees: int
    allowable_error: float
    n_simulations: int

    runtimes: np.ndarray
    conclusions: np.ndarray


@cache.memoize()
def simulate_forest(n_trees, n_positive_trees, allowable_error, n_simulations, random_seed) -> ForestSimulationResults:
    rng = random.Random()
    rng.seed(random_seed)

    envelope = get_envelope_by_eb_greedily(n_trees, allowable_error)

    fwe = ForestWithEnvelope.create(n_trees, n_positive_trees, envelope)

    runtimes = np.zeros(n_simulations)
    results = np.zeros(n_simulations)

    for i_simulation in range(n_simulations):
        runtimes[i_simulation], results[i_simulation] = fwe.simulate(rng=rng)

    return ForestSimulationResults(
        n_trees,
        n_positive_trees,
        allowable_error,
        n_simulations,

        runtimes,
        results
    )


def simulation_scatterplot(n_forests, n_simulations_per_forest, min_n_trees, max_n_trees, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    fig, (ax_runtimes, ax_error_rates) = plt.subplots(1, 2)

    expected_runtimes = np.zeros(n_forests)
    expected_error_rates = np.zeros(n_forests)
    observed_mean_runtimes = np.zeros(n_forests)
    observed_error_rates = np.zeros(n_forests)

    n_surprising_runtimes = 0
    n_surprising_error_rates = 0

    for i_forest in range(n_forests):
        with TimerContext(f"Forest #{i_forest + 1}"):
            n_trees = random.choice(range(min_n_trees, max_n_trees + 1, 2))
            n_positive_trees = random.choice(range(n_trees // 5, n_trees // 3))  # random.choice(range(n_trees))
            allowable_error = random.random() * 0.05

            print(f"Forest {i_forest + 1} / {n_forests}: {n_trees=}, {n_positive_trees=}, {allowable_error=}")

            envelope = get_envelope_by_eb_greedily(n_trees, allowable_error)

            fwe = ForestWithEnvelope.create(n_trees, n_positive_trees, envelope)

            expected_runtime = expected_runtimes[i_forest] = fwe.analyse().expected_runtime
            expected_error_rate = expected_error_rates[i_forest] = fwe.analyse().prob_error

            correct_conclusion = (n_positive_trees > n_trees / 2)

            result: ForestSimulationResults = simulate_forest(
                n_trees,
                n_positive_trees,
                allowable_error,
                n_simulations_per_forest,
                random_seed=random.random()
            )

            wrong_conclusions = result.conclusions != correct_conclusion

            observed_mean_runtimes[i_forest] = np.mean(result.runtimes)
            observed_error_rates[i_forest] = np.mean(wrong_conclusions)

            is_runtime_surprising = is_mean_surprising(result.runtimes, expected_runtime)
            is_error_rate_surprising = is_proportion_surprising(wrong_conclusions, expected_error_rate)

            n_surprising_runtimes += is_runtime_surprising
            n_surprising_error_rates += is_error_rate_surprising

            print(f"  {'* ' if is_runtime_surprising else ''}Runtime: E = {expected_runtime}, O = {observed_mean_runtimes[i_forest]}")
            print(f"  {'* ' if is_error_rate_surprising else ''}Error rate: E = {expected_error_rate}, O = {observed_error_rates[i_forest]}")

    ax_runtimes.title.set_text(f"Runtimes ({n_surprising_runtimes} surprises / {n_forests})")
    ax_runtimes.scatter(expected_runtimes, observed_mean_runtimes)
    ax_runtimes.plot([min_n_trees, max_n_trees], [min_n_trees, max_n_trees], linestyle="-")

    ax_error_rates.title.set_text(f"Error Rates ({n_surprising_error_rates} surprises / {n_forests})")
    ax_error_rates.scatter(expected_error_rates, observed_error_rates)
    ax_error_rates.plot([0, max(expected_error_rates)], [0, max(expected_error_rates)], linestyle="-")

    plt.show()


def forest_with_zero_envelope(n_total, n_positive):
    envelope = get_envelope_by_eb_greedily(n_total, 0)
    return ForestWithEnvelope.create(n_total, n_positive, envelope)


def get_expected_run_proportion_approx_1(n_total, prop_positive, allowable_error, n_stops=1):
    base = get_expected_run_proportion_with_interpolation(n_total, prop_positive, 0)
    log_half = -np.log(2)
    log_aer_remaining = np.log(allowable_error)
    stop_steps = []

    for i_stop in range(n_stops):
        approx_steps_to_stop = np.ceil(-np.log2(allowable_error))

        if approx_steps_to_stop == 0:
            break

        log_aer_remaining = logsumexp([log_aer_remaining, approx_steps_to_stop * log_half], b=[1, -1])

        if log_aer_remaining <= -np.inf:
            break

        stop_steps.append(approx_steps_to_stop)

    stop_steps_cumulative = np.cumsum(stop_steps)

    log_stop_prob = -np.inf

    for i_stop, cumulative_steps in enumerate(stop_steps_cumulative):
        log_stop_prob = logsumexp([log_stop_prob, binom(cumulative_steps, prop_positive).logpmf(i_stop)])
        log_stop_prob = logsumexp([log_stop_prob, binom(cumulative_steps, 1 - prop_positive).logpmf(i_stop)])

    return base * (1 - np.exp(log_stop_prob))


def get_expected_run_proportion_approx_2(n_total, prop_positive, allowable_error):
    p0 = ((n_total - 1) // 2) / n_total
    q0 = 1 - p0

    b = np.log(1 / allowable_error - 1) / (np.log(q0) - np.log(p0))

    p = prop_positive
    q = 1 - p

    expected_steps = abs(q - p) ** (-1) * b * abs(1 - 2 / (1 + (q / p) ** b))

    return expected_steps / n_total


def get_expected_run_proportion_approx_3(n_total, prop_positive, allowable_error):
    base = get_expected_run_proportion_with_interpolation(n_total, 1 / 2, allowable_error)
    base_entropy = bernoulli(0.5).entropy()
    my_entropy = bernoulli(prop_positive).entropy()

    return base * (my_entropy / base_entropy)


def plot_state_probabilities_and_envelope(fwe: ForestWithEnvelope, ax, envelope=None, min_value=None):
    if envelope is None:
        envelope = fwe.envelope

    data = np.array([
        [fwe.get_log_state_probability(n_seen, n_seen_good) for n_seen_good in range(fwe.n_total_positive + 1)]
        for n_seen in range(fwe.n_steps)
    ]).T

    masked_data = np.ma.array(data, mask=np.isinf(data))

    cmap = plt.get_cmap("viridis")
    cmap.set_bad(color="k")

    im = ax.imshow(masked_data, cmap=cmap, interpolation='nearest', origin='lower', vmin=min_value)

    ax.plot(envelope[:, 0] - 0.5, color="red")

    plt.colorbar(im, ax=ax)

    return ax


def show_state_probabilities_and_envelopes_separately(n_total, n_positive, allowable_error_rates):
    n_rows = len(allowable_error_rates)
    n_columns = 2

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_columns)

    null_fwe = ForestWithEnvelope.create(n_total, n_positive, get_null_envelope(n_total))
    fwes = [
        ForestWithEnvelope.create(n_total, n_positive, get_envelope_by_eb_greedily(n_total, aer))
        for aer in allowable_error_rates
    ]

    min_value = min(fwe.get_lowest_finite_log_probability() for fwe in (fwes + [null_fwe]))

    for i, fwe in enumerate(fwes):
        plot_state_probabilities_and_envelope(fwes[0], axs[i, 0], envelope=fwe.envelope, min_value=min_value)
        plot_state_probabilities_and_envelope(fwe, axs[i, 1], min_value=min_value)

    plt.show()


def show_base_forest_state_probabilities_with_envelopes(n_total, n_positive, allowable_error_rates):
    null_fwe = ForestWithEnvelope.create(n_total, n_positive, get_null_envelope(n_total))
    fwes = [
        ForestWithEnvelope.create(n_total, n_positive, get_envelope_by_eb_greedily(n_total, aer))
        for aer in allowable_error_rates
    ]

    ax = plt.subplot()
    plot_state_probabilities_and_envelope(null_fwe, ax)

    for fwe in fwes:
        ax.plot(fwe.envelope[:, 0] - 0.5)

    plt.show()


def main():
    simulation_scatterplot(
        n_forests=100,
        n_simulations_per_forest=1_000,
        min_n_trees=6001,
        max_n_trees=7001,
        random_seed=10259
    )


if __name__ == "__main__":
    main()
