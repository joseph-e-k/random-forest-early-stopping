import os
import random

import numpy as np
from matplotlib import pyplot as plt
from diskcache import Cache
from scipy.special import logsumexp
from scipy.stats import nhypergeom, binom, bernoulli

from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope, ForestAnalysis
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import plot_function_many_curves, plot_function, timed, TimerContext, is_deviant_value

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


def get_expected_run_proportion(n_total, prop_positive, allowable_error):
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


def get_lower_envelope_at_proportion(n_total, proportional_step, allowable_error):
    envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
    step = int(proportional_step * n_total)
    return envelope[step][0] / step


def simulation_scatterplot():
    random.seed(10259)

    fig, (ax_runtimes, ax_error_rates) = plt.subplots(1, 2)

    n_forests = 100
    n_simulations_per_forest = 10_000
    min_n_trees = 6001
    max_n_trees = 7001

    expected_runtimes = np.zeros(n_forests)
    expected_error_rates = np.zeros(n_forests)
    observed_mean_runtimes = np.zeros(n_forests)
    observed_error_rates = np.zeros(n_forests)

    for i_forest in range(n_forests):
        with TimerContext(f"Forest #{i_forest + 1}"):
            n_trees = random.choice(range(min_n_trees, max_n_trees + 1, 2))
            n_positive_trees = random.choice(range(n_trees // 5, n_trees // 3))  # random.choice(range(n_trees))
            allowable_error = random.random() * 0.05

            print(f"Forest {i_forest + 1} / {n_forests}: {n_trees=}, {n_positive_trees=}, {allowable_error=}")

            envelope = get_envelope_by_eb_greedily(n_trees, allowable_error)

            fwe = ForestWithEnvelope.create(n_trees, n_positive_trees, envelope)

            correct_result = (n_positive_trees > n_trees / 2)

            runtimes = np.zeros(n_simulations_per_forest)
            results = np.zeros(n_simulations_per_forest)

            expected_runtime = expected_runtimes[i_forest] = fwe.analyse().expected_runtime
            expected_error_rate = expected_error_rates[i_forest] = fwe.analyse().prob_error

            for i_simulation in range(n_simulations_per_forest):
                runtimes[i_simulation], results[i_simulation] = fwe.simulate()

            observed_mean_runtime = observed_mean_runtimes[i_forest] = np.mean(runtimes)
            observed_error_rate = observed_error_rates[i_forest] = np.mean(results != correct_result)

            is_runtime_deviant = is_deviant_value(expected_runtime, observed_mean_runtime)
            is_error_rate_deviant = is_deviant_value(expected_error_rate, observed_error_rate)

            print(f"  {'* ' if is_runtime_deviant else ''}Runtime: E = {expected_runtime}, O = {observed_mean_runtime}")
            print(f"  {'* ' if is_error_rate_deviant else ''}Error rate: E = {expected_error_rate}, O = {observed_error_rate}")

    ax_runtimes.title.set_text("Runtimes")
    ax_runtimes.scatter(expected_runtimes, observed_mean_runtimes)
    ax_runtimes.plot([min_n_trees, max_n_trees], [min_n_trees, max_n_trees], linestyle="-")

    ax_error_rates.title.set_text("Error Rates")
    ax_error_rates.scatter(expected_error_rates, observed_error_rates)
    ax_error_rates.plot([0, max(expected_error_rates)], [0, max(expected_error_rates)], linestyle="-")

    plt.show()


def forest_with_zero_envelope(n_total, n_positive):
    envelope = get_envelope_by_eb_greedily(n_total, 0)
    return ForestWithEnvelope.create(n_total, n_positive, envelope)


def get_expected_run_proportion_approx_1(n_total, prop_positive, allowable_error, n_stops=1):
    base = get_expected_run_proportion(n_total, prop_positive, 0)
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
    base = get_expected_run_proportion(n_total, 1/2, allowable_error)
    base_entropy = bernoulli(0.5).entropy()
    my_entropy = bernoulli(prop_positive).entropy()

    return base * (my_entropy / base_entropy)


def main():
    ax = plt.subplot()
    ax.grid(visible=True)

    prop_positive = 0.25

    n_total_min = 101
    n_total_max = 3400
    allowable_error = 10**-3

    plot_function_many_curves(
        ax=ax,
        x_axis_arg_name="n_total",
        distinct_curves_arg_name="allowable_error",
        function=get_expected_run_proportion,
        function_kwargs=dict(
            n_total=list(range(n_total_min, n_total_max + 2, 2)),
            prop_positive=prop_positive,
            allowable_error=[10**-2, 10**-3]
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()
