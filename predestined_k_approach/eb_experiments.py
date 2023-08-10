import numpy as np
from matplotlib import pyplot as plt
from diskcache import Cache

from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope, ForestAnalysis
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import plot_function_many_curves, plot_function, timed, rolling_average

cache = Cache("./.cache")


@cache.memoize()
@timed
def analyse_fwe_or_get_cached(n_total, n_positive, allowable_error) -> ForestAnalysis:
    envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
    fwe = ForestWithEnvelope.create(n_total, n_positive, envelope)
    return fwe.analyse()


def get_expected_runtime(n_total, n_positive, allowable_error):
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


def main():
    ax = plt.subplot()

    window_length = 100

    plot_function(
        ax=ax,
        x_axis_arg_name="n_total",
        function=get_expected_run_proportion,
        function_kwargs=dict(
            n_total=list(range(101, 5_001 + 1, 2)),
            prop_positive=0.1,
            allowable_error=10**-3,
        ),
        plot_kwargs=None,
        results_transform=lambda ys: rolling_average(ys, window_length),
        x_axis_values_transform=lambda xs: xs[window_length-1:]
    )

    ax.grid(visible=True)

    plt.show()


if __name__ == "__main__":
    main()
