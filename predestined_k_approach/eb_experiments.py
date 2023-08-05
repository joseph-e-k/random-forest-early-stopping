import os

import numpy as np
from matplotlib import pyplot as plt
from diskcache import Cache

from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.optimization import get_envelope_by_eb_greedily
from predestined_k_approach.utils import plot_function_many_curves


cache = Cache("./.cache")


@cache.memoize()
def get_expected_runtime(n_total, n_positive, allowable_error):
    envelope = get_envelope_by_eb_greedily(n_total, allowable_error)
    fwe = ForestWithEnvelope.create(n_total, n_positive, envelope)
    return fwe.analyse().expected_runtime


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


def main():
    ax = plt.subplot()

    plot_function_many_curves(
        ax=ax,
        x_axis_arg_name="prop_positive",
        distinct_curves_arg_name="n_total",
        function=get_expected_run_proportion,
        function_kwargs=dict(
            n_total=[101, 1001],
            prop_positive=np.linspace(0, 1, 1002),
            allowable_error=10**-3,
        ),
        plot_kwargs=None
    )

    plt.show()


if __name__ == "__main__":
    main()
