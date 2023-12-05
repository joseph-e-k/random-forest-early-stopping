import dataclasses
import subprocess
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, stats
from diskcache import Cache

from .Forest import Forest
from .optimization import get_optimal_stopping_strategy
from .ForestWithEnvelope import ForestWithEnvelope
from .ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from .utils import TimerContext, plot_function, plot_functions, timed


cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))

def time_computation_of_optimal_stopping_strategy(n_total, aer):
    with TimerContext(f"get_optimal_stopping_strategy({n_total, aer})") as timer:
        get_optimal_stopping_strategy(n_total, aer)
    return timer.elapsed_time

@timed
@cache.memoize()
def optimal_expected_runtime(n_total, prop_positive, aer, relative=False):
    stopping_strategy = get_optimal_stopping_strategy(n_total, aer)
    fwss = ForestWithGivenStoppingStrategy(Forest(n_total, int(n_total * prop_positive)), stopping_strategy)
    expected_runtime = fwss.analyse().expected_runtime

    if relative:
        return expected_runtime / n_total
    return expected_runtime


@timed
@cache.memoize()
def greedy_envelope_expected_runtime(n_total, prop_positive, aer, relative=False):
    fwe = ForestWithEnvelope.create_greedy(n_total, int(n_total * prop_positive), aer)
    expected_runtime = fwe.analyse().expected_runtime

    if relative:
        return expected_runtime / n_total
    return expected_runtime


def main():
    n_total_min = 11
    n_total_max = 251
    n_total_step = 2
    prop_positive = 0.25
    aer = 10**-6

    plot_functions(
        ax=plt.subplot(),
        x_axis_arg_name="n_total",
        functions=[optimal_expected_runtime, greedy_envelope_expected_runtime],
        function_kwargs=dict(
            n_total=range(n_total_min, n_total_max, n_total_step),
            prop_positive=prop_positive,
            aer=aer,
            relative=False,
        )
    )

    plt.show()


if __name__ == "__main__":
    main()
