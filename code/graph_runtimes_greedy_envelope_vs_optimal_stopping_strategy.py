import os

from matplotlib import pyplot as plt
from diskcache import Cache

from code.Forest import Forest
from code.optimization import get_optimal_stopping_strategy
from code.ForestWithEnvelope import ForestWithEnvelope
from code.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from code.utils import TimerContext, timed
from code.figure_utils import plot_functions

cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))

def time_computation_of_optimal_stopping_strategy(n_total, aer):
    with TimerContext(f"get_optimal_stopping_strategy({n_total, aer})") as timer:
        get_optimal_stopping_strategy(n_total=n_total, allowable_error=aer, precise=True)
    return timer.elapsed_time


@timed
@cache.memoize()
def optimal_expected_runtime(n_total, prop_positive, aer, relative=False):
    stopping_strategy = get_optimal_stopping_strategy(n_total=n_total, allowable_error=aer, precise=True)
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
    n_total_max = 201
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
