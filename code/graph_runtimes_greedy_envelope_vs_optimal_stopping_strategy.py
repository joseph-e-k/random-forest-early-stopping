from matplotlib import pyplot as plt

from code.Forest import Forest
from code.optimization import get_optimal_stopping_strategy
from code.ForestWithEnvelope import ForestWithEnvelope
from code.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from code.utils import timed, memoize
from code.figure_utils import plot_functions


@timed
@memoize()
def optimal_expected_runtime(n_total, prop_positive, aer, relative=False):
    stopping_strategy = get_optimal_stopping_strategy(n_total=n_total, allowable_error=aer, precise=True)
    fwss = ForestWithGivenStoppingStrategy(Forest(n_total, int(n_total * prop_positive)), stopping_strategy)
    expected_runtime = fwss.analyse().expected_runtime

    if relative:
        return expected_runtime / n_total
    return expected_runtime


@timed
@memoize()
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
