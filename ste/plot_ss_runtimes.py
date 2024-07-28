import argparse
import os

from matplotlib import pyplot as plt

from ste.Forest import Forest
from ste.utils.logging import configure_logging
from ste.optimization import get_optimal_stopping_strategy
from ste.ForestWithEnvelope import ForestWithEnvelope
from ste.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from ste.utils.misc import get_output_path, timed
from ste.utils.caching import memoize
from ste.utils.figures import plot_functions


@timed
@memoize()
def optimal_expected_runtime(n_total, prop_positive, aer, relative=False):
    stopping_strategy = get_optimal_stopping_strategy(n=n_total, alpha=aer)
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


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower-bound", "-l", type=int, default=11)
    parser.add_argument("--n-upper-bound", "-u", type=int, required=True)
    parser.add_argument("--n-step", "-s", type=int, default=2)
    parser.add_argument("--alpha", "--aer", "-a", type=float, default=1e-6)
    parser.add_argument("--prop-positive", "-p", type=float, default=0.25)
    return parser.parse_args()


def main():
    configure_logging()

    args = _parse_args()

    plot_functions(
        ax=plt.subplot(),
        x_axis_arg_name="n_total",
        functions=[optimal_expected_runtime, greedy_envelope_expected_runtime],
        function_kwargs=dict(
            n_total=range(args.n_lower_bound, args.n_upper_bound, args.n_step),
            prop_positive=args.prop_positive,
            aer=args.alpha,
            relative=False,
        )
    )

    this_module_name = os.path.splitext(os.path.basename(__file__))[0]
    output_path = get_output_path(f"{this_module_name}_{args.n_lower_bound}_to_{args.n_upper_bound}")
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
