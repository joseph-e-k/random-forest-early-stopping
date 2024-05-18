import argparse
import itertools

from code.multiprocessing_utils import parallelize
from code.optimization import get_optimal_stopping_strategy
from code.utils import TimerContext


DEFAULT_AERS = (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _compute_oss(n, a):
    return get_optimal_stopping_strategy(n_total=n, allowable_error=a, precise=True)


def compute_optimal_stopping_strategies(low_n_total, high_n_total, aers):
    n_totals = range(low_n_total, high_n_total + 1)
    with TimerContext("total"):
        for ((n_total, aer), success, result, duration) in parallelize(_compute_oss, itertools.product(n_totals, aers)):
            if success:
                print(f"success ({duration:.1f}s): {n_total=}, {aer=}")
            else:
                print(f"error ({duration:.1f}s): {n_total=}, {aer=}, error={result!r}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower-bound", "-l", type=int, default=1)
    parser.add_argument("--n-upper-bound", "-u", type=int, required=True)
    parser.add_argument("--alphas", "-a", type=float, nargs="+", default=DEFAULT_AERS)
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = _parse_args()
    compute_optimal_stopping_strategies(
        cmd_args.n_lower_bound,
        cmd_args.n_upper_bound,
        cmd_args.alphas
    )
