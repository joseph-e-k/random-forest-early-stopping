import argparse

from ste.multiprocessing_utils import parallelize
from ste.optimization import get_optimal_stopping_strategy
from ste.utils import TimerContext


DEFAULT_AERS = (0.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def compute_optimal_stopping_strategies(low_n_total, high_n_total, aers):
    n_totals = range(low_n_total, high_n_total + 1)
    task_outcomes = parallelize(
        get_optimal_stopping_strategy,
        argses_to_combine=(n_totals, aers),
        reraise_exceptions=False
    )
    with TimerContext("total"):
        for outcome in task_outcomes:
            n_total, aer = outcome.args_or_kwargs
            if outcome.exception is None:
                print(f"success ({outcome.duration:.1f}s): {n_total=}, {aer=}")
            else:
                print(f"error ({outcome.duration:.1f}s): {n_total=}, {aer=}, {outcome.exception=}")


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
