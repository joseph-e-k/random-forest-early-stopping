import argparse
import pickle

import numpy as np

from .utils.logging import configure_logging, get_module_logger
from .utils.multiprocessing import parallelize
from .optimization import get_optimal_stopping_strategy
from .utils.misc import TimerContext, get_output_path


_logger = get_module_logger()


DEFAULT_AERS = (0.0, 1e-1, 1e-2, 5e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def compute_optimal_stopping_strategies(low_n_total, high_n_total, aers, should_pickle=False):
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
                _logger.info(f"Success ({outcome.duration:.1f}s): {n_total=}, {aer=}")
                if should_pickle:
                    path = get_output_path(f"../precomputed_stopping_strategies/oss_n_{n_total}_aer_{aer}", ".pickle")
                    with open(path, "wb") as result_file:
                        pickle.dump(np.asarray(outcome.result, dtype=float), result_file)
            else:
                _logger.error(f"Error ({outcome.duration:.1f}s): {n_total=}, {aer=}, {outcome.exception=}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lower-bound", "-l", type=int, default=1)
    parser.add_argument("--n-upper-bound", "-u", type=int, required=True)
    parser.add_argument("--alphas", "-a", type=float, nargs="+", default=DEFAULT_AERS)
    parser.add_argument("--pickle", "-p", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    cmd_args = _parse_args()
    compute_optimal_stopping_strategies(
        cmd_args.n_lower_bound,
        cmd_args.n_upper_bound,
        cmd_args.alphas,
        cmd_args.pickle
    )
