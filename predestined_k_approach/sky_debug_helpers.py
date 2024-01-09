import argparse
import itertools
import multiprocessing as mp
from typing import Callable

from predestined_k_approach.Forest import Forest
from predestined_k_approach.optimization import *
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from predestined_k_approach.utils import TimerContext


def make_sky_from_truncated_theta(truncated_theta):
    n = truncated_theta.shape[0] - 1
    theta = np.ones((n + 1, n + 1))
    theta[:, :truncated_theta.shape[1]] = truncated_theta

    p = np.zeros_like(theta)

    theta_bar = 1 - theta

    p[0, 0] = 1
    for i in range(n):
        p[i + 1, 0] = p[i, 0] * theta_bar[i, 0]

        for j in range(i + 1):
            p[i + 1, j + 1] = (
                ((i - j) / (i + 1)) * p[i, j + 1] * theta_bar[i, j + 1]
                + ((j + 1) / (i + 1)) * p[i, j] * theta_bar[i, j]
            )

    return Sky(p, p * theta, p * theta_bar)


def get_expected_runtimes(n_total, aer=10**-6):
    n_positive_low = n_total // 2
    n_positive_high = n_positive_low + 1

    low_forest = Forest(n_total, n_positive_low)
    high_forest = Forest(n_total, n_positive_high)

    low_fwe = ForestWithEnvelope.create_greedy(n_total, n_positive_low, aer)
    high_fwe = ForestWithEnvelope(forest=high_forest, envelope=low_fwe.envelope)

    fwss_sky = make_and_solve_optimal_stopping_problem(n_total, aer)
    optimal_stopping_strategy = make_theta_from_sky(fwss_sky)

    low_fwss = ForestWithGivenStoppingStrategy(low_forest, optimal_stopping_strategy)
    high_fwss = ForestWithGivenStoppingStrategy(high_forest, optimal_stopping_strategy)

    low_fwss_time = low_fwss.analyse().expected_runtime
    high_fwss_time = high_fwss.analyse().expected_runtime
    low_fwe_time = low_fwe.analyse().expected_runtime
    high_fwe_time = high_fwe.analyse().expected_runtime

    return low_fwss_time, high_fwss_time, low_fwe_time, high_fwe_time

@dataclasses.dataclass(frozen=True)
class Worker:
    function: Callable

    def __call__(self, args):
        try:
            result = self.function(*args)
        except Exception as e:
            return args, False, e
        return args, True, result


def search_for_impossibilities(n_processes, low_n_total, high_n_total):
    n_totals = range(low_n_total, high_n_total + 1, 2)
    aers = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]
    with TimerContext("total"):
        with mp.Pool(n_processes) as pool:
            for (args, success, result) in pool.imap(Worker(get_expected_runtimes), itertools.product(n_totals, aers)):
                n_total, aer = args

                if success:
                    low_fwss_time, high_fwss_time, low_fwe_time, high_fwe_time = result
                    if low_fwss_time > low_fwe_time and high_fwss_time > high_fwe_time:
                        print(f"impossible: {n_total=}, {aer=}, {low_fwss_time=}, {high_fwss_time=}, {low_fwe_time=}, {high_fwe_time=}")
                    else:
                        print(f"possible: {n_total=}, {aer=}, {low_fwss_time=}, {high_fwss_time=}, {low_fwe_time=}, {high_fwe_time=}")

                else:
                    print(f"error: {n_total=}, {aer=}, error={result!r}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("low_n_total", type=int)
    parser.add_argument("high_n_total", type=int)
    parser.add_argument("n_processes", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = _parse_args()
    search_for_impossibilities(cmd_args.n_processes, cmd_args.low_n_total, cmd_args.high_n_total)
