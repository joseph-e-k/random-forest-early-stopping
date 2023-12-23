import dataclasses

import numpy as np
from scipy import optimize, stats

from predestined_k_approach.Forest import Forest
from predestined_k_approach.optimization import get_optimal_stopping_strategy
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from predestined_k_approach.utils import TimerContext


def main1():
    res = optimize.linprog(
        c=-np.array([1, 2]),
        A_ub=np.array([[1, 1]]),
        b_ub=1
    )
    print(res.x)
    print(-res.fun)


def main2():
    n_total = np.array([20, 30])
    n_good = np.array([10, 5])
    n_selected = np.array([3, 4])

    rvs = stats.hypergeom(n_total, n_good, n_selected)
    print(rvs)


@dataclasses.dataclass
class DummyProblem:
    conditions_with_labels: list[tuple[bool, str]] = dataclasses.field(default_factory=list)

    def __iadd__(self, condition_with_label: tuple[bool, str]):
        self.conditions_with_labels.append(condition_with_label)
        return self

    def check_conditions(self):
        for condition, label in self.conditions_with_labels:
            assert condition, label


def get_pi_and_pi_bar_from_theta(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_orig = theta

    theta = np.zeros((theta_orig.shape[0], theta_orig.shape[0]))

    for i in range(theta_orig.shape[0]):
        for j in range(theta_orig.shape[1]):
            theta[i, j] = theta_orig[i, j]

    theta_bar = 1 - theta
    pi = np.zeros((theta.shape[0] - 1, theta.shape[0]))
    pi_bar = np.ones_like(pi)

    for c in range(pi.shape[1]):
        pi[0, c] = theta[0, c]
        pi_bar[0, c] = theta_bar[0, c]

    for n in range(1, pi.shape[0]):
        for c in range(pi.shape[1]):
            combinatoric_factor = (c / n * pi_bar[n-1, c-1] + (n - c) / n * pi_bar[n-1, c])
            pi[n, c] = theta[n, c] * combinatoric_factor
            pi_bar[n, c] = theta_bar[n, c] * combinatoric_factor

    return pi, pi_bar


def get_expected_runtime_coefficients(t, t_plus):
    m_pi = np.zeros((t, t + 1))

    for n in range(m_pi.shape[0]):
        for c in range(m_pi.shape[1]):
            m_pi[n, c] = n * stats.hypergeom(t, t_plus, n).pmf(c)

    return m_pi


def get_expected_runtime(t, t_plus, pi, pi_bar):
    assert pi.shape == pi_bar.shape

    m_pi = get_expected_runtime_coefficients(t, t_plus)

    r = np.sum(pi * m_pi)

    for c in range(pi.shape[1]):
        combinatoric_factor = (c / t * pi_bar[t - 1, c - 1] + (t - c) / t * pi_bar[t - 1, c])
        r += combinatoric_factor * stats.hypergeom(t, t_plus, t).pmf(c) * t

    return r

def get_error_rate(t, t_plus, pi, pi_bar):
    assert pi.shape == pi_bar.shape

    is_error = np.empty_like(pi, dtype=bool)
    for n in range(is_error.shape[0]):
        for c in range(is_error.shape[1]):
            is_error[n, c] = ((c > n / 2) != (t_plus > t / 2))

    s = 0

    for n in range(pi.shape[0]):
        for c in range(pi.shape[1]):
            if is_error[n, c]:
                s += stats.hypergeom(t, t_plus, n).pmf(c) * pi[n, c]

    for c in range(pi_bar.shape[1]):
        if is_error[t, c]:
            s += stats.hypergeom(t, t_plus, t).pmf(c) * pi_bar[t, c]

    return s


def get_greedy_fwe_pi_and_pi_bar():
    n_total = 185
    fwe = ForestWithEnvelope.create_greedy(n_total, n_total // 2, 0.001)
    return get_pi_and_pi_bar_from_theta(np.exp(fwe._get_log_prob_stop()))


@dataclasses.dataclass
class SecondaryOutputObject:
    pi: np.ndarray = None
    pi_bar: np.ndarray= None


def main():
    aer = 10**-6

    for n_total in range(11, 301, 2):
        n_positive = n_total // 2

        forest = Forest(n_total, n_positive)

        with TimerContext(f"find optimal stopping strategy ({n_total=}, {aer=})"):
            optimal_stopping_strategy = get_optimal_stopping_strategy(n_total, aer)

        fwss = ForestWithGivenStoppingStrategy(forest, optimal_stopping_strategy)
        fwe = ForestWithEnvelope.create_greedy(n_total, n_positive, aer)

        print(f"{n_total=}")
        print(f"{fwss.analyse().expected_runtime=}")
        print(f"{fwe.analyse().expected_runtime=}")
        if fwss.analyse().expected_runtime > fwe.analyse().expected_runtime:
            print("^-- *** PANIC TIME ***")
        print()

        # fwss_pi, fwss_pi_bar = get_pi_and_pi_bar_from_theta(fwss.stopping_strategy)
        #
        # fwe_pi, fwe_pi_bar = get_pi_and_pi_bar_from_theta(np.exp(fwe._get_log_prob_stop()))
        #
        # print(f"{get_expected_runtime(n_total, n_total // 2, fwss_pi, fwss_pi_bar)=}")
        # print(f"{get_expected_runtime(n_total, n_total // 2, fwe_pi, fwe_pi_bar)=}")
        # print(f"{get_expected_runtime(n_total, n_total // 2 + 1, fwss_pi, fwss_pi_bar)=}")
        # print(f"{get_expected_runtime(n_total, n_total // 2 + 1, fwe_pi, fwe_pi_bar)=}")


if __name__ == "__main__":
    main()
