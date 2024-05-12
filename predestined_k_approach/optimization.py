from __future__ import annotations

import argparse
import dataclasses
import os
from fractions import Fraction

import numpy as np
from diskcache import Cache
from scipy import stats
from scipy.special import comb

from linear_programming_utils import Problem, OptimizationResult, ArithmeticExpression

cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))


@dataclasses.dataclass(frozen=True)
class Sky:
    p: np.ndarray
    pi: np.ndarray
    pi_bar: np.ndarray


def get_optimal_stopping_strategy(n_total, allowable_error, precise=False):
    pi_solution, objective_value = make_and_solve_optimal_stopping_problem(n_total, allowable_error, precise)
    return make_theta_from_sky(pi_solution)


# @cache.memoize()
def make_and_solve_optimal_stopping_problem(n: int, alpha: float, precise: bool = False) -> tuple[Sky, float]:
    problem = Problem()

    p, pi, pi_bar = _make_decision_variables(n, problem)
    n_plus = np.array([n // 2, n // 2 + 1])
    a = make_abstract_probability_matrix(n, n_plus, precise)
    beta = a * pi

    prob_B_equals = np.sum(beta, axis=2)
    expected_B = np.sum(prob_B_equals * np.arange(n + 1), axis=1)
    max_expected_B = problem.add_variable("max_expected_B")
    problem.set_objective(max_expected_B)
    for k in range(len(n_plus)):
        problem.add_constraint(max_expected_B >= expected_B[k])

    for decision_variable in [p, pi, pi_bar]:
        for i in range(n + 1):
            for j in range(i + 1):
                problem.add_constraint(decision_variable[i, j] >= 0)
                problem.add_constraint(decision_variable[i, j] <= 1)

    e = _make_error_mask(n, n_plus)
    prob_error = np.sum(e * beta, axis=(1, 2))
    for k in range(len(n_plus)):
        problem.add_constraint(prob_error[k] <= alpha)

    problem.add_constraint(p[0, 0] == 1)

    for i in range(n):
        problem.add_constraint(p[i + 1, 0] == pi_bar[i, 0])

    for i in range(n):
        for j in range(i + 1):
            if precise:
                constraint = (
                        p[i + 1, j + 1] ==
                            Fraction(i - j, i + 1) * pi_bar[i, j + 1]
                            + Fraction(j + 1, i + 1) * pi_bar[i, j]
                )
            else:
                constraint = (
                    p[i + 1, j + 1] ==
                        ((i - j) / (i + 1)) * pi_bar[i, j + 1]
                        + ((j + 1) / (i + 1)) * pi_bar[i, j]
                )
            problem.add_constraint(constraint)

    for j in range(n + 1):
        problem.add_constraint(pi[n, j] == p[n, j])

    if precise:
        solution = problem.solve_with_soplex()
    else:
        solution = problem.solve_with_pulp()

    sky_solution = Sky(
        _get_decision_variable_values(solution, p),
        _get_decision_variable_values(solution, pi),
        _get_decision_variable_values(solution, pi_bar)
    )

    return sky_solution, solution.objective_value


def _make_decision_variables(n, problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = _make_decision_variable_matrix(n, "pi", problem)
    pi_bar = _make_decision_variable_matrix(n, "pi_bar", problem)
    p = pi + pi_bar
    return p, pi, pi_bar


def _make_decision_variable_matrix(n, variable_name, problem: Problem) -> np.ndarray:
    matrix = np.zeros((n + 1, n + 1), dtype=object)
    for i in range(n + 1):
        for j in range(i + 1):
            matrix[i, j] = problem.add_variable(f"{variable_name}_{i}_{j}")
    return matrix


def make_abstract_probability_matrix(n, n_plus, precise=False):
    """Probability matrices for the abstract process. Shape is (len(n_plus), n + 1, n + 1).
    The first index corresponds to the case (the value of n_plus); the second and third correspond to i and j."""
    if precise:
        out = np.zeros(shape=(len(n_plus), n + 1, n + 1), dtype=object)
        for k, i, j in np.ndindex(out.shape):
            out[k, i, j] = _precise_hypergeometric_probability_mass(n, n_plus[k], i, j)
        return out

    return stats.hypergeom(
        n,
        n_plus.reshape((-1, 1, 1)),
        np.arange(n + 1).reshape((1, -1, 1))
    ).pmf(np.arange(n + 1).reshape((1, 1, -1)))


def _precise_hypergeometric_probability_mass(n_total, n_good, n_draws, n_good_draws):
    n_bad = n_total - n_good
    n_bad_draws = n_draws - n_good_draws
    return Fraction(
        comb(n_good, n_good_draws, exact=True) * comb(n_bad, n_bad_draws, exact=True),
        comb(n_total, n_draws, exact=True)
    )


def _make_error_mask(n, n_plus) -> np.ndarray:
    i_arange = np.arange(n + 1).reshape(1, -1, 1)
    j_arange = np.arange(n + 1).reshape(1, 1, -1)
    R = (j_arange > (i_arange / 2))
    r = n_plus > (n / 2)
    e = (R != r.reshape(-1, 1, 1))
    return e


def _get_decision_variable_values(solution: OptimizationResult, decision_variable_matrix, precise=False):
    values = np.empty_like(decision_variable_matrix, dtype=object if precise else float)
    for i in range(decision_variable_matrix.shape[0]):
        for j in range(decision_variable_matrix.shape[1]):
            values[i, j] = ArithmeticExpression.evaluate(decision_variable_matrix[i, j], solution)
    return values


def make_theta_from_sky(sky):
    theta = np.ones_like(sky.p)
    np.divide(sky.pi, sky.p, out=theta, where=(sky.p!=0))

    # TODO: Find a less hacky way to deal with floating-point errors and decision variable values exceeding their bounds
    np.clip(theta, 0, 1, out=theta)

    return theta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("alpha", type=float, default=0.05)
    parser.add_argument("--precise", "-p", action="store_true")
    args = parser.parse_args()

    sky, objective_value = make_and_solve_optimal_stopping_problem(args.n, args.alpha, args.precise)
    print(f"{objective_value=}")
    theta_values = make_theta_from_sky(sky)
    print(theta_values)
