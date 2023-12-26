from __future__ import annotations

import dataclasses
import os

import numpy as np
from diskcache import Cache
from pulp import LpProblem, LpMinimize, LpVariable, PULP_CBC_CMD, pulp
from scipy import stats


cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))


@dataclasses.dataclass(frozen=True)
class Sky:
    p: np.ndarray
    pi: np.ndarray
    pi_bar: np.ndarray


def get_optimal_stopping_strategy(n_total, allowable_error):
    sky = make_and_solve_optimal_stopping_problem(n_total, allowable_error)
    return make_theta_from_sky(sky)


# @cache.memoize()
def make_and_solve_optimal_stopping_problem(n: int, alpha: float) -> Sky:
    problem = LpProblem(sense=LpMinimize)

    p, pi, pi_bar = _make_decision_variables(n)
    n_plus = np.array([n // 2, n // 2 + 1])
    a = make_abstract_probability_matrix(n, n_plus)
    beta = a * pi

    prob_B_equals = np.sum(beta, axis=2)
    expected_B = np.sum(prob_B_equals * np.arange(n + 1), axis=1)
    max_expected_B = LpVariable("max_expected_B")
    problem += max_expected_B, "Objective: minimize maximal expected runtime"
    for k in range(len(n_plus)):
        problem += (
            (max_expected_B >= expected_B[k]),
            f"max_expected_runtime >= expected_B[{k}]"
        )

    e = _make_error_mask(n, n_plus)
    prob_error = np.sum(e * beta, axis=(1, 2))
    for k in range(len(n_plus)):
        problem += (
            (prob_error[k] <= alpha),
            f"prob_error[{k}] <= alpha"
        )

    problem += (
        (p[0, 0] == 1),
        "p[0, 0] == 1"
    )

    for i in range(n):
        problem += (
            (p[i + 1, 0] == pi_bar[i, 0]),
            f"p[{i} + 1, 0] == pi_bar[{i}, 0]"
        )

    for i in range(n):
        for j in range(i + 1):
            problem += (
                p[i + 1, j + 1] ==
                    ((i - j) / (i + 1)) * pi_bar[i, j + 1]
                    + ((j + 1) / (i + 1)) * pi_bar[i, j],
                f"p[i + 1, j + 1] == ((i - j) / (i + 1)) * pi_bar[i, j + 1] + ((j + 1) / (i + 1)) * pi_bar[i, j]"
                f" for {i=}, {j=}"
            )

    for j in range(n + 1):
        problem += (
            pi[n, j] == p[n, j],
            f"pi[n, {j}] == p[n, {j}]"
        )


    problem.solve(solver=PULP_CBC_CMD(msg=False))
    print(f"{problem.status=}")

    return Sky(
        _get_decision_variable_values(p),
        _get_decision_variable_values(pi),
        _get_decision_variable_values(pi_bar)
    )


def _make_decision_variables(n) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = _make_decision_variable_matrix(n, "pi")
    pi_bar = _make_decision_variable_matrix(n, "pi_bar")
    p = pi + pi_bar
    return p, pi, pi_bar


def _make_decision_variable_matrix(n, variable_name) -> np.ndarray:
    matrix = np.zeros((n + 1, n + 1), dtype=object)
    for i in range(n + 1):
        for j in range(i + 1):
            matrix[i, j] = LpVariable(
                f"{variable_name}_{i}_{j}",
                lowBound=0,
                upBound=1
            )
    return matrix


def make_abstract_probability_matrix(n, n_plus):
    # Probability matrices for the abstract process. Shape is (len(n_plus), n + 1, n + 1).
    # The first index corresponds to the case (the value of n_plus); the second and third correspond
    # to i and j.
    return stats.hypergeom(
        n,
        n_plus.reshape((-1, 1, 1)),
        np.arange(n + 1).reshape((1, -1, 1))
    ).pmf(np.arange(n + 1).reshape((1, 1, -1)))


def _make_error_mask(n, n_plus) -> np.ndarray:
    i_arange = np.arange(n + 1).reshape(1, -1, 1)
    j_arange = np.arange(n + 1).reshape(1, 1, -1)
    R = (j_arange > (i_arange / 2))
    r = n_plus > (n / 2)
    e = (R != r.reshape(-1, 1, 1))
    return e


def _get_decision_variable_values(decision_variable_matrix):
    values = np.empty_like(decision_variable_matrix, dtype=float)
    for i in range(decision_variable_matrix.shape[0]):
        for j in range(decision_variable_matrix.shape[1]):
            values[i, j] = pulp.value(decision_variable_matrix[i, j])
    return values


def make_theta_from_sky(sky):
    theta = np.ones_like(sky.p)
    np.divide(sky.pi, sky.p, out=theta, where=(sky.p!=0))

    # TODO: Find a less hacky way to deal with floating-point errors and decision variable values exceeding their bounds
    np.clip(theta, 0, 1, out=theta)

    return theta
