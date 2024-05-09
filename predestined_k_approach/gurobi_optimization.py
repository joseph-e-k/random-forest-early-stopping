from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
from diskcache import Cache
from scipy import stats
import gurobipy as gp
from gurobipy import GRB


cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))


@dataclasses.dataclass(frozen=True)
class PiSolution:
    p: np.ndarray
    pi: np.ndarray
    pi_bar: np.ndarray


@cache.memoize("gurobi_optimization.get_optimal_stopping_strategy")
def get_optimal_stopping_strategy(n_total, allowable_error, scale: float = 1.0):
    model, pi_solution = make_and_solve_optimal_stopping_problem(n_total, allowable_error, scale)
    return make_theta_from_pi(pi_solution)


def make_and_solve_optimal_stopping_problem(n: int, alpha: float, scale: float = 1.0) -> tuple[gp.Model, PiSolution]:
    model = gp.Model()
    model.Params.NumericFocus = 3
    model.Params.Aggregate = 0
    model.Params.BarConvTol = 0
    model.Params.FeasibilityTol = 1e-9

    p, pi, pi_bar = _make_decision_variables(n, model, scale)
    n_plus = np.array([n // 2, n // 2 + 1])
    a = make_abstract_probability_matrix(n, n_plus)
    beta = a * pi

    prob_B_equals = np.sum(beta, axis=2)
    expected_B = np.sum(prob_B_equals * np.arange(n + 1), axis=1)
    max_expected_B = model.addVar(name="max_expected_B")
    model.setObjective(max_expected_B, sense=GRB.MINIMIZE)
    for k in range(len(n_plus)):
        model.addConstr(max_expected_B >= expected_B[k], f"max_expected_B >= expected_B[k] with {k=}")

    e = _make_error_state_mask(n, n_plus)
    prob_error = np.sum(e * beta, axis=(1, 2))
    for k in range(len(n_plus)):
        model.addConstr(prob_error[k] <= alpha, f"prob_error[k] <= alpha with {k=}, {alpha=}")

    model.addConstr(p[0, 0] == scale, "p[0, 0] == 1.0")

    for i in range(n):
        model.addConstr(p[i + 1, 0] == pi_bar[i, 0], f"p[i + 1, 0] == pi_bar[i, 0] with {i=}")

    for i in range(n):
        for j in range(i + 1):
            model.addConstr(
                p[i + 1, j + 1] ==
                    ((i - j) / (i + 1)) * pi_bar[i, j + 1]
                    + ((j + 1) / (i + 1)) * pi_bar[i, j],
                "p[i + 1, j + 1] =="
                " ((i - j) / (i + 1)) * pi_bar[i, j + 1]"
                " + ((j + 1) / (i + 1)) * pi_bar[i, j]"
                f" with {i=}, {j=}"
            )

    for j in range(n + 1):
        model.addConstr(pi[n, j] == p[n, j], f"pi[n, j] == p[n, j] with {n=}, {j=}")

    model.optimize()

    pi_solution = PiSolution(
        _get_decision_variable_values(p),
        _get_decision_variable_values(pi),
        _get_decision_variable_values(pi_bar)
    )

    return model, pi_solution


def _make_decision_variables(n: int, model: gp.Model, scale: float=1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = _make_decision_variable_matrix(n, "pi", model, ub=scale)
    pi_bar = _make_decision_variable_matrix(n, "pi_bar", model, ub=scale)
    p = pi + pi_bar
    return p, pi, pi_bar


def _make_decision_variable_matrix(n, variable_name, model: gp.Model, lb=0.0, ub=1.0) -> np.ndarray:
    matrix = np.zeros((n + 1, n + 1), dtype=object)
    for i in range(n + 1):
        for j in range(i + 1):
            matrix[i, j] = model.addVar(name=f"{variable_name}_{i}_{j}", lb=lb, ub=ub)
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


def _make_error_state_mask(n, n_plus) -> np.ndarray:
    i_arange = np.arange(n + 1).reshape(1, -1, 1)
    j_arange = np.arange(n + 1).reshape(1, 1, -1)
    R = (j_arange > (i_arange / 2))
    r = n_plus > (n / 2)
    e = np.not_equal(R, r.reshape(-1, 1, 1))
    return e


def _get_decision_variable_values(decision_variable_matrix: np.ndarray):
    values = np.empty_like(decision_variable_matrix, dtype=float)
    for i, j in np.ndindex(decision_variable_matrix.shape):
        expression = decision_variable_matrix[i, j] * 1
        if isinstance(expression, gp.LinExpr):
            values[i, j] = expression.getValue()
        else:
            values[i, j] = float(expression)
    return values


def make_theta_from_pi(pi_solution):
    theta = np.ones_like(pi_solution.p)
    np.divide(pi_solution.pi, pi_solution.p, out=theta, where=(pi_solution.p != 0))

    # TODO: Find a less hacky way to deal with floating-point errors and decision variable values exceeding their bounds
    np.clip(theta, 0, 1, out=theta)

    return theta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("alpha", type=float, default=0.05)
    args = parser.parse_args()

    stopping_strategy = get_optimal_stopping_strategy(args.n, args.alpha)
    print(stopping_strategy)


if __name__ == "__main__":
    main()
