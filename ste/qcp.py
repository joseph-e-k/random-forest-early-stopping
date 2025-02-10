import argparse

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def make_and_solve_qcp(n, alpha):
    model, theta_vars = make_qcp(n, alpha)
    model.optimize()

    theta_values = np.empty_like(theta_vars)
    for index in np.ndindex(theta_vars.shape):
        theta_values[index] = theta_vars[index].X

    return theta_values


def make_and_time_qcp(n, alpha):
    model, _ = make_qcp(n, alpha)
    model.optimize()
    return model.Runtime


def make_qcp(n, alpha):
    n_positive = np.array([n // 2, n // 2 + 1]).reshape((-1, 1, 1))
    p = _make_positive_observation_probability_matrix(n, n_positive)

    model = gp.Model()

    theta = _make_decision_variable_matrix((n + 1, n + 1), "theta", model, lb=0, ub=1)
    for j in range(n + 1):
        model.addConstr(theta[n, j] == 1)

    b = _make_decision_variable_matrix(p.shape, "b", model)
    beta = b * theta
    beta_bar = b - beta

    for i_case in range(b.shape[0]):
        model.addConstr(b[i_case, 0, 0] == 1)

        for i_step in range(1, b.shape[1]):
            model.addConstr(b[i_case, i_step, 0] == beta_bar[i_case, i_step - 1, 0] * (1 - p[i_case, i_step - 1, 0]))

            for i_value in range(1, b.shape[2]):
                model.addConstr(
                    b[i_case, i_step, i_value] == (
                        beta_bar[i_case, i_step - 1, i_value] * (1 - p[i_case, i_step - 1, i_value])
                        + beta_bar[i_case, i_step - 1, i_value - 1] * p[i_case, i_step - 1, i_value - 1]
                    )
                )

    prob_stop_at_step = np.sum(beta, axis=2)
    expected_runtime = np.sum(np.arange(n + 1) * prob_stop_at_step, axis=1)
    max_expected_runtime = model.addVar(name="max_expected_runtime")
    for expected_runtime_case in expected_runtime:
        model.addConstr(max_expected_runtime >= expected_runtime_case)

    model.setObjective(max_expected_runtime, sense=GRB.MINIMIZE)

    e = _make_error_indicator_matrix(n, n_positive)

    prob_error = np.sum(beta * e, axis=(1, 2))
    for prob_error_case in prob_error:
        model.addConstr(prob_error_case <= alpha)

    return model, theta

def _make_decision_variable_matrix(shape, variable_name, model: gp.Model, **kwargs) -> np.ndarray:
    matrix = np.zeros(shape, dtype=object)
    for index in np.ndindex(shape):
        index_suffix = "_".join(str(coordinate) for coordinate in index)
        matrix[index] = model.addVar(name=f"{variable_name}_{index_suffix}", **kwargs)
    return matrix

def _make_positive_observation_probability_matrix(n, n_positive):
    # The first index corresponds to the case (the value of n_positive); the second and third correspond
    # to n_seen and n_seen_positive.
    p = np.zeros((n_positive.shape[0], n + 1, n + 1))

    n_seen = np.arange(n + 1).reshape((1, -1, 1))
    n_seen_positive = np.arange(n + 1).reshape((1, 1, -1)).clip(0, n_positive)
    n_unseen = n - n_seen
    n_unseen_positive = n_positive - n_seen_positive

    return np.divide(n_unseen_positive, n_unseen, where=n_unseen != 0, out=p)

def _make_error_indicator_matrix(n, n_positive):
    correct_result = n_positive > n / 2

    n_seen = np.arange(n + 1).reshape((1, -1, 1))
    n_seen_positive = np.arange(n + 1).reshape((1, 1, -1)).clip(0, n_positive)

    result = n_seen_positive > n_seen / 2
    return result != correct_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("--alpha", "--adr", "-a", type=float, default=0.05)
    args = parser.parse_args()

    print(make_and_solve_qcp(args.n, args.alpha))
