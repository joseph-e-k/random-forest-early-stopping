import argparse

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .optimization import make_disagreement_mask


def make_and_solve_qcp(N, alpha):
    """Create and solve a quadratic programming model to find the optimal stopping strategy.
    
    Args:
        N (int): Number of base models.
        alpha (float): Allowable disagreement rate.
    
    Returns:
        np.ndarray: The optimal stopping strategy, a 2D array of shape (N + 1, N + 1).
    """
    model, theta_vars = make_qcp(N, alpha)
    model.optimize()

    theta_values = np.empty_like(theta_vars)
    for index in np.ndindex(theta_vars.shape):
        theta_values[index] = theta_vars[index].X

    return theta_values


def make_and_time_qcp(N, alpha):
    """Create and time a quadratic programming model to find the optimal stopping strategy.
    
    Args:
        N (int): Number of base models.
        alpha (float): Allowable disagreement rate.
        
    Returns:
        float: The time taken to solve the model (in seconds)."""
    model, _ = make_qcp(N, alpha)
    model.optimize()
    return model.Runtime


def make_qcp(N, alpha):
    """Create a quadratic programming model to find the optimal stopping strategy.
    
    Args:
        N (int): Number of base models.
        alpha (float): Allowable disagreement rate.
        
    Returns:
        gurobipy.Model: A Gurobi model representing the quadratic programming problem whose solution is the optimal stopping strategy.
    """
    values_of_n = np.array([N // 2, N // 2 + 1]).reshape((-1, 1, 1))
    prob_see_yes = _get_prob_see_yes(N, values_of_n)

    model = gp.Model()

    theta = _make_decision_variable_matrix((N + 1, N + 1), "theta", model, lb=0, ub=1)
    for j in range(N + 1):
        model.addConstr(theta[N, j] == 1)

    prob_reach = _make_decision_variable_matrix(prob_see_yes.shape, "b", model)
    prob_reach_and_stop = prob_reach * theta
    prob_reach_and_continue = prob_reach - prob_reach_and_stop

    for case in range(prob_reach.shape[0]):
        model.addConstr(prob_reach[case, 0, 0] == 1)

        for i in range(1, prob_reach.shape[1]):
            model.addConstr(prob_reach[case, i, 0] == prob_reach_and_continue[case, i - 1, 0] * (1 - prob_see_yes[case, i - 1, 0]))

            for j in range(1, prob_reach.shape[2]):
                model.addConstr(
                    prob_reach[case, i, j] == (
                        prob_reach_and_continue[case, i - 1, j] * (1 - prob_see_yes[case, i - 1, j])
                        + prob_reach_and_continue[case, i - 1, j - 1] * prob_see_yes[case, i - 1, j - 1]
                    )
                )

    prob_stop_at_step = np.sum(prob_reach_and_stop, axis=2)
    expected_runtime = np.sum(np.arange(N + 1) * prob_stop_at_step, axis=1)
    max_expected_runtime = model.addVar(name="max_expected_runtime")
    for expected_runtime_case in expected_runtime:
        model.addConstr(max_expected_runtime >= expected_runtime_case)

    model.setObjective(max_expected_runtime, sense=GRB.MINIMIZE)

    d = make_disagreement_mask(N, values_of_n)

    prob_error = np.sum(prob_reach_and_stop * d, axis=(1, 2))
    for prob_error_case in prob_error:
        model.addConstr(prob_error_case <= alpha)

    return model, theta


def _make_decision_variable_matrix(shape, variable_name, model: gp.Model, **kwargs) -> np.ndarray:
    matrix = np.zeros(shape, dtype=object)
    for index in np.ndindex(shape):
        index_suffix = "_".join(str(coordinate) for coordinate in index)
        matrix[index] = model.addVar(name=f"{variable_name}_{index_suffix}", **kwargs)
    return matrix


def _get_prob_see_yes(n_total, values_of_n_yes):
    p = np.zeros((values_of_n_yes.shape[0], n_total + 1, n_total + 1))

    n_seen = np.arange(n_total + 1).reshape((1, -1, 1))
    n_seen_yes = np.arange(n_total + 1).reshape((1, 1, -1)).clip(0, values_of_n_yes)
    n_unseen = n_total - n_seen
    n_unseen_yes = values_of_n_yes - n_seen_yes

    return np.divide(n_unseen_yes, n_unseen, where=n_unseen != 0, out=p)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int)
    parser.add_argument("--alpha", "--adr", "-a", type=float, default=0.05)
    args = parser.parse_args(args)

    print(make_and_solve_qcp(args.N, args.alpha))


if __name__ == "__main__":
    main()
