import pytest
from ste.utils.linear_programming import FinalizedProblemError, OptimizationSense, Problem


def test_sanity():
    prob = Problem()
    x = prob.add_variable("x", lower_bound=0)
    y = prob.add_variable("y", lower_bound=0)
    objective = -(50 * x + 60 * y)
    prob.set_objective(objective)
    prob.add_constraint(5 * x + 8 * y <= 180)
    prob.add_constraint(5 * x + 4 * y <= 120)
    solution = prob.solve_with_soplex()
    assert solution.variable_values["x"] == 12
    assert solution.variable_values["y"] == 15
    assert solution.objective_value == -1500


def test_maximization_sanity():
    prob = Problem()
    x = prob.add_variable("x", lower_bound=0)
    y = prob.add_variable("y", lower_bound=0)
    objective = 50 * x + 60 * y
    prob.set_objective(objective, OptimizationSense.Maximize)
    prob.add_constraint(5 * x + 8 * y <= 180)
    prob.add_constraint(5 * x + 4 * y <= 120)
    solution = prob.solve_with_soplex()
    assert solution.variable_values["x"] == 12
    assert solution.variable_values["y"] == 15
    assert solution.objective_value == 1500


def test_problem_reuse():
    prob = Problem()
    x = prob.add_variable("x", lower_bound=0, upper_bound=1)
    prob.set_objective(x)
    assert prob.solve_with_soplex().variable_values["x"] == 0
    assert prob.solve_with_soplex().variable_values["x"] == 0

    with pytest.raises(FinalizedProblemError):
        y = prob.add_variable("y", lower_bound=0, upper_bound=1)
        prob.set_objective(y, OptimizationSense.Maximize)
        prob.solve_with_soplex()
