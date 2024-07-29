from __future__ import annotations

import argparse
import dataclasses
import os
from fractions import Fraction

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.special import comb

from ste.ForestWithEnvelope import get_greedy_stopping_strategy
from ste.utils.figures import create_subplot_grid, plot_stopping_strategy
from ste.utils.linear_programming import Problem, OptimizationResult, ArithmeticExpression, OptimizationFailure
from ste.utils.logging import configure_logging, get_module_logger
from ste.utils.misc import forwards_to, get_output_path


_logger = get_module_logger()


@dataclasses.dataclass(frozen=True)
class PiSolution:
    p: np.ndarray
    pi: np.ndarray
    pi_bar: np.ndarray


def make_and_solve_optimal_stopping_problem(n: int, alpha: float, freqs_n_plus: np.ndarray = None, disagreement_minimax=True, runtime_minimax=True) -> tuple[PiSolution, float]:
    problem = Problem()

    if disagreement_minimax and runtime_minimax and freqs_n_plus is not None:
        raise ValueError("frequencies were provided for n_plus but both disagreement_minimax and runtime_minimax are True, so those frequencies cannot be used")

    if disagreement_minimax and runtime_minimax:
        # In minimax mode, we need only consider the worst-case scenario, which is a balanced ensemble
        n_plus = np.array([n // 2, n // 2 + 1])
    else:
        n_plus = np.arange(0, n + 1)

    p, pi, pi_bar = _make_decision_variables(n, problem)
    a = make_abstract_probability_matrix(n, n_plus)
    beta = a * pi

    prob_B_equals = np.sum(beta, axis=2)
    expected_B = np.sum(prob_B_equals * np.arange(n + 1), axis=1)

    if runtime_minimax:
        max_expected_B = problem.add_variable("max_expected_B")
        for k in range(len(n_plus)):
            problem.add_constraint(max_expected_B >= expected_B[k])
        problem.set_objective(max_expected_B)
    else:
        expected_expected_B = np.sum(expected_B * freqs_n_plus)
        problem.set_objective(expected_expected_B)

    for decision_variable in [p, pi, pi_bar]:
        for i in range(n + 1):
            for j in range(i + 1):
                problem.add_constraint(decision_variable[i, j] >= 0)
                problem.add_constraint(decision_variable[i, j] <= 1)

    d = _make_disagreement_mask(n, n_plus)
    prob_disagreement = np.sum(d * beta, axis=(1, 2))

    if disagreement_minimax:
        # In minimax mode, disagreement probability must be controlled in each scenario separately
        for k in range(len(n_plus)):
            problem.add_constraint(prob_disagreement[k] <= alpha)
    else:
        expected_prob_disagreement = np.sum(prob_disagreement * freqs_n_plus) / np.sum(freqs_n_plus)
        problem.add_constraint(expected_prob_disagreement <= alpha)


    problem.add_constraint(p[0, 0] == 1)

    for i in range(n):
        problem.add_constraint(p[i + 1, 0] == pi_bar[i, 0])

    for i in range(n):
        for j in range(i + 1):
            constraint = (
                    p[i + 1, j + 1] ==
                        Fraction(i - j, i + 1) * pi_bar[i, j + 1]
                        + Fraction(j + 1, i + 1) * pi_bar[i, j]
            )
            problem.add_constraint(constraint)

    for j in range(n + 1):
        problem.add_constraint(pi[n, j] == p[n, j])

    solution = problem.solve_with_soplex()

    pi_solution = PiSolution(
        _get_decision_variable_values(solution, p),
        _get_decision_variable_values(solution, pi),
        _get_decision_variable_values(solution, pi_bar)
    )

    return pi_solution, solution.objective_value


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


def make_abstract_probability_matrix(n, n_plus):
    """Probability matrices for the abstract process. Shape is (len(n_plus), n + 1, n + 1).
    The first index corresponds to the case (the value of n_plus); the second and third correspond to i and j."""
    out = np.zeros(shape=(len(n_plus), n + 1, n + 1), dtype=object)
    for k, i, j in np.ndindex(out.shape):
        out[k, i, j] = _precise_hypergeometric_probability_mass(n, n_plus[k], i, j)
    return out


def _precise_hypergeometric_probability_mass(n_total, n_good, n_draws, n_good_draws):
    n_bad = n_total - n_good
    n_bad_draws = n_draws - n_good_draws
    return Fraction(
        comb(n_good, n_good_draws, exact=True) * comb(n_bad, n_bad_draws, exact=True),
        comb(n_total, n_draws, exact=True)
    )


def _make_disagreement_mask(n, n_plus) -> np.ndarray:
    i_arange = np.arange(n + 1).reshape(1, -1, 1)
    j_arange = np.arange(n + 1).reshape(1, 1, -1)
    R = (j_arange > (i_arange / 2))
    r = n_plus > (n / 2)
    e = (R != r.reshape(-1, 1, 1))
    return e


def _get_decision_variable_values(solution: OptimizationResult, decision_variable_matrix):
    values = np.empty_like(decision_variable_matrix, dtype=object)
    for i in range(decision_variable_matrix.shape[0]):
        for j in range(decision_variable_matrix.shape[1]):
            values[i, j] = ArithmeticExpression.evaluate(decision_variable_matrix[i, j], solution)
    return values


def make_theta_from_pi(pi_solution):
    theta = np.ones_like(pi_solution.p)
    np.divide(pi_solution.pi, pi_solution.p, out=theta, where=(pi_solution.p != 0))

    # TODO: Find a less hacky way to deal with floating-point errors and decision variable values exceeding their bounds
    np.clip(theta, 0, 1, out=theta)

    return theta


def show_stopping_strategies(stopping_strategies, titles, n_rows=None, n_columns=None):
    fig, axs = create_subplot_grid(len(stopping_strategies), n_rows, n_columns)
    axs = axs.reshape(-1)
    for i, (ss, title) in enumerate(zip(stopping_strategies, titles)):
        plot_stopping_strategy(ss, ax=axs[i])
        axs[i].title.set_text(title)

    plt.show()


@forwards_to(make_and_solve_optimal_stopping_problem)
def get_optimal_stopping_strategy(*args, **kwargs):
    pi_solution, objective_value = make_and_solve_optimal_stopping_problem(*args, **kwargs)
    return make_theta_from_pi(pi_solution)



def main():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("--alpha", "--adr", "-a", type=float, default=0.05)
    parser.add_argument("--graph", "-g", action="store_true")
    parser.add_argument("--output-path", "-o", type=str, default=None)
    args = parser.parse_args()

    try:
        oss = get_optimal_stopping_strategy(args.n, args.alpha)
    except OptimizationFailure as e:
        _logger.exception()
        return
    else:
        _logger.info(f"{np.asarray(oss, dtype=float)=}")

    if not args.graph:
        return

    greedy_ss = get_greedy_stopping_strategy(args.n, args.alpha)
    show_stopping_strategies([oss, greedy_ss], ["Optimal stopping strategy", "Greedy envelope"])

    output_path = args.output_path or get_output_path(f"ss_visualization_{args.n}_submodels_{args.alpha}_adr")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
