from __future__ import annotations

import argparse
import dataclasses
import os
from fractions import Fraction

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import comb

from .EnsembleVote import EnsembleVote, EnsembleVoteWithStoppingStrategy
from .utils.figures import plot_evwss
from .utils.linear_programming import Problem, OptimizationResult, ArithmeticExpression
from .utils.logging import configure_logging, get_module_logger
from .utils.misc import forwards_to, get_output_path


_logger = get_module_logger()


@dataclasses.dataclass(frozen=True)
class PiSolution:
    """A solution to the optimal-stopping problem in Pi form"""
    p: np.ndarray
    pi: np.ndarray
    pi_bar: np.ndarray


def make_and_solve_optimal_stopping_problem(N: int, alpha: float, D_hat: np.ndarray = None, disagreement_minimax=True, runtime_minimax=True) -> tuple[PiSolution, float]:
    """Constructs and solves the optimal-stopping problem in Pi form for the given parameters.

    Args:
        N (int): Number of base models.
        alpha (float): Allowable disagreement rate.
        D_hat (np.ndarray, optional): Estimated distribution of `n`, as an array of relative frequencies. Unused if `disagreement_minimax` and `runtime_minimax` are both True (the default).
        disagreement_minimax (bool, optional): If True (the default), the disagreement probability is minimized for each value of `n` separately. If False, the disagreement probability is minimized for the given distribution of `n`.
        runtime_minimax (bool, optional): If True (the default), the expected runtime is minimized for each value of `n` separately. If False, the expected runtime is minimized for the given distribution of `n`.

    Returns:
        tuple[PiSolution, float]: The solution in Pi form and the expected runtime of that solution.
    """
    problem, p, pi, pi_bar = make_optimal_stopping_problem(N, alpha, D_hat, disagreement_minimax, runtime_minimax)

    solution = problem.solve_with_soplex()

    pi_solution = PiSolution(
        _get_decision_variable_values(solution, p),
        _get_decision_variable_values(solution, pi),
        _get_decision_variable_values(solution, pi_bar)
    )

    return pi_solution, solution.objective_value


def make_optimal_stopping_problem(N: int, alpha: float, D_hat: np.ndarray = None, disagreement_minimax=True, runtime_minimax=True) -> tuple[Problem, np.ndarray, np.ndarray, np.ndarray]:
    """Constructs the optimal stopping problem for the given parameters.

    Args:
        N (int): Number of base models.
        alpha (float): Allowable disagreement rate.
        D_hat (np.ndarray, optional): Estimated frequencies of different values of n. Unused if `disagreement_minimax` and `runtime_minimax` are both True (the default).
        disagreement_minimax (bool, optional): If True (the default), the disagreement probability is minimized for each value of `n` separately. If False, the disagreement probability is minimized for the given distribution of `n`.
        runtime_minimax (bool, optional): If True (the default), the expected runtime is minimized for each value of `n` separately. If False, the expected runtime is minimized for the given distribution of `n`.

    Returns:
        tuple[Problem, np.ndarray, np.ndarray, np.ndarray]: The problem object and the decision variable matrices p, pi, and pi_bar.
    """
    if disagreement_minimax and runtime_minimax and D_hat is not None:
        raise ValueError("frequencies were provided for n but both disagreement_minimax and runtime_minimax are True, so those frequencies cannot be used")

    stringified_args = f"{N=}, {alpha=}"
    if D_hat is not None:
        if (D_hat == 1).all():
            stringified_args += ", D_hat=<uniform>"
        else:
            stringified_args += f", {D_hat=}"
        stringified_args += f", {disagreement_minimax=}, {runtime_minimax=}"

    problem = Problem()
    _logger.info(f"Constructing optimal-stopping problem for {stringified_args}, tagged {problem.tag!r}...")

    # `values_of_n` is an array of all possible values of `n`
    values_of_n = np.arange(0, N + 1)

    # Declare decision variables. Technically this is doable with only two of these three, but this makes the code cleaner
    p, pi, pi_bar = _make_decision_variables(N, problem)

    # Probability matrix for W_{ij}, that is, the the probability that an unstopped ensemble would reach a certain state.
    # Its dimensions are (len(values_of_n), N + 1, N + 1), where the first index corresponds to the case (the value of `n`) and the second and third correspond to i and j.
    prob_W = make_abstract_probability_matrix(N, values_of_n)

    # Probability matrix for Pr(reach and stop at (i, j)).
    # Its dimensions are (len(values_of_n), N + 1, N + 1), where the first index corresponds to the case (the value of `n`) and the second and third correspond to i and j.
    prob_reach_and_stop = prob_W * pi

    # `B` is the expected number of base models that are executed in the stopped ensemble,
    # so Prob(B = i | n) is the sum of `prob_reach_and_stop`s in a single column
    prob_B_equals = np.sum(prob_reach_and_stop, axis=2)

    # `expected_B` is the expected number of base models that are executed in the stopped ensemble
    # This is still an array, now with only 1 dimension corresponding to the different values of `n`
    expected_B = np.sum(prob_B_equals * np.arange(N + 1), axis=1)

    if runtime_minimax:
        # If we are minimaxing the expected runtime, we need to constrain it for each value of `n` separately
        max_expected_B = problem.add_variable("max_expected_B")
        for n in values_of_n:
            problem.add_constraint(max_expected_B >= expected_B[n])
        problem.set_objective(max_expected_B)
    else:
        # Otherwise, we minimize E[E[B | n]] over the given distribution of `n`, which is `D_hat`
        expected_expected_B = np.sum(expected_B * D_hat)
        problem.set_objective(expected_expected_B)

    # `d` is the disagreement mask, which is 1 for states where the stopped ensemble would disagree with the complete ensemble and 0 elsewhere
    # Its dimensions are (len(values_of_n), N + 1, N + 1), where the first index corresponds to the case (the value of `n`) and the second and third correspond to i and j
    d = make_disagreement_mask(N, values_of_n)

    # `prob_disagreement` is the probability that the stopped ensemble disagrees with the complete ensemble
    # It has only 1 dimension, corresponding to the different values of `n`
    prob_disagreement = np.sum(d * prob_reach_and_stop, axis=(1, 2))

    if disagreement_minimax:
        # If we are minimaxing the disagreement probability, we need to constrain it for each value of `n` separately
        for n in range(len(values_of_n)):
            problem.add_constraint(prob_disagreement[n] <= alpha)
    else:
        # Otherwise, we minimize E[Prob(disagreement | N^+)] over the given distribution of `n`
        expected_prob_disagreement = np.sum(prob_disagreement * D_hat) / np.sum(D_hat)
        problem.add_constraint(expected_prob_disagreement <= alpha)

    # Now the constraints to force our decision variables to be in Pi

    # All our decision variables represent probabilities and therefore must be between 0 and 1
    for decision_variable in [p, pi, pi_bar]:
        for i in range(N + 1):
            for j in range(i + 1):
                problem.add_constraint(decision_variable[i, j] >= 0)
                problem.add_constraint(decision_variable[i, j] <= 1)

    # Before any base models are executed, the probability of being in state (0, 0) is 1
    problem.add_constraint(p[0, 0] == 1)

    # The only way to reach state (i + 1, 0) is to reach state (i, 0) and not stop there
    for i in range(N):
        problem.add_constraint(p[i + 1, 0] == pi_bar[i, 0])

    # All other states (i, j) have two antecedents: (i - 1, j) and (i - 1, j + 1)
    for i in range(N):
        for j in range(i + 1):
            constraint = (
                    p[i + 1, j + 1] ==
                        Fraction(i - j, i + 1) * pi_bar[i, j + 1]
                        + Fraction(j + 1, i + 1) * pi_bar[i, j]
            )
            problem.add_constraint(constraint)

    # Once the ensemble is exhausted, we must stop, so the probability of reaching and
    # stopping at a state (N, j) is equal to the probability of reaching it
    for j in range(N + 1):
        problem.add_constraint(pi[N, j] == p[N, j])

    _logger.info(f"Finished constructing optimal-stopping problem for {stringified_args}")

    return problem, p, pi, pi_bar


def _make_decision_variables(N, problem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = _make_decision_variable_matrix(N, "pi", problem)
    pi_bar = _make_decision_variable_matrix(N, "pi_bar", problem)
    p = pi + pi_bar
    return p, pi, pi_bar


def _make_decision_variable_matrix(N, variable_name, problem: Problem) -> np.ndarray:
    matrix = np.zeros((N + 1, N + 1), dtype=object)
    for i in range(N + 1):
        for j in range(i + 1):
            matrix[i, j] = problem.add_variable(f"{variable_name}_{i}_{j}")
    return matrix


def make_abstract_probability_matrix(N, values_of_n):
    """Probability matrices for the abstract process. Shape is (len(values_of_n), N + 1, N + 1).
    The first index corresponds to the case (the value of n); the second and third correspond to i and j."""
    out = np.zeros(shape=(len(values_of_n), N + 1, N + 1), dtype=object)
    for n, i, j in np.ndindex(out.shape):
        out[n, i, j] = _precise_hypergeometric_probability_mass(N, n, i, j)
    return out


def _precise_hypergeometric_probability_mass(n_total, n_good, n_draws, n_good_draws):
    n_bad = n_total - n_good
    n_bad_draws = n_draws - n_good_draws
    return Fraction(
        comb(n_good, n_good_draws, exact=True) * comb(n_bad, n_bad_draws, exact=True),
        comb(n_total, n_draws, exact=True)
    )


def make_disagreement_mask(N, values_of_n) -> np.ndarray:
    """For each state, determine whether stopping in that state would give a result different from that of the complete ensemble.
    
    Args:
        N (int): Number of base models.
        values_of_n (np.ndarray): Possible values of n (number of 'yes' votes).

    Returns:
        np.ndarray: A mask of shape (len(values_of_n), N + 1, N + 1) where the first index corresponds to the case (the value of n) and the second and third correspond to i and j.
    """
    i_arange = np.arange(N + 1).reshape(1, -1, 1)
    j_arange = np.arange(N + 1).reshape(1, 1, -1)
    stopped_result = (j_arange > (i_arange / 2))
    unstopped_result = values_of_n > (N / 2)
    disagreement = (stopped_result != unstopped_result.reshape(-1, 1, 1))
    return disagreement


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


def make_pi_from_theta(theta):
    p = np.empty_like(theta)
    pi = np.empty_like(theta)
    pi_bar = np.empty_like(theta)

    N = theta.shape[0] - 1

    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            p[i, j] = 0
            pi[i, j] = 0
            pi_bar[i, j] = 0

    p[0, 0] = 1
    pi[0, 0] = theta[0, 0]
    pi_bar[0, 0] = 1 - theta[0, 0]

    for i in range(N):
        for j in range(i + 1):
            pi[i, j] = p[i, j] * theta[i, j]
            pi_bar[i, j] = p[i, j] - pi[i, j]

        p[i + 1, 0] = pi_bar[i, 0]
        for j in range(i + 1):
            p[i + 1, j + 1] = (
                Fraction(i - j, i + 1) * pi_bar[i, j + 1]
                + Fraction(j + 1, i + 1) * pi_bar[i, j]
            )

    for j in range(N + 1):
        pi[N, j] = p[N, j]
        pi_bar[N, j] = 0

    return PiSolution(p, pi, pi_bar)


def show_stopping_strategy(ss, save_to_folder=None):
    N = ss.shape[0] - 1
    values_of_n = [N, N // 2]
    fig_width = 2 * N + 2
    fig_height = N + 1
    for n in values_of_n:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        evwss = EnsembleVoteWithStoppingStrategy(EnsembleVote(N, n), ss)
        plot_evwss(evwss, ax=ax)

        if save_to_folder is not None:
            if not os.path.exists(save_to_folder):
                os.mkdir(save_to_folder)
            path = os.path.join(save_to_folder, f"n_{n}.pdf")
            fig.savefig(path)

    plt.show()


@forwards_to(make_and_solve_optimal_stopping_problem)
def get_optimal_stopping_strategy(*args, **kwargs):
    """Get the optimal stopping strategy for the given parameters.
    This is a wrapper around `make_and_solve_optimal_stopping_problem` that converts the solution to Theta form.

    Returns:
        np.ndarray: matrix of stopping probabilities
    """
    pi_solution, objective_value = make_and_solve_optimal_stopping_problem(*args, **kwargs)
    return make_theta_from_pi(pi_solution)


def main():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("--alpha", "--adr", "-a", type=float, default=0.05)
    parser.add_argument("--graph", "-g", action="store_true")
    parser.add_argument("--output-path", "-o", type=str, default=None)
    args = parser.parse_args()

    oss = get_optimal_stopping_strategy(args.N, args.alpha)
    _logger.info(f"{np.asarray(oss, dtype=float)=}")

    if not args.graph:
        return

    output_path = args.output_path or get_output_path(f"ss_visualization_{args.N}_submodels_{args.alpha}_adr", file_name_suffix="")
    show_stopping_strategy(oss, save_to_folder=output_path)


if __name__ == "__main__":
    main()
