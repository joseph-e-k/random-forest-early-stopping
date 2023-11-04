from __future__ import annotations

import itertools
import os

import numpy as np
from diskcache import Cache
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from scipy import stats


cache = Cache(os.path.join(os.path.dirname(__file__), ".cache"))


@cache.memoize()
def get_optimal_stopping_strategy(n_total, allowable_error):
    n_steps = n_total + 1
    n_values = n_steps

    # Two values of t^+: one just below t/2, and one just above it
    n_good = np.array([n_total // 2, n_total // 2 + 1])
    n_cases = len(n_good)

    prob_abstract = _get_abstract_probabilities(n_total, n_good)

    problem = LpProblem(sense=LpMinimize)

    max_expected_runtime = LpVariable(name="max_expected_runtime")

    # Objective function must be added to the problem first
    problem += max_expected_runtime, "Objective: minimize maximal expected runtime"

    # Our decision variables are pi and pi_bar, which we later translate into actual stopping-probabilities
    pi, pi_bar = _create_decision_variables(n_steps, n_values)

    prob_concrete_reach_cond = _get_probabilities_concrete_reach_conditional(pi, pi_bar)
    prob_concrete_stop = _get_concrete_stop_probabilities(pi, pi_bar, prob_abstract, prob_concrete_reach_cond)

    _add_inherent_probability_constraints(problem, pi, pi_bar, prob_concrete_reach_cond)
    _add_max_runtime_constraints(problem, max_expected_runtime, prob_concrete_stop, n_cases, n_steps, n_values)
    _add_error_rate_constraints(problem, prob_concrete_stop, allowable_error, n_cases, n_good, n_steps, n_total,
                                n_values)

    problem.solve(solver=PULP_CBC_CMD(msg=False))

    pi, pi_bar = _recover_decision_variables(problem, n_steps, n_values)

    return _to_decision_probabilities(n_total, pi, pi_bar)


def _recover_decision_variables(problem, n_steps, n_values):
    variable_values_by_name = {
        name: variable.varValue
        for (name, variable) in problem.variablesDict().items()
    }
    pi = np.empty((n_steps, n_values))
    pi_bar = np.empty((n_steps, n_values))
    for i_step, i_value in itertools.product(range(n_steps - 1), range(n_values)):
        pi[i_step, i_value] = variable_values_by_name.get(f"pi_{i_step}_{i_value}", 0)
        pi_bar[i_step, i_value] = variable_values_by_name.get(f"pi_bar_{i_step}_{i_value}", 1)
    return pi, pi_bar


def _add_error_rate_constraints(problem, prob_concrete_stop, allowable_error, n_cases, n_good, n_steps, n_total,
                                n_values):
    state_result = np.array([
        [
            i_value > i_step / 2
            for i_value in range(n_values)
        ]
        for i_step in range(n_steps)
    ])
    correct_result = n_good.reshape((-1, 1, 1)) > n_total / 2
    state_is_error = (correct_result != state_result)
    for i_case in range(n_cases):
        error_rate = lpSum(
            lpSum(
                state_is_error[i_case, i_step, i_value] * prob_concrete_stop[i_case, i_step, i_value]
                for i_value in range(n_values)
            )
            for i_step in range(n_steps)
        )

        problem += (error_rate <= allowable_error), f"error rate (case {i_case}) <= {allowable_error}"


def _add_max_runtime_constraints(problem, max_expected_runtime, prob_concrete_stop, n_cases, n_steps, n_values):
    for i_case in range(n_cases):
        expected_runtime = lpSum(
            i_step * lpSum(prob_concrete_stop[i_case, i_step, i_value] for i_value in range(n_values))
            for i_step in range(n_steps)
        )

        problem += (
            (max_expected_runtime >= expected_runtime),
            f"max_expected_runtime >= expected runtime (case {i_case})"
        )


def _get_concrete_stop_probabilities(pi, pi_bar, prob_abstract, prob_concrete_reach_cond):
    # Define expressions for the probability of the concrete process reaching and stopping at a given state:
    #   $a_{n,c} \pi_{n,c}$ if $n < t$
    #   $b_{n, c}$ if $n = t$
    # Note that these are expressions composed of existing variables, not new variables

    n_cases, n_steps, n_values = prob_abstract.shape
    i_final_step = n_steps - 1
    prob_concrete_stop = np.empty(shape=prob_abstract.shape, dtype=np.dtype(object))

    for i_case in range(n_cases):
        for (i_step, i_value) in itertools.product(range(pi.shape[0]), range(pi.shape[1])):
            prob_concrete_stop[i_case, i_step, i_value] = prob_abstract[i_case, i_step, i_value] * pi[i_step, i_value]

        for i_value in range(n_values):
            prob_concrete_stop[i_case, i_final_step, i_value] = (
                    prob_abstract[i_case, i_final_step, i_value] * prob_concrete_reach_cond[i_final_step, i_value]
            )

    return prob_concrete_stop


def _create_decision_variables(n_steps, n_values):
    pi = np.zeros((n_steps - 1, n_values), np.dtype(object))
    pi_bar = np.ones_like(pi)

    for i_step in range(pi.shape[0]):
        for i_value in range(i_step + 1):
            pi[i_step, i_value] = LpVariable(
                name=f"pi_{i_step}_{i_value}",
                lowBound=0
            )

            pi_bar[i_step, i_value] = LpVariable(
                name=f"pi_bar_{i_step}_{i_value}",
                lowBound=0
            )

    return pi, pi_bar


def _get_probabilities_concrete_reach_conditional(pi, pi_bar):
    p = np.zeros((pi.shape[0] + 1, pi.shape[1]), np.dtype(object))

    p[0, :] = 1

    for i_step in range(1, pi.shape[0] + 1):
        for i_value in range(i_step + 1):
            p[i_step, i_value] = (i_step - i_value) / i_step * pi_bar[i_step - 1, i_value]

            if i_value >= 1:
                p[i_step, i_value] += i_value / i_step * pi_bar[i_step - 1, i_value - 1]

    return p


def _add_inherent_probability_constraints(problem, pi, pi_bar, prob_concrete_reach_cond):
    # In the first row, we have:
    #   $\pi_{0, c} = \theta_{0, c}$
    #   $\bar\pi_{0, c} = \bar\theta_{0, c} = 1 - \theta_{0, c}$
    # Therefore we constrain $\pi_{0, c} + \bar\pi_{0, c} = 1$
    for i_value in range(pi.shape[1]):
        problem += (pi[0, i_value] + pi_bar[0, i_value] == 1), f"pi[0, {i_value}] + pi_bar[0, {i_value}] == 1"

    # In subsequent rows we have:
    #   $a_{n, c} (\pi_{n, c} + \bar\pi_{n, c}) = b_{n, c}$
    # Or in other words:
    #  $\pi_{n, c} + \bar\pi_{n, c} = \frac{b_{n, c}}{a_{n, c}}$
    for i_step in range(1, pi.shape[0]):
        for i_value in range(i_step + 1):
            lhs = pi[i_step, i_value] + pi_bar[i_step, i_value]
            rhs = prob_concrete_reach_cond[i_step, i_value]
            problem += (lhs == rhs), f"pi[{i_step}, {i_value}] + pi_bar[{i_step}, {i_value}]"


def _get_abstract_probabilities(n_total, n_good):
    # Probability matrices for the abstract process. Shape is (2, t+1, t+1).
    # The first index corresponds to the case (low t^+ or high t^+); the second and third correspond
    # to n and c.
    return stats.hypergeom(
        n_total,
        n_good.reshape((-1, 1, 1)),
        np.arange(n_total + 1).reshape((1, -1, 1))
    ).pmf(np.arange(n_total + 1).reshape((1, 1, -1)))


def _to_decision_probabilities(n_total, pi, pi_bar):
    n_steps = n_values = n_total + 1
    decision_probability = np.zeros((n_steps, n_values))

    decision_probability[0, :] = pi[0, :]

    for i_step in range(1, n_steps - 1):
        for i_value in range(i_step + 1):
            if pi[i_step, i_value] == 0:
                decision_probability[i_step, i_value] = 0
            else:
                conversion_factor = (
                    # Note that we don't have to worry about the wraparound from `i_value - 1` here;
                    # when i_value - 1 < 0, then i_value = 0, so the coefficient is 0 anyway
                    (i_value / i_step) * pi_bar[i_step - 1, i_value - 1]
                    + ((i_step - i_value) / i_step) * pi_bar[i_step - 1, i_value]
                )
                decision_probability[i_step, i_value] = pi[i_step, i_value] / conversion_factor

    # If we reach the last step, we must stop
    decision_probability[n_total, :] = 1

    # Floating-point errors may sometimes cause probabilities to slightly exceed the legal range.
    # We just clamp these, as allowing them to persist will NaN-poison our entire program.
    np.clip(decision_probability, 0, 1, out=decision_probability)

    return decision_probability
