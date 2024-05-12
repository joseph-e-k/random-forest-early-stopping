from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from enum import Enum
from fractions import Fraction
from io import StringIO

import pulp
import scipy
from frozendict import frozendict
from scipy.sparse import dok_array

SparseArray = dok_array

Constant = int | float | Fraction

CONSTANT_COEFF_KEY = None

GUROBI_CL_PATH = "/home/josephkalman/gurobi1100/linux64/bin/gurobi_cl"
GUROBI_LIB_PATH = "/home/josephkalman/gurobi1100/linux64/lib"
SOPLEX_CL_FORMAT = "soplex {} --real:feastol=0 --real:opttol=0 --int:solvemode=2 --int:syncmode=1 --int:readmode=1 --int:checkmode=2 -X={}"


class OptimizationFailure(Exception):
    pass


class Coefficients(frozendict[str | None, Constant]):
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            return 0


class ComparisonOperator(Enum):
    Eq = "=="
    LEq = "<="


@dataclasses.dataclass(frozen=True)
class ArithmeticExpression:
    coefficients: Coefficients

    def evaluate(self, context):
        # This insanity allows us to call ArithmeticExpression.evaluate() as if it were a static function,
        # and have it work for constants as well
        if isinstance(self, Constant):
            return self

        return sum(context[name] * value for (name, value) in self.coefficients.items())

    @classmethod
    def from_constant(cls, const: Constant):
        return cls(Coefficients([(CONSTANT_COEFF_KEY, const)]))

    def __add__(self, other: ArithmeticExpression | Constant):
        if isinstance(other, ArithmeticExpression):
            return self._add_arithmetic_expression(other)
        if isinstance(other, Constant):
            return self._add_constant(other)
        raise TypeError(f"In {type(self)}.__add__, `other` must be an ArithmeticExpression or Constant")

    def _add_arithmetic_expression(self, other: ArithmeticExpression):
        return ArithmeticExpression(Coefficients(
            (index, (self.coefficients[index] + other.coefficients[index]))
            for index in (self.coefficients.keys() | other.coefficients.keys())
        ))

    def _add_constant(self, const: Constant):
        return ArithmeticExpression(Coefficients(
            self.coefficients.set(
                CONSTANT_COEFF_KEY,
                self.coefficients[CONSTANT_COEFF_KEY] + const)
        ))

    def __radd__(self, other: Constant):
        return self._add_constant(other)

    def __mul__(self, other: Constant):
        return ArithmeticExpression(Coefficients(
            (key, other * value)
            for (key, value) in self.coefficients.items()
        ))

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other: Constant):
        return ArithmeticExpression(Coefficients(
            (key, value / other)
            for (key, value) in self.coefficients.items()
        ))

    def __eq__(self, other: ArithmeticExpression | Constant) -> LogicalExpression:
        if isinstance(other, Constant):
            other = ArithmeticExpression.from_constant(other)
        return LogicalExpression(self, other, ComparisonOperator.Eq)

    def __le__(self, other) -> LogicalExpression:
        if isinstance(other, Constant):
            other = ArithmeticExpression.from_constant(other)
        return LogicalExpression(self, other, ComparisonOperator.LEq)

    def __ge__(self, other) -> LogicalExpression:
        if isinstance(other, Constant):
            other = ArithmeticExpression.from_constant(other)
        return LogicalExpression(other, self, ComparisonOperator.LEq)

    def __repr__(self):
        items = sorted(self.coefficients.items(), key=(lambda kv: (kv[0] is not None, kv[0])))
        item_strings = []

        for name, value in items:
            if value == 0:
                continue
            if name is None:
                name = ""
            if value < 0:
                item_strings.append(f"-{abs(value)}{name}")
            else:
                item_strings.append(f"+{value}{name}")
        if item_strings:
            return " ".join(item_strings)
        return f"{type(self).__name__}.{self.from_constant.__name__}(0)"


@dataclasses.dataclass(frozen=True)
class LogicalExpression:
    lhs: ArithmeticExpression
    rhs: ArithmeticExpression
    operator: ComparisonOperator

    def isolate_constants_on_rhs(self) -> LogicalExpression:
        lhs_coeffs = (self.lhs - self.rhs).coefficients.set(CONSTANT_COEFF_KEY, 0)
        rhs_value = self.rhs.coefficients[CONSTANT_COEFF_KEY] - self.lhs.coefficients[CONSTANT_COEFF_KEY]
        return LogicalExpression(
            lhs=ArithmeticExpression(lhs_coeffs),
            rhs=ArithmeticExpression.from_constant(rhs_value),
            operator=self.operator
        )

    def __repr__(self):
        return f"{self.lhs!r} {self.operator.value} {self.rhs!r}"


@dataclasses.dataclass
class Problem:
    variable_names_to_indices: dict[str, int] = dataclasses.field(default_factory=dict)
    constraints: set[LogicalExpression] = dataclasses.field(default_factory=set)
    objective: ArithmeticExpression = None
    # TODO: Allow maximization problems as well

    @property
    def n_variables(self):
        return len(self.variable_names_to_indices)

    def add_variable(self, variable_name: str, lower_bound: Constant = None, upper_bound: Constant = None) -> ArithmeticExpression:
        if variable_name in self.variable_names_to_indices:
            raise ValueError(f"Variable name {variable_name} already in use")

        self.variable_names_to_indices[variable_name] = len(self.variable_names_to_indices)

        variable = ArithmeticExpression(Coefficients([(variable_name, 1)]))

        if lower_bound is not None:
            self.add_constraint(variable >= lower_bound)
        if upper_bound is not None:
            self.add_constraint(variable <= upper_bound)

        return variable

    def add_constraint(self, constraint: LogicalExpression):
        self.constraints.add(constraint)

    def set_objective(self, objective: ArithmeticExpression, override: bool = False):
        if self.objective is not None and not override:
            raise ValueError(f"Objective is already set to ({self.objective}); use override=True to override")

        self.objective = objective

    def expression_as_sparse_array(self, expression: ArithmeticExpression) -> SparseArray:
        array = SparseArray((1, self.n_variables), dtype=float)

        for name, value in expression.coefficients.items():
            if value == 0:
                continue
            index = self.variable_names_to_indices[name]
            array[0, index] = value

        return array


    def solve_with_scipy(self, method="highs"):
        if self.objective is None:
            raise ValueError("Must set an objective function before converting to SciPy format")

        A_ub, b_ub, A_eq, b_eq = self._get_constraints_scipy_format()

        scipy_result = scipy.optimize.linprog(
            c=self._get_objective_coeffs_array().toarray(),
            A_ub=A_ub.toarray(),
            b_ub=b_ub.toarray(),
            A_eq=A_eq.toarray(),
            b_eq=b_eq.toarray(),
            bounds=(None, None),
            method=method
        )

        return OptimizationResult.from_scipy_format(scipy_result, self)

    def _get_objective_coeffs_array(self):
        return self.expression_as_sparse_array(self.objective)

    def _get_constraints_scipy_format(self):
        constraints_by_operator = {
            ComparisonOperator.LEq: set(),
            ComparisonOperator.Eq: set()
        }

        for constraint in self.constraints:
            constraints_by_operator[constraint.operator].add(constraint)

        for operator in [ComparisonOperator.LEq, ComparisonOperator.Eq]:
            yield from self._filtered_constraints_to_scipy_format(constraints_by_operator[operator])

    def _filtered_constraints_to_scipy_format(self, constraints: set[LogicalExpression]):
        lhs_coeffs_matrix = SparseArray((len(constraints), self.n_variables))
        rhs_constants_array = SparseArray((len(constraints), 1))

        for i_constraint, constraint in enumerate(constraints):
            constraint: LogicalExpression = constraint.isolate_constants_on_rhs()
            lhs_coeffs_matrix[i_constraint, :] = self.expression_as_sparse_array(constraint.lhs)
            rhs_constants_array[i_constraint, 0] = constraint.rhs.coefficients[CONSTANT_COEFF_KEY]

        return lhs_coeffs_matrix, rhs_constants_array

    def solve_with_pulp(self):
        pulp_problem = pulp.LpProblem(sense=pulp.LpMinimize)
        pulp_vars_by_name = {
            name: pulp.LpVariable(name)
            for name in self.variable_names_to_indices
        }

        pulp_problem += (
            self._arithmetic_expression_to_pulp_format(self.objective, pulp_vars_by_name),
            "objective"
        )

        for i_constraint, constraint in enumerate(self.constraints):
            pulp_problem += (
                self._logical_expression_to_pulp_format(constraint, pulp_vars_by_name),
                f"constraint_{i_constraint}"
            )

        if os.path.exists(GUROBI_CL_PATH):
            os.putenv("LD_LIBRARY_PATH", GUROBI_LIB_PATH)
            solver = pulp.GUROBI_CMD(path=GUROBI_CL_PATH, msg=False)
        else:
            solver = pulp.PULP_CBC_CMD(msg=False)
        pulp_problem.solve(solver=solver)
        return OptimizationResult.from_pulp_format(pulp_problem, pulp_vars_by_name)

    @staticmethod
    def _arithmetic_expression_to_pulp_format(expression: ArithmeticExpression, pulp_vars_by_name: dict):
        coeffs = expression.coefficients

        return sum(
            coeff * pulp_vars_by_name[name]
            for (name, coeff) in coeffs.items()
            if name is not None
        ) + coeffs[CONSTANT_COEFF_KEY]

    @staticmethod
    def _logical_expression_to_pulp_format(expression: LogicalExpression, pulp_vars_by_name: dict):
        pulp_lhs = Problem._arithmetic_expression_to_pulp_format(expression.lhs, pulp_vars_by_name)
        pulp_rhs = Problem._arithmetic_expression_to_pulp_format(expression.rhs, pulp_vars_by_name)

        if expression.operator == ComparisonOperator.LEq:
            return pulp_lhs <= pulp_rhs

        if expression.operator == ComparisonOperator.Eq:
            return pulp_lhs == pulp_rhs

        raise ValueError(f"Unknown comparison operator: {expression.operator}")

    def solve_with_soplex(self) -> OptimizationResult:
        lp_file = tempfile.NamedTemporaryFile("wt")
        solution_file = tempfile.NamedTemporaryFile("rt")

        with lp_file:
            buffer = StringIO()
            self.save_as_lp_format(buffer)
            lp_text = buffer.getvalue()
            print("*** BEGIN LP FILE ***")
            print(lp_text)
            print("*** END LP FILE ***")
            lp_file.write(lp_text)
            lp_file.flush()
            process = subprocess.run(
                SOPLEX_CL_FORMAT.format(lp_file.name, solution_file.name),
                shell=True,
                capture_output=True,
                text=True
            )

        print("*** BEGIN SOPLEX STDOUT ***")
        print(process.stdout)
        print("*** END SOPLEX STDOUT ***")
        print("*** BEGIN SOPLEX STDERR ***")
        print(process.stderr)
        print("*** END SOPLEX STDERR ***")

        for line in process.stdout.splitlines():
            if m := re.match(r"Objective value\s*: (\S+)$", line):
                objective_value = float(m.group(1))
                break
        else:
            raise OptimizationFailure("Could not get objective value from SoPlex output")

        variable_values = dict()
        with solution_file:
            for line in solution_file:
                if m := re.match(r"(\S+)\s+(\S+)$", line):
                    variable_values[m.group(1)] = Fraction(m.group(2))

        return OptimizationResult(
            objective_value=objective_value,
            variable_values=Coefficients(variable_values)
        )

    def save_as_lp_format(self, file):
        file.write("Minimize\n")
        file.write(f" {self._arithmetic_expression_to_lp_format(self.objective)}\n")
        file.write("Subject To\n")
        for i_constraint, constraint in enumerate(self.constraints):
            constraint_str = self._logical_expression_to_lp_format(constraint)
            if constraint_str == "0 <= 0":
                continue
            file.write(f" c{i_constraint}: {constraint_str}\n")
        file.write("Generals\n")
        for var_name in self.variable_names_to_indices.keys():
            file.write(f" {var_name}")
        file.write("\n")
        file.write("End\n")

    @staticmethod
    def _arithmetic_expression_to_lp_format(expression: ArithmeticExpression):
        parts = []
        for var_name, coef in expression.coefficients.items():
            if coef == 0:
                continue
            parts.append("+" if coef > 0 else "-")
            if abs(coef) != 1 or var_name == CONSTANT_COEFF_KEY:
                parts.append(str(abs(coef)))
            if var_name != CONSTANT_COEFF_KEY:
                parts.append(var_name)
        if len(parts) == 0:
            return 0
        if parts[0] == "+":
            parts = parts[1:]
        return " ".join(parts)

    @classmethod
    def _logical_expression_to_lp_format(cls, expr: LogicalExpression):
        expr = expr.isolate_constants_on_rhs()
        lhs_str = cls._arithmetic_expression_to_lp_format(expr.lhs)
        operator_str = {
            ComparisonOperator.LEq: "<=",
            ComparisonOperator.Eq: "="
        }[expr.operator]
        rhs_str = cls._arithmetic_expression_to_lp_format(expr.rhs)
        ret = f"{lhs_str} {operator_str} {rhs_str}"
        if re.match(r".*<=\s*$", ret):
            import pdb
            pdb.set_trace()
        return ret


@dataclasses.dataclass(frozen=True)
class OptimizationResult:
    variable_values: Coefficients
    objective_value: Constant

    def __getitem__(self, item):
        return self.variable_values[item]

    @classmethod
    def from_scipy_format(cls, scipy_result, problem):
        if not scipy_result.success:
            raise OptimizationFailure()

        variable_values = dict()

        for variable_name, variable_index in problem.variable_names_to_indices.items():
            variable_value = scipy_result.x[variable_index]
            if variable_value != 0:
                variable_values[variable_name] = scipy_result.x[variable_index]

        return cls(
            variable_values=Coefficients(variable_values),
            objective_value=scipy_result.fun
        )

    @classmethod
    def from_pulp_format(cls, pulp_problem, pulp_vars_by_name):
        if pulp_problem.status != 1:
            raise OptimizationFailure()

        variable_values = {
            name: pulp.value(pulp_var)
            for (name, pulp_var) in pulp_vars_by_name.items()
        }

        return cls(
            variable_values=Coefficients(variable_values),
            objective_value=pulp.value(pulp_problem.objective)
        )


if __name__ == "__main__":
    prob = Problem()
    x = prob.add_variable("x", lower_bound=0)
    y = prob.add_variable("y", lower_bound=0)
    objective = -(50 * x + 60 * y)
    prob.set_objective(objective)
    prob.add_constraint(5 * x + 8 * y <= 180)
    prob.add_constraint(5 * x + 4 * y <= 120)

    print("Solving with SciPy:")
    solution = prob.solve_with_scipy()
    print(f"x = {x.evaluate(solution)}, y = {y.evaluate(solution)}")
    print(f"objective = {objective.evaluate(solution)}")
    print(solution)
    print()

    print("Solving with PuLP:")
    solution = prob.solve_with_pulp()
    print(f"x = {x.evaluate(solution)}, y = {y.evaluate(solution)}")
    print(f"objective = {objective.evaluate(solution)}")
    print(solution)
    print()

    print("Solving with SoPlex:")
    solution = prob.solve_with_soplex()
    print(f"x = {x.evaluate(solution)}, y = {y.evaluate(solution)}")
    print(f"objective = {objective.evaluate(solution)}")
    print(solution)