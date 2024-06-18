from __future__ import annotations

import dataclasses
import re
import subprocess
import tempfile
from enum import Enum
from fractions import Fraction
from io import StringIO

from frozendict import frozendict

Constant = int | float | Fraction

CONSTANT_COEFF_KEY = None

GUROBI_CL_PATH = "/home/josephkalman/gurobi1100/linux64/bin/gurobi_cl"
GUROBI_LIB_PATH = "/home/josephkalman/gurobi1100/linux64/lib"
SOPLEX_CL_FORMAT = "soplex {} --real:feastol=0 --real:opttol=0 --int:solvemode=2 --int:syncmode=1 --int:readmode=1 --int:checkmode=2 --int:multiprecision_limit=3000 -X={}"


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

    def solve_with_soplex(self) -> OptimizationResult:
        lp_file = tempfile.NamedTemporaryFile("wt")
        solution_file = tempfile.NamedTemporaryFile("rt")

        with lp_file:
            self.save_as_lp_format(lp_file)
            lp_file.flush()
            process = subprocess.run(
                SOPLEX_CL_FORMAT.format(lp_file.name, solution_file.name),
                shell=True,
                capture_output=True,
                text=True
            )

        return OptimizationResult.from_soplex_output(
            StringIO(process.stdout),
            StringIO(process.stderr),
            solution_file
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
            return "0"
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
        return ret


@dataclasses.dataclass(frozen=True)
class OptimizationResult:
    variable_values: Coefficients
    objective_value: Constant

    def __getitem__(self, item):
        return self.variable_values[item]

    @classmethod
    def from_soplex_output(cls, stdout, stderr, solution_file):
        success = False
        objective_value = None

        for line in stdout:
            if re.match(r"SoPlex status\s*:\s*problem is solved \[optimal\]\s*$", line):
                success = True
            if m := re.match(r"Objective value\s*: (\S+)$", line):
                objective_value = float(m.group(1))
            if success and objective_value is not None:
                break
        else:
            stdout.seek(0)
            output_text = stdout.read()
            error_text = stderr.read()

            if not success:
                message = "SoPlex status was not 'optimal' or could not be parsed from output"
            else:
                message = "objective value could not be parsed from SoPlex output"

            raise OptimizationFailure(
                message,
                output_text,
                error_text
            )

        variable_values = dict()
        with solution_file:
            for line in solution_file:
                if m := re.match(r"(\S+)\s+(\S+)$", line):
                    variable_values[m.group(1)] = Fraction(m.group(2))

        return OptimizationResult(
            objective_value=objective_value,
            variable_values=Coefficients(variable_values)
        )


if __name__ == "__main__":
    prob = Problem()
    x = prob.add_variable("x", lower_bound=0)
    y = prob.add_variable("y", lower_bound=0)
    objective = -(50 * x + 60 * y)
    prob.set_objective(objective)
    prob.add_constraint(5 * x + 8 * y <= 180)
    prob.add_constraint(5 * x + 4 * y <= 120)

    print("Solving with SoPlex:")
    solution = prob.solve_with_soplex()
    print(f"x = {x.evaluate(solution)}, y = {y.evaluate(solution)}")
    print(f"objective = {objective.evaluate(solution)}")
    print(solution)
