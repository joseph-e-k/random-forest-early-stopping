from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import dataclasses
import os
import re
import shutil
import subprocess
import tempfile
from enum import Enum
from fractions import Fraction
from io import StringIO
from typing import Self

from .logging import get_module_logger


CONSTANT_COEFF_KEY = None

SOPLEX_EXECUTABLE_PATH = os.getenv("SOPLEX_PATH", None)


_logger = get_module_logger()


class UnspecifiedExecutableError(Exception):
    pass


class OptimizationFailure(Exception):
    pass


class FinalizedProblemError(Exception):
    pass


class ComparisonOperator(Enum):
    Eq = "=="
    LEq = "<="


class OptimizationSense(Enum):
    Minimize = "Minimize"
    Maximize = "Maximize"


Constant = int | float | Fraction


class ArithmeticExpression(defaultdict[str | None, Constant]):
    """A constant, variable, or linear combination of constants and variables"""
    def __init__(self, items_source=(), /, **kwargs):
        super().__init__(lambda: 0, items_source, **kwargs)

    @staticmethod
    def evaluate(expression: Self | Constant, context: Mapping[str, Constant]):
        if isinstance(expression, Constant):
            return expression

        return sum(context[name] * value for (name, value) in expression.items())

    @classmethod
    def from_constant(cls, const: Constant):
        return cls.from_coeffs([(CONSTANT_COEFF_KEY, const)])
    
    @classmethod
    def from_coeffs(cls, coeffs_source=()):
        if isinstance(coeffs_source, Mapping):
            coeff_items = coeffs_source.items()
        else:
            coeff_items = coeffs_source
        return cls(
            [
                (key, value)
                for (key, value) in coeff_items
                if value != 0
            ]
        )

    def __add__(self, other: ArithmeticExpression | Constant):
        if isinstance(other, ArithmeticExpression):
            return self._add_arithmetic_expression(other)
        if isinstance(other, Constant):
            return self._add_constant(other)
        raise TypeError(f"In {type(self)}.__add__, `other` must be an ArithmeticExpression or Constant")

    def _add_arithmetic_expression(self, other: ArithmeticExpression):
        new = ArithmeticExpression(self)
        for key, value in other.items():
            new[key] += value
        return new

    def _add_constant(self, const: Constant):
        return self + ArithmeticExpression.from_constant(const)

    def __radd__(self, other: Constant):
        return self._add_constant(other)

    def __mul__(self, other: Constant):
        return ArithmeticExpression.from_coeffs(
            (key, other * value)
            for (key, value) in self.items()
        )

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other: Constant):
        return ArithmeticExpression.from_coeffs(
            (key, value / other)
            for (key, value) in self.items()
        )

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
        items = sorted(self.items(), key=(lambda kv: (kv[0] is not None, kv[0])))
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
    """Two arithmetic expressions joined connected by a comparison operator (== or <=)"""
    lhs: ArithmeticExpression
    rhs: ArithmeticExpression
    operator: ComparisonOperator

    def isolate_constants_on_rhs(self) -> LogicalExpression:
        lhs = self.lhs - self.rhs
        lhs.pop(CONSTANT_COEFF_KEY, None)
        rhs_value = self.rhs[CONSTANT_COEFF_KEY] - self.lhs[CONSTANT_COEFF_KEY]
        return LogicalExpression(
            lhs=lhs,
            rhs=ArithmeticExpression.from_constant(rhs_value),
            operator=self.operator
        )
    
    def is_tautology(self) -> bool:
        return len(self.isolate_constants_on_rhs().lhs.keys()) == 0

    def __repr__(self):
        return f"{self.lhs!r} {self.operator.value} {self.rhs!r}"


class Problem:
    """A linear programming problem consisting of variables, constraints, and an objective"""

    def __init__(self):
        self.tag = os.urandom(8).hex()
        self._file_prefix = f"lp_problem_{self.tag}"
        self._objective_file = tempfile.NamedTemporaryFile("wt", prefix=f"{self._file_prefix}_objective_", delete_on_close=False)
        self._constraints_file = tempfile.NamedTemporaryFile("wt", prefix=f"{self._file_prefix}_constraints_", delete_on_close=False)
        self._n_constraints = 0
        self._variables_file = tempfile.NamedTemporaryFile("wt", prefix=f"{self._file_prefix}_variables_", delete_on_close=False)
        self._optimization_sense = OptimizationSense.Minimize
        self._is_finalized = False
        self._optimization_result = None

    is_finalized = property(lambda self: self._is_finalized)

    def add_variable(self, variable_name: str, lower_bound: Constant = None, upper_bound: Constant = None) -> ArithmeticExpression:
        """Add a variable to the problem.

        Args:
            variable_name (str): Name of the variable to be created.
            lower_bound (Constant, optional): Lower bound of the variable's feasible range. Defaults to -infinity.
            upper_bound (Constant, optional): Upper bound of the variable's feasible range. Defaults to +infinity.

        Raises:
            FinalizedProblemError: The problem has already been finalized (by trying to solve it) and cannot be changed further.

        Returns:
            ArithmeticExpression: The newly created variable.
        """
        if self._is_finalized:
            raise FinalizedProblemError("This problem has been finalized and cannot be changed")

        variable = ArithmeticExpression.from_coeffs([(variable_name, 1)])
        self._variables_file.write(f" {variable_name}\n")

        if lower_bound is not None:
            self.add_constraint(variable >= lower_bound)
        if upper_bound is not None:
            self.add_constraint(variable <= upper_bound)

        return variable

    def add_constraint(self, constraint: LogicalExpression):
        """Add a constraint to the problem.

        Args:
            constraint (LogicalExpression): Constraint to be added. Must not include any variables that do not belong to this problem.

        Raises:
            FinalizedProblemError: The problem has already been finalized (by trying to solve it) and cannot be changed further.
        """
        if self._is_finalized:
            raise FinalizedProblemError("This problem has been finalized and cannot be changed")

        if constraint.is_tautology():
            return
        
        self._n_constraints += 1
        constraint_str = self._logical_expression_to_lp_format(constraint)
        self._constraints_file.write(f" c{self._n_constraints}: {constraint_str}\n")

    def set_objective(self, objective: ArithmeticExpression, sense: OptimizationSense = OptimizationSense.Minimize):
        """Set the the expression to minimized or maximized

        Args:
            objective (ArithmeticExpression): Expression to be minimized or maximized
            sense (OptimizationSense, optional): Whether the expression should be minimized or maximized. Defaults to OptimizationSense.Minimize.

        Raises:
            FinalizedProblemError: The problem has already been finalized (by trying to solve it) and cannot be changed further.
        """
        if self._is_finalized:
            raise FinalizedProblemError("This problem has been finalized and cannot be changed")
        
        self._objective_file.write(f" {self._arithmetic_expression_to_lp_format(objective)}\n")
        self._optimization_sense = sense

    def solve_with_soplex(self) -> OptimizationResult:
        """Solve this linear programming problem using SoPlex's precise rational solver.

        Raises:
            UnspecifiedExecutableError: The SOPLEX_PATH environment variable was not set.

        Returns:
            OptimizationResult
        """
        if self._is_finalized:
            return self._optimization_result

        if SOPLEX_EXECUTABLE_PATH is None:
            raise UnspecifiedExecutableError("SOPLEX_PATH environment variable must be set to point to the SoPlex executable file")
        
        self._is_finalized = True

        lp_file = tempfile.NamedTemporaryFile("wt", prefix=f"{self._file_prefix}_problem_")
        solution_file = tempfile.NamedTemporaryFile("rt", prefix=f"{self._file_prefix}_solution_")

        with lp_file:
            _logger.info("Serializing problem to disk...")
            self._save_as_lp_format(lp_file)
            lp_file.flush()
            _logger.info("Running SoPlex to solve problem...")
            process = subprocess.run(
                [
                    SOPLEX_EXECUTABLE_PATH,
                    lp_file.name,
                    "--real:feastol=0",
                    "--real:opttol=0",
                    "--int:solvemode=2",
                    "--int:syncmode=1",
                    "--int:readmode=1",
                    "--int:checkmode=2",
                    "--int:multiprecision_limit=2147483647",
                    "-X={}".format(solution_file.name)
                ],
                capture_output=True,
                text=True
            )

        _logger.info("Parsing SoPlex output...")
        self._optimization_result = OptimizationResult.from_soplex_output(
            StringIO(process.stdout),
            StringIO(process.stderr),
            solution_file
        )
        return self._optimization_result
    
    @staticmethod
    def _close_and_copy_file(source_file, dest_file, mode="t"):
        source_file.close()
        with open(source_file.name, "r" + mode) as source_file:
            shutil.copyfileobj(source_file, dest_file)

    def _save_as_lp_format(self, lp_file):
        lp_file.write(f"{self._optimization_sense.value}\n")
        self._close_and_copy_file(self._objective_file, lp_file)
        
        lp_file.write("Subject To\n")
        self._close_and_copy_file(self._constraints_file, lp_file)

        lp_file.write("Generals\n")
        self._close_and_copy_file(self._variables_file, lp_file)

        lp_file.write("End\n")

    @staticmethod
    def _arithmetic_expression_to_lp_format(expression: ArithmeticExpression):
        parts = []
        for var_name, coef in expression.items():
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
        return f"{lhs_str} {operator_str} {rhs_str}"


@dataclasses.dataclass(frozen=True)
class OptimizationResult:
    """
    Represents the result of solving a linear programming problem.

    Attributes:
        variable_values (Mapping[str, Constant]): A mapping of variable names to their optimized values.
        objective_value (Constant): The value of the objective function at the optimal solution.
        seconds_to_solve (float): The time taken to solve the problem, in seconds.
    """

    variable_values: Mapping[str, Constant]
    objective_value: Constant
    seconds_to_solve: float

    def __getitem__(self, item):
        """
        Retrieve the value of a specific variable by its name.

        Args:
            item (str): The name of the variable.

        Returns:
            Constant: The value of the variable.
        """
        return self.variable_values[item]

    @classmethod
    def from_soplex_output(cls, stdout, stderr, solution_file):
        """
        Parse the output of the SoPlex solver to create an OptimizationResult instance.

        Args:
            stdout (StringIO): The standard output from the SoPlex solver.
            stderr (StringIO): The standard error output from the SoPlex solver.
            solution_file (file): The file containing the solution values.

        Returns:
            OptimizationResult: An instance containing the parsed results.

        Raises:
            OptimizationFailure: If the solver did not return an optimal solution or if parsing failed.
        """
        success = False
        objective_value = None
        seconds_to_solve = None

        for line in stdout:
            if m := re.match(r"Solving time \(sec\)\s*: ([\d.]+)$", line):
                seconds_to_solve = float(m.group(1))
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

        variable_values = defaultdict(lambda: 0)
        with solution_file:
            for line in solution_file:
                if m := re.match(r"(\S+)\s+(\S+)$", line):
                    variable_values[m.group(1)] = Fraction(m.group(2))

        return OptimizationResult(
            objective_value=objective_value,
            variable_values=variable_values,
            seconds_to_solve=seconds_to_solve
        )
