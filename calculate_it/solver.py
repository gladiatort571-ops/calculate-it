"""Equation solver using sympy."""

from typing import List, Optional, Tuple, Union

import sympy
from sympy import symbols, Eq, solve, sympify, Symbol, simplify, Integral, Derivative, diff, integrate


# Priority variables for solving
PRIORITY_VARS = ['x', 'y', 'z', 't']


class SolverError(Exception):
    """Raised when solving fails."""
    pass


class EquationSolver:
    """Solves mathematical equations using sympy."""

    def __init__(self):
        self.x, self.y, self.z, self.t = symbols('x y z t')
        self.var_map = {
            'x': self.x, 'y': self.y, 'z': self.z, 't': self.t,
        }

    def solve(self, ast_string: str, target_var: Optional[str] = None) -> str:
        """Solve an equation given its AST string.

        Args:
            ast_string: The parsed expression string (e.g., "2*x + 3 = 7")
            target_var: Optional variable to solve for. If None, solves for
                        priority variables (x, y, z, t).

        Returns:
            String representation of the solution.
        """
        # Split on '='
        if '=' not in ast_string:
            # No equation, just evaluate
            return self._evaluate(ast_string)

        parts = ast_string.split('=')
        if len(parts) != 2:
            raise SolverError("Invalid equation: multiple '=' signs")

        left_str = parts[0].strip()
        right_str = parts[1].strip()

        try:
            left_expr = sympify(left_str)
            right_expr = sympify(right_str)
        except Exception as e:
            raise SolverError(f"Could not parse expression: {e}")

        equation = Eq(left_expr, right_expr)

        # Determine which variable(s) to solve for
        if target_var:
            if target_var not in self.var_map:
                # Create symbol on the fly
                target = symbols(target_var)
            else:
                target = self.var_map[target_var]
            return self._solve_for(equation, target, target_var)

        # Find all priority variables in the equation
        found_vars = []
        for var_name in PRIORITY_VARS:
            var_symbol = self.var_map[var_name]
            if equation.has(var_symbol):
                found_vars.append((var_name, var_symbol))

        if not found_vars:
            # Try to find any symbol in the equation
            free_symbols = equation.free_symbols
            if free_symbols:
                for sym in free_symbols:
                    found_vars.append((str(sym), sym))

        if not found_vars:
            # No variables, just evaluate
            result = simplify(left_expr - right_expr)
            if result == 0:
                return "True (identity)"
            else:
                return f"False (contradiction: {result} != 0)"

        # Solve for all found variables
        solutions = []
        for var_name, var_symbol in found_vars:
            sol = self._solve_for(equation, var_symbol, var_name)
            solutions.append(sol)

        return '\n'.join(solutions)

    def _solve_for(self, equation: Eq, target: Symbol, target_name: str) -> str:
        """Solve equation for a specific target variable."""
        try:
            raw_solutions = solve(equation, target)
        except Exception as e:
            raise SolverError(f"Could not solve for {target_name}: {e}")

        if not raw_solutions:
            return f"{target_name} = (no solution)"

        # Format solutions
        formatted = []
        for sol in raw_solutions:
            simplified = simplify(sol)
            # Try to get both exact and approximate
            try:
                approx = float(simplified.evalf())
                if approx == int(approx):
                    formatted.append(f"{target_name} = {int(approx)}")
                else:
                    formatted.append(f"{target_name} = {simplified} | {target_name} ≈ {approx:.6g}")
            except (TypeError, ValueError):
                formatted.append(f"{target_name} = {simplified}")

        return '\n'.join(formatted)

    def _evaluate(self, expr_string: str) -> str:
        """Evaluate a non-equation expression."""
        try:
            # Check if the expression contains Integral or Derivative before sympify
            # to prevent automatic evaluation
            has_integral = 'Integral(' in expr_string
            has_derivative = 'Derivative(' in expr_string

            if has_integral or has_derivative:
                # Parse without evaluating
                expr = sympify(expr_string, evaluate=False)
            else:
                expr = sympify(expr_string)

            # Handle Integral expressions
            if isinstance(expr, Integral):
                return self._evaluate_integral(expr)

            # Handle Derivative expressions
            if isinstance(expr, Derivative):
                return self._evaluate_derivative(expr)

            simplified = simplify(expr)
            try:
                approx = float(simplified.evalf())
                if approx == int(approx):
                    return f"{int(approx)}"
                return f"{approx:.10g}"
            except (TypeError, ValueError):
                return str(simplified)
        except Exception as e:
            raise SolverError(f"Could not evaluate: {e}")

    def _evaluate_integral(self, expr: Integral) -> str:
        """Evaluate an integral expression.

        For definite integrals, shows the antiderivative and the result.
        For indefinite integrals, shows the antiderivative.
        """
        try:
            # Get the function and integration variable(s)
            func = expr.function
            limits = expr.limits

            if len(limits) == 1 and len(limits[0]) == 1:
                # Indefinite integral: Integral(f(x), x)
                var = limits[0][0]
                result = integrate(func, var)
                result = simplify(result)
                return f"{result} + C"

            elif len(limits) == 1 and len(limits[0]) == 3:
                # Definite integral: Integral(f(x), (x, a, b))
                var, a, b = limits[0]

                # Compute the antiderivative
                antiderivative = integrate(func, var)
                antiderivative = simplify(antiderivative)

                # Evaluate at bounds
                result = antiderivative.subs(var, b) - antiderivative.subs(var, a)
                result = simplify(result)

                # Format the output
                try:
                    numeric = float(result.evalf())
                    if numeric == int(numeric):
                        return f"{antiderivative} | from {a} to {b} = {int(numeric)}"
                    return f"{antiderivative} | from {a} to {b} ≈ {numeric:.10g}"
                except (TypeError, ValueError):
                    return f"{antiderivative} | from {a} to {b} = {result}"

            else:
                # Multiple integrals or other cases
                result = expr.doit()
                result = simplify(result)
                return str(result)

        except Exception as e:
            raise SolverError(f"Could not evaluate integral: {e}")

    def _evaluate_derivative(self, expr: Derivative) -> str:
        """Evaluate a derivative expression."""
        try:
            func = expr.function
            vars = expr.variables

            # Compute the derivative
            result = func
            for var in vars:
                result = diff(result, var)

            result = simplify(result)
            return str(result)

        except Exception as e:
            raise SolverError(f"Could not evaluate derivative: {e}")

    def solve_system(self, equations: List[str]) -> str:
        """Solve a system of equations."""
        sympy_eqs = []
        all_vars = set()

        for eq_str in equations:
            if '=' not in eq_str:
                continue
            parts = eq_str.split('=')
            if len(parts) != 2:
                continue
            try:
                left = sympify(parts[0].strip())
                right = sympify(parts[1].strip())
                sympy_eqs.append(Eq(left, right))
                all_vars.update(left.free_symbols)
                all_vars.update(right.free_symbols)
            except Exception:
                continue

        if not sympy_eqs:
            raise SolverError("No valid equations provided")

        try:
            solutions = solve(sympy_eqs, list(all_vars))
            if isinstance(solutions, dict):
                return '\n'.join(f"{k} = {v}" for k, v in solutions.items())
            elif isinstance(solutions, list):
                return '\n'.join(str(s) for s in solutions)
            else:
                return str(solutions)
        except Exception as e:
            raise SolverError(f"Could not solve system: {e}")
