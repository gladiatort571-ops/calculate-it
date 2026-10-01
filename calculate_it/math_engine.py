"""Math mode engine for calculate-it."""

import re
from typing import Optional

from .parser import parse_expression, Mode
from .solver import EquationSolver, SolverError
from .ai_engine import AIEngine


# Math constants (9 digits)
MATH_CONSTANTS = {
    'pi': '3.14159265',
    'eu': '2.71828183',
    'tau': '6.28318531',
}


class MathEngine:
    """Engine for math (default) mode calculations."""

    def __init__(self):
        self.solver = EquationSolver()

    def solve(self, expression: str) -> str:
        """Solve a math expression.

        Args:
            expression: The input expression string.

        Returns:
            String representation of the result.
        """
        if not expression or not expression.strip():
            return "Error: Empty expression"

        # Check for target variable specification (e.g., "xy-1=0, x")
        target_var = None
        if ',' in expression:
            parts = expression.rsplit(',', 1)
            if len(parts) == 2:
                expression = parts[0].strip()
                target_var = parts[1].strip()

        # Parse the expression
        parse_result = parse_expression(expression, mode=Mode.MATH)

        # Substitute constants
        ast_string = parse_result.ast_string
        for const_name, const_value in MATH_CONSTANTS.items():
            # Replace whole-word matches only
            ast_string = re.sub(r'\b' + const_name + r'\b', f'({const_value})', ast_string)

        # Handle imaginary unit (i -> I in sympy)
        ast_string = re.sub(r'\bi\b', 'I', ast_string)

        # Solve
        try:
            result = self.solver.solve(ast_string, target_var)
            # Check if the result indicates failure and fall back to AI
            if result and ("Error" in result or "no solution" in result or "Could not" in result):
                ai = AIEngine()
                return ai.solve(expression, mode="math")
            return result
        except SolverError as e:
            # Fall back to AI for complex expressions
            try:
                ai = AIEngine()
                return ai.solve(expression, mode="math")
            except Exception:
                return f"Error: {e}"
        except Exception as e:
            # Fall back to AI for complex expressions
            try:
                ai = AIEngine()
                return ai.solve(expression, mode="math")
            except Exception:
                return f"Error: Unexpected error: {e}"
