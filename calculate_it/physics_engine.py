"""Physics mode engine for calculate-it."""

from typing import Dict, List, Optional, Tuple
import re
import sympy
from sympy import symbols, Eq, solve, sympify, Symbol, simplify, Rational

from .parser import parse_expression, Mode
from .ai_engine import AIEngine


# Physics constants
PHYSICS_CONSTANTS: Dict[str, str] = {
    'me': '9.11e-31', 'mp': '1.67e-27', 'mn': '1.67e-27',
    'amu': '1.66e-27', 'ro': '5.29e-11', 'Eo': '-13.6',
    'h': '6.63e-34', 'hb': '1.05e-34',
    'qe': '1.6e-19', 'mu': '1.26e-6', 'e': '8.85e-12',
    'lp': '1.6e-35', 'tp': '5.39e-44',
    'c': '299792458', 'R': '8.314', 'g': '9.81',
}

# Physics symbols and their meanings
PHYSICS_SYMBOLS: Dict[str, list] = {
    'a': ['acceleration'], 'v': ['velocity'], 'd': ['distance'],
    'x': ['displacement'], 'u': ['friction coefficient'],
    'l': ['angular momentum'], 'p': ['density', 'momentum'],
    'P': ['pressure', 'power'], 'r': ['position vector', 'radius'],
    'm': ['mass'], 'ke': ['kinetic energy'], 'pe': ['potential energy'],
    'M': ['total energy'], 'q': ['charge'], 'Q': ['charge'],
    'V': ['voltage', 'volume'], 'R': ['resistance', 'ideal gas constant'],
    'z': ['impedance'], 'Z': ['impedance'],
    'i': ['current'], 'I': ['current'], 'C': ['capacitance'],
    'E': ['energy', 'electric field'], 'f': ['frequency'],
    'F': ['force'], 't': ['time'], 'w': ['angular velocity'],
    'W': ['work'], 'n': ['number of moles'], 'T': ['temperature'],
    'k': ['spring constant'], 'G': ['gravitational constant'],
    'B': ['magnetic field'],
}

# Physics units for each variable (disambiguated)
PHYSICS_UNITS: Dict[str, str] = {
    'a': 'm/s²', 'v': 'm/s', 'd': 'm', 'x': 'm',
    'u': '(dimensionless)', 'l': 'kg·m²/s',
    'p': 'kg/m³ or kg·m/s', 'P': 'Pa or W',
    'r': 'm', 'm': 'kg', 'ke': 'J', 'pe': 'J',
    'M': 'J', 'q': 'C', 'Q': 'C',
    'V': 'V (voltage) or m³ (volume)', 'R': 'Ω (resistance) or J/mol·K (gas constant)',
    'z': 'Ω', 'Z': 'Ω', 'i': 'A', 'I': 'A',
    'C': 'F', 'E': 'J or V/m', 'f': 'Hz',
    'F': 'N', 't': 's', 'w': 'rad/s', 'W': 'J',
    'n': 'mol', 'T': 'K', 'k': 'N/m', 'G': 'N·m²/kg²',
    'g': 'm/s²', 'c': 'm/s', 'h': 'J·s', 'hb': 'J·s', 'B': 'T',
    'me': 'kg', 'mp': 'kg', 'mn': 'kg', 'amu': 'kg',
    'ro': 'm', 'Eo': 'eV', 'qe': 'C', 'mu': 'N/A²',
    'e': 'F/m', 'lp': 'm', 'tp': 's',
}

# Physics formulas: target -> [(formula_str, required_vars, description, unit)]
# formula_str uses Python/sympy syntax with Rational for fractions
PHYSICS_FORMULAS: Dict[str, List[Tuple[str, List[str], str, str]]] = {
    # Mechanics
    'ke': [('Rational(1,2)*m*v**2', ['m', 'v'], 'kinetic energy', 'J')],
    'pe': [('m*g*h', ['m', 'g', 'h'], 'potential energy', 'J')],
    'p': [('m*v', ['m', 'v'], 'momentum', 'kg·m/s')],
    'F': [('m*a', ['m', 'a'], "Newton's second law", 'N'), ('q*v*B', ['q', 'v', 'B'], 'Lorentz force', 'N'), ('q*E', ['q', 'E'], 'electric force', 'N'), ('k*q1*q2/r**2', ['k', 'q1', 'q2', 'r'], "Coulomb's law", 'N'), ('G*m1*m2/r**2', ['G', 'm1', 'm2', 'r'], "Newton's law of gravitation", 'N'), ('k*x', ['k', 'x'], "Hooke's law", 'N')],
    'E': [('m*c**2', ['m', 'c'], 'rest energy', 'J'), ('h*f', ['h', 'f'], 'photon energy', 'J'), ('Rational(1,2)*m*v**2', ['m', 'v'], 'kinetic energy', 'J'), ('m*g*h', ['m', 'g', 'h'], 'potential energy', 'J')],
    'W': [('F*d', ['F', 'd'], 'work', 'J'), ('ke', ['ke'], 'work', 'J'), ('m*g*h', ['m', 'g', 'h'], 'work against gravity', 'J')],
    'P': [('V*I', ['V', 'I'], 'power', 'W'), ('F/A', ['F', 'A'], 'pressure', 'Pa'), ('R*T/V', ['R', 'T', 'V'], 'ideal gas law', 'Pa'), ('V**2/R', ['V', 'R'], 'power from voltage and resistance', 'W'), ('F*v', ['F', 'v'], 'power', 'W'), ('W/t', ['W', 't'], 'power', 'W')],
    'T': [('P*V/R', ['P', 'V', 'R'], 'temperature from ideal gas law', 'K')],
    'V': [('I*R', ['I', 'R'], "Ohm's law", 'V'), ('W/q', ['W', 'q'], 'voltage', 'V'), ('R*T/P', ['R', 'T', 'P'], 'volume from ideal gas law', 'm³')],
    'v': [('d/t', ['d', 't'], 'velocity', 'm/s'), ('a*t', ['a', 't'], 'velocity', 'm/s'), ('f*lambda', ['f', 'lambda'], 'wave velocity', 'm/s')],
    'a': [('v/t', ['v', 't'], 'acceleration', 'm/s²'), ('F/m', ['F', 'm'], 'acceleration', 'm/s²')],
    'd': [('v*t', ['v', 't'], 'distance', 'm')],
    't': [('d/v', ['d', 'v'], 'time', 's')],
    'm': [('ke/(Rational(1,2)*v**2)', ['ke', 'v'], 'mass', 'kg'), ('F/a', ['F', 'a'], 'mass', 'kg'), ('p/v', ['p', 'v'], 'mass', 'kg')],
    'f': [('E/h', ['E', 'h'], 'frequency', 'Hz'), ('c/lambda', ['c', 'lambda'], 'frequency', 'Hz'), ('v/lambda', ['v', 'lambda'], 'frequency', 'Hz')],
    'l': [('m*v*r', ['m', 'v', 'r'], 'angular momentum', 'kg·m²/s')],
    'k': [('F/x', ['F', 'x'], 'spring constant', 'N/m')],
    'w': [('v/r', ['v', 'r'], 'angular velocity', 'rad/s'), ('2*pi*f', ['f'], 'angular velocity', 'rad/s')],
    # Electromagnetism
    'V': [('I*R', ['I', 'R'], "Ohm's law", 'V'), ('W/q', ['W', 'q'], 'voltage', 'V'), ('R*T/P', ['R', 'T', 'P'], 'volume from ideal gas law', 'm³')],
    'I': [('V/R', ['V', 'R'], 'current', 'A'), ('Q/t', ['Q', 't'], 'current', 'A')],
    'R': [('V/I', ['V', 'I'], 'resistance', 'Ω'), ('rho*l/A', ['rho', 'l', 'A'], 'resistance', 'Ω')],
    'C': [('Q/V', ['Q', 'V'], 'capacitance', 'F')],
    'q': [('C*V', ['C', 'V'], 'charge', 'C'), ('I*t', ['I', 't'], 'charge', 'C')],
    'Q': [('C*V', ['C', 'V'], 'charge', 'C'), ('I*t', ['I', 't'], 'charge', 'C')],
    'B': [('F/(q*v)', ['F', 'q', 'v'], 'magnetic field from Lorentz force', 'T'), ('mu*I/(2*pi*r)', ['mu', 'I', 'r'], 'magnetic field around wire', 'T')],
    # Thermodynamics
    'n': [('m/M', ['m', 'M'], 'moles', 'mol')],
    'M': [('m/n', ['m', 'n'], 'molar mass', 'g/mol')],
    # Waves and Optics
    'lambda': [('v/f', ['v', 'f'], 'wavelength', 'm'), ('c/f', ['c', 'f'], 'wavelength', 'm')],
}

CONSTANT_SYMBOLS = {'c', 'g', 'h', 'hb', 'me', 'mp', 'mn', 'amu', 'ro', 'Eo', 'qe', 'mu', 'e', 'lp', 'tp', 'R'}


class PhysicsEngine:
    """Engine for physics mode calculations."""

    def __init__(self):
        self.constants = PHYSICS_CONSTANTS
        self.symbols = PHYSICS_SYMBOLS
        self.formulas = PHYSICS_FORMULAS
        self.units = PHYSICS_UNITS

    def solve(self, expression: str) -> str:
        if not expression or not expression.strip():
            return "Error: Empty expression"
        expression = expression.strip()
        if expression in self.constants:
            return self._lookup_constant(expression)
        if expression in self.symbols:
            return self._lookup_symbol(expression)
        if '=' in expression:
            result = self._solve_equation(expression)
            # Fall back to AI if physics engine can't solve it
            if result and ("Error" in result or "Could not" in result):
                try:
                    ai = AIEngine()
                    return ai.solve(expression, mode="physics")
                except Exception:
                    pass
            return result
        if any(op in expression for op in ['+', '-', '*', '/', '^', '**']):
            result = self._evaluate_expression(expression)
            # Fall back to AI if physics engine can't evaluate it
            if result and ("Error" in result or "Could not" in result):
                try:
                    ai = AIEngine()
                    return ai.solve(expression, mode="physics")
                except Exception:
                    pass
            return result
        return self._describe_capabilities()

    def _lookup_constant(self, symbol: str) -> str:
        if symbol in self.constants:
            value = self.constants[symbol]
            unit = self.units.get(symbol, '')
            return f"{symbol} = {value} {unit}".strip()
        return f"Unknown constant: {symbol}"

    def _lookup_symbol(self, symbol: str) -> str:
        if symbol in self.symbols:
            meanings = ', '.join(self.symbols[symbol])
            unit = self.units.get(symbol, '')
            return f"{symbol} ({meanings}) [{unit}]".strip()
        return f"Unknown symbol: {symbol}"

    def _parse_value(self, value_str: str) -> float:
        """Parse a value string that may contain units and constant references.
        
        Examples:
            '1.5' -> 1.5
            'qe' -> 1.6e-19 (constant reference)
            '0.4c' -> 0.4 * 299792458 (constant reference)
            '0.001T' -> 0.001 (unit stripped)
            '1e-2' -> 0.01
        """
        value_str = value_str.strip()
        
        # Try direct float conversion first
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Insert * between digit and letter (e.g., 0.4c -> 0.4*c)
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', value_str)
        # Replace constants with their values (longest first to avoid partial matches)
        for const_name, const_value in sorted(self.constants.items(), key=lambda x: len(x[0]), reverse=True):
            expr = re.sub(r'(?<!\w)' + re.escape(const_name) + r'(?!\w)', f'({const_value})', expr)
        # Fix implicit multiplication: digit followed by ( -> digit*(
        expr = re.sub(r'(\d)\(', r'\1*(', expr)
        
        # Strip units - remove trailing unit symbols (longest first)
        unit_patterns = [
            'm/s²', 'm/s', 'kg·m²/s', 'kg·m/s', 'kg/m³', 'kg/m/s',
            'N·m²/kg²', 'J/mol·K', 'J/K', 'mol⁻¹', 'mL/L', 'cm³',
            'g/cm³', 'kg·m/s²', 'rad/s', 'N/m', 'Pa', 'W', 'J',
            'N', 'V', 'A', 'Ω', 'F', 'C', 'T', 'Hz', 'kg', 'g',
            'm', 's', 'K', 'mol', 'u', 'eV', '°C'
        ]
        for unit in sorted(unit_patterns, key=len, reverse=True):
            expr = re.sub(r'\*?\s*' + re.escape(unit) + r'$', '', expr)
        
        # Try to evaluate
        try:
            return float(sympify(expr))
        except Exception:
            raise ValueError(f"Could not parse value: {value_str}")

    def _solve_equation(self, expression: str) -> str:
        parts = [p.strip() for p in expression.split(',')]
        known_values: Dict[str, float] = {}
        target_var = None

        for part in parts:
            if '=' in part:
                key, _, value = part.partition('=')
                key = key.strip()
                value = value.strip()
                try:
                    known_values[key] = self._parse_value(value)
                except ValueError as e:
                    return f"Error: {e}"
            else:
                target_var = part.strip()

        if not target_var:
            return self._solve_for_all(known_values)

        # Try to solve with formulas first
        if target_var in self.formulas:
            result = self._solve_with_formulas(target_var, known_values)
            if result:
                return result

        # If target is a constant, show its value
        if target_var in self.constants:
            return self._lookup_constant(target_var)

        if target_var not in self.symbols:
            return f"Error: Unknown variable '{target_var}'"

        return f"Error: Could not solve for '{target_var}' with given values"

    def _solve_with_formulas(self, target_var: str, known_values: Dict[str, float]) -> Optional[str]:
        if target_var not in self.formulas:
            return None

        for formula_str, required_vars, description, unit in self.formulas[target_var]:
            missing = [v for v in required_vars if v not in known_values and v not in self.constants]
            if missing:
                continue

            try:
                # Use a simple namespace with unique names to avoid sympy built-in conflicts
                # Replace variable names with unique identifiers
                namespace = {}
                for v in required_vars:
                    if v in known_values:
                        namespace[f"var_{v}"] = known_values[v]
                    elif v in self.constants:
                        namespace[f"var_{v}"] = float(sympify(self.constants[v]))

                # Replace variable names in formula string (using word boundaries to avoid partial matches)
                modified_formula = formula_str
                for v in sorted(required_vars, key=len, reverse=True):
                    modified_formula = re.sub(r'\b' + re.escape(v) + r'\b', f"var_{v}", modified_formula)

                # Evaluate the formula
                result = eval(modified_formula, {"__builtins__": {}}, {"Rational": Rational, **namespace})

                try:
                    numeric = float(result)
                    if numeric == int(numeric):
                        return f"{target_var} = {int(numeric)} {unit}".strip()
                    return f"{target_var} = {numeric:.10g} {unit}".strip()
                except (TypeError, ValueError):
                    return f"{target_var} = {result}"

            except Exception:
                continue

        return None

    def _solve_for_all(self, known_values: Dict[str, float]) -> str:
        results = []
        user_vars = set(known_values.keys()) - CONSTANT_SYMBOLS

        for target_var, formulas in self.formulas.items():
            if target_var in known_values:
                continue

            is_relevant = False
            for formula_str, required_vars, description in formulas:
                formula_vars = set(required_vars) - CONSTANT_SYMBOLS
                if formula_vars & user_vars:
                    is_relevant = True
                    break

            if not is_relevant:
                continue

            result = self._solve_with_formulas(target_var, known_values)
            if result:
                results.append(result)

        if results:
            return '\n'.join(results)

        return "Error: Could not solve for any variable with the given known values"

    def _evaluate_expression(self, expression: str) -> str:
        try:
            expr = expression
            for const_name, const_value in self.constants.items():
                expr = re.sub(r'\b' + const_name + r'\b', f'({const_value})', expr)
            expr = re.sub(r'\bj\b', 'I', expr)
            result = sympify(expr)
            result = simplify(result)
            try:
                numeric = float(result.evalf())
                if numeric == int(numeric):
                    return f"{int(numeric)}"
                return f"{numeric:.10g}"
            except (TypeError, ValueError):
                return str(result)
        except Exception as e:
            return f"Error: Could not evaluate expression: {e}"

    def _describe_capabilities(self) -> str:
        return (
            "Physics mode (-p)\n\n"
            "Usage: c -p \"m=1.5, v=8.94, ke\"\n\n"
            "Supported formulas:\n"
            "  ke = 1/2*m*v^2 (kinetic energy)\n"
            "  pe = m*g*h (potential energy)\n"
            "  p = m*v (momentum)\n"
            "  F = m*a (Newton's second law)\n"
            "  E = m*c^2 (rest energy)\n"
            "  E = h*f (photon energy)\n"
            "  V = I*R (Ohm's law)\n"
            "  P = V*I (power)\n"
            "  v = d/t (velocity)\n"
            "  a = v/t (acceleration)\n"
            "  W = F*d (work)\n"
            "  l = m*v*r (angular momentum)\n\n"
            "Constants: h, c, me, mp, mn, g, R, etc."
        )
