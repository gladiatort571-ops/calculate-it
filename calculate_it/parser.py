"""Expression parser for calculate-it.

Handles implicit multiplication, power notation, function grouping,
and spacing rules as defined in the specification.

Mode-aware: math uses 'i' for imaginary, physics uses 'j'.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict, Any


class Mode(Enum):
    MATH = auto()
    PHYSICS = auto()
    CHEMISTRY = auto()


class TokenType(Enum):
    NUMBER = auto()
    VARIABLE = auto()
    FUNCTION = auto()
    OPERATOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    SPACE = auto()
    COMMA = auto()
    EQUALS = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    position: int = 0


# Known functions
FUNCTIONS = {
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'asin', 'acos', 'atan', 'acot', 'asec', 'acsc',
    'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
    'exp', 'ln', 'log', 'sqrt', 'abs',
    'floor', 'ceil', 'round',
    'sr', 'cr',  # sr = square root, cr = cubic root
    'int', 'diff',  # int = integral, diff = derivative
}

# Multi-character constants
CONSTANTS = {
    'pi', 'eu', 'tau',
    'me', 'mp', 'mn', 'amu', 'ro', 'Eo', 'hb',
    'qe', 'mu', 'e', 'lp', 'tp',
    'na', 'NA',
}

# Known multi-letter variables (physics/chemistry)
MULTI_LETTER_VARS = {
    'ke', 'pe',  # kinetic energy, potential energy
}

# Mode-specific imaginary unit
IMAGINARY_UNIT = {
    Mode.MATH: 'i',
    Mode.PHYSICS: 'j',
    Mode.CHEMISTRY: 'i',
}


@dataclass
class ParseResult:
    """Result of parsing an expression."""
    tokens: List[Token] = field(default_factory=list)
    ast_string: str = ""
    variables: List[str] = field(default_factory=list)
    mode: Mode = Mode.MATH
    warnings: List[str] = field(default_factory=list)
    guessed: bool = False


class ExpressionParser:
    """Parses implicit notation into standard mathematical expressions."""

    def __init__(self, mode: Mode = Mode.MATH):
        self.mode = mode
        self.tokens: List[Token] = []
        self.pos = 0
        self.variables: List[str] = []
        self.warnings: List[str] = []
        self.guessed = False
        self.imaginary = IMAGINARY_UNIT[mode]

    def parse(self, expression: str) -> ParseResult:
        """Parse an expression string into tokens and AST."""
        self.tokens = []
        self.pos = 0
        self.variables = []
        self.warnings = []
        self.guessed = False

        expression = expression.strip()

        if not expression:
            return ParseResult(
                tokens=[], ast_string="", variables=[],
                mode=self.mode, warnings=[], guessed=False,
            )

        # Tokenize
        self.tokens = self._tokenize(expression)

        # Build AST string using a two-pass approach
        ast_string = self._build_ast(self.tokens)

        return ParseResult(
            tokens=self.tokens,
            ast_string=ast_string,
            variables=self.variables,
            mode=self.mode,
            warnings=self.warnings,
            guessed=self.guessed,
        )

    def _tokenize(self, expr: str) -> List[Token]:
        """Convert expression string into tokens."""
        tokens = []
        i = 0
        n = len(expr)

        while i < n:
            char = expr[i]

            if char == ' ':
                tokens.append(Token(TokenType.SPACE, ' ', i))
                i += 1
                continue

            if char == ',':
                tokens.append(Token(TokenType.COMMA, ',', i))
                i += 1
                continue

            if char == '=':
                tokens.append(Token(TokenType.EQUALS, '=', i))
                i += 1
                continue

            if char == '(':
                tokens.append(Token(TokenType.LPAREN, '(', i))
                i += 1
                continue

            if char == ')':
                tokens.append(Token(TokenType.RPAREN, ')', i))
                i += 1
                continue

            if char in '+-*/^':
                tokens.append(Token(TokenType.OPERATOR, char, i))
                i += 1
                continue

            if char.isdigit() or (char == '.' and i + 1 < n and expr[i + 1].isdigit()):
                num, i = self._read_number(expr, i)
                tokens.append(Token(TokenType.NUMBER, num, i))
                continue

            if char.isalpha() or char == '_':
                # Greedy matching: try to match longest known word first
                word, end_pos = self._read_word(expr, i)
                matched = False

                # Try functions (longest first)
                for func in sorted(FUNCTIONS, key=len, reverse=True):
                    if word.startswith(func):
                        tokens.append(Token(TokenType.FUNCTION, func, i))
                        i += len(func)
                        matched = True
                        break

                if matched:
                    continue

                # Try constants (longest first)
                for const in sorted(CONSTANTS, key=len, reverse=True):
                    if word.startswith(const):
                        tokens.append(Token(TokenType.VARIABLE, const, i))
                        if const not in self.variables:
                            self.variables.append(const)
                        i += len(const)
                        matched = True
                        break

                if matched:
                    continue

                # Try multi-letter variables (longest first)
                for var in sorted(MULTI_LETTER_VARS, key=len, reverse=True):
                    if word.startswith(var):
                        tokens.append(Token(TokenType.VARIABLE, var, i))
                        if var not in self.variables:
                            self.variables.append(var)
                        i += len(var)
                        matched = True
                        break

                if matched:
                    continue

                # No match: emit first letter as variable, continue with rest
                tokens.append(Token(TokenType.VARIABLE, word[0], i))
                if word[0] not in self.variables:
                    self.variables.append(word[0])
                i += 1
                continue

            if char == self.imaginary:
                tokens.append(Token(TokenType.NUMBER, char, i))
                i += 1
                continue

            i += 1

        return tokens

    def _read_number(self, expr: str, start: int) -> Tuple[str, int]:
        """Read a number from the expression.

        Note: Does NOT consume scientific notation (e.g., 2e3).
        Scientific notation is handled by _handle_number instead.
        """
        i = start
        n = len(expr)

        while i < n and expr[i].isdigit():
            i += 1

        if i < n and expr[i] == '.':
            i += 1
            while i < n and expr[i].isdigit():
                i += 1

        return expr[start:i], i

    def _read_word(self, expr: str, start: int) -> Tuple[str, int]:
        """Read a word (function name, constant, or variable) from the expression."""
        i = start
        n = len(expr)

        while i < n and (expr[i].isalpha() or expr[i] == '_'):
            i += 1

        return expr[start:i], i

    def _build_ast(self, tokens: List[Token]) -> str:
        """Build an AST string from tokens, handling implicit notation rules."""
        if not tokens:
            return ""

        # Pass 1: Group tokens into logical units
        items = self._group_tokens(tokens)

        # Pass 2: Emit items as a string with implicit multiplication
        ast_string = self._emit_items(items)

        return ast_string

    def _group_tokens(self, tokens: List[Token], in_function: bool = False) -> List[Dict[str, Any]]:
        """Group tokens into logical units.

        Args:
            tokens: The tokens to group.
            in_function: Whether we're inside a function call (affects space grouping).
        """
        items = []
        i = 0
        n = len(tokens)
        space_before = False

        while i < n:
            token = tokens[i]

            if token.type == TokenType.SPACE:
                space_before = True
                i += 1
                continue

            if token.type == TokenType.COMMA:
                items.append({'type': 'comma', 'value': ',', 'space_before': space_before})
                space_before = False
                i += 1
                continue

            if token.type == TokenType.EQUALS:
                items.append({'type': 'equals', 'value': '=', 'space_before': space_before})
                space_before = False
                i += 1
                continue

            if token.type == TokenType.OPERATOR:
                # Don't preserve space before operators (they add their own spaces)
                # Exception: * doesn't get spaces (e.g., cos2 *2 -> cos(2)*2)
                items.append({'type': 'operator', 'value': token.value, 'space_before': False})
                space_before = False
                i += 1
                continue

            if token.type == TokenType.LPAREN:
                paren_item, i = self._collect_parenthesized(tokens, i)
                paren_item['space_before'] = space_before
                items.append(paren_item)
                space_before = False
                continue

            if token.type == TokenType.RPAREN:
                items.append({'type': 'rparen', 'value': ')', 'space_before': space_before})
                space_before = False
                i += 1
                continue

            if token.type == TokenType.FUNCTION:
                func_item, i = self._handle_function(tokens, i)
                func_item['space_before'] = space_before
                items.append(func_item)
                space_before = False
                continue

            if token.type == TokenType.NUMBER:
                num_item, i = self._handle_number(tokens, i, in_function)
                num_item['space_before'] = space_before
                items.append(num_item)
                space_before = False
                continue

            if token.type == TokenType.VARIABLE:
                var_item, i = self._handle_variable(tokens, i)
                var_item['space_before'] = space_before
                items.append(var_item)
                space_before = False
                continue

            i += 1

        return items

    def _collect_parenthesized(self, tokens: List[Token], start: int) -> Tuple[Dict[str, Any], int]:
        """Collect all tokens inside matching parentheses."""
        i = start + 1
        n = len(tokens)
        depth = 1
        inner_tokens = []

        while i < n and depth > 0:
            token = tokens[i]
            if token.type == TokenType.LPAREN:
                depth += 1
            elif token.type == TokenType.RPAREN:
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            inner_tokens.append(token)
            i += 1

        inner_items = self._group_tokens(inner_tokens)

        return {
            'type': 'paren_group',
            'value': '(',
            'inner': inner_items,
        }, i

    def _handle_function(self, tokens: List[Token], start: int) -> Tuple[Dict[str, Any], int]:
        """Handle function token and its arguments with absorption rules.

        Rules:
        - sinax+b -> sin(a*x+b)  [no spaces, function absorbs all]
        - sinax + b -> sin(a*x) + b  [space before operator ends function]
        - cos2t -> cos(2*t)  [number after function is multiplied inside]
        - cos2 t+a -> cos(t+a)**2  [space after number makes it exponent]
        - sr6 -> sqrt(6)  [sr takes only a single number/variable]
        - cr8 -> 8**(1/3)  [cr takes only a single number/variable]
        """
        func_token = tokens[start]
        func_name = func_token.value
        i = start + 1
        n = len(tokens)

        # Special handling for sr (square root) and cr (cubic root)
        # These take only a single number or variable as their argument
        if func_name in ('sr', 'cr'):
            arg_tokens = []
            while i < n:
                token = tokens[i]
                if token.type in (TokenType.EQUALS, TokenType.COMMA, TokenType.OPERATOR):
                    break
                if token.type == TokenType.SPACE:
                    # Check if next non-space token is an operator or equals
                    j = i + 1
                    while j < n and tokens[j].type == TokenType.SPACE:
                        j += 1
                    if j < n and tokens[j].type in (TokenType.OPERATOR, TokenType.EQUALS, TokenType.COMMA):
                        break
                arg_tokens.append(token)
                i += 1

            arg_items = self._group_tokens(arg_tokens)
            return {
                'type': 'function',
                'name': func_name,
                'exponent': None,
                'args': arg_items,
            }, i

        # Special handling for int (integral) and diff (derivative)
        # These use comma-separated arguments: int f(x), x, a, b or diff f(x), x
        if func_name in ('int', 'diff'):
            # Collect all tokens until equals, splitting on commas
            arg_groups = [[]]
            while i < n:
                token = tokens[i]
                if token.type == TokenType.EQUALS:
                    break
                if token.type == TokenType.COMMA:
                    arg_groups.append([])
                    i += 1
                    continue
                arg_groups[-1].append(token)
                i += 1

            # Parse each argument group
            parsed_args = []
            for group in arg_groups:
                if group:
                    parsed_args.append(self._group_tokens(group))

            return {
                'type': 'calculus',
                'name': func_name,
                'args': parsed_args,
            }, i

        # Check if there's a number immediately after the function (potential exponent)
        if i < n and tokens[i].type == TokenType.NUMBER:
            # Look ahead: is there a space after this number?
            if i + 1 < n and tokens[i + 1].type == TokenType.SPACE:
                # Check what comes after the space
                j = i + 2
                while j < n and tokens[j].type == TokenType.SPACE:
                    j += 1

                # If next non-space is an operator, the number is the argument (not exponent)
                # e.g., cos2 *2 -> cos(2)*2
                if j < n and tokens[j].type == TokenType.OPERATOR:
                    # Number is the argument, not the exponent
                    pass  # Fall through to normal absorption
                else:
                    # This number is the exponent of the function
                    # e.g., cos2 t+a -> cos(t+a)**2
                    exponent = tokens[i].value
                    i += 2  # skip number and space

                    # Now collect the actual argument
                    arg_tokens = []
                    while i < n:
                        token = tokens[i]
                        if token.type == TokenType.EQUALS:
                            break
                        if token.type == TokenType.COMMA:
                            break
                        if token.type == TokenType.OPERATOR:
                            if i > 0 and tokens[i - 1].type == TokenType.SPACE:
                                break
                        arg_tokens.append(token)
                        i += 1

                    arg_items = self._group_tokens(arg_tokens)
                    return {
                        'type': 'function',
                        'name': func_name,
                        'exponent': exponent,
                        'args': arg_items,
                    }, i

        # Normal case: function absorbs until space+operator or equals
        arg_tokens = []
        while i < n:
            token = tokens[i]

            if token.type == TokenType.EQUALS:
                break

            if token.type == TokenType.COMMA:
                break

            if token.type == TokenType.OPERATOR:
                # Check if there's a space before this operator
                if i > 0 and tokens[i - 1].type == TokenType.SPACE:
                    break
                # No space before operator - still inside function
                arg_tokens.append(token)
                i += 1
                continue

            if token.type == TokenType.SPACE:
                # Check if next non-space token is an operator
                j = i + 1
                while j < n and tokens[j].type == TokenType.SPACE:
                    j += 1

                if j < n and tokens[j].type == TokenType.OPERATOR:
                    # Function ends before the space
                    break
                elif j < n and tokens[j].type == TokenType.EQUALS:
                    break
                else:
                    # Space is inside function args
                    arg_tokens.append(token)
                    i += 1
                    continue

            arg_tokens.append(token)
            i += 1

        arg_items = self._group_tokens(arg_tokens)

        return {
            'type': 'function',
            'name': func_name,
            'exponent': None,
            'args': arg_items,
        }, i

    def _handle_number(self, tokens: List[Token], start: int, in_function: bool = False) -> Tuple[Dict[str, Any], int]:
        """Handle number token, checking for power notation.

        Rules:
        - 2x3 -> 2*x**3  [number-var-number = coefficient * var^power]
        - 2e3 -> 2*10**3  [scientific notation]
        - 2t+a -> 2*(t+a)  [space grouping inside functions: no space before operator]
        """
        num_token = tokens[start]
        i = start + 1
        n = len(tokens)

        # Check for scientific notation: 2e3 -> 2*10**3, 2e-3 -> 2*10**(-3)
        if i < n and tokens[i].type == TokenType.VARIABLE and tokens[i].value == 'e':
            if i + 1 < n and tokens[i + 1].type == TokenType.NUMBER:
                exp_num = tokens[i + 1].value
                i += 2
                return {
                    'type': 'scientific',
                    'mantissa': num_token.value,
                    'exponent': exp_num,
                }, i
            if i + 2 < n and tokens[i + 1].type == TokenType.OPERATOR and tokens[i + 1].value in '+-' and tokens[i + 2].type == TokenType.NUMBER:
                exp_sign = tokens[i + 1].value
                exp_num = tokens[i + 2].value
                i += 3
                return {
                    'type': 'scientific',
                    'mantissa': num_token.value,
                    'exponent': f"({exp_sign}{exp_num})",
                }, i

        # Check if followed by a variable (with optional space before it)
        # e.g., 2t+a -> 2*(t+a), 2 t+a -> 2*(t+a)
        if i < n and tokens[i].type == TokenType.SPACE:
            # Look ahead: is the next non-space token a variable?
            j = i + 1
            while j < n and tokens[j].type == TokenType.SPACE:
                j += 1
            if j < n and tokens[j].type == TokenType.VARIABLE:
                # Number followed by space and variable: treat as coefficient * group
                var_token = tokens[j]
                i = j + 1

                # Check for space grouping: variable followed by operator with no space
                # Only apply inside functions (e.g., cos2 2t+a -> cos(2*(t+a))**2)
                if i < n and tokens[i].type == TokenType.OPERATOR:
                    group_tokens = [var_token]
                    while i < n:
                        token = tokens[i]
                        if token.type == TokenType.EQUALS:
                            break
                        if token.type == TokenType.COMMA:
                            break
                        if token.type == TokenType.SPACE:
                            k = i + 1
                            while k < n and tokens[k].type == TokenType.SPACE:
                                k += 1
                            if k < n and tokens[k].type == TokenType.OPERATOR:
                                break
                        group_tokens.append(token)
                        i += 1

                    group_items = self._group_tokens(group_tokens, in_function=True)
                    return {
                        'type': 'product',
                        'parts': [num_token.value, f"({self._emit_items(group_items, in_function=True)})"],
                    }, i

                # Check if followed by more variables/numbers (exponent)
                exponent_parts = self._collect_exponent_parts(tokens, i)
                i = exponent_parts[1]
                exponent_parts = exponent_parts[0]

                if exponent_parts:
                    exponent = '*'.join(exponent_parts)
                    return {
                        'type': 'power',
                        'base': var_token.value,
                        'coefficient': num_token.value,
                        'exponent': exponent,
                    }, i
                else:
                    return {
                        'type': 'product',
                        'parts': [num_token.value, var_token.value],
                    }, i

        if i < n and tokens[i].type == TokenType.VARIABLE:
            var_token = tokens[i]
            i += 1

            # Check for space grouping: variable followed by operator with no space
            # Only apply inside functions (e.g., cos2 2t+a -> cos(2*(t+a))**2)
            if in_function and i < n and tokens[i].type == TokenType.OPERATOR:
                # No space before operator - variable and everything after form a group
                group_tokens = [var_token]
                while i < n:
                    token = tokens[i]
                    if token.type == TokenType.EQUALS:
                        break
                    if token.type == TokenType.COMMA:
                        break
                    if token.type == TokenType.SPACE:
                        # Check if next non-space is an operator
                        j = i + 1
                        while j < n and tokens[j].type == TokenType.SPACE:
                            j += 1
                        if j < n and tokens[j].type == TokenType.OPERATOR:
                            break
                    group_tokens.append(token)
                    i += 1

                group_items = self._group_tokens(group_tokens, in_function=True)
                return {
                    'type': 'product',
                    'parts': [num_token.value, f"({self._emit_items(group_items, in_function=True)})"],
                }, i

            # Check if followed by more variables/numbers (exponent)
            exponent_parts = self._collect_exponent_parts(tokens, i)
            i = exponent_parts[1]
            exponent_parts = exponent_parts[0]

            if exponent_parts:
                exponent = '*'.join(exponent_parts)
                return {
                    'type': 'power',
                    'base': var_token.value,
                    'coefficient': num_token.value,
                    'exponent': exponent,
                }, i
            else:
                # Just implicit multiplication: 2x -> 2*x
                return {
                    'type': 'product',
                    'parts': [num_token.value, var_token.value],
                }, i

        # Regular number
        return {
            'type': 'number',
            'value': num_token.value,
        }, i

    def _collect_exponent_parts(self, tokens: List[Token], start: int) -> Tuple[List[str], int]:
        """Collect exponent parts, handling scientific notation.

        Returns:
            Tuple of (exponent_parts, new_position)
        """
        exponent_parts = []
        i = start
        n = len(tokens)

        while i < n and tokens[i].type in (TokenType.VARIABLE, TokenType.NUMBER):
            if tokens[i].type == TokenType.SPACE:
                break

            # Check for scientific notation: number + 'e' + number
            if (tokens[i].type == TokenType.NUMBER and
                i + 2 < n and
                tokens[i + 1].type == TokenType.VARIABLE and
                tokens[i + 1].value == 'e' and
                tokens[i + 2].type == TokenType.NUMBER):
                mantissa = tokens[i].value
                exp = tokens[i + 2].value
                exponent_parts.append(f"{mantissa}*10**{exp}")
                i += 3
                continue

            exponent_parts.append(tokens[i].value)
            i += 1

        return exponent_parts, i

    def _handle_variable(self, tokens: List[Token], start: int) -> Tuple[Dict[str, Any], int]:
        """Handle variable token, checking for power notation.

        Rules:
        - axn -> a*x**n  [var-var-var = first * second^third]
        - axbc -> a*x**(b*c)  [multiple vars after = power of product]
        - x3 -> x**3  [var-number = var^power]
        """
        var_token = tokens[start]
        i = start + 1
        n = len(tokens)

        # Check if followed by a number (power notation: x3 -> x**3)
        if i < n and tokens[i].type == TokenType.NUMBER:
            # Check if there's a space after the number
            if i + 1 < n and tokens[i + 1].type == TokenType.SPACE:
                # Space after number - just implicit multiplication
                return {
                    'type': 'product',
                    'parts': [var_token.value, tokens[i].value],
                }, i + 1

            # No space - this is power notation: x3 -> x**3
            exponent = tokens[i].value
            i += 1
            return {
                'type': 'power',
                'base': var_token.value,
                'coefficient': None,
                'exponent': exponent,
            }, i

        # Check if followed by more variables (power notation)
        if i < n and tokens[i].type == TokenType.VARIABLE:
            next_var = tokens[i].value

            # Check if there's a space after the next variable
            if i + 1 < n and tokens[i + 1].type == TokenType.SPACE:
                # Space after next var - just implicit multiplication
                return {
                    'type': 'product',
                    'parts': [var_token.value, next_var],
                }, i + 1

            # Check for more variables/numbers after
            exponent_parts = [next_var]
            i += 1
            more_parts, i = self._collect_exponent_parts(tokens, i)
            exponent_parts.extend(more_parts)

            if len(exponent_parts) > 1:
                # axn -> a*x**n (first is coefficient, second is base, rest is exponent)
                coefficient = var_token.value
                base = exponent_parts[0]
                exponent = '*'.join(exponent_parts[1:])
                return {
                    'type': 'power',
                    'base': base,
                    'coefficient': coefficient,
                    'exponent': exponent,
                }, i
            else:
                # Just two variables: ax -> a*x
                return {
                    'type': 'product',
                    'parts': [var_token.value, next_var],
                }, i

        # Single variable
        return {
            'type': 'variable',
            'value': var_token.value,
        }, i

    def _emit_items(self, items: List[Dict[str, Any]], in_function: bool = False) -> str:
        """Convert items to a sympy-compatible string.

        Args:
            items: The items to emit.
            in_function: Whether we're inside a function call (no spaces around operators).
        """
        parts = []

        for i, item in enumerate(items):
            part = self._emit_item(item, in_function)

            # Check if we need implicit multiplication between this and previous
            if i > 0:
                # Don't add space if previous item is an operator (already has spaces)
                if item.get('space_before', False) and items[i - 1]['type'] != 'operator':
                    parts.append(' ')
                    if self._needs_implicit_mult(items[i - 1], item):
                        parts.append('*')
                elif self._needs_implicit_mult(items[i - 1], item):
                    parts.append('*')

            parts.append(part)

        return ''.join(parts)

    def _emit_item(self, item: Dict[str, Any], in_function: bool = False) -> str:
        """Emit a single item as a string.

        Args:
            item: The item to emit.
            in_function: Whether we're inside a function call (no spaces around operators).
        """
        itype = item['type']

        if itype == 'number':
            return item['value']

        if itype == 'variable':
            return item['value']

        if itype == 'operator':
            if in_function:
                return item['value']
            elif item['value'] == '*':
                return '*'
            else:
                return f" {item['value']} "

        if itype == 'comma':
            return ', '

        if itype == 'equals':
            return ' = '

        if itype == 'paren_group':
            inner = self._emit_items(item['inner'], in_function)
            return f"({inner})"

        if itype == 'function':
            args_str = self._emit_items(item['args'], in_function=True)
            func_name = item['name']
            # Handle sr (square root) and cr (cubic root) specially
            if func_name == 'sr':
                func_str = f"sqrt({args_str})"
            elif func_name == 'cr':
                func_str = f"({args_str})**(1/3)"
            else:
                func_str = f"{func_name}({args_str})"
            if item.get('exponent'):
                func_str = f"{func_str}**{item['exponent']}"
            return func_str

        if itype == 'calculus':
            func_name = item['name']
            args = item['args']
            if func_name == 'int':
                # int f(x), x, a, b -> Integral(f(x), (x, a, b))
                if len(args) >= 4:
                    f_str = self._emit_items(args[0], in_function=True)
                    var_str = self._emit_items(args[1], in_function=True)
                    a_str = self._emit_items(args[2], in_function=True)
                    b_str = self._emit_items(args[3], in_function=True)
                    return f"Integral({f_str}, ({var_str}, {a_str}, {b_str}))"
                elif len(args) >= 2:
                    f_str = self._emit_items(args[0], in_function=True)
                    var_str = self._emit_items(args[1], in_function=True)
                    return f"Integral({f_str}, {var_str})"
            elif func_name == 'diff':
                # diff f(x), x -> Derivative(f(x), x)
                if len(args) >= 2:
                    f_str = self._emit_items(args[0], in_function=True)
                    var_str = self._emit_items(args[1], in_function=True)
                    return f"Derivative({f_str}, {var_str})"
            return f"{func_name}({', '.join(self._emit_items(a, in_function=True) for a in args)})"

        if itype == 'power':
            base = item['base']
            exponent = item['exponent']
            # Wrap exponent in parens if it's a product
            if '*' in exponent:
                exponent = f"({exponent})"
            if item.get('coefficient'):
                return f"{item['coefficient']}*{base}**{exponent}"
            else:
                return f"{base}**{exponent}"

        if itype == 'product':
            return '*'.join(item['parts'])

        if itype == 'scientific':
            return f"{item['mantissa']}*10**{item['exponent']}"

        if itype == 'rparen':
            return ')'

        return ''

    def _needs_implicit_mult(self, prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
        """Check if implicit multiplication is needed between two items."""
        # Don't add * after operators, commas, equals, or opening parens
        if prev['type'] in ('operator', 'comma', 'equals', 'paren_group'):
            return False

        # Don't add * before operators, commas, equals, or closing parens
        if curr['type'] in ('operator', 'comma', 'equals', 'rparen'):
            return False

        # Add * between: number/variable/product/power/scientific/paren_group/function
        # and number/variable/product/power/scientific/paren_group/function
        if prev['type'] in ('number', 'variable', 'product', 'power', 'scientific', 'paren_group', 'function'):
            if curr['type'] in ('number', 'variable', 'product', 'power', 'scientific', 'paren_group', 'function'):
                return True

        return False


def parse_expression(expression: str, mode: Mode = Mode.MATH) -> ParseResult:
    """Convenience function to parse an expression."""
    parser = ExpressionParser(mode=mode)
    return parser.parse(expression)
