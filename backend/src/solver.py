"""
Equation parser and solver using SymPy.

Handles two modes:
  1. Expressions (no '='): evaluate and auto-append '= result'
  2. Equations  (with '='): solve for variable x

Includes strict input validation to prevent code injection and
catch invalid syntax before it reaches the parser.
"""

import re
from sympy import symbols, Eq, solve, oo, zoo, nan, S
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

x = symbols('x')

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Strict whitelist: digits, variable x, operators, parens, equals, decimal, spaces, Unicode math
ALLOWED_PATTERN = re.compile(r'^[0-9x+\-*/^()=.\s\u00d7\u00f7]+$')


def solve_equation(equation_str):
    """
    Parse, validate, and solve an equation or expression string.

    Args:
        equation_str: e.g. "2x+4=10" or "(2*5)+3"

    Returns:
        dict with: success (bool), result (str), type (str), error (str)
    """
    equation_str = equation_str.strip()

    if not equation_str:
        return {'success': False, 'error': 'Empty equation'}

    # Replace display Unicode with math operators
    math_str = equation_str.replace('\u00d7', '*').replace('\u00f7', '/')

    # Input validation
    if not ALLOWED_PATTERN.match(math_str):
        return {'success': False, 'error': 'Invalid characters in equation'}

    # Validate balanced parentheses
    depth = 0
    for ch in math_str:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if depth < 0:
            return {'success': False, 'error': 'Unbalanced parentheses'}
    if depth != 0:
        return {'success': False, 'error': 'Unbalanced parentheses'}

    try:
        if '=' in math_str:
            return _solve_equation_mode(math_str)
        else:
            return _evaluate_expression_mode(math_str)

    except (SyntaxError, TypeError, ValueError):
        return {'success': False, 'error': 'Invalid syntax: could not parse the equation'}
    except ZeroDivisionError:
        return {'success': False, 'error': 'Division by zero'}
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 120:
            error_msg = error_msg[:120] + '...'
        return {'success': False, 'error': f'Could not solve: {error_msg}'}


def _solve_equation_mode(math_str):
    """Handle equations containing '=' — solve for x."""
    parts = math_str.split('=')
    if len(parts) != 2:
        return {'success': False, 'error': 'Invalid equation: multiple equals signs'}

    lhs_str = parts[0].strip()
    rhs_str = parts[1].strip()

    if not lhs_str or not rhs_str:
        return {'success': False, 'error': 'Incomplete equation (empty side)'}

    lhs = parse_expr(lhs_str, transformations=TRANSFORMATIONS, local_dict={'x': x})
    rhs = parse_expr(rhs_str, transformations=TRANSFORMATIONS, local_dict={'x': x})

    eq = Eq(lhs, rhs)
    solutions = solve(eq, x)

    if not solutions:
        # Check for identity (e.g., x = x)
        simplified = lhs - rhs
        if simplified.simplify() == 0:
            return {'success': True, 'result': 'Identity (true for all x)', 'type': 'equation'}
        return {'success': False, 'error': 'No solution found'}

    result_parts = [f"x = {s}" for s in solutions]
    return {'success': True, 'result': ', '.join(result_parts), 'type': 'equation'}


def _evaluate_expression_mode(math_str):
    """Handle pure expressions (no '=') — evaluate and auto-append '='."""
    expr = parse_expr(math_str, transformations=TRANSFORMATIONS, local_dict={'x': x})

    if expr.free_symbols:
        return {
            'success': False,
            'error': 'Expression contains variable x but no equation to solve (missing =)'
        }

    if not expr.is_number:
        return {'success': False, 'error': 'Could not evaluate expression to a number'}

    # Check for undefined values
    if expr in (oo, -oo, zoo, nan) or expr == S.ComplexInfinity:
        return {'success': False, 'error': 'Undefined result (division by zero or infinity)'}

    # Format: keep exact integer if possible, else evaluate numerically
    if expr.is_integer:
        formatted = str(int(expr))
    else:
        float_val = float(expr.evalf())
        if float_val == int(float_val):
            formatted = str(int(float_val))
        else:
            formatted = str(round(float_val, 10))

    return {'success': True, 'result': f'= {formatted}', 'type': 'expression'}
