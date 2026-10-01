"""CLI interface for calculate-it."""

import argparse
import sys

from .math_engine import MathEngine
from .physics_engine import PhysicsEngine
from .chemistry_engine import ChemistryEngine
from .ai_engine import AIEngine, should_use_ai


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for calculate-it."""
    parser = argparse.ArgumentParser(
        prog="c",
        description="A terminal calculator with math, physics, and chemistry modes.",
        epilog='Example: c "m=1.5, v=8.94, ke" (AI auto-detects math/physics/chemistry mode)',
    )

    parser.add_argument(
        "-p", "--physics",
        action="store_true",
        help="Physics mode",
    )
    parser.add_argument(
        "-c", "--chemistry",
        action="store_true",
        help="Chemistry mode",
    )

    # Chemistry sub-modes
    parser.add_argument(
        "-g", "--general",
        action="store_true",
        help="General chemistry formula solver (use with -c)",
    )
    parser.add_argument(
        "-o", "--organic",
        action="store_true",
        help="Organic chemistry formula solver (use with -c)",
    )
    parser.add_argument(
        "-r", "--redox",
        action="store_true",
        help="Redox and equilibrium reactions (use with -c)",
    )

    parser.add_argument(
        "expression",
        nargs="?",
        default=None,
        help='The expression to evaluate, e.g. "2x+3=7"',
    )

    parser.add_argument(
        "-a", "--ai",
        action="store_true",
        help="Force AI mode (AI also auto-triggers on keywords like solve/find/balance, and as a fallback when offline engines fail)",
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s 1.0.0",
    )

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.expression and not (args.physics or args.chemistry or args.ai):
        parser.print_help()
        sys.exit(0)

    expression = args.expression or ""

    # Determine mode: explicit flags win; otherwise let the AI classify
    # the problem so the user can just write: c "expression"
    explicit_mode = args.physics or args.chemistry
    if explicit_mode:
        mode = "physics" if args.physics else "chemistry"
    elif expression.strip():
        mode = AIEngine().classify(expression)
    else:
        mode = "math"

    try:
        # AI mode (forced via -a, or auto-triggered by keywords like
        # "solve", "find", "balance", ...)
        if args.ai or should_use_ai(expression):
            engine = AIEngine()
            result = engine.solve(expression, mode=mode)
            print(result)
            return

        # Offline engines
        if mode == "chemistry":
            engine = ChemistryEngine()
            if args.general:
                result = engine.solve_general(expression)
            elif args.organic:
                result = engine.solve_organic(expression)
            elif args.redox:
                result = engine.solve_redox(expression)
            else:
                result = engine.solve(expression)
        elif mode == "physics":
            engine = PhysicsEngine()
            result = engine.solve(expression)
        else:
            engine = MathEngine()
            result = engine.solve(expression)

        # Offline engines couldn't handle it -> let the AI answer
        if isinstance(result, str) and (
            "Coming soon" in result or "Error:" in result or "Could not" in result
        ):
            from .ai_engine import AIEngine as _AI
            print(_AI().solve(expression, mode=mode))
            return

        print(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
