"""Phase 1 -- Build the Bivariate Bicycle (BB) qLDPC code and its distance bound.

We use the [[144, 12, d]] BB code (cyclic group orders {x:12, y:6}), the same code
family used in recent real-time qLDPC decoding papers (GARI, Beam Search decoder,
DART-Q). A 72-qubit `code_small` is also provided for fast local debugging.
"""

from __future__ import annotations

from sympy.abc import x, y

from qldpc import codes
from qldpc.objects import Pauli

DISTANCE_TRIALS_DEFAULT = 1000


def build_codes() -> tuple:
    """Return (code_small, code_144)."""
    code_small = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    code_144 = codes.BBCode({x: 12, y: 6}, x**3 + y**2 + y, x**2 + x + y**3)
    return code_small, code_144


def compute_distance_bound(code, num_trials: int = DISTANCE_TRIALS_DEFAULT) -> dict:
    """Randomized decoder-based distance UPPER BOUND (arXiv:2308.07915).

    This deliberately avoids qldpc's GAP-based `get_distance()`, which hangs in
    headless environments (Colab, CI, plain servers) because GAP is not installed
    and qldpc's GAP interface blocks waiting for interactive copy/paste input.

    IMPORTANT: this is a randomized UPPER BOUND, not a GAP-certified exact
    distance. Report it as such (e.g. "d <= 12, decoder-based bound"), not as an
    exact computed value, in any paper/report that cites it.
    """
    dx_bound = code.get_distance_bound_with_decoder(Pauli.X, num_trials=num_trials)
    dz_bound = code.get_distance_bound_with_decoder(Pauli.Z, num_trials=num_trials)
    d = min(dx_bound, dz_bound)
    return {
        "num_trials": num_trials,
        "dx_bound": dx_bound,
        "dz_bound": dz_bound,
        "d": d,
        "n": len(code),
        "k": code.dimension,
    }


def print_distance_report(result: dict) -> None:
    print(f"Distance upper bound ({result['num_trials']} trials, decoder-based, GAP-free):")
    print(f"  X-distance bound: {result['dx_bound']}")
    print(f"  Z-distance bound: {result['dz_bound']}")
    print(
        f"  d (min of X,Z bounds) = {result['d']}  ->  code is "
        f"[[{result['n']},{result['k']},{result['d']}]]"
    )
    print("  NOTE: this is a randomized UPPER BOUND, not a GAP-certified exact distance.")
