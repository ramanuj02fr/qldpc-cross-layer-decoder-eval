"""Phase 2A -- Code-capacity logical error rate (LER) curves.

Uses qLDPC's official high-level helper `code.get_logical_error_rate_func(...)`,
which handles sampling + decoding internally for BP-OSD / BP-LSD.

Fix carried over from the original notebook: `error_rates` is built *before*
calling `get_logical_error_rate_func`, and `max_error_rate` is derived from
`max(error_rates)` instead of a hardcoded constant. A hardcoded value smaller
than the top of `error_rates` raises
`ValueError: This ErrorRateFunc does not cover physical error rates greater
than 0.5.` when regenerating figures later with a wider sweep.
"""

from __future__ import annotations

import numpy as np


def code_capacity_ler_curve(code_obj, error_rates=None, num_samples: int = 200, **decoding_kwargs):
    """Returns (p_phys, p_log, stderr) arrays for one decoder configuration.

    `decoding_kwargs` is passed straight through to
    `code_obj.get_logical_error_rate_func`, e.g. `with_BP_OSD=True` or
    `with_BP_LSD=True`.
    """
    if error_rates is None:
        error_rates = list(np.logspace(-2, -0.3, 20))
    get_ler = code_obj.get_logical_error_rate_func(num_samples, max(error_rates), **decoding_kwargs)
    logical_rates, stderrs = get_ler(error_rates)
    return np.array(error_rates), np.array(logical_rates), np.array(stderrs)
