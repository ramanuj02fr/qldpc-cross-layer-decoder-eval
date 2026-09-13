"""Phase 2B -- Classic decoders (BP / BP-OSD / BP-LSD) from the `ldpc` package
(Roffe et al.), plus helpers to extract the Z-type parity-check matrix and
logical-X operators from a qldpc code object.
"""

from __future__ import annotations

import numpy as np

from qldpc.objects import Pauli


def get_z_check_matrix(code_obj) -> np.ndarray:
    """Return the Z-type parity check matrix (used to decode X-type errors).

    Tries a few plausible attribute names across qLDPC versions. If none match,
    run `print(dir(CODE))` and look for something like 'matrix_z' / 'code_z' /
    'Hz', then set it manually.
    """
    for name in ("matrix_z", "code_z"):
        if hasattr(code_obj, name):
            attr = getattr(code_obj, name)
            mat = attr.matrix if hasattr(attr, "matrix") else attr
            return np.array(mat).astype(np.uint8)
    raise AttributeError(
        "Could not auto-find the Z-type check matrix. Run `print(dir(CODE))` and look for "
        "something like 'matrix_z' / 'code_z' / 'Hz', then set Hz manually."
    )


def get_logical_x_ops(code_obj) -> np.ndarray:
    return np.array(code_obj.get_logical_ops(Pauli.X)).astype(np.uint8)


def make_decoders(Hz: np.ndarray, p: float) -> dict:
    """Build fresh BP / BP-OSD / BP-LSD decoder instances for physical error
    rate `p`. Decoders must be rebuilt per physical-error-rate point -- the
    `error_rate` parameter is baked in at construction time.
    """
    from ldpc.bp_decoder import BpDecoder
    from ldpc.bposd_decoder import BpOsdDecoder
    from ldpc.bplsd_decoder import BpLsdDecoder

    return {
        "BP": BpDecoder(
            Hz, error_rate=p, bp_method="ms", max_iter=30, schedule="serial",
        ),
        "BP-OSD": BpOsdDecoder(
            Hz, error_rate=p, bp_method="ms", max_iter=30, schedule="serial",
            osd_method="osd0", osd_order=0,
        ),
        "BP-LSD": BpLsdDecoder(
            Hz, error_rate=p, bp_method="ms", max_iter=30, schedule="serial",
            lsd_method="lsd_cs", lsd_order=0,
        ),
    }
