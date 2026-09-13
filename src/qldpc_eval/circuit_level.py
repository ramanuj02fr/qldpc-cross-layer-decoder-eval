"""Phase 3 -- Circuit-level noise memory experiment (the MAIN result).

Everything in Phase 2 used a code-capacity (i.i.d.) noise model, which is a
good first pass but not the same as realistic circuit-level noise (gate
errors, measurement errors, idling).

Fixes carried over from the original notebook (this was the weakest part of
earlier versions):

1. Adaptive shot counts (`shots_for_p`) -- a flat shot count for every physical
   error rate produces several 0-error points at low p, which are statistically
   meaningless (they only bound LER from above, they don't estimate it).
2. Wilson score confidence intervals (`wilson_ci`), not a `clip(lower=1e-6)`
   hack. A 0-error point now correctly shows as "LER upper-bounded by X at 95%
   confidence," rather than being silently drawn as if it were a real point
   estimate.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import sinter

from qldpc import circuits, decoders as qdecoders


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (phat, lower, upper).

    Far more reliable than a normal-approximation stderr when k is 0 or close
    to n, which happens frequently in QEC logical-error-rate sampling at low
    physical error rates.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)


def shots_for_p(p: float, low_thresh: float = 0.002, mid_thresh: float = 0.004,
                 low_shots: int = 8000, mid_shots: int = 5000, high_shots: int = 3000) -> int:
    """Adaptive shot allocation: rarer events (low p) need more shots to be
    sampled at all.
    """
    if p < low_thresh:
        return low_shots
    elif p < mid_thresh:
        return mid_shots
    else:
        return high_shots


def run_circuit_level_experiment_adaptive(code_obj, basis, error_rate_shots: dict,
                                           max_errors: int = 100, num_rounds: int = 3,
                                           decoder_obj=None, **decoding_kwargs):
    """error_rate_shots: dict {physical_error_rate: num_shots_for_that_rate}.
    Runs one sinter.collect call per physical error rate so each can have its
    own shot budget.

    `num_rounds` is fixed at 3 by default -- originally a workaround for a
    GAP-based distance computation that hung; now that a distance bound is
    available (see code_setup.compute_distance_bound), num_rounds could be
    tied to it (e.g. num_rounds ~ d), but this has not been done yet. See
    Threats to Validity in README.md.
    """
    circuit = circuits.get_memory_experiment(code_obj, basis=basis, num_rounds=num_rounds)
    all_stats = []
    for p, n_shots in error_rate_shots.items():
        noise_model = circuits.DepolarizingNoiseModel(p, include_idling_error=True)
        noisy_circuit = noise_model.noisy_circuit(circuit)
        task = sinter.Task(circuit=noisy_circuit, json_metadata={"prob": p})
        decoder = decoder_obj if decoder_obj is not None else qdecoders.SinterDecoder(**decoding_kwargs)
        stats = sinter.collect(
            tasks=[task],
            decoders=["custom"],
            custom_decoders={"custom": decoder},
            num_workers=max(1, (os.cpu_count() or 2) - 1),
            max_shots=n_shots,
            max_errors=max_errors,
        )
        all_stats.extend(stats)
    return all_stats


def stats_to_df(stats_list, decoder_label: str) -> pd.DataFrame:
    rows = []
    for s in stats_list:
        shots, errors = s.shots, s.errors
        p = s.json_metadata["prob"]
        phat, ci_lo, ci_hi = wilson_ci(errors, shots)
        rows.append({
            "decoder": decoder_label, "prob": p, "shots": shots, "errors": errors,
            "logical_error_rate": phat, "ci_lower": ci_lo, "ci_upper": ci_hi,
            "is_upper_bound_only": errors == 0,
        })
    return pd.DataFrame(rows)
