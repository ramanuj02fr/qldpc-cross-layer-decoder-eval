"""Phase 2B -- Per-shot runtime benchmark, swept across physical error rates.

`code.get_logical_error_rate_func` (Phase 2A) hides per-shot timing. For the
real-time question we need the runtime of every individual decode call, at
several physical error rates (BP-OSD/BP-LSD runtime is not error-rate
independent: more errors typically means more OSD/LSD post-processing work).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def sample_iid_error(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) < p).astype(np.uint8)


def run_benchmark(Hz, Lx, decoder_objs: dict, num_shots: int, p: float, seed: int = 0) -> pd.DataFrame:
    """Sample i.i.d. bit-flip errors and decode each shot individually with every
    decoder in `decoder_objs`, timing each call with `time.perf_counter()`.
    """
    rng = np.random.default_rng(seed)
    n = Hz.shape[1]
    records = []
    for shot in range(num_shots):
        error = sample_iid_error(n, p, rng)
        syndrome = (Hz @ error) % 2
        for name, dec in decoder_objs.items():
            t0 = time.perf_counter()
            guess = dec.decode(syndrome)
            runtime_s = time.perf_counter() - t0
            residual = (guess.astype(np.uint8) + error) % 2
            logical_fail = bool(np.any((Lx @ residual) % 2))
            records.append({
                "decoder": name,
                "shot": shot,
                "runtime_s": runtime_s,
                "logical_fail": logical_fail,
            })
    return pd.DataFrame.from_records(records)


def summarise_with_ci(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise: logical error rate (with CI across seeds) + runtime
    percentiles, broken out per (decoder, physical error rate).
    """
    rows = []
    for (name, p), g in bench_df.groupby(["decoder", "phys_error_rate"]):
        per_seed_ler = g.groupby("seed")["logical_fail"].mean()
        rt = g["runtime_s"].values * 1e6  # convert to microseconds
        n_seeds = len(per_seed_ler)
        rows.append({
            "decoder": name,
            "phys_error_rate": p,
            "logical_error_rate": per_seed_ler.mean(),
            "ler_stderr": per_seed_ler.std(ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else float("nan"),
            "n_seeds": n_seeds,
            "mean_us": rt.mean(),
            "p50_us": np.percentile(rt, 50),
            "p95_us": np.percentile(rt, 95),
            "p99_us": np.percentile(rt, 99),
            "p99.9_us": np.percentile(rt, 99.9),
            "max_us": rt.max(),
        })
    return pd.DataFrame(rows)
