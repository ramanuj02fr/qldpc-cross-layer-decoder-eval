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


def bootstrap_percentile_ci(values, percentile: float, n_bootstrap: int = 1000,
                             ci: float = 95.0, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a sample percentile (e.g. p99 runtime).

    A single p99 estimate from a few thousand shots is itself a noisy
    statistic -- exactly how noisy depends on the shape of the tail, which
    varies by decoder. This resamples `values` with replacement `n_bootstrap`
    times, recomputes the target percentile each time, and returns the
    `ci`% interval of that bootstrap distribution.

    Vectorized: builds an (n_bootstrap, n) index array in one shot rather than
    looping in Python, which matters since this runs once per (decoder,
    physical_error_rate, percentile) combination in `summarise_with_ci`.
    """
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_stats = np.percentile(values[idx], percentile, axis=1)
    alpha = (100.0 - ci) / 2.0
    return (float(np.percentile(boot_stats, alpha)), float(np.percentile(boot_stats, 100.0 - alpha)))


def summarise_with_ci(bench_df: pd.DataFrame, n_bootstrap: int = 1000,
                       bootstrap_ci: float = 95.0, bootstrap_seed: int = 0) -> pd.DataFrame:
    """Summarise: logical error rate (with CI across seeds) + runtime
    percentiles (each with a bootstrap CI), broken out per (decoder,
    physical error rate).

    The runtime percentiles (p50/p95/p99/p99.9) are themselves point
    estimates from a finite number of shots -- this is especially relevant
    for p99/p99.9, since a paper's central real-time argument typically rests
    on tail latency, not mean runtime. `{pct}_ci_lower` / `{pct}_ci_upper`
    columns report the bootstrap CI for each percentile so a claim like
    "BP-LSD's p99 is higher than BP-OSD's" can be checked for whether the
    CIs actually separate, not just compared as two point estimates.
    """
    rows = []
    for (name, p), g in bench_df.groupby(["decoder", "phys_error_rate"]):
        per_seed_ler = g.groupby("seed")["logical_fail"].mean()
        rt = g["runtime_s"].values * 1e6  # convert to microseconds
        n_seeds = len(per_seed_ler)
        row = {
            "decoder": name,
            "phys_error_rate": p,
            "logical_error_rate": per_seed_ler.mean(),
            "ler_stderr": per_seed_ler.std(ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else float("nan"),
            "n_seeds": n_seeds,
            "n_shots": len(rt),
            "mean_us": rt.mean(),
            "max_us": rt.max(),
        }
        for pct, label in [(50, "p50"), (95, "p95"), (99, "p99"), (99.9, "p99.9")]:
            row[f"{label}_us"] = np.percentile(rt, pct)
            ci_lo, ci_hi = bootstrap_percentile_ci(
                rt, pct, n_bootstrap=n_bootstrap, ci=bootstrap_ci, seed=bootstrap_seed,
            )
            row[f"{label}_ci_lower_us"] = ci_lo
            row[f"{label}_ci_upper_us"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)
