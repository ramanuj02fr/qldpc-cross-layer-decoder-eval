"""Phase 4 -- Workload / Queue simulator (FIFO baseline + DART-Q-style EDF).

Pure Python, no external QEC library needed. Each decode "job" arrives over
time (Poisson arrivals), waits in a queue, and is served for a duration drawn
from a measured runtime distribution. Each job has a deadline; if it isn't
finished by then, it's a "deadline miss".

Two scheduling policies:
- FIFO -- first-in-first-out, the simple baseline.
- EDF (earliest-deadline-first) with admission control -- serve whichever
  waiting job is closest to its deadline, drop jobs immediately once their
  deadline has already passed. A simplified but faithful single-server analog
  of DART-Q's deadline-driven scheduling idea (see README.md Threats to
  Validity for what this simplifies away).
"""

from __future__ import annotations

import heapq

import numpy as np


def simulate_queue(service_times_s, arrival_rate_hz: float, deadline_s: float,
                    sim_duration_s: float, seed: int = 0, warmup_s: float = 0.0) -> dict:
    """Single-server FIFO queue, deadline-based miss tracking.

    `warmup_s`: jobs that *arrive* before this time still pass through the
    server (so the queue is in a realistic, non-empty state once the
    measurement window starts), but are excluded from the reported metrics.
    This matters most for the "burst" (mild-overload) scenario, where the
    queue starting empty at t=0 can bias miss-rate/response-time numbers low
    for a while. Pass e.g. `warmup_s=0.2 * sim_duration_s` and check that
    results don't meaningfully change vs `warmup_s=0`.
    """
    rng = np.random.default_rng(seed)
    arrivals = []
    t = 0.0
    while t < sim_duration_s:
        t += rng.exponential(1.0 / arrival_rate_hz)
        if t < sim_duration_s:
            arrivals.append(t)
    arrivals = np.array(arrivals)
    n_jobs = len(arrivals)
    service_times = rng.choice(service_times_s, size=n_jobs, replace=True)

    start_times = np.zeros(n_jobs)
    finish_times = np.zeros(n_jobs)
    queue_len_at_arrival = np.zeros(n_jobs)
    server_free_at = 0.0
    for i in range(n_jobs):
        start = max(arrivals[i], server_free_at)
        finish = start + service_times[i]
        start_times[i] = start
        finish_times[i] = finish
        server_free_at = finish
        queue_len_at_arrival[i] = np.sum((arrivals[:i] <= arrivals[i]) & (finish_times[:i] > arrivals[i]))

    response_times = finish_times - arrivals
    deadlines = arrivals + deadline_s
    missed = finish_times > deadlines

    keep = arrivals >= warmup_s
    n_kept = int(keep.sum())
    return {
        "n_jobs": n_kept,
        "n_jobs_total_incl_warmup": n_jobs,
        "deadline_miss_rate": missed[keep].mean() if n_kept else float("nan"),
        "p50_response_s": np.percentile(response_times[keep], 50) if n_kept else float("nan"),
        "p99_response_s": np.percentile(response_times[keep], 99) if n_kept else float("nan"),
        "max_queue_len": queue_len_at_arrival[keep].max() if n_kept else 0,
        "mean_queue_len": queue_len_at_arrival[keep].mean() if n_kept else 0.0,
        "goodput_hz": (~missed[keep]).sum() / (sim_duration_s - warmup_s) if n_kept else 0.0,
    }


def simulate_queue_edf(service_times_s, arrival_rate_hz: float, deadline_s: float,
                        sim_duration_s: float, seed: int = 0,
                        drop_if_already_late: bool = True, warmup_s: float = 0.0) -> dict:
    """Single-server EDF (Earliest-Deadline-First) queue with admission control.

    See `simulate_queue`'s docstring for what `warmup_s` does and why.
    """
    rng = np.random.default_rng(seed)
    arrivals = []
    t = 0.0
    while t < sim_duration_s:
        t += rng.exponential(1.0 / arrival_rate_hz)
        if t < sim_duration_s:
            arrivals.append(t)
    arrivals = np.array(arrivals)
    n_jobs = len(arrivals)
    if n_jobs == 0:
        return {"n_jobs": 0, "n_jobs_total_incl_warmup": 0, "n_dropped_early": 0,
                "deadline_miss_rate": float("nan"),
                "p50_response_s": float("nan"), "p99_response_s": float("nan"),
                "max_queue_len": 0, "mean_queue_len": 0.0, "goodput_hz": 0.0}

    service_times = rng.choice(service_times_s, size=n_jobs, replace=True)
    deadlines = arrivals + deadline_s

    waiting = []  # min-heap of (deadline, job_index)
    finish_times = np.full(n_jobs, np.nan)
    dropped = np.zeros(n_jobs, dtype=bool)
    served = np.zeros(n_jobs, dtype=bool)
    server_free_at = 0.0
    next_arrival_idx = 0
    queue_len_samples = []
    queue_len_samples_arrival_time = []

    while next_arrival_idx < n_jobs or waiting:
        while next_arrival_idx < n_jobs and arrivals[next_arrival_idx] <= server_free_at:
            heapq.heappush(waiting, (deadlines[next_arrival_idx], next_arrival_idx))
            next_arrival_idx += 1

        if not waiting:
            if next_arrival_idx < n_jobs:
                server_free_at = arrivals[next_arrival_idx]
            continue

        queue_len_samples.append(len(waiting))
        queue_len_samples_arrival_time.append(server_free_at)
        deadline_i, idx = heapq.heappop(waiting)

        if drop_if_already_late and server_free_at > deadline_i:
            dropped[idx] = True
            finish_times[idx] = server_free_at
            continue

        start = server_free_at
        finish = start + service_times[idx]
        finish_times[idx] = finish
        served[idx] = True
        server_free_at = finish

    response_times = finish_times - arrivals
    missed = dropped | (finish_times > deadlines)

    keep = arrivals >= warmup_s
    n_kept = int(keep.sum())
    qlen_keep = np.array(queue_len_samples_arrival_time) >= warmup_s
    qlen_kept = np.array(queue_len_samples)[qlen_keep] if len(queue_len_samples) else np.array([])
    return {
        "n_jobs": n_kept,
        "n_jobs_total_incl_warmup": n_jobs,
        "n_dropped_early": int(dropped[keep].sum()) if n_kept else 0,
        "deadline_miss_rate": missed[keep].mean() if n_kept else float("nan"),
        "p50_response_s": np.nanpercentile(response_times[keep], 50) if n_kept else float("nan"),
        "p99_response_s": np.nanpercentile(response_times[keep], 99) if n_kept else float("nan"),
        "max_queue_len": qlen_kept.max() if len(qlen_kept) else 0,
        "mean_queue_len": float(qlen_kept.mean()) if len(qlen_kept) else 0.0,
        "goodput_hz": served[keep].sum() / (sim_duration_s - warmup_s) if n_kept else 0.0,
    }


def simulate_queue_multi_seed(sim_fn, service_times_s, arrival_rate_hz: float, deadline_s: float,
                               sim_duration_s: float, seeds=range(10), warmup_s: float = 0.0,
                               **kwargs) -> dict:
    """Run `sim_fn` (either `simulate_queue` or `simulate_queue_edf`) across
    multiple independent arrival-sequence seeds and aggregate.

    A single seed's `deadline_miss_rate` etc. is one draw of a random
    arrival process -- a headline number like "60% relative reduction" from
    one seed could just be luck of the draw. This runs `len(seeds)`
    independent simulations and reports mean +/- standard error (normal
    approx across seeds) for each scalar metric, plus the per-seed values
    for inspection.
    """
    per_seed = [sim_fn(service_times_s, arrival_rate_hz, deadline_s, sim_duration_s,
                        seed=s, warmup_s=warmup_s, **kwargs) for s in seeds]
    n = len(per_seed)
    scalar_keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float, np.floating, np.integer))]
    agg = {"n_seeds": n, "per_seed": per_seed}
    for key in scalar_keys:
        vals = np.array([d[key] for d in per_seed], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_stderr"] = float("nan")
            continue
        agg[f"{key}_mean"] = vals.mean()
        agg[f"{key}_stderr"] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return agg
