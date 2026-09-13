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
                    sim_duration_s: float, seed: int = 0) -> dict:
    """Single-server FIFO queue, deadline-based miss tracking."""
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
    return {
        "n_jobs": n_jobs,
        "deadline_miss_rate": missed.mean(),
        "p50_response_s": np.percentile(response_times, 50),
        "p99_response_s": np.percentile(response_times, 99),
        "max_queue_len": queue_len_at_arrival.max() if n_jobs else 0,
        "mean_queue_len": queue_len_at_arrival.mean() if n_jobs else 0.0,
        "goodput_hz": (~missed).sum() / sim_duration_s,
    }


def simulate_queue_edf(service_times_s, arrival_rate_hz: float, deadline_s: float,
                        sim_duration_s: float, seed: int = 0,
                        drop_if_already_late: bool = True) -> dict:
    """Single-server EDF (Earliest-Deadline-First) queue with admission control."""
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
        return {"n_jobs": 0, "n_dropped_early": 0, "deadline_miss_rate": float("nan"),
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

    while next_arrival_idx < n_jobs or waiting:
        while next_arrival_idx < n_jobs and arrivals[next_arrival_idx] <= server_free_at:
            heapq.heappush(waiting, (deadlines[next_arrival_idx], next_arrival_idx))
            next_arrival_idx += 1

        if not waiting:
            if next_arrival_idx < n_jobs:
                server_free_at = arrivals[next_arrival_idx]
            continue

        queue_len_samples.append(len(waiting))
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
    return {
        "n_jobs": n_jobs,
        "n_dropped_early": int(dropped.sum()),
        "deadline_miss_rate": missed.mean(),
        "p50_response_s": np.nanpercentile(response_times, 50),
        "p99_response_s": np.nanpercentile(response_times, 99),
        "max_queue_len": max(queue_len_samples) if queue_len_samples else 0,
        "mean_queue_len": float(np.mean(queue_len_samples)) if queue_len_samples else 0.0,
        "goodput_hz": served.sum() / sim_duration_s,
    }
