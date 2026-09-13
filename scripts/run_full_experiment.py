#!/usr/bin/env python3
"""Run the full cross-layer qLDPC decoder evaluation end to end.

This is a plain-Python port of qldpc_realtime_experiment_v7_beamsearch.ipynb,
runnable from the command line (no Colab / Jupyter required). It reproduces,
in order: Phase 1 (code + distance bound), Phase 2A (code-capacity LER),
Phase 2B (runtime benchmark across error rates, + optional Beam Search),
Phase 3 (circuit-level LER, the main result), Phase 4 (FIFO/EDF queue
simulator), and Phase 5 (all comparison figures + a zipped reproducibility
bundle).

Usage:
    python scripts/run_full_experiment.py                # full run (slow, ~tens of min)
    python scripts/run_full_experiment.py --quick         # small debug run, fast
    python scripts/run_full_experiment.py --skip-beamsearch
    python scripts/run_full_experiment.py --output-dir results/

The original notebook's Colab-only step (`google.colab.files.download`) has
been removed; figures + environment_info.txt are zipped locally instead.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qldpc_eval import beam_search_setup, benchmark, circuit_level, code_setup  # noqa: E402
from qldpc_eval import environment_info, plotting, queue_sim  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=".", help="Directory to write figures/ and environment_info.txt into.")
    p.add_argument("--quick", action="store_true",
                   help="Small shot counts for a fast sanity-check run (not for reportable numbers).")
    p.add_argument("--skip-beamsearch", action="store_true",
                   help="Skip building/benchmarking the optional Beam Search decoder.")
    p.add_argument("--distance-trials", type=int, default=1000,
                   help="Number of trials for the randomized distance-bound computation (Phase 1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    np.random.seed(0)

    # ---------------- Phase 1: code + distance bound ----------------
    print("\n=== Phase 1: building code + distance bound ===")
    _, code_144 = code_setup.build_codes()
    CODE = code_144
    print(CODE)
    print("n, k:", len(CODE), CODE.dimension)

    dist_result = code_setup.compute_distance_bound(CODE, num_trials=args.distance_trials)
    code_setup.print_distance_report(dist_result)
    CODE_DISTANCE = dist_result["d"]

    # ---------------- Phase 2A: code-capacity LER curve ----------------
    print("\n=== Phase 2A: code-capacity LER curve ===")
    plotting.plot_code_capacity_ler(CODE, figures_dir, num_samples=200 if args.quick else 2000)

    # ---------------- Phase 2B: runtime benchmark ----------------
    print("\n=== Phase 2B: runtime benchmark across physical error rates ===")
    from qldpc_eval import decoders as decoders_mod

    Hz = decoders_mod.get_z_check_matrix(CODE)
    Lx = decoders_mod.get_logical_x_ops(CODE)
    print("Hz shape:", Hz.shape, "Lx shape:", Lx.shape)

    if args.quick:
        NUM_SHOTS, SEEDS, BENCH_ERROR_RATES = 50, [0], [0.03]
    else:
        NUM_SHOTS, SEEDS, BENCH_ERROR_RATES = 3000, [0, 1, 2, 3, 4], [0.01, 0.03, 0.05, 0.08]
    REFERENCE_P = 0.03
    assert REFERENCE_P in BENCH_ERROR_RATES

    all_bench_dfs = []
    for p in BENCH_ERROR_RATES:
        decoder_objs_p = decoders_mod.make_decoders(Hz, p)
        print(f"Physical error rate p={p} -- decoders rebuilt: {list(decoder_objs_p.keys())}")
        for seed in SEEDS:
            df = benchmark.run_benchmark(Hz, Lx, decoder_objs_p, NUM_SHOTS, p, seed=seed)
            df["seed"] = seed
            df["phys_error_rate"] = p
            all_bench_dfs.append(df)
    bench_df = pd.concat(all_bench_dfs, ignore_index=True)

    # ---------------- Phase 2C: optional Beam Search ----------------
    beam_commit = None
    if not args.skip_beamsearch:
        print("\n=== Phase 2C: building optional Beam Search decoder ===")
        build_result = beam_search_setup.build_beam_search()
        if build_result.available:
            beam_commit = build_result.commit
            print("Beam Search decoder built and imported successfully.")
            print("Cloned commit:", beam_commit)
            beam_bench_dfs = []
            for p in BENCH_ERROR_RATES:
                beam_only = {"BeamSearch": beam_search_setup.BeamSearchCodeCapacity(Hz, p)}
                for seed in SEEDS:
                    df = benchmark.run_benchmark(Hz, Lx, beam_only, NUM_SHOTS, p, seed=seed)
                    df["seed"] = seed
                    df["phys_error_rate"] = p
                    beam_bench_dfs.append(df)
            bench_df = pd.concat([bench_df] + beam_bench_dfs, ignore_index=True)
            print("Added BeamSearch. Current decoders:", sorted(bench_df["decoder"].unique()))
        else:
            print("Beam Search build/import failed -- skipping this optional section.")
            print("Error was:", repr(build_result.error))

    summary_df = benchmark.summarise_with_ci(bench_df)
    plotting.plot_runtime_vs_error_rate(summary_df, figures_dir)

    ref_bench_df = bench_df[bench_df["phys_error_rate"] == REFERENCE_P]
    ref_summary_df = summary_df[summary_df["phys_error_rate"] == REFERENCE_P].set_index("decoder")
    plotting.plot_runtime_ecdf(ref_bench_df, REFERENCE_P, figures_dir)
    plotting.plot_ler_vs_p99(ref_summary_df, REFERENCE_P, figures_dir)

    # ---------------- Phase 3: circuit-level experiment (MAIN result) ----------------
    print("\n=== Phase 3: circuit-level noise memory experiment ===")
    from qldpc import decoders as qdecoders
    from qldpc.objects import Pauli

    CIRCUIT_MAX_ERRORS = 100
    if args.quick:
        circuit_error_rates = list(np.logspace(-3, -2.1, 3))
        error_rate_shots = {p: 300 for p in circuit_error_rates}
    else:
        circuit_error_rates = list(np.logspace(-3, -2.1, 6))
        error_rate_shots = {p: circuit_level.shots_for_p(p) for p in circuit_error_rates}
    print("Shot budget per physical error rate:")
    for p, n in sorted(error_rate_shots.items()):
        print(f"  p={p:.5f}  ->  {n} shots")

    circuit_results = []
    for label, kwargs in [("BP-OSD", dict(with_BP_OSD=True)), ("BP-LSD", dict(with_BP_LSD=True))]:
        print(f"Running circuit-level experiment for {label} ...")
        stats = circuit_level.run_circuit_level_experiment_adaptive(
            CODE, Pauli.X, error_rate_shots, max_errors=CIRCUIT_MAX_ERRORS, **kwargs,
        )
        circuit_results.append(circuit_level.stats_to_df(stats, label))
    circuit_df = pd.concat(circuit_results, ignore_index=True)

    if beam_commit is not None:
        print("Running circuit-level experiment for BeamSearch ...")
        sinter_beam_cls = beam_search_setup.import_sinter_beam_search_decoder()
        stats_beam = circuit_level.run_circuit_level_experiment_adaptive(
            CODE, Pauli.X, error_rate_shots, max_errors=CIRCUIT_MAX_ERRORS,
            decoder_obj=sinter_beam_cls(beam_width=beam_search_setup.BEAM_WIDTH_DEFAULT),
        )
        circuit_df = pd.concat([circuit_df, circuit_level.stats_to_df(stats_beam, "BeamSearch")], ignore_index=True)

    plotting.plot_circuit_level_ler(circuit_df, CODE, figures_dir)

    # ---------------- Phase 4: queue simulator ----------------
    print("\n=== Phase 4: FIFO / EDF queue simulator ===")
    DEADLINE_VALUES_S = [200e-6, 500e-6, 1000e-6, 2000e-6, 5000e-6, 10000e-6]
    SIM_DURATION_S = 0.5 if args.quick else 2.0
    WARMUP_FRAC = 0.1  # discard the first 10% of sim time from metrics (queue starts empty)
    QUEUE_SEEDS = range(3) if args.quick else range(10)
    DEADLINE_S = 500e-6

    mean_us_by_decoder = ref_summary_df["mean_us"]
    workload_scenarios = {
        "normal": 1.0 / mean_us_by_decoder.max() * 1e6 * 0.3,
        "moderate": 1.0 / mean_us_by_decoder.max() * 1e6 * 0.7,
        "burst": 1.0 / mean_us_by_decoder.max() * 1e6 * 1.05,
    }
    print("Workload arrival rates (Hz):", workload_scenarios)
    print(f"Queue sim: {len(list(QUEUE_SEEDS))} seeds per (decoder, scenario, deadline), "
          f"warmup={WARMUP_FRAC*100:.0f}% of sim duration discarded from metrics.")

    # Each (decoder, scenario, deadline, policy) point is run across QUEUE_SEEDS
    # independent arrival sequences (see queue_sim.simulate_queue_multi_seed) --
    # a single-seed run's deadline-miss-rate is one draw of a random process,
    # and a headline number like "X% relative reduction" needs a CI to be
    # trustworthy. `_mean` columns are used for plotting (drop-in replacement
    # for the old single-seed values); `_stderr` columns give the CI.
    queue_records = []
    for decoder_name, g in ref_bench_df.groupby("decoder"):
        service_times_s = g["runtime_s"].values
        for scenario_name, arrival_rate in workload_scenarios.items():
            for deadline_s in DEADLINE_VALUES_S:
                warmup_s = WARMUP_FRAC * SIM_DURATION_S
                for policy, sim_fn in [("FIFO", queue_sim.simulate_queue), ("EDF", queue_sim.simulate_queue_edf)]:
                    agg = queue_sim.simulate_queue_multi_seed(
                        sim_fn, service_times_s, arrival_rate, deadline_s, SIM_DURATION_S,
                        seeds=QUEUE_SEEDS, warmup_s=warmup_s,
                    )
                    row = {
                        "decoder": decoder_name, "scenario": scenario_name,
                        "policy": policy, "deadline_us": deadline_s * 1e6,
                        "n_seeds": agg["n_seeds"],
                    }
                    for key in ("deadline_miss_rate", "p50_response_s", "p99_response_s",
                                "max_queue_len", "mean_queue_len", "goodput_hz"):
                        row[key] = agg.get(f"{key}_mean", float("nan"))
                        row[f"{key}_stderr"] = agg.get(f"{key}_stderr", float("nan"))
                    queue_records.append(row)
    queue_df = pd.DataFrame(queue_records)

    # Fix #4 sanity check: confirm the burst (mild-overload) scenario's metrics
    # are stable w.r.t. simulation duration / warm-up, not a transient artifact
    # of starting from an empty queue. Only runs once (fixed deadline/decoder),
    # since it's a spot-check, not swept across the whole grid.
    print("\n--- Warm-up / duration sensitivity check (burst scenario) ---")
    check_decoder = ref_bench_df["decoder"].iloc[0]
    check_service_times = ref_bench_df[ref_bench_df["decoder"] == check_decoder]["runtime_s"].values
    check_arrival_rate = workload_scenarios["burst"]
    for duration_s, warmup_frac in [(SIM_DURATION_S, 0.0), (SIM_DURATION_S, WARMUP_FRAC),
                                     (SIM_DURATION_S * 5, WARMUP_FRAC)]:
        agg = queue_sim.simulate_queue_multi_seed(
            queue_sim.simulate_queue_edf, check_service_times, check_arrival_rate, DEADLINE_S,
            duration_s, seeds=QUEUE_SEEDS, warmup_s=warmup_frac * duration_s,
        )
        print(f"  duration={duration_s:.1f}s warmup_frac={warmup_frac:.2f} -> "
              f"miss_rate={agg['deadline_miss_rate_mean']:.4f} +/- {agg['deadline_miss_rate_stderr']:.4f}")
    print("If these three numbers don't agree within ~1-2 stderr, the default "
          "SIM_DURATION_S/WARMUP_FRAC above are too small and should be increased.")

    # ---------------- Phase 5: final comparison plots ----------------
    print("\n=== Phase 5: final comparison plots ===")
    plotting.plot_missrate_vs_workload(queue_df, DEADLINE_S, REFERENCE_P, figures_dir)
    plotting.plot_ler_vs_p99_system(queue_df, ref_summary_df, DEADLINE_S, figures_dir)
    plotting.plot_crossover(queue_df, figures_dir)

    # ---------------- Save raw data + environment info ----------------
    bench_df.to_csv(os.path.join(args.output_dir, "bench_df.csv"), index=False)
    summary_df.to_csv(os.path.join(args.output_dir, "summary_df.csv"), index=False)
    circuit_df.to_csv(os.path.join(args.output_dir, "circuit_df.csv"), index=False)
    queue_df.to_csv(os.path.join(args.output_dir, "queue_df.csv"), index=False)

    env_path = os.path.join(args.output_dir, "environment_info.txt")
    environment_info.write_environment_info(
        env_path,
        extra={
            "CODE_DISTANCE (decoder-based upper bound)": CODE_DISTANCE,
            "BENCH_ERROR_RATES": BENCH_ERROR_RATES,
            "NUM_SHOTS per (decoder, p, seed)": NUM_SHOTS,
            "SEEDS": SEEDS,
            "CIRCUIT error_rate_shots": error_rate_shots,
            "CIRCUIT_MAX_ERRORS": CIRCUIT_MAX_ERRORS,
        },
        beamsearch_commit=beam_commit,
    )
    shutil.copy(env_path, os.path.join(figures_dir, "environment_info.txt"))
    archive_path = shutil.make_archive(os.path.join(args.output_dir, "all_figures"), "zip", figures_dir)
    print(f"\nDone. Figures + environment_info.txt + CSVs are in {args.output_dir}/")
    print(f"Zipped bundle: {archive_path}")


if __name__ == "__main__":
    main()
