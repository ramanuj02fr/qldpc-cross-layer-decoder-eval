"""Phase 5 -- Final comparison plots.

Headline plots:
1. Deadline miss rate vs workload intensity, per decoder, FIFO vs EDF.
2. Logical error rate vs p99 system-level response time (does the
   fastest-average decoder still win once queueing is included?).
3. Deadline miss rate vs deadline threshold (crossover plot).

Every function here takes the dataframes produced by the other phases and a
`figures_dir` to save into; nothing is regenerated implicitly, unlike the
original notebook's Phase 5 "regenerate everything" cell -- call the
individual phase functions again first if you want fresh data.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def _save(fig, figures_dir: str, name: str) -> str:
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    return path


def plot_code_capacity_ler(code_obj, figures_dir: str, num_samples: int = 2000):
    """Phase 2A figure: code-capacity LER vs physical error rate."""
    from .ler_curves import code_capacity_ler_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, kwargs in [("BP-OSD", dict(with_BP_OSD=True)), ("BP-LSD", dict(with_BP_LSD=True))]:
        p_phys, p_log, err = code_capacity_ler_curve(code_obj, num_samples=num_samples, **kwargs)
        line, = ax.plot(p_phys, p_log, label=label)
        ax.fill_between(p_phys, p_log - err, p_log + err, color=line.get_color(), alpha=0.2)
    ax.axline((0, 0), slope=1, color="k", linestyle=":", label=r"$p_{log}=p_{phys}$")
    ax.loglog()
    ax.set_xlabel("physical error rate")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"Code-capacity LER -- n={len(code_obj)}, k={code_obj.dimension}")
    ax.legend()
    ax.grid(which="both")
    plt.tight_layout()
    return _save(fig, figures_dir, "01_phase2A_code_capacity_ler")


def plot_runtime_vs_error_rate(summary_df, figures_dir: str):
    """Phase 2B figure: p50/p99 runtime vs physical error rate."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pct_col, pct_label in zip(axes, ["p50_us", "p99_us"], ["p50", "p99"]):
        for decoder_name, g in summary_df.groupby("decoder"):
            g = g.sort_values("phys_error_rate")
            ax.plot(g["phys_error_rate"], g[pct_col], marker="o", label=decoder_name)
        ax.set_xlabel("physical error rate")
        ax.set_title(f"{pct_label} runtime vs physical error rate")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("runtime (microseconds)")
    axes[1].legend()
    plt.tight_layout()
    return _save(fig, figures_dir, "02_phase2B_runtime_vs_error_rate")


def plot_runtime_ecdf(ref_bench_df, reference_p: float, figures_dir: str):
    """Phase 2B figure: runtime ECDF at REFERENCE_P."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, g in ref_bench_df.groupby("decoder"):
        rt_us = np.sort(g["runtime_s"].values * 1e6)
        ecdf = np.arange(1, len(rt_us) + 1) / len(rt_us)
        ax.plot(rt_us, ecdf, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("runtime (microseconds)")
    ax.set_ylabel("cumulative fraction of shots")
    ax.set_title(f"Decoder runtime ECDF (p = {reference_p})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return _save(fig, figures_dir, "03_phase2B_ecdf")


def plot_ler_vs_p99(ref_summary_df, reference_p: float, figures_dir: str):
    """Phase 2B figure: the 'killer graph' -- LER vs p99 runtime, at REFERENCE_P."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, row in ref_summary_df.iterrows():
        ax.scatter(row["p99_us"], row["logical_error_rate"], s=80, label=name)
        ax.annotate(name, (row["p99_us"], row["logical_error_rate"]),
                    textcoords="offset points", xytext=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p99 runtime (microseconds)")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"Logical Error Rate vs Tail Latency (p = {reference_p})")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return _save(fig, figures_dir, "04_phase2B_ler_vs_p99")


def plot_circuit_level_ler(circuit_df, code_obj, figures_dir: str):
    """Phase 3 figure: circuit-level LER vs physical error rate (the MAIN
    result), with Wilson 95% CI bands and 0-error points marked as upper bounds.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, g in circuit_df.groupby("decoder"):
        g = g.sort_values("prob")
        line, = ax.plot(g["prob"], g["logical_error_rate"].clip(lower=1e-7), marker="o", label=label)
        ax.fill_between(
            g["prob"],
            g["ci_lower"].clip(lower=1e-7),
            g["ci_upper"].clip(lower=1e-7),
            color=line.get_color(), alpha=0.2,
        )
        ub_only = g[g["is_upper_bound_only"]]
        if len(ub_only):
            ax.scatter(ub_only["prob"], ub_only["ci_upper"].clip(lower=1e-7),
                       marker="v", s=60, color=line.get_color(), zorder=5)
    ax.axline((0, 0), slope=1, color="k", linestyle=":", label=r"$p_{log}=p_{phys}$")
    ax.loglog()
    ax.set_xlabel("physical error rate (per-gate depolarizing)")
    ax.set_ylabel("logical error rate")
    ax.set_title(
        f"Circuit-level LER -- n={len(code_obj)}, k={code_obj.dimension} code\n"
        "(adaptive shots, Wilson 95% CI; downward triangle = 0-error upper bound)"
    )
    ax.legend()
    ax.grid(which="both")
    plt.tight_layout()
    path = _save(fig, figures_dir, "05_phase3_circuit_level_ler")

    n_ub_only = circuit_df["is_upper_bound_only"].sum()
    print(f"NOTE: {n_ub_only} of {len(circuit_df)} points had 0 observed logical errors "
          f"-- these are upper bounds on LER, not point estimates. Consider more shots "
          f"at those (decoder, p) combinations before finalizing numbers for a report.")
    return path


def plot_missrate_vs_workload(queue_df, deadline_s: float, reference_p: float,
                               figures_dir: str, scenario_order=("normal", "moderate", "burst")):
    """Phase 5 figure: deadline miss rate vs workload scenario, FIFO vs EDF."""
    default_slice = queue_df[queue_df["deadline_us"] == deadline_s * 1e6]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, policy in zip(axes, ["FIFO", "EDF"]):
        sub = default_slice[default_slice["policy"] == policy]
        for decoder_name, g in sub.groupby("decoder"):
            g = g.set_index("scenario").loc[list(scenario_order)]
            ax.plot(scenario_order, g["deadline_miss_rate"], marker="o", label=decoder_name)
        ax.set_xlabel("workload scenario")
        ax.set_title(f"{policy} scheduling")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("deadline miss rate")
    axes[1].legend()
    fig.suptitle(f"Deadline Miss Rate vs Workload (deadline = {deadline_s*1e6:.0f} us, p={reference_p})")
    plt.tight_layout()
    return _save(fig, figures_dir, "06_phase5_missrate_vs_workload")


def plot_ler_vs_p99_system(queue_df, ref_summary_df, deadline_s: float, figures_dir: str):
    """Phase 5 figure: does the fastest-average decoder still win once EDF
    queueing (burst workload) is included? Includes the readable-x-tick fix
    for narrow log-scale ranges where decoders' p99 values land close together.
    """
    default_slice = queue_df[queue_df["deadline_us"] == deadline_s * 1e6]
    burst_df = default_slice[
        (default_slice["scenario"] == "burst") & (default_slice["policy"] == "EDF")
    ].set_index("decoder")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for decoder_name in burst_df.index:
        ler = ref_summary_df.loc[decoder_name, "logical_error_rate"]
        p99_system_us = burst_df.loc[decoder_name, "p99_response_s"] * 1e6
        ax.scatter(p99_system_us, ler, s=80, label=decoder_name)
        ax.annotate(decoder_name, (p99_system_us, ler), textcoords="offset points", xytext=(6, 4))

    ax.set_xscale("log")
    ax.set_yscale("log")

    x_vals = (burst_df["p99_response_s"] * 1e6).values
    xmin, xmax = x_vals.min() * 0.9, x_vals.max() * 1.1
    ax.set_xlim(xmin, xmax)
    nice_ticks = np.linspace(x_vals.min(), x_vals.max(), 4)
    ax.set_xticks(nice_ticks)
    ax.set_xticklabels([f"{t:,.0f}" for t in nice_ticks])
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel("p99 SYSTEM-level response time (microseconds, burst workload, EDF)")
    ax.set_ylabel("logical error rate")
    ax.set_title("Does the fastest decoder win once EDF queueing is included?")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return _save(fig, figures_dir, "07_phase5_ler_vs_p99_system")


def plot_crossover(queue_df, figures_dir: str):
    """Phase 5 figure: deadline miss rate vs deadline threshold (moderate
    workload, EDF scheduling) -- the crossover plot.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    crossover_slice = queue_df[(queue_df["scenario"] == "moderate") & (queue_df["policy"] == "EDF")]
    for decoder_name, g in crossover_slice.groupby("decoder"):
        g = g.sort_values("deadline_us")
        ax.plot(g["deadline_us"], g["deadline_miss_rate"], marker="o", label=decoder_name)
    ax.set_xscale("log")
    ax.set_xlabel("deadline (microseconds)")
    ax.set_ylabel("deadline miss rate")
    ax.set_title("Deadline Miss Rate vs Deadline Threshold\n(moderate workload, EDF scheduling)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return _save(fig, figures_dir, "08_phase5_crossover")
