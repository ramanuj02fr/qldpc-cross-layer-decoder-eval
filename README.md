# qLDPC Cross-Layer Decoder Evaluation

Cross-layer evaluation of three qLDPC decoders (BP, BP-OSD, BP-LSD; optionally
also the Beam Search decoder) on a `[[144,12,12]]` Bivariate Bicycle (BB) code:

1. Code-capacity + circuit-level logical error rate (LER), with statistical
   rigor (Wilson score confidence intervals, adaptive shot allocation).
2. Per-shot runtime distributions across multiple physical error rates.
3. FIFO vs. EDF (deadline-driven) queueing simulation, to test whether the
   fastest-average decoder is actually the best real-time decoder.

**Central finding:** BP-OSD and BP-LSD are statistically tied on LER at every
physical error rate tested, but BP-LSD's runtime grows substantially faster
with physical error rate -- invisible from LER alone, only visible once
runtime and queueing are jointly analyzed. EDF scheduling recovers a large
fraction of real-time performance without changing the decoder at all.

This repo is a plain-Python port of the original Colab notebook
(no `.ipynb` file is shipped here) -- see [`PROJECT_STATUS.md`](PROJECT_STATUS.md)
for the full history of fixes and open items.

## Repo layout

```
src/qldpc_eval/
    code_setup.py          Phase 1  -- build the BB code + distance bound
    ler_curves.py           Phase 2A -- code-capacity LER curve
    decoders.py             Phase 2B -- BP / BP-OSD / BP-LSD decoder factory
    benchmark.py             Phase 2B -- per-shot runtime benchmark loop
    beam_search_setup.py    Phase 2C -- optional Beam Search decoder build
    circuit_level.py        Phase 3  -- circuit-level LER (MAIN result)
    queue_sim.py             Phase 4  -- FIFO / EDF queue simulator
    plotting.py              Phase 5  -- all comparison figures
    environment_info.py     reproducibility record (CPU + pinned versions)
scripts/
    run_full_experiment.py  CLI entry point, runs everything end to end
    install_beamsearch.sh   standalone installer for the optional decoder
requirements.txt
requirements-beamsearch.txt
PROJECT_STATUS.md           full history of review findings + fixes
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Fast sanity check (small shot counts, skips Beam Search, a couple of minutes):

```bash
python scripts/run_full_experiment.py --quick --skip-beamsearch --output-dir results/
```

Full run (reportable numbers -- this is slow, budget tens of minutes on a
typical machine, longer on a shared/low-core VM):

```bash
python scripts/run_full_experiment.py --output-dir results/
```

Output: `results/figures/*.png` (8 comparison figures), `results/*.csv` (raw
per-decoder dataframes), `results/environment_info.txt` (CPU + pinned package
versions for reproducibility), and `results/all_figures.zip` bundling the above.

## Optional: Beam Search decoder

The [Beam Search decoder](https://github.com/ionq-publications/BeamSearchDecoder)
(Ye, Wecker, Delfosse -- PRX Quantum, July 2026) is a recent BP-guided search
decoder reporting large LER improvements over BP-OSD on this exact
`[[144,12,12]]` BB code. It's optional because it requires compiling a
C++/Cython extension from an external repo; the rest of the pipeline runs
fine without it.

`scripts/run_full_experiment.py` builds it automatically unless you pass
`--skip-beamsearch`. To build it standalone instead:

```bash
bash scripts/install_beamsearch.sh
```

**Known dependency pin issue (fixed here):** `PyMatching==2.2.1`'s own PyPI
metadata hard-pins `numpy==1.*`, which is stale -- it predates numpy 2.0, and
the BeamSearchDecoder repo's own `requirements.txt` asks for `numpy>=2.0.0`.
On environments that already have numpy>=2 installed for other reasons (e.g.
Colab), pip's resolver can't satisfy both constraints and aborts the install
with a non-zero exit code. This was verified to be a stale-metadata problem,
not a real incompatibility -- PyMatching 2.2.1's compiled wheel imports and
decodes correctly on numpy 2.5.x. The fix, applied in both
`beam_search_setup.py` and `install_beamsearch.sh`, is to install
`PyMatching` and `stimbposd` with `--no-deps` so pip doesn't try to "fix"
numpy's version.

## Threats to Validity

Be upfront about these if reporting the numbers this pipeline produces:

1. **Software latency, not hardware latency.** All runtimes are the
   wall-clock time of `decode()` calls in Python, on a single CPU core (see
   `environment_info.txt`). This is *not* directly comparable to FPGA/ASIC
   real-time decoder latencies reported in hardware-decoder papers (e.g.
   DART-Q, GARI). Any "real-time feasibility" claim should be scoped to
   "software decoding on commodity hardware."
2. **Single code family, single distance.** Results are for one
   `[[144,12,12]]` BB code; its distance is a decoder-based randomized
   **upper bound** (see `code_setup.compute_distance_bound`), not a
   GAP-certified exact value. Decoder ranking may differ for other code
   families or sizes.
3. **EDF simulator is a simplified single-server analog of DART-Q's ideas**
   (deadline-based scheduling + admission control), not a reproduction of
   DART-Q's full formulation. Results are "consistent with" the idea, not a
   validation of DART-Q's own reported numbers.
4. **Runtime is confounded with the physical error rate it was measured at**
   in the queue simulator (Phase 4), which uses a single reference physical
   error rate's runtime distribution for all workload scenarios, not the
   circuit-level timing.
5. **Circuit-level (Phase 3) shot counts, while adaptive, are still modest**
   (3,000-8,000 depending on p). Points flagged `is_upper_bound_only` should
   not be treated as point estimates.
6. **Beam Search decoder depends on an external, separately-maintained
   repository** built from source at run time. Even with the commit hash
   pinned and logged to `environment_info.txt`, the build environment (system
   compiler, OS packages) is not fully controlled.
7. **`num_rounds=3` is fixed** in the circuit-level memory experiment; it has
   not yet been tied to the code's distance bound.
8. **No correlated/biased noise model tested** -- only i.i.d. bit-flip
   (code-capacity) and i.i.d. depolarizing (circuit-level).

See `PROJECT_STATUS.md` for the full paper-review findings and priority order
of open items.
