# qLDPC Cross-Layer Decoder Evaluation — Project Status

This document summarizes everything reviewed and fixed so far, so work can continue
on GitHub without losing context. Paste this whole file as context in a new
conversation (with Claude, or any assistant) to resume exactly where this left off.

## Files in this bundle

- `qldpc_cross_layer_paper.docx` — the original paper draft (unmodified).
- `qldpc_realtime_experiment_v6_distance.ipynb` — the experiment notebook, with the
  distance-computation fix applied (see below). This is v6, built on top of the
  uploaded `qldpc_realtime_experiment_v5_Q2level.ipynb`.
- `PROJECT_STATUS.md` — this file.

## What the paper is about

Cross-layer evaluation of three qLDPC decoders (BP, BP-OSD, BP-LSD) on a
`[[144,12,12]]` Bivariate Bicycle (BB) code:
1. Code-capacity + circuit-level logical error rate (LER), with statistical rigor
   (Wilson score CIs, adaptive shot allocation).
2. Per-shot runtime distributions across multiple physical error rates.
3. FIFO vs. EDF (deadline-driven) queueing simulation to test whether the
   fastest-average decoder is actually the best real-time decoder.

**Central finding:** BP-OSD and BP-LSD are statistically tied on LER at every
physical error rate tested, but BP-LSD's runtime grows substantially faster with
physical error rate — invisible from LER alone, only visible once runtime and
queueing are jointly analyzed. EDF scheduling recovers a large fraction of
real-time performance without changing the decoder at all.

## Review findings — paper draft (qldpc_cross_layer_paper.docx)

### High priority (fix before any submission)
1. **Code distance `d` was missing** — title and text use `[[144,12,d]]`. **RESOLVED**
   — see "Fix #1" below. Now `[[144,12,12]]`.
2. **References [8] (DART-Q) and [9] (beam-search decoder, Ye/Wecker/Delfosse)
   are unverified** — the paper itself flags them as needing verification against
   primary sources before formal citation. Not yet resolved.
3. **Author/affiliation are placeholders** ("Author Name¹", "Affiliation... edit as
   appropriate"). Not yet resolved.
4. **Beam-search decoder comparator is missing** — build failed with a pip
   dependency conflict in the notebook. This is arguably the biggest gap: the
   paper's recommendation (BP-OSD over BP-LSD) doesn't account for the beam-search
   decoder, which reportedly beats BP-OSD on LER. Root cause identified (see
   "Beam-search build failure" below) but not yet fixed.

### Medium priority
5. **Statistical rigor** — "statistically indistinguishable" is asserted by eye
   (CI overlap) rather than via a formal test (e.g., two-proportion z-test).
6. **Queueing simulator confound** — the queue simulator (Phase 4 / Section 3.5)
   uses a single reference physical error rate (p=0.03) code-capacity runtime
   distribution for all workload scenarios, not the circuit-level timing. Paper's
   own Threats to Validity (point 4) already flags this.
7. **Single code family / single distance** — no generalization test yet.
8. **Shared 2-vCPU Colab VM timing noise** — not yet quantified/controlled.
9. **No correlated/biased noise model tested** — only i.i.d. bit-flip and i.i.d.
   depolarizing.

### Lower priority / polish
10. Code/data listed as "available upon request" — consider a public GitHub repo
    with a Zenodo DOI instead, for credibility at a good venue.
11. Related work section could discuss hardware-decoder papers (e.g. GARI) more
    directly, not just in the limitations section.

## Discrepancies found between the paper text and the actual notebook data

These were found by actually reading and executing the notebook
(`qldpc_realtime_experiment_v5_Q2level.ipynb`, as uploaded), and comparing its
real output to what the paper draft claims. **Not yet fixed in the paper text.**

1. **Circuit-level crossing point (Section 4.4) is wrong.** Paper text says the
   `p_log = p_phys` crossing point is "in the vicinity of p ≈ 3–4 × 10⁻³". Actual
   notebook data (from `circuit_df`, cell 24 output):

   | decoder | p        | LER      |
   |---------|----------|----------|
   | BP-OSD  | 0.001000 | 0.000500 |
   | BP-OSD  | 0.001514 | 0.002250 |
   | BP-LSD  | 0.001000 | 0.000125 |
   | BP-LSD  | 0.001514 | 0.002250 |

   Both decoders cross `p_log = p_phys` between p=0.001 and p=0.001514 (~1.2×10⁻³),
   **not** 3–4×10⁻³. This needs to be corrected in Section 4.4 of the paper.

2. **"Adaptive shot allocation (3,000–8,000 shots per point)" caption (Figure 5) is
   inaccurate.** Because `sinter.collect` is called with both `max_shots` (the
   budget) AND `max_errors=100`, most points stop far short of their budget once
   100 errors are observed. Actual shots used (BP-OSD):

   | p        | budget | actual shots used | errors |
   |----------|--------|--------------------|--------|
   | 0.001000 | 8000   | 8000 (full)        | 4      |
   | 0.001514 | 8000   | 8000 (full)        | 18     |
   | 0.002291 | 5000   | 4679               | 100    |
   | 0.003467 | 5000   | 636                | 100    |
   | 0.005248 | 3000   | 169                | 100    |
   | 0.007943 | 3000   | 103                | 100    |

   (BP-LSD numbers are very similar.) This early-stopping is statistically valid
   (standard practice), but the paper should report actual shots used, not the
   budget, in the Figure 5 caption and Section 3.4 text.

3. **Sparse low-p point.** At p=0.001, BP-LSD saw only **1 observed error** out of
   8000 shots (Wilson CI: 0.000022–0.000708, a >30× span). This point is nearly
   uninformative for comparing BP-OSD vs. BP-LSD at that p. Recommend either
   raising `max_errors` specifically for the lowest 1–2 physical error rates, or
   running more shots there specifically.

4. **Table 1 and LER/runtime numbers DO match** between the paper and the notebook
   — no data fabrication issue, just the two discrepancies above (crossing point
   text, and the shots caption).

## Fixes made so far

### Fix #1: Code distance (DONE)

**Problem:** `code.get_distance()` (exact, GAP-based) hangs indefinitely in Colab
because GAP isn't installed there, and qldpc's GAP interface blocks waiting for
interactive copy/paste input rather than failing cleanly.

**Solution:** Use `code.get_distance_bound_with_decoder(pauli, num_trials=...)`
instead. This is a randomized decoder-based algorithm (arXiv:2308.07915) that
never touches GAP, and converges fast:

```python
from qldpc.objects import Pauli

DISTANCE_TRIALS = 1000
dx_bound = CODE.get_distance_bound_with_decoder(Pauli.X, num_trials=DISTANCE_TRIALS)
dz_bound = CODE.get_distance_bound_with_decoder(Pauli.Z, num_trials=DISTANCE_TRIALS)
CODE_DISTANCE = min(dx_bound, dz_bound)
```

**Result (verified by actually running this):** with `num_trials=1000`, both
X-distance and Z-distance bounds converge to **12** in about 5 seconds total. This
matches the known/expected value for this BB code family (the well-known
`[[144,12,12]]` "gross code" from Bravyi et al. 2024). Convergence by trial count:

| num_trials | X-bound | Z-bound | time |
|------------|---------|---------|------|
| 10         | 28      | 20      | <1s  |
| 50         | 12      | 20      | ~1s  |
| 200        | 12      | 22      | ~2s  |
| 1000       | 12      | 12      | ~9s  |

**Important caveat:** This is a randomized **upper bound**, not a GAP-certified
exact distance. The paper/notebook should describe it that way, not as an exact
computed value.

**Where this was applied in the notebook (v6):**
- Cell 5 (code definition cell): distance computation appended after `CODE = code_144`.
- Cell 22: fixed a stale comment that blamed `num_rounds=3` on "GAP hanging" —
  updated to note that distance is now known but `num_rounds` has not yet been
  tied to it.
- Cell 38 (Threats to Validity, markdown): updated point 2 (single code family)
  to say `[[144,12,12]]` instead of `[[144,12,d]]` and to note it's a decoder-based
  bound; updated point 7 (`num_rounds=3` fixed) to remove the now-inaccurate "GAP
  was hanging" framing.

Cell outputs were cleared in the v6 file to keep file size manageable — re-run
top-to-bottom in Colab to regenerate all outputs and figures.

### Fix #2: Beam-search build failure (DONE)

**Problem:** the beam-search install cell failed at:
```
pip install -q Cython==3.0.10 PyMatching==2.2.1 stimbposd==0.1.0 networkx==3.3
```
with a `CalledProcessError`.

**Root cause (confirmed by actually reproducing the install and inspecting package
metadata):** `PyMatching==2.2.1`'s own PyPI metadata hard-pins `Requires-Dist: numpy
==1.*`. This pin is stale — it predates numpy 2.0. The BeamSearchDecoder repo's own
`requirements.txt` asks for `numpy>=2.0.0`, directly contradicting PyMatching's pin.
On Colab (which has numpy>=2-dependent packages already installed), pip's resolver
cannot satisfy `numpy==1.*` and `numpy>=2.0.0` simultaneously and aborts with a
non-zero exit code — that's the `CalledProcessError`.

**Verified this is a metadata problem, not a real incompatibility:** installed
PyMatching 2.2.1 fresh alongside numpy 2.5.3 (ignoring its pin) and confirmed:
- `import pymatching` succeeds
- a real `Matching().decode(...)` call runs correctly
- `import stimbposd`, `from stimbposd import BPOSD, SinterDecoder_BPOSD` succeed
- the full `BeamSearchDecoder` Cython/C++ extension builds cleanly with
  `python setup.py build_ext --inplace` (only harmless signedness compiler warnings)
- `from decoder.beam_search_decoder import BeamSearchDecoder` imports and a
  `.decode(...)` smoke test call runs and returns a result
- `from sinter_beamsearch import SinterDecoder_BeamSearch` (used later in cell 25)
  also imports fine

**Solution:** install `PyMatching==2.2.1` and `stimbposd==0.1.0` with `--no-deps`,
so pip doesn't try to "fix" numpy's version. `Cython==3.0.10` and `networkx==3.3`
have no such conflicting pin and still install normally with their own deps.
Added an explicit `import pymatching` / `import stimbposd` sanity check right after
install so any *real* incompatibility on a given runtime fails loudly there,
instead of surfacing later as a confusing `ImportError` inside the beam-search
import.

```python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "Cython==3.0.10", "networkx==3.3"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
     "PyMatching==2.2.1", "stimbposd==0.1.0"],
    check=True,
)
import pymatching as _pymatching_check  # noqa: F401
import stimbposd as _stimbposd_check  # noqa: F401
```

**Where this was applied:** `qldpc_realtime_experiment_v7_beamsearch.ipynb`, cell 19
(the beam-search install cell), built on top of v6. No other cells changed.

**Caveat / what's still unverified:** all of the above was verified in a Linux
sandbox with network access to PyPI/GitHub, matching Colab's Linux x86_64 + Python
3.12 environment closely enough that the same wheels apply — but the actual Colab
notebook run (with the real `[[144,12,12]]` code and real circuit-level data) has
**not yet been done**. Next step is to run cell 19 top-to-bottom in Colab itself to
confirm it builds there too, then run cells 20 and 25 (which add BeamSearch to the
Phase 2B and Phase 3 comparisons) to get real LER numbers for BeamSearch vs BP-OSD
vs BP-LSD on the actual code.

## Next steps (in priority order)

1. **Run `qldpc_realtime_experiment_v7_beamsearch.ipynb` in Colab top-to-bottom**
   to confirm the beam-search install fix (Fix #2 above) also works there, then
   check whether BeamSearch actually beats BP-OSD on LER on the real
   `[[144,12,12]]` code as the paper reports it does elsewhere — this directly
   affects the paper's central recommendation.
2. **Fix Section 4.4 text and Figure 5 caption** in the paper draft to match the
   actual notebook data (see "Discrepancies" above).
3. **Add a formal statistical test** (e.g., two-proportion z-test) for the
   "statistically indistinguishable" claims instead of eyeballing CI overlap.
4. **Verify references [8] and [9]** against primary sources.
5. Medium/lower priority items from the paper review list above, as time allows.

## Working style notes (for whoever/whatever continues this)

- The user works in Hinglish; response style has been informal, in Hindi/Hinglish,
  with concrete numbers pulled from actually running code rather than generic advice.
- The user prefers incremental, one-fix-at-a-time work ("ek ek karke karte hai"),
  verified end-to-end (actually run the code, don't just describe the fix) before
  moving to the next item.
- The user's Google Drive (`ramanuj02fr@gmail.com`) has all notebook versions
  (v1 through v5) in a shared folder (Drive folder ID:
  `1cEQ-IjeNNIhQVP3mxVJhOVUb7N0zik0Y`) — useful if reconnecting a Drive connector
  in a future session instead of GitHub.
- Going forward, the user wants to work via a **GitHub repo** instead of
  manual download/upload — push this bundle there and continue from a clone of
  that repo.
