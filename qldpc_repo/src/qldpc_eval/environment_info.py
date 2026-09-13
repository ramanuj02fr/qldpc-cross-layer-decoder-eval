"""Environment / reproducibility record.

Records CPU info and exact package versions used for a run, and saves them
into the figures bundle. Without this, "decoder X's runtime is Y microseconds"
is not a reproducible or comparable claim -- it is tied to whatever machine
happened to run it.
"""

from __future__ import annotations

import platform
import subprocess
import sys

_RELEVANT_PREFIXES = (
    "ldpc==", "qldpc==", "stim==", "sinter==", "numpy==", "scipy==",
    "pandas==", "matplotlib==", "cython==", "pymatching==", "networkx==",
    "stimbposd==",
)


def write_environment_info(path: str, extra: dict | None = None,
                            beamsearch_commit: str | None = None) -> None:
    """Write platform info, pinned package versions, and any extra run
    parameters (e.g. BENCH_ERROR_RATES, NUM_SHOTS, SEEDS) to `path`.
    """
    lines = []
    lines.append(f"Platform: {platform.platform()}")
    lines.append(f"Processor: {platform.processor()}")
    lines.append("")
    try:
        lscpu_out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        lines.append("lscpu:")
        lines.append(lscpu_out)
    except Exception as e:  # noqa: BLE001
        lines.append(f"lscpu unavailable: {e}")

    lines.append("Pinned package versions:")
    freeze_out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    relevant = [l for l in freeze_out.splitlines() if l.lower().startswith(_RELEVANT_PREFIXES)]
    lines.extend(relevant)

    if beamsearch_commit:
        lines.append("")
        lines.append(f"BeamSearchDecoder commit: {beamsearch_commit}")

    if extra:
        lines.append("")
        for key, value in extra.items():
            lines.append(f"{key}: {value}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")
