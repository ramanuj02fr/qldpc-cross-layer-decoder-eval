"""Phase 2C -- Optional extension: Beam Search decoder.

The Beam Search decoder (Ye, Wecker, Delfosse -- PRX Quantum, July 2026,
https://github.com/ionq-publications/BeamSearchDecoder) is a recent BP-guided
search decoder reporting large LER improvements over BP-OSD on the exact
[[144,12,12]] BB code, at tunable speed/accuracy tradeoffs via `beam_width`.

This module is optional and riskier than the rest of the package -- it
requires compiling a C++/Cython extension from an external repo. If the build
fails, the rest of the pipeline still runs fine using BP / BP-OSD / BP-LSD only.

--- Fix history ---
The original pinned install command,

    pip install -q Cython==3.0.10 PyMatching==2.2.1 stimbposd==0.1.0 networkx==3.3

failed with a CalledProcessError on Colab. Root cause (confirmed by reproducing
the install and inspecting package metadata): PyMatching==2.2.1's own PyPI
metadata hard-pins `numpy==1.*`, which is stale -- it predates numpy 2.0, and
this repo's own requirements.txt asks for numpy>=2.0.0. Forcing numpy down to
1.x collides with other numpy>=2-requiring packages already installed on
Colab, so pip's resolver aborts.

Verified directly: PyMatching 2.2.1's compiled wheel imports and decodes
correctly on numpy 2.5.x despite the stale pin -- the metadata constraint is
simply wrong, not a real runtime incompatibility. So PyMatching and stimbposd
are installed with `--no-deps` below, which stops pip from "fixing" numpy while
still pinning their own versions for reproducibility.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

BEAM_WIDTH_DEFAULT = 8  # paper reports beam_width=8 roughly matches BP-OSD's LER
                         # with much better tail behavior
REPO_URL = "https://github.com/ionq-publications/BeamSearchDecoder.git"


@dataclass
class BeamSearchBuildResult:
    available: bool
    commit: str | None = None
    error: Exception | None = None


def install_beam_search_dependencies() -> None:
    """Install the extra pinned dependencies the BeamSearchDecoder repo needs,
    beyond what the rest of the pipeline already installs (qldpc, ldpc, stim,
    sinter, matplotlib, pandas, numpy, scipy).

    See the module docstring for why PyMatching/stimbposd use --no-deps.
    """
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "Cython==3.0.10", "networkx==3.3"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps",
         "PyMatching==2.2.1", "stimbposd==0.1.0"],
        check=True,
    )
    # Sanity check: confirm both imports actually work post-install (fail loudly
    # and early here, rather than a confusing ImportError deep inside the
    # beam-search import below, if some target runtime really is incompatible).
    import pymatching as _pymatching_check  # noqa: F401
    import stimbposd as _stimbposd_check  # noqa: F401


def build_beam_search(clone_dir: str = "BeamSearchDecoder") -> BeamSearchBuildResult:
    """Clone (if needed), pin the commit, install deps, and build the Cython
    extension in place. Returns a BeamSearchBuildResult; does not raise on
    failure -- check `.available` and `.error`.
    """
    try:
        if not os.path.exists(clone_dir):
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, clone_dir],
                check=True,
            )
        result = subprocess.run(
            ["git", "-C", clone_dir, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        commit = result.stdout.strip()

        install_beam_search_dependencies()

        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=os.path.join(clone_dir, "decoder"), check=True,
        )
        sys.path.insert(0, os.path.abspath(clone_dir))
        sys.path.insert(0, os.path.abspath(os.path.join(clone_dir, "decoder")))
        return BeamSearchBuildResult(available=True, commit=commit)
    except Exception as e:  # noqa: BLE001 -- intentionally broad, this is an optional extension
        return BeamSearchBuildResult(available=False, error=e)


def import_raw_beam_search_decoder():
    """Call only after a successful `build_beam_search()`."""
    from decoder.beam_search_decoder import BeamSearchDecoder as _RawBeamSearchDecoder
    return _RawBeamSearchDecoder


def import_sinter_beam_search_decoder():
    """Call only after a successful `build_beam_search()`. Requires the repo
    root (not just decoder/) on sys.path -- `build_beam_search` already does this.
    """
    from sinter_beamsearch import SinterDecoder_BeamSearch
    return SinterDecoder_BeamSearch


class BeamSearchCodeCapacity:
    """Wraps the raw BeamSearchDecoder so it has the same `.decode(syndrome)`
    interface as the `ldpc`-package decoders, for the code-capacity benchmark
    in benchmark.run_benchmark().
    """

    def __init__(self, Hz, p, beam_width: int = BEAM_WIDTH_DEFAULT, **kwargs):
        raw_cls = import_raw_beam_search_decoder()
        n = Hz.shape[1]
        self._dec = raw_cls(pcm=Hz, error_channel=[p] * n, beam_width=beam_width, **kwargs)

    def decode(self, syndrome):
        return self._dec.decode(syndrome)
