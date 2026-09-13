#!/usr/bin/env bash
# Installs and builds the optional Beam Search decoder (Phase 2C).
#
# Run this AFTER `pip install -r requirements.txt`. It exists as a standalone
# script (in addition to qldpc_eval.beam_search_setup.build_beam_search()) so
# it can be run once, outside of a Python session, e.g. in a Dockerfile or CI
# step.
#
# See README.md "Optional: Beam Search decoder" for why PyMatching/stimbposd
# use --no-deps below.
set -euo pipefail

REPO_URL="https://github.com/ionq-publications/BeamSearchDecoder.git"
CLONE_DIR="${1:-BeamSearchDecoder}"

if [ ! -d "$CLONE_DIR" ]; then
    git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
fi

echo "Cloned commit: $(git -C "$CLONE_DIR" rev-parse HEAD)"

pip install -q Cython==3.0.10 networkx==3.3
pip install -q --no-deps PyMatching==2.2.1 stimbposd==0.1.0

python -c "import pymatching, stimbposd; print('pymatching + stimbposd import OK')"

( cd "$CLONE_DIR/decoder" && python setup.py build_ext --inplace )

echo "Beam Search decoder built successfully in $CLONE_DIR/decoder/beam_search_decoder/"
echo "Add '$CLONE_DIR' and '$CLONE_DIR/decoder' to sys.path to import it, e.g.:"
echo "  sys.path.insert(0, os.path.abspath('$CLONE_DIR'))"
echo "  sys.path.insert(0, os.path.abspath('$CLONE_DIR/decoder'))"
echo "  from decoder.beam_search_decoder import BeamSearchDecoder"
