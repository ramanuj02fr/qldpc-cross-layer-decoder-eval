"""
qldpc_eval -- Cross-layer evaluation of qLDPC decoders (BP / BP-OSD / BP-LSD /
optional Beam Search) on a Bivariate Bicycle (BB) code, across code-capacity LER,
circuit-level LER, per-shot runtime distributions, and a FIFO/EDF queue simulator.

This package is a plain-.py port of the original Colab notebook
(qldpc_realtime_experiment_v7_beamsearch.ipynb) so it can live in a git repo
without shipping a .ipynb. See scripts/run_full_experiment.py for the CLI
entry point that reproduces the notebook end to end.
"""

__version__ = "0.7.0"
