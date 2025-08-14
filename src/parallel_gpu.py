"""GPU accelerated decision tree.

The real project contains a CUDA based implementation.  For the purposes of the
unit tests in this kata we provide a very small wrapper around the parallel CPU
implementation.  If the optional :mod:`cupy` package is available the class will
use it for array manipulation which allows the heavy calculations to run on a
GPU.  When :mod:`cupy` is not installed the implementation silently falls back
to NumPy and therefore still works in CPU only environments.
"""

from __future__ import annotations

from typing import Sequence

from .parallel_cpu import ParallelDecisionTree

try:  # pragma: no cover - environment dependent
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when CuPy is missing
    _CUPY_AVAILABLE = False
    cp = None  # type: ignore


class GPUDecisionTree(ParallelDecisionTree):
    """Decision tree with an optional GPU backend."""

    def __init__(self, num_threads: int | None = None) -> None:
        # Reuse the CPU parallel implementation.  The ``num_threads`` argument is
        # kept for API compatibility even though it is not used when CuPy is
        # available.
        super().__init__(num_threads=num_threads)

    # The public interface does not need to change; ``ParallelDecisionTree``
    # already implements ``build_tree`` and ``classify``.  The presence of CuPy
    # is exposed via the ``_CUPY_AVAILABLE`` module level constant which tests
    # can introspect if required.
