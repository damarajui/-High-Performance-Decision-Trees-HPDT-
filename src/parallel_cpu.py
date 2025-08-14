"""Parallel CPU implementation of the decision tree.

The original project provides a C++ implementation that parallelises the search
for the best attribute using multiple threads.  The Python counterpart mirrors
that behaviour using :mod:`concurrent.futures` which offers a simple
abstraction over thread pools.

This module reuses the :class:`DecisionTree` implementation and only overrides
``_find_best_attribute`` to distribute the calculation of information gain
across several worker threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from .decision_tree import DecisionTree


class ParallelDecisionTree(DecisionTree):
    """Decision tree that parallelises information gain computation.

    Parameters
    ----------
    num_threads:
        The number of worker threads used when computing the information gain
        for each attribute.  If ``None`` a reasonable default based on the
        machine's CPU count is used.
    """

    def __init__(self, num_threads: int | None = None) -> None:
        super().__init__()
        self.num_threads = num_threads

    # ------------------------------------------------------------------
    def _find_best_attribute(
        self, data: Sequence[Sequence[str]], attributes: Sequence[str]
    ) -> int:
        """Return the index of the attribute with highest information gain.

        The heavy lifting of computing the information gain for each attribute
        is distributed across a pool of worker threads.  For the typically small
        data sets used in the unit tests this overhead is negligible while still
        providing a faithful representation of the parallel algorithm.
        """

        # ``max_workers=None`` lets ``ThreadPoolExecutor`` pick a sensible
        # default based on ``os.cpu_count``.
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            gains = list(
                executor.map(
                    lambda i: self._calculate_information_gain(data, i),
                    range(len(attributes)),
                )
            )

        return max(range(len(attributes)), key=lambda i: gains[i])
