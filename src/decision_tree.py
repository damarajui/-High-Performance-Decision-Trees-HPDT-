from __future__ import annotations

"""Basic decision tree implementation.

This module provides a very small implementation of the ID3 decision tree
algorithm.  The C++ sources bundled with the project contain a complete
implementation but the Python modules were left empty.  The tests in this kata
exercise the Python version, therefore we implement a clear and reasonably
small port of the C++ version.

The tree expects the training data to be provided as a list of lists where each
inner list represents a single instance.  Every instance should contain the
attribute values followed by the label as the last element.  A separate list of
attribute names in the same order is also required.

Example
-------
>>> data = [
...     ["sunny", "hot", "high", "FALSE", "no"],
...     ["sunny", "hot", "high", "TRUE", "no"],
... ]
>>> attributes = ["outlook", "temperature", "humidity", "windy"]
>>> tree = DecisionTree()
>>> tree.build_tree(data, attributes)
>>> tree.classify(["sunny", "hot", "high", "FALSE"])
'no'

The implementation aims to be easy to read rather than extremely efficient –
perfect for the unit tests included with this kata.
"""

from collections import Counter
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class Node:
    """A node in the decision tree.

    Parameters
    ----------
    attribute: str | None
        Name of the attribute used for splitting at this node.  ``None`` for
        leaf nodes.
    label: str | None
        Majority label for this node.  For leaf nodes this is the class label;
        for internal nodes it is used as a fall back when an attribute value is
        not present in the training data.
    is_leaf: bool
        Whether this node is a leaf.
    children: Dict[str, Node]
        Mapping from attribute value to the child node.
    """

    attribute: str | None = None
    label: str | None = None
    is_leaf: bool = False
    children: Dict[str, "Node"] = field(default_factory=dict)


class DecisionTree:
    """Decision tree using the ID3 algorithm.

    The public API mirrors the C++ implementation included in the repository.
    The tree operates on simple categorical data and is intended only for the
    unit tests supplied with this exercise.
    """

    def __init__(self) -> None:
        self.root: Node | None = None
        # Original order of attributes used during training – required for
        # classification where we receive only the attribute values.
        self._attributes: List[str] | None = None

    # ------------------------------------------------------------------
    # Building the tree
    # ------------------------------------------------------------------
    def build_tree(
        self, data: Sequence[Sequence[str]], attributes: Sequence[str]
    ) -> None:
        """Build the decision tree from ``data``.

        Parameters
        ----------
        data:
            Training instances.  Each instance is a sequence whose last element
            is the class label.
        attributes:
            Names of the attributes for the instances.  ``len(attributes)`` must
            be ``len(instance) - 1``.
        """

        if not data:
            raise ValueError("`data` must contain at least one instance")
        if not attributes:
            raise ValueError("`attributes` must contain at least one entry")

        self._attributes = list(attributes)
        self.root = self._build_tree_recursive(list(data), list(attributes))

    # ------------------------------------------------------------------
    def _build_tree_recursive(
        self, data: List[List[str]], attributes: List[str]
    ) -> Node:
        node = Node()
        labels = [row[-1] for row in data]

        # If all instances have the same label we can create a leaf node.
        if len(set(labels)) == 1:
            node.is_leaf = True
            node.label = labels[0]
            return node

        # If there are no attributes left we create a leaf with the majority
        # label.
        if not attributes:
            node.is_leaf = True
            node.label = Counter(labels).most_common(1)[0][0]
            return node

        # Store the majority label for use when classification encounters an
        # unseen value.
        node.label = Counter(labels).most_common(1)[0][0]

        best_index = self._find_best_attribute(data, attributes)
        best_attr = attributes[best_index]
        node.attribute = best_attr

        # Partition the data by the best attribute's values.
        subsets: Dict[str, List[List[str]]] = {}
        for row in data:
            value = row[best_index]
            subsets.setdefault(value, []).append(row)

        # Remove the used attribute for the recursive call.
        new_attrs = attributes[:best_index] + attributes[best_index + 1 :]

        for value, subset in subsets.items():
            # Remove the attribute value from each instance as the attribute is
            # no longer considered deeper in the tree.
            reduced_subset = [
                r[:best_index] + r[best_index + 1 :] for r in subset
            ]
            node.children[value] = self._build_tree_recursive(
                reduced_subset, new_attrs
            )

        return node

    # ------------------------------------------------------------------
    def _calculate_entropy(self, data: Sequence[Sequence[str]]) -> float:
        label_count = Counter(row[-1] for row in data)
        total = float(len(data))
        entropy = 0.0
        for count in label_count.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    # ------------------------------------------------------------------
    def _calculate_information_gain(
        self, data: Sequence[Sequence[str]], attribute_index: int
    ) -> float:
        total_entropy = self._calculate_entropy(data)
        total_instances = float(len(data))

        # Partition by attribute value
        subsets: Dict[str, List[Sequence[str]]] = {}
        for row in data:
            subsets.setdefault(row[attribute_index], []).append(row)

        weighted_entropy = 0.0
        for subset in subsets.values():
            weight = len(subset) / total_instances
            weighted_entropy += weight * self._calculate_entropy(subset)

        return total_entropy - weighted_entropy

    # ------------------------------------------------------------------
    def _find_best_attribute(
        self, data: Sequence[Sequence[str]], attributes: Sequence[str]
    ) -> int:
        gains = [
            self._calculate_information_gain(data, i)
            for i in range(len(attributes))
        ]
        # ``max`` with ``enumerate`` returns the index of the largest gain.
        return max(range(len(attributes)), key=lambda i: gains[i])

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify(self, instance: Sequence[str]) -> str:
        """Classify a previously unseen instance.

        ``instance`` should contain the attribute values in the same order as
        used during training.
        """

        if self.root is None or self._attributes is None:
            raise ValueError("The tree has not been built yet")
        if len(instance) != len(self._attributes):
            raise ValueError(
                "Instance has %d attributes but %d were expected"
                % (len(instance), len(self._attributes))
            )

        node = self.root
        while not node.is_leaf:
            attr = node.attribute
            assert attr is not None  # for type checkers
            index = self._attributes.index(attr)
            value = instance[index]
            if value not in node.children:
                # Unseen attribute value – fall back to the majority label stored
                # at the node.
                return node.label if node.label is not None else ""
            node = node.children[value]

        return node.label if node.label is not None else ""
