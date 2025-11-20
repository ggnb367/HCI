#!/usr/bin/env python3
"""Command-line driver for a simplified Pruned Highway Labeling (PHL) workflow.

This script bundles a lightweight adaptation of the PHL preprocessing and query
routines so it can be executed as a single file. It supports loading either the
Planetoid *PubMed* citation network (via `torch_geometric`) or text-based
knowledge graphs stored in the ``WN18`` / ``WN50k`` format described by the
user. The resulting graph is treated as undirected and edge weights default to
``1.0`` unless explicitly provided.

Example usages::

    python phl.py --dataset pubmed --data-root ./data
    python phl.py --dataset wn18 --data-path ./data/WN18.txt
    python phl.py --dataset wn50k --data-path ./data/WN50k.txt --samples 800

The script reports preprocessing time, average query time over sampled pairs and
mean absolute error (MAE) between the PHL answer and an exact Dijkstra
computation.  The implementation purposefully keeps the decomposition step
simple by selecting hub vertices based on node degree, which allows it to scale
reasonably to the requested datasets while remaining faithful to the pruned
Dijkstra labelling technique.
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


INF = float("inf")


@dataclass
class LabelEntry:
    """Single entry within a PHL label."""

    path: int
    dist_origin: float
    dist_node: float


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def build_graph_from_edges(
    edges: Iterable[Tuple[int, int, float]],
    *,
    num_nodes_hint: Optional[int] = None,
) -> Tuple[List[List[Tuple[int, float]]], Dict[int, int], List[int]]:
    """Construct an undirected adjacency list from an edge iterator.

    Parameters
    ----------
    edges:
        Iterable containing ``(u, v, weight)`` tuples. The function keeps the
        minimum weight when multiple edges connect the same endpoints.
    num_nodes_hint:
        Optional hint for the number of distinct nodes. Only used to pre-size
        containers when provided.

    Returns
    -------
    adjacency:
        List where ``adjacency[u]`` stores ``(v, weight)`` tuples.
    to_index:
        Mapping from original node identifier to the new contiguous index.
    to_node:
        Reverse mapping from contiguous indices back to the original node ids.
    """

    edge_map: Dict[Tuple[int, int], float] = {}
    nodes: set[int] = set()

    for u, v, w in edges:
        if u == v:
            continue
        if w < 0:
            raise ValueError("Negative weight edges are not supported.")
        a, b = (u, v) if u <= v else (v, u)
        nodes.add(a)
        nodes.add(b)
        key = (a, b)
        if key in edge_map:
            edge_map[key] = min(edge_map[key], w)
        else:
            edge_map[key] = w

    ordered_nodes = sorted(nodes)
    to_index = {node: idx for idx, node in enumerate(ordered_nodes)}
    to_node = ordered_nodes

    adjacency: List[List[Tuple[int, float]]] = [
        [] for _ in range(len(ordered_nodes) if num_nodes_hint is None else max(len(ordered_nodes), num_nodes_hint))
    ]

    for (u, v), weight in edge_map.items():
        ui = to_index[u]
        vi = to_index[v]
        adjacency[ui].append((vi, weight))
        adjacency[vi].append((ui, weight))

    # Trim potential over-allocation if a hint was supplied but actual nodes were fewer.
    return adjacency[: len(ordered_nodes)], to_index, to_node


# ---------------------------------------------------------------------------
# Core PHL routines (simplified)
# ---------------------------------------------------------------------------


def query(u: int, v: int, label: Sequence[List[LabelEntry]]) -> float:
    """Return the labelled distance between ``u`` and ``v``.

    If the nodes do not share a labelled path, ``INF`` is returned.
    """

    label_u = label[u]
    label_v = label[v]
    if not label_u or not label_v:
        return INF

    by_path = {entry.path: entry for entry in label_v}
    best = INF
    for entry in label_u:
        match = by_path.get(entry.path)
        if match is None:
            continue
        dist = entry.dist_node + abs(match.dist_origin - entry.dist_origin) + match.dist_node
        if dist < best:
            best = dist
    return best


def pruned_dijkstra_search(
    graph: Sequence[Sequence[Tuple[int, float]]],
    hub: int,
    label: List[List[LabelEntry]],
    path_id: int,
) -> None:
    """Perform a pruned Dijkstra search rooted at ``hub`` and update labels."""

    distances: Dict[int, float] = {hub: 0.0}
    heap: List[Tuple[float, int]] = [(0.0, hub)]

    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > distances[u]:
            continue

        if query(u, hub, label) <= dist_u:
            continue

        label[u].append(LabelEntry(path=path_id, dist_origin=0.0, dist_node=dist_u))

        for v, weight in graph[u]:
            alt = dist_u + weight
            if alt < distances.get(v, INF):
                distances[v] = alt
                heapq.heappush(heap, (alt, v))


def preprocess(
    graph: Sequence[Sequence[Tuple[int, float]]],
    hubs: Sequence[int],
) -> List[List[LabelEntry]]:
    """Run the simplified PHL preprocessing for a collection of hub nodes."""

    label: List[List[LabelEntry]] = [[] for _ in range(len(graph))]
    for path_id, hub in enumerate(hubs):
        pruned_dijkstra_search(graph, hub, label, path_id)
    return label


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def dijkstra(
    graph: Sequence[Sequence[Tuple[int, float]]],
    source: int,
) -> Dict[int, float]:
    """Exact single-source Dijkstra distances."""

    dist: Dict[int, float] = {source: 0.0}
    heap: List[Tuple[float, int]] = [(0.0, source)]

    while heap:
        dist_u, u = heapq.heappop(heap)
        if dist_u > dist[u]:
            continue
        for v, weight in graph[u]:
            alt = dist_u + weight
            if alt < dist.get(v, INF):
                dist[v] = alt
                heapq.heappush(heap, (alt, v))
    return dist


def evaluate(
    graph: Sequence[Sequence[Tuple[int, float]]],
    label: Sequence[List[LabelEntry]],
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[float, float, int]:
    """Compute average query time (seconds) and MAE over sampled pairs."""

    cache: Dict[int, Dict[int, float]] = {}
    total_abs_error = 0.0
    considered = 0
    total_query_time = 0.0

    for u, v in pairs:
        if u == v:
            continue
        if u not in cache:
            cache[u] = dijkstra(graph, u)
        exact = cache[u].get(v, INF)

        start = time.perf_counter()
        predicted = query(u, v, label)
        total_query_time += time.perf_counter() - start

        if math.isinf(exact) and math.isinf(predicted):
            continue
        if math.isinf(exact) and not math.isinf(predicted):
            # Underestimation when the pair is disconnected: count as large error.
            total_abs_error += predicted
            considered += 1
            continue
        if math.isinf(predicted) and not math.isinf(exact):
            total_abs_error += exact
            considered += 1
            continue

        total_abs_error += abs(predicted - exact)
        considered += 1

    average_query_time = total_query_time / max(len(pairs), 1)
    mae = total_abs_error / considered if considered > 0 else INF
    return average_query_time, mae, considered


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_wn_dataset(path: Path) -> Iterable[Tuple[int, int, float]]:
    """Load edges from a WN-style triple file."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf8") as handle:
        first_line = handle.readline()
        try:
            _ = int(first_line.strip())
        except ValueError:
            # The first line might be a triple already.
            handle.seek(0)

        for line in handle:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            src = int(parts[0])
            dst = int(parts[1])
            weight = 1.0
            if len(parts) >= 3:
                try:
                    weight = float(parts[2])
                except ValueError:
                    weight = 1.0
            yield src, dst, weight


def load_pubmed_dataset(root: Path) -> Iterable[Tuple[int, int, float]]:
    """Load the Planetoid PubMed dataset via torch_geometric."""

    try:
        from torch_geometric.datasets import Planetoid
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "torch_geometric is required to load the PubMed dataset. "
            "Install it via 'pip install torch-geometric' and retry."
        ) from exc

    dataset = Planetoid(root=str(root), name="PubMed")
    data = dataset[0]
    edge_index = data.edge_index

    if edge_index.size(0) != 2:
        raise RuntimeError("Unexpected edge_index shape for PubMed dataset.")

    for idx in range(edge_index.size(1)):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        if src == dst:
            continue
        yield src, dst, 1.0


# ---------------------------------------------------------------------------
# Hub selection and sampling
# ---------------------------------------------------------------------------


def select_hubs(graph: Sequence[Sequence[Tuple[int, float]]], max_hubs: Optional[int] = None) -> List[int]:
    """Select hub vertices with a simple degree-based heuristic."""

    degrees = [(idx, len(neighbors)) for idx, neighbors in enumerate(graph)]
    degrees.sort(key=lambda item: item[1], reverse=True)

    if max_hubs is None:
        max_hubs = max(1, int(math.sqrt(len(graph))))
    max_hubs = max(1, min(max_hubs, len(graph)))

    hubs = [idx for idx, _ in degrees[:max_hubs] if len(graph[idx]) > 0]
    if not hubs:
        raise RuntimeError("No suitable hubs found; the graph might be empty.")
    return hubs


def sample_pairs(
    graph: Sequence[Sequence[Tuple[int, float]]],
    sample_size: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """Randomly sample node pairs from the set of non-isolated vertices."""

    candidates = [idx for idx, neighbors in enumerate(graph) if neighbors]
    if len(candidates) < 2:
        raise RuntimeError("Not enough connected vertices to sample pairs.")

    pairs: List[Tuple[int, int]] = []
    for _ in range(sample_size):
        u = rng.choice(candidates)
        v = rng.choice(candidates)
        while v == u:
            v = rng.choice(candidates)
        pairs.append((u, v))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["pubmed", "wn18", "wn50k"],
        required=True,
        help="Dataset identifier to load.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Root directory for Planetoid datasets (PubMed).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the WN-style triple file (WN18 / WN50k).",
    )
    parser.add_argument(
        "--num-hubs",
        type=int,
        default=None,
        help="Maximum number of hub vertices used during preprocessing.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of (source, target) query pairs to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.dataset == "pubmed":
        edges = load_pubmed_dataset(args.data_root)
    else:
        if args.data_path is None:
            default_path = args.data_root / ("WN18.txt" if args.dataset == "wn18" else "WN50k.txt")
            args.data_path = default_path
        edges = load_wn_dataset(args.data_path)

    adjacency, to_index, _ = build_graph_from_edges(edges)
    if not adjacency:
        raise RuntimeError("The constructed graph is empty.")

    rng = random.Random(args.seed)

    start_pre = time.perf_counter()
    hubs = select_hubs(adjacency, args.num_hubs)
    label = preprocess(adjacency, hubs)
    preprocessing_time = time.perf_counter() - start_pre

    pairs = sample_pairs(adjacency, args.samples, rng)
    avg_query_time, mae, considered = evaluate(adjacency, label, pairs)

    print("PHL evaluation summary")
    print("=======================")
    print(f"Nodes: {len(adjacency)}")
    print(f"Edges: {sum(len(neigh) for neigh in adjacency) // 2}")
    print(f"Hubs used: {len(hubs)}")
    print(f"Preprocessing time: {preprocessing_time:.4f} s")
    print(f"Average query time: {avg_query_time * 1e3:.6f} ms over {len(pairs)} pairs")
    if math.isinf(mae):
        print("MAE: not defined (no finite reference distances)")
    else:
        print(f"MAE: {mae:.6f} based on {considered} comparable pairs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
