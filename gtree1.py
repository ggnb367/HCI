#!/usr/bin/env python3
"""Python port of the original G-Tree C++ workflow.

The goal of this module is to mirror the behaviour of ``gtree_build`` and
``gtree_query`` as closely as possible while keeping everything self-contained
in a single Python file.  The implementation purposely keeps the same
terminology as the C++ source (``Nodes``, ``GTree`` etc.) so that the code can
be cross-referenced easily when studying the original project.

The script retains the convenient dataset loaders that were added earlier:
WN18/WN50k text files, Planetoid's PubMed graph and Hugging Face's YAGO3-10
release.  Once the graph is loaded, the build pipeline strictly follows the
original order:

1. build the G-Tree hierarchy using a METIS-style partitioner (``pymetis``);
2. pre-compute border distance matrices for every tree node;
3. answer shortest-path queries using the pre-computed hierarchy while still
   falling back to Dijkstra if anything unexpected occurs.

Progress bars are shown during the expensive phases (dataset ingestion,
pre-computation, query evaluation) so that long runs provide immediate visual
feedback, just like the timing prints in the C++ executables.
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:  # Optional dependency mirroring the C++ METIS usage
    import pymetis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pymetis = None


# ---------------------------------------------------------------------------
# Constants that match the C++ macros
# ---------------------------------------------------------------------------

FILE_NODE = "cal.cnode"
FILE_EDGE = "cal.cedge"
FILE_GTREE = "cal.gtree"
FILE_PATHS = "cal.paths"
FILE_MIND = "cal.minds"
WEIGHT_INFLATE_FACTOR = 100000
ADJWEIGHT_SET_TO_ALL_ONE = True
PARTITION_PART = 4
LEAF_CAP = 32


# ---------------------------------------------------------------------------
# Utility helpers (progress, timing, dijkstra)
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Console progress bar used to mirror the feedback from the C++ tools."""

    def __init__(self, total: int, description: str, *, width: int = 30) -> None:
        self.total = max(int(total), 0)
        self.description = description
        self.width = max(int(width), 10)
        self.count = 0
        self.start = time.perf_counter()
        self.last_print = self.start
        self._closed = False
        self.enabled = self.total > 0
        if self.enabled:
            print(f"{self.description} 开始，共 {self.total} 步…", flush=True)

    def update(self, step: int = 1) -> None:
        if not self.enabled or self._closed:
            return
        self.count = min(self.count + step, self.total)
        now = time.perf_counter()
        if self.count == self.total or (now - self.last_print) >= 0.1:
            ratio = 0.0 if self.total == 0 else self.count / self.total
            filled = int(ratio * self.width)
            bar = "█" * filled + "-" * (self.width - filled)
            print(
                f"\r{self.description} [{bar}] {self.count}/{self.total}",
                end="",
                flush=True,
            )
            self.last_print = now
            if self.count == self.total:
                self.close()

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        duration = time.perf_counter() - self.start
        print(
            f"\r{self.description} 完成 {self.count}/{self.total}"
            f" （{duration:.2f} 秒）" + " " * 10,
            flush=True,
        )


@dataclasses.dataclass
class NodeRecord:
    """Structure equivalent to ``Node`` in the C++ code."""

    x: float = 0.0
    y: float = 0.0
    adjnodes: List[int] = dataclasses.field(default_factory=list)
    adjweight: List[int] = dataclasses.field(default_factory=list)
    isborder: bool = False
    gtreepath: List[int] = dataclasses.field(default_factory=list)

    def copy(self) -> "NodeRecord":
        return NodeRecord(
            x=self.x,
            y=self.y,
            adjnodes=list(self.adjnodes),
            adjweight=list(self.adjweight),
            isborder=self.isborder,
            gtreepath=list(self.gtreepath),
        )


@dataclasses.dataclass
class TreeNodeRecord:
    """Structure mirroring the ``TreeNode`` struct in the C++ project."""

    borders: List[int] = dataclasses.field(default_factory=list)
    children: List[int] = dataclasses.field(default_factory=list)
    isleaf: bool = False
    leafnodes: List[int] = dataclasses.field(default_factory=list)
    father: int = -1
    union_borders: List[int] = dataclasses.field(default_factory=list)
    mind: List[int] = dataclasses.field(default_factory=list)
    # The following arrays are populated after the hierarchy is built
    current_pos: List[int] = dataclasses.field(default_factory=list)
    up_pos: List[int] = dataclasses.field(default_factory=list)


class Graph:
    """Undirected weighted graph used as input for the G-Tree builder."""

    def __init__(self) -> None:
        self.adj: List[List[int]] = []
        self.weight: List[List[int]] = []

    def __len__(self) -> int:
        return len(self.adj)

    def ensure_node(self, nid: int) -> None:
        while len(self.adj) <= nid:
            self.adj.append([])
            self.weight.append([])

    def add_edge(self, u: int, v: int, w: int) -> None:
        self.ensure_node(max(u, v))
        self.adj[u].append(v)
        self.weight[u].append(w)
        self.adj[v].append(u)
        self.weight[v].append(w)

    def neighbours(self, node: int) -> Iterator[Tuple[int, int]]:
        for nid, w in zip(self.adj[node], self.weight[node]):
            yield nid, w


# ---------------------------------------------------------------------------
# Dijkstra routines (global and restricted)
# ---------------------------------------------------------------------------


def dijkstra_full(graph: Graph, source: int, target: int) -> Optional[int]:
    if source == target:
        return 0
    n = len(graph)
    dist = [math.inf] * n
    dist[source] = 0
    pq: List[Tuple[int, int]] = [(0, source)]
    visited = [False] * n
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == target:
            return d
        for v, w in graph.neighbours(u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return None


def dijkstra_restricted(
    graph: List[NodeRecord],
    source: int,
    targets: Sequence[int],
    allowed: Optional[set[int]] = None,
) -> Dict[int, int]:
    """Closest equivalent to ``dijkstra_candidate`` from the C++ sources."""

    todo = set(targets)
    dist: Dict[int, int] = {source: 0}
    visited: set[int] = set()
    pq: List[Tuple[int, int]] = [(0, source)]
    while pq and todo:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u in todo:
            todo.remove(u)
        for v, w in zip(graph[u].adjnodes, graph[u].adjweight):
            if allowed is not None and v not in allowed:
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return {t: dist[t] for t in targets if t in dist}


# ---------------------------------------------------------------------------
# G-Tree construction mirroring the original C++ implementation
# ---------------------------------------------------------------------------


class GTreeIndex:
    def __init__(self, nodes: List[NodeRecord], gtree: List[TreeNodeRecord]) -> None:
        self.Nodes = nodes
        self.GTree = gtree
        self._leaf_pos_cache: Dict[int, Dict[int, int]] = {}
        self._union_pos_cache: Dict[int, Dict[int, int]] = {}
        self._graph = Graph()
        for nid in range(len(nodes)):
            self._graph.ensure_node(nid)
        for nid, node in enumerate(nodes):
            for adj, w in zip(node.adjnodes, node.adjweight):
                if nid <= adj:
                    self._graph.add_edge(nid, adj, w)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        graph: Graph,
        *,
        leaf_cap: int = LEAF_CAP,
        fanout: int = PARTITION_PART,
        precompute_boundaries: bool = True,
    ) -> "GTreeIndex":
        if fanout <= 1:
            raise ValueError("fanout must be >= 2 to construct a hierarchy")
        if pymetis is None:
            raise RuntimeError(
                "pymetis 未安装。请先 `pip install pymetis` 以便使用与原始 C++ 相同的划分逻辑。"
            )

        # ----- init Nodes -----
        nodes = [NodeRecord() for _ in range(len(graph))]
        for nid in range(len(graph)):
            nodes[nid].adjnodes = list(graph.adj[nid])
            nodes[nid].adjweight = list(graph.weight[nid])

        # ----- build tree -----
        gtree: List[TreeNodeRecord] = [TreeNodeRecord()]
        status_stack: List[Tuple[int, set[int]]] = [(0, set(range(len(graph))))]

        build_start = time.perf_counter()
        print("BUILD 阶段: 递归划分图构建 G-Tree…")

        while status_stack:
            tnid, nset = status_stack.pop()
            for nid in nset:
                nodes[nid].gtreepath.append(tnid)

            if len(nset) <= leaf_cap:
                leaf = gtree[tnid]
                leaf.isleaf = True
                leaf.leafnodes = sorted(nset)
                continue

            partition_map = cls._partition_with_metis(nodes, nset, fanout)
            child_sets: List[set[int]] = [set() for _ in range(fanout)]
            for nid in nset:
                slot = partition_map[nid]
                child_sets[slot].add(nid)

            for child_set in child_sets:
                child = TreeNodeRecord()
                child.father = tnid
                child_idx = len(gtree)
                gtree.append(child)
                gtree[tnid].children.append(child_idx)

                borders: List[int] = []
                for nid in child_set:
                    is_border = False
                    for adj in nodes[nid].adjnodes:
                        if adj not in child_set:
                            is_border = True
                            break
                    if is_border:
                        borders.append(nid)
                        nodes[nid].isborder = True
                child.borders = sorted(borders)

                if child_set:
                    status_stack.append((child_idx, child_set))
                else:
                    child.isleaf = True
                    child.leafnodes = []

        build_time = time.perf_counter() - build_start
        print(f"BUILD 阶段完成，用时 {build_time:.2f} 秒。")

        index = cls(nodes, gtree)

        if precompute_boundaries:
            index._hierarchy_shortest_path_calculation()
        else:
            print("警告: 未进行边界距离预计算，查询将退化为全图 Dijkstra。")

        index._prepare_border_position_tables()
        return index

    @staticmethod
    def _partition_with_metis(
        nodes: List[NodeRecord],
        nset: set[int],
        fanout: int,
    ) -> Dict[int, int]:
        order = sorted(nset)
        position = {nid: idx for idx, nid in enumerate(order)}
        adjacency: List[List[int]] = []
        for nid in order:
            neighs = [position[v] for v in nodes[nid].adjnodes if v in nset]
            adjacency.append(neighs)
        _, part = pymetis.part_graph(fanout, adjacency=adjacency)
        return {order[i]: part[i] for i in range(len(order))}

    # ------------------------------------------------------------------
    # Pre-computation of boundary distances
    # ------------------------------------------------------------------

    def _hierarchy_shortest_path_calculation(self) -> None:
        print("MIND 阶段: 预计算各层边界距离…")
        level_nodes: List[List[int]] = []
        current = [0]
        while current:
            level_nodes.append(current)
            nxt: List[int] = []
            for tn in current:
                nxt.extend(self.GTree[tn].children)
            current = nxt

        temp_graph = [node.copy() for node in self.Nodes]
        vertex_pairs: Dict[int, Dict[int, int]] = {}
        total_work = sum(len(level) for level in level_nodes)
        tracker = ProgressTracker(total_work, "预计算边界")

        for depth in reversed(range(len(level_nodes))):
            for tn in level_nodes[depth]:
                tracker.update()
                gtn = self.GTree[tn]
                if gtn.isleaf:
                    candidates = gtn.leafnodes
                    gtn.union_borders = list(gtn.borders)
                else:
                    union_set: set[int] = set()
                    for child in gtn.children:
                        union_set.update(self.GTree[child].borders)
                    candidates = sorted(union_set)
                    gtn.union_borders = list(candidates)

                vertex_pairs.clear()
                gtn.mind = []
                for border in gtn.union_borders:
                    result = dijkstra_restricted(temp_graph, border, candidates)
                    row: List[int] = []
                    row_map: Dict[int, int] = {}
                    for c in candidates:
                        val = result.get(c, math.inf)
                        row.append(val)
                        if val < math.inf:
                            row_map[c] = val
                    gtn.mind.extend(row)
                    vertex_pairs[border] = row_map

                # Degenerate the graph exactly as in the C++ code
                if gtn.borders:
                    border_set = set(gtn.borders)
                    for border in gtn.borders:
                        filtered_nodes: List[int] = []
                        filtered_weight: List[int] = []
                        for adj, w in zip(temp_graph[border].adjnodes, temp_graph[border].adjweight):
                            if len(self.Nodes[adj].gtreepath) <= depth or self.Nodes[adj].gtreepath[depth] != tn:
                                filtered_nodes.append(adj)
                                filtered_weight.append(w)
                        temp_graph[border].adjnodes = filtered_nodes
                        temp_graph[border].adjweight = filtered_weight

                    for a in gtn.borders:
                        for b in gtn.borders:
                            if a == b:
                                continue
                            if b in vertex_pairs.get(a, {}):
                                temp_graph[a].adjnodes.append(b)
                                temp_graph[a].adjweight.append(vertex_pairs[a][b])
        tracker.close()
        print("MIND 阶段完成。")

    def _prepare_border_position_tables(self) -> None:
        for tnid, tnode in enumerate(self.GTree):
            if tnode.isleaf:
                self._leaf_pos_cache[tnid] = {nid: idx for idx, nid in enumerate(tnode.leafnodes)}
            self._union_pos_cache[tnid] = {nid: idx for idx, nid in enumerate(tnode.union_borders)}
            current_pos: List[int] = []
            for border in tnode.borders:
                current_pos.append(self._union_pos_cache[tnid].get(border, -1))
            tnode.current_pos = current_pos
            if tnode.father != -1:
                parent_union = self._union_pos_cache[tnode.father]
                tnode.up_pos = [parent_union.get(border, -1) for border in tnode.borders]
            else:
                tnode.up_pos = []

    # ------------------------------------------------------------------
    # Query interfaces
    # ------------------------------------------------------------------

    def _border_dp(self, node_id: int) -> Dict[int, List[int]]:
        path = self.Nodes[node_id].gtreepath
        dp: Dict[int, List[int]] = {}
        leaf_tn = path[-1]
        leaf = self.GTree[leaf_tn]
        if leaf_tn not in self._leaf_pos_cache:
            raise RuntimeError("leaf position cache missing")
        pos = self._leaf_pos_cache[leaf_tn].get(node_id)
        if pos is None:
            raise RuntimeError("节点不在所属叶子节点中")
        distances = []
        width = len(leaf.leafnodes) if leaf.leafnodes else 1
        for idx in range(len(leaf.borders)):
            offset = idx * width + pos
            distances.append(leaf.mind[offset])
        dp[leaf_tn] = distances

        for depth in range(len(path) - 2, -1, -1):
            tn = path[depth]
            child = path[depth + 1]
            parent_node = self.GTree[tn]
            child_node = self.GTree[child]
            child_dist = dp[child]
            arr: List[int] = []
            union_size = len(parent_node.union_borders)
            for idx, border in enumerate(parent_node.borders):
                posa = parent_node.current_pos[idx]
                best = math.inf
                for jdx in range(len(child_node.borders)):
                    posb = child_node.up_pos[jdx]
                    if posa == -1 or posb == -1:
                        continue
                    mind_idx = posa * union_size + posb
                    value = child_dist[jdx] + parent_node.mind[mind_idx]
                    if value < best:
                        best = value
                arr.append(best)
            dp[tn] = arr
        return dp

    def _combine_at_lca(
        self,
        lca: int,
        dp_s: Dict[int, List[int]],
        dp_t: Dict[int, List[int]],
        source: int,
        target: int,
    ) -> Optional[int]:
        node = self.GTree[lca]
        if node.isleaf:
            allowed = set(node.leafnodes)
            if source not in allowed or target not in allowed:
                return None
            res = dijkstra_restricted(self.Nodes, source, [target], allowed)
            return res.get(target)

        union_size = len(node.union_borders)
        if union_size == 0:
            return None
        best = math.inf
        s_dist = dp_s[lca]
        t_dist = dp_t[lca]
        for i, border_s in enumerate(node.borders):
            posa = node.current_pos[i]
            if posa == -1:
                continue
            for j, border_t in enumerate(node.borders):
                posb = node.current_pos[j]
                if posb == -1:
                    continue
                mid = node.mind[posa * union_size + posb]
                total = s_dist[i] + mid + t_dist[j]
                if total < best:
                    best = total
        return None if best == math.inf else best

    def shortest_path_distance(self, source: int, target: int) -> Optional[float]:
        if source < 0 or target < 0 or source >= len(self.Nodes) or target >= len(self.Nodes):
            return None
        if source == target:
            return 0.0
        try:
            dp_s = self._border_dp(source)
            dp_t = self._border_dp(target)
        except Exception:
            dist = dijkstra_full(self._graph, source, target)
            return None if dist is None else float(dist)

        path_s = self.Nodes[source].gtreepath
        path_t = self.Nodes[target].gtreepath
        lca_depth = 0
        while (
            lca_depth < len(path_s)
            and lca_depth < len(path_t)
            and path_s[lca_depth] == path_t[lca_depth]
        ):
            lca_depth += 1
        lca_depth -= 1
        lca = path_s[lca_depth]

        combined = self._combine_at_lca(lca, dp_s, dp_t, source, target)
        if combined is None:
            dist = dijkstra_full(self._graph, source, target)
            return None if dist is None else float(dist)
        return float(combined)

    # Helper to rebuild Graph view on demand
    def _to_graph(self) -> Graph:
        return self._graph


# ---------------------------------------------------------------------------
# Dataset ingestion utilities
# ---------------------------------------------------------------------------


def load_knowledge_graph(
    path: Path,
    *,
    weight_strategy: str = "relation",
) -> Tuple[Graph, Dict[int, int]]:
    """Load WN-style triples and convert them to a symmetric graph."""

    graph = Graph()
    entity_to_id: Dict[str, int] = {}
    mapping: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip()
        try:
            total = int(first)
        except ValueError as exc:  # pragma: no cover - input guard
            raise ValueError("知识图谱文件第一行必须是三元组数量") from exc
        tracker = ProgressTracker(total, "读取三元组")
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            src_raw, dst_raw, rel_raw = parts
            if src_raw not in entity_to_id:
                entity_to_id[src_raw] = len(entity_to_id)
            if dst_raw not in entity_to_id:
                entity_to_id[dst_raw] = len(entity_to_id)
            src = entity_to_id[src_raw]
            dst = entity_to_id[dst_raw]
            if weight_strategy == "unit":
                weight = 1
            else:
                try:
                    weight = int(rel_raw)
                except ValueError:
                    weight = abs(hash(rel_raw)) % 1000 + 1
            graph.add_edge(src, dst, weight)
            tracker.update()
        tracker.close()
    mapping = {idx: raw for raw, idx in entity_to_id.items()}
    return graph, mapping


def load_planetoid_pubmed(root: Path) -> Tuple[Graph, Dict[int, int]]:
    try:
        from torch_geometric.datasets import Planetoid  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("需要安装 torch-geometric 才能加载 Planetoid 数据集") from exc
    dataset = Planetoid(root=str(root), name="PubMed")
    data = dataset[0]
    graph = Graph()
    edges = data.edge_index.numpy().T
    for src, dst in edges:
        graph.add_edge(int(src), int(dst), 1)
    mapping = {i: i for i in range(len(graph))}
    return graph, mapping


def load_yago3(split: str) -> Tuple[Graph, Dict[int, int]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("需要安装 datasets 才能加载 Hugging Face YAGO3-10") from exc
    ds = load_dataset("marius-team/yago3-10", split=split)
    graph = Graph()
    entity_to_id: Dict[int, int] = {}
    tracker = ProgressTracker(len(ds), f"读取 YAGO3-10({split})")
    for row in ds:
        src_raw = row["head"]
        dst_raw = row["tail"]
        rel_raw = row["relation"]
        if src_raw not in entity_to_id:
            entity_to_id[src_raw] = len(entity_to_id)
        if dst_raw not in entity_to_id:
            entity_to_id[dst_raw] = len(entity_to_id)
        src = entity_to_id[src_raw]
        dst = entity_to_id[dst_raw]
        graph.add_edge(src, dst, int(rel_raw) + 1)
        tracker.update()
    tracker.close()
    mapping = {idx: ent for ent, idx in entity_to_id.items()}
    return graph, mapping


# ---------------------------------------------------------------------------
# Evaluation logic (500 random samples, timings, MAE)
# ---------------------------------------------------------------------------


def evaluate_random_queries(
    index: GTreeIndex,
    graph: Graph,
    samples: int,
    *,
    seed: int = 13,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(graph)
    dijkstra_times: List[float] = []
    gtree_times: List[float] = []
    errors: List[float] = []
    tracker = ProgressTracker(samples, "随机查询测试")
    for _ in range(samples):
        s = rng.randrange(n)
        t = rng.randrange(n)
        start = time.perf_counter()
        gtree_dist = index.shortest_path_distance(s, t)
        gtree_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        dijkstra_dist_raw = dijkstra_full(graph, s, t)
        dijkstra_times.append(time.perf_counter() - start)
        if gtree_dist is None or dijkstra_dist_raw is None:
            errors.append(float("inf"))
        else:
            dijkstra_dist = float(dijkstra_dist_raw)
            errors.append(abs(gtree_dist - dijkstra_dist))
        tracker.update()
    tracker.close()
    return (
        statistics.mean(gtree_times),
        statistics.mean(dijkstra_times),
        statistics.mean(errors),
    )


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def build_index_cmd(args: argparse.Namespace) -> None:
    dataset = args.dataset
    if dataset == "wn18":
        path = Path(args.data or "./data/WN18.txt")
        graph, _ = load_knowledge_graph(path, weight_strategy=args.weight_strategy)
    elif dataset == "wn50k":
        path = Path(args.data or "./data/WN50k.txt")
        graph, _ = load_knowledge_graph(path, weight_strategy=args.weight_strategy)
    elif dataset == "pubmed":
        graph, _ = load_planetoid_pubmed(Path(args.planetoid_root or "./planetoid"))
    elif dataset == "yago3-10":
        graph, _ = load_yago3(args.hf_split)
    else:
        graph, _ = load_knowledge_graph(Path(args.data), weight_strategy=args.weight_strategy)

    init_start = time.perf_counter()
    index = GTreeIndex.build(
        graph,
        leaf_cap=args.leaf_size,
        fanout=args.fanout,
        precompute_boundaries=not args.no_precompute,
    )
    init_time = time.perf_counter() - init_start
    print(f"索引构建完成，总耗时 {init_time:.2f} 秒。")

    if args.output:
        with open(args.output, "wb") as fh:
            import pickle

            pickle.dump(index, fh)
        print(f"索引已保存至 {args.output}")


def evaluate_cmd(args: argparse.Namespace) -> None:
    dataset = args.dataset
    if dataset == "wn18":
        path = Path(args.data or "./data/WN18.txt")
        graph, _ = load_knowledge_graph(path, weight_strategy=args.weight_strategy)
    elif dataset == "wn50k":
        path = Path(args.data or "./data/WN50k.txt")
        graph, _ = load_knowledge_graph(path, weight_strategy=args.weight_strategy)
    elif dataset == "pubmed":
        graph, _ = load_planetoid_pubmed(Path(args.planetoid_root or "./planetoid"))
    elif dataset == "yago3-10":
        graph, _ = load_yago3(args.hf_split)
    else:
        graph, _ = load_knowledge_graph(Path(args.data), weight_strategy=args.weight_strategy)

    start = time.perf_counter()
    index = GTreeIndex.build(
        graph,
        leaf_cap=args.leaf_size,
        fanout=args.fanout,
        precompute_boundaries=not args.no_precompute,
    )
    preprocess_time = time.perf_counter() - start
    gtree_t, dijkstra_t, mae = evaluate_random_queries(index, graph, args.samples)
    print("评估结果：")
    print(f"  预处理耗时: {preprocess_time:.2f} 秒")
    print(f"  G-Tree 平均查询耗时: {gtree_t * 1000:.3f} ms")
    print(f"  Dijkstra 平均查询耗时: {dijkstra_t * 1000:.3f} ms")
    print(f"  平均绝对误差 (MAE): {mae:.6f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G-Tree 原版逻辑的 Python 复刻")
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", choices=["wn18", "wn50k", "pubmed", "yago3-10", "custom"], default="wn18")
    common.add_argument("--data")
    common.add_argument("--weight-strategy", choices=["relation", "unit"], default="relation")
    common.add_argument("--planetoid-root")
    common.add_argument("--hf-split", default="train")
    common.add_argument("--leaf-size", type=int, default=LEAF_CAP)
    common.add_argument("--fanout", type=int, default=PARTITION_PART)
    common.add_argument("--no-precompute", action="store_true")

    build_parser = sub.add_parser("build", parents=[common], help="仅构建索引")
    build_parser.add_argument("--output", help="将索引序列化保存到指定路径")
    build_parser.set_defaults(func=build_index_cmd)

    eval_parser = sub.add_parser("evaluate", parents=[common], help="构建并评估 G-Tree")
    eval_parser.add_argument("--samples", type=int, default=500)
    eval_parser.set_defaults(func=evaluate_cmd)

    eval_parser_alias = sub.add_parser(
        "evaluate-kg",
        parents=[common],
        help="兼容旧命令：构建并评估 G-Tree",
    )
    eval_parser_alias.add_argument("--samples", type=int, default=500)
    eval_parser_alias.set_defaults(func=evaluate_cmd)

    eval_parser_wn = sub.add_parser(
        "evaluate-wn18",
        parents=[common],
        help="兼容旧命令：针对 WN18 的评估",
    )
    eval_parser_wn.add_argument("--samples", type=int, default=500)
    eval_parser_wn.set_defaults(func=evaluate_cmd)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()