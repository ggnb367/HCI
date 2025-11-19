#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAGO3-10 分层 Leiden（modularity）+ 叶簇收缩图 + GH 树分层
并新增：
- 整图节点平均度数与度数分布导出（支持 log–log、不等距对数分桶）
- 可选将 relation 当作节点（bipartite：h—r—t），但在度数分布统计中可排除 relation 节点

运行示例：
  python3 yago310_hl_gh_degree_loglog.py \
      --degree_loglog \
      --degree_log_bins \
      --degree_bins 80

若需要将 relation 作为节点（而度分布只统计实体degree）：
  python3 yago310_hl_gh_degree_loglog.py \
      --relation_as_nodes \
      --exclude_relation_in_degree \
      --degree_loglog --degree_log_bins

依赖：matplotlib, datasets, graspologic, networkx, numpy, tqdm
"""

from __future__ import annotations
import os
import json
import csv
from collections import defaultdict
from typing import Callable, Dict, Any

import networkx as nx
import numpy as np
from tqdm import tqdm

# ---- 绘图（无显示环境）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 依赖检查
try:
    from datasets import load_dataset
except Exception as e:  # pragma: no cover
    raise RuntimeError("请先安装 datasets： pip install datasets") from e

try:
    from graspologic.partition import hierarchical_leiden
except Exception as e:  # pragma: no cover
    raise RuntimeError("请先安装 graspologic： pip install graspologic") from e


# ----------------------- 工具：relation 节点命名 -----------------------
REL_PREFIX = "REL::"


def is_relation_node(n: str) -> bool:
    return isinstance(n, str) and n.startswith(REL_PREFIX)


# ----------------------- 数据加载 -----------------------

def load_yago310_from_hf(
    cache_dir: str | None = None,
    relation_as_nodes: bool = False,
    drop_self_loops: bool = True,
) -> nx.Graph:
    """
    从 HF 加载 YAGO3-10：VLyb/YAGO3-10
    relation_as_nodes=False：将多关系并到实体-实体边属性 {rels,multi}
    relation_as_nodes=True ：将 relation 作为节点（bipartite：h—r—t）
    """
    print("[INFO] Loading YAGO3-10 from Hugging Face (VLyb/YAGO3-10) ...")
    ds = load_dataset("VLyb/YAGO3-10", cache_dir=cache_dir)
    triples = []
    for split in ("train", "validation", "test"):
        if split in ds:
            for row in ds[split]:
                h = str(row["head"]) ; r = str(row["relation"]) ; t = str(row["tail"])
                if drop_self_loops and h == t:
                    continue
                triples.append((h, r, t))
    print(f"[INFO] Total triples loaded: {len(triples):,}")

    if not relation_as_nodes:
        # 聚合多关系为单条无向边（默认权重 1）
        pair2rels = defaultdict(set)
        for h, r, t in triples:
            a, b = (h, t) if h < t else (t, h)
            pair2rels[(a, b)].add(r)
        G = nx.Graph()
        for (u, v), rels in pair2rels.items():
            G.add_edge(u, v, weight=1.0, rels=tuple(sorted(rels)), multi=len(rels))
    else:
        # 构造二部图：实体—关系—实体（无向）
        G = nx.Graph()
        for h, r, t in triples:
            rn = REL_PREFIX + r
            G.add_edge(h, rn)
            G.add_edge(rn, t)

    # 移除孤立点
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        print(f"[INFO] Removed {len(isolates)} isolated nodes.")

    print(f"[INFO] Graph constructed: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
    return G


# ----------------------- 从 HL 压缩“最终簇”映射 -----------------------

def final_partition_from_hl(hl_result, G_sub: nx.Graph) -> dict:
    """将 graspologic 的层次输出压缩为 node->local_cluster_id（仅取最终簇）。"""
    node_to_lc: Dict[Any, Any] = {}
    for h in hl_result:
        if getattr(h, "is_final_cluster", False):
            node_to_lc[h.node] = (h.level, h.cluster)

    # 兜底：若有节点未出现，设为独立簇
    for n in G_sub.nodes():
        if n not in node_to_lc:
            node_to_lc[n] = (-1, f"iso_{n}")

    lc2cid, next_id = {}, 0
    part = {}
    for n, lc in node_to_lc.items():
        if lc not in lc2cid:
            lc2cid[lc] = next_id
            next_id += 1
        part[n] = lc2cid[lc]
    return part  # node -> 0..K-1


# ----------------------- 从 HL 构建“树状层级关系（原始 HL）” -----------------------

def build_tree_from_hl(hl_result, comp_id: int) -> dict:
    membership: dict[str, dict[int, int | str]] = defaultdict(dict)
    cluster_size: dict[tuple[int, int | str], int] = defaultdict(int)
    levels_present: set[int] = set()

    for h in hl_result:
        l = int(h.level)
        c = h.cluster
        n = h.node
        levels_present.add(l)
        if l not in membership[n]:
            membership[n][l] = c
            cluster_size[(l, c)] += 1

    edges_set: set[tuple[tuple[int, int | str], tuple[int, int | str]]] = set()
    if levels_present:
        max_level = max(levels_present)
        for n, lv_map in membership.items():
            for l in range(0, max_level):
                if (l in lv_map) and ((l + 1) in lv_map):
                    parent = (l, lv_map[l])
                    child = (l + 1, lv_map[l + 1])
                    edges_set.add((parent, child))

    def node_id(level: int, cluster: int | str) -> str:
        return f"{comp_id}:{level}:{cluster}"

    nodes_out = []
    levels_counts = defaultdict(int)
    for (l, c), sz in cluster_size.items():
        levels_counts[l] += 1
        nodes_out.append({
            "id": node_id(l, c),
            "comp": comp_id,
            "level": l,
            "cluster": c,
            "size": int(sz),
        })

    edges_out = [{"parent": node_id(pl, pc), "child": node_id(cl, cc)} for (pl, pc), (cl, cc) in sorted(edges_set)]

    return {
        "component": comp_id,
        "nodes": nodes_out,
        "edges": edges_out,
        "levels": {str(k): int(v) for k, v in sorted(levels_counts.items())},
    }


# ----------------------- 单分量跑 HL（modularity） -----------------------

def run_hierarchical_leiden_component(
    G_sub: nx.Graph,
    comp_id: int,
    resolution: float,
    max_cluster_size: int,
    seed: int
) -> tuple[dict, dict]:
    """
    在单个连通分量上执行 hierarchical_leiden（modularity）
    返回:
      - local_part: node -> f"{comp_id}_{local_cid}"（最终叶簇）
      - tree_dict: 该分量的树状结构（用于导出 JSON）
    """
    print(f"[INFO] ▶ Running Leiden (modularity) on component #{comp_id} (|V|={len(G_sub)}) ...")
    hl = hierarchical_leiden(
        G_sub,
        use_modularity=True,
        resolution=resolution,
        max_cluster_size=max_cluster_size,
        random_seed=seed,
        check_directed=False,
    )

    local_part0toK = final_partition_from_hl(hl, G_sub)   # node -> 0..K-1
    local_part = {n: f"{comp_id}_{cid}" for n, cid in local_part0toK.items()}
    tree_dict = build_tree_from_hl(hl, comp_id=comp_id)

    print(f"[INFO] Component #{comp_id}: levels={list(tree_dict['levels'].keys())}, "
          f"clusters_per_level={list(tree_dict['levels'].values())}, "
          f"edges={len(tree_dict['edges'])}")
    return local_part, tree_dict


# ----------------------- 统计（叶簇内部） -----------------------

def analyze_clusters(G: nx.Graph, partition: dict):
    cid2nodes = defaultdict(set)
    for n, c in partition.items():
        cid2nodes[c].add(n)

    results = []
    for cid, nodes in tqdm(cid2nodes.items(), desc="Analyzing clusters"):
        subg = G.subgraph(nodes)
        internal_edges = subg.number_of_edges()
        borders = sum(
            1 for n in nodes
            if any((nbr not in nodes) for nbr in G.neighbors(n))
        )
        results.append({
            "cid": cid,
            "num_nodes": len(nodes),
            "num_edges": internal_edges,
            "num_borders": borders
        })
    return results


# ----------------------- 构建“叶簇收缩图” -----------------------

def build_leaf_graph(G: nx.Graph, partition: dict) -> nx.Graph:
    """
    根据最终叶簇划分 'partition: node->cluster_id' 构建收缩图：
      - 每个叶簇是一个点（节点属性 size=簇大小）
      - 对所有跨簇的原始边聚合为一条边，记录：
          weight_sum: 原权重之和（默认 1.0）
          edge_count: 跨簇边条数
          multi_sum : 原 multi 之和（聚合关系类型数量的总和）
    """
    # 簇大小
    cluster_size = defaultdict(int)
    for n, c in partition.items():
        cluster_size[c] += 1

    # 边聚合
    agg = defaultdict(lambda: {"weight_sum": 0.0, "edge_count": 0, "multi_sum": 0})
    skipped_edges = 0
    for u, v, data in G.edges(data=True):
        cu = partition.get(u) ; cv = partition.get(v)
        if (cu is None) or (cv is None):
            skipped_edges += 1
            continue
        if cu == cv:
            continue
        a, b = (cu, cv) if str(cu) < str(cv) else (cv, cu)
        entry = agg[(a, b)]
        entry["weight_sum"] += float(data.get("weight", 1.0))
        entry["edge_count"] += 1
        entry["multi_sum"]  += int(data.get("multi", 1))

    # 生成叶簇图
    LG = nx.Graph()
    for cid, sz in cluster_size.items():
        LG.add_node(cid, size=int(sz))
    for (a, b), val in agg.items():
        LG.add_edge(a, b, **val)

    print(f"[INFO] Leaf graph: |V|={LG.number_of_nodes()}, |E|={LG.number_of_edges()} (skipped_original_edges={skipped_edges})")
    return LG


def leaf_graph_topk_by_degree(LG: nx.Graph, top_k: int = 10):
    deg = dict(LG.degree())
    weighted_deg = {}
    for n in LG.nodes():
        s = 0.0
        for nbr in LG.neighbors(n):
            s += float(LG[n][nbr].get("weight_sum", 1.0))
        weighted_deg[n] = s
    size_attr = {n: int(LG.nodes[n].get("size", 0)) for n in LG.nodes()}
    order = sorted(LG.nodes(), key=lambda x: (-deg[x], -weighted_deg[x], str(x)))
    top = order[:top_k]
    rows = []
    for i, cid in enumerate(top, 1):
        rows.append({
            "rank": i,
            "cluster_id": cid,
            "degree": deg[cid],
            "weighted_degree": weighted_deg[cid],
            "size": size_attr.get(cid, 0)
        })
    return rows


def save_leaf_graph_as_csv(LG: nx.Graph, out_prefix: str):
    nodes_csv = f"{out_prefix}_nodes.csv"
    edges_csv = f"{out_prefix}_edges.csv"

    with open(nodes_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["cluster_id", "size"])
        for n, data in LG.nodes(data=True):
            w.writerow([n, int(data.get("size", 0))])

    with open(edges_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["u", "v", "weight_sum", "edge_count", "multi_sum"])
        for u, v, data in LG.edges(data=True):
            w.writerow([
                u, v,
                float(data.get("weight_sum", 0.0)),
                int(data.get("edge_count", 0)),
                int(data.get("multi_sum", 0)),
            ])

    print(f"[INFO] Saved leaf graph nodes to: {nodes_csv}")
    print(f"[INFO] Saved leaf graph edges to: {edges_csv}")


# ----------------------- GH 树：构建 & 导出 -----------------------

def build_gomory_hu_tree(LG: nx.Graph, capacity_attr: str = "weight_sum") -> nx.Graph:
    Gcap = nx.Graph()
    Gcap.add_nodes_from(LG.nodes(data=True))
    for u, v, d in LG.edges(data=True):
        cap = float(d.get(capacity_attr, 1.0))
        cap = max(cap, 0.0)
        Gcap.add_edge(u, v, capacity=cap)
    T = nx.gomory_hu_tree(Gcap, capacity="capacity")
    return T


def export_tree_json(T: nx.Graph, path: str, root=None):
    if root is None:
        def wdeg(n):
            return sum(float(T[n][nbr].get("weight", 1.0)) for nbr in T.neighbors(n))
        root = max(T.nodes(), key=wdeg)

    parent = {root: None}
    from collections import deque
    dq = deque([root])
    visited = {root}
    edges = []
    while dq:
        u = dq.popleft()
        for v in T.neighbors(u):
            if v in visited:
                continue
            visited.add(v)
            w = float(T[u][v].get("weight", 1.0))
            edges.append({"parent": u, "child": v, "mincut": w})
            dq.append(v)

    out = {"root": root, "nodes": [{"id": n} for n in T.nodes()], "edges": edges}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[INFO] Saved GH tree to: {path}")


# ----------------------- GH 树 → 分层：按最小割从小到大切 -----------------------

def gh_levels_by_cuts(T: nx.Graph):
    edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in T.edges(data=True)]
    if not edges:
        only = next(iter(T.nodes()))
        return [{"level": 0, "cut_weight": None, "num_clusters": 1, "partition": {only: 0}}]

    weights = sorted(set(w for _, _, w in edges))
    levels = []
    H = T.copy()
    levels.append({"level": 0, "cut_weight": None, "num_clusters": 1, "partition": {n: 0 for n in H.nodes()}})

    w2edges = defaultdict(list)
    for u, v, w in edges:
        w2edges[w].append((u, v))

    level_id = 1
    for w in weights:
        H.remove_edges_from(w2edges[w])
        comps = list(nx.connected_components(H))
        partition = {}
        for bid, comp_nodes in enumerate(comps):
            for n in comp_nodes:
                partition[n] = bid
        levels.append({"level": level_id, "cut_weight": w, "num_clusters": len(comps), "partition": partition})
        level_id += 1

    return levels


# ----------------------- 每层的跨 cluster 边界统计 -----------------------

def compute_level_border_stats(LG: nx.Graph, partition: dict):
    border_pairs = 0
    border_edge_count = 0
    border_weight_sum = 0.0

    block_border_edge_count = defaultdict(int)
    block_border_weight_sum = defaultdict(float)
    block_leaf_size = defaultdict(int)
    block_entity_size = defaultdict(int)

    for n in LG.nodes():
        b = partition.get(n)
        if b is None:
            continue
        block_leaf_size[b] += 1
        block_entity_size[b] += int(LG.nodes[n].get("size", 0))

    for u, v, d in LG.edges(data=True):
        bu = partition.get(u); bv = partition.get(v)
        if bu is None or bv is None:
            continue
        if bu != bv:
            border_pairs += 1
            ec = int(d.get("edge_count", 0))
            ws = float(d.get("weight_sum", 0.0))
            border_edge_count += ec
            border_weight_sum += ws
            block_border_edge_count[bu] += ec
            block_border_edge_count[bv] += ec
            block_border_weight_sum[bu] += ws
            block_border_weight_sum[bv] += ws

    per_block = []
    for b in sorted(block_leaf_size.keys()):
        per_block.append({
            "block_id": b,
            "leaf_clusters": block_leaf_size[b],
            "entities": block_entity_size[b],
            "border_edge_count": block_border_edge_count.get(b, 0),
            "border_weight_sum": block_border_weight_sum.get(b, 0.0),
        })

    return {
        "border_pairs": border_pairs,
        "border_edge_count": border_edge_count,
        "border_weight_sum": border_weight_sum,
        "per_block": per_block,
    }


# ----------------------- 新增：整图度数统计与分布图导出 -----------------------

def compute_and_save_degree_stats(
    G: nx.Graph,
    out_img_path: str,
    out_csv_path: str,
    bins: int = 50,
    loglog: bool = True,
    log_bins: bool = False,
    node_include_pred: Callable[[str], bool] | None = None,
) -> dict:
    """
    统计整图 G 的节点度数并导出：
      - 平均度数（2|E|/|V|，在“被统计的节点子集”上近似）
      - 直方图：支持对数不等距分桶（logspace bins）与 y 轴 log（即 log–log 展示）
      - CSV：degree -> count
      - 附加两张更适合长尾分布的图：
          1) CCDF（P(Degree ≥ x)）——看尾部更清楚
          2) Rank–Frequency（Zipf 图）：按度降序排序后的 (rank, degree)
      - 附加一个 tail 概览 CSV：top-K 度、若干分位数

    node_include_pred：若提供，仅统计返回 True 的节点（如排除 relation 节点）。
    """
    if node_include_pred is None:
        node_include_pred = lambda x: True

    # 仅统计被选择的节点的度（按原图度）
    selected_nodes = [n for n in G.nodes() if node_include_pred(n)]
    deg_map = dict(G.degree(selected_nodes))
    degrees = np.array(list(deg_map.values()), dtype=np.int64)

    n = len(selected_nodes)
    m = float(degrees.sum()) / 2.0 if n > 0 else 0.0  # 仅就被选子图的度和估算对应边数
    avg_deg = (2.0 * m / n) if n > 0 else 0.0
    max_deg = int(degrees.max()) if degrees.size else 0

    # ---- CSV: 度 -> 频次
    from collections import Counter
    cnt = Counter(degrees.tolist())
    with open(out_csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["degree", "count"])
        for k in sorted(cnt.keys()):
            w.writerow([k, cnt[k]])

    # ---- 直方图
    plt.figure(figsize=(7, 5))
    if degrees.size:
        if log_bins:
            # 对数不等距分桶（忽略 0 度，因为 log 无法定义）
            pos = degrees[degrees > 0]
            if pos.size == 0:
                edges = np.arange(0.5, 1.5, 1.0)
            else:
                min_pos = int(pos.min())
                max_pos = int(pos.max())
                edges = np.logspace(np.log10(min_pos), np.log10(max_pos), bins)
            plt.hist(degrees, bins=edges)
            plt.xscale("log")
        else:
            plt.hist(degrees, bins=bins)
        if loglog:
            plt.yscale("log")
    plt.xlabel("Node Degree" + (" (log)" if log_bins else ""))
    plt.ylabel("Count" + (" (log)" if loglog else ""))
    title = f"Degree Distribution (n={n}, avg={avg_deg:.2f})"
    if log_bins and loglog:
        title = f"Degree Distribution (log–log) (n={n}, avg={avg_deg:.2f})"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_img_path, dpi=160)
    plt.close()

    # ---- CCDF：P(Degree ≥ x)
    ccdf_path = out_img_path.replace('.png', '_ccdf.png')
    if degrees.size:
        xs = np.sort(np.unique(degrees))
        tail_counts = np.array([(degrees >= x).sum() for x in xs], dtype=np.float64)
        ccdf = tail_counts / float(n)
        plt.figure(figsize=(7, 5))
        plt.plot(xs, ccdf)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Degree (log)')
        plt.ylabel('P(Degree ≥ x) (log)')
        plt.title('Degree CCDF (log–log)')
        plt.tight_layout()
        plt.savefig(ccdf_path, dpi=160)
        plt.close()

    # ---- Rank–Frequency（Zipf）：按度降序
    rank_path = out_img_path.replace('.png', '_rank.png')
    if degrees.size:
        sdeg = np.sort(degrees)[::-1]
        ranks = np.arange(1, sdeg.size + 1)
        plt.figure(figsize=(7, 5))
        plt.plot(ranks, sdeg)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Rank (log)')
        plt.ylabel('Degree (log)')
        plt.title('Rank–Frequency (Zipf) of Degrees (log–log)')
        plt.tight_layout()
        plt.savefig(rank_path, dpi=160)
        plt.close()

    # ---- Tail 概览：高分位与 Top-K
    tail_csv = out_img_path.replace('.png', '_tail.csv')
    if degrees.size:
        percentiles = [50, 75, 90, 95, 99, 99.9]
        perc_vals = {p: int(np.percentile(degrees, p)) for p in percentiles}
        topk = 20 if degrees.size >= 20 else degrees.size
        top_vals = sdeg[:topk].tolist()
        with open(tail_csv, 'w', newline='', encoding='utf-8') as ft:
            w = csv.writer(ft)
            w.writerow(['metric', 'value'])
            w.writerow(['n_nodes', n])
            w.writerow(['avg_degree', f"{avg_deg:.4f}"])
            w.writerow(['max_degree', max_deg])
            for p in percentiles:
                w.writerow([f'p{p}', perc_vals[p]])
            for i, v in enumerate(top_vals, 1):
                w.writerow([f'top{i}', v])

    print(f"[INFO] Degree subset: nodes={n:,}, edges≈{int(m):,} | avg_degree={avg_deg:.4f} | max_degree={max_deg}")
    print(f"[INFO] Saved degree histogram to: {out_img_path}")
    print(f"[INFO] Saved degree histogram CSV to: {out_csv_path}")
    if degrees.size:
        print(f"[INFO] Saved CCDF to: {ccdf_path}")
        print(f"[INFO] Saved Rank–Frequency to: {rank_path}")
        print(f"[INFO] Saved tail summary to: {tail_csv}")
    return {"avg_degree": avg_deg, "num_nodes": n, "approx_edges": int(m), "max_degree": max_deg}


# ----------------------- 主程序 -----------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    # 图构建
    ap.add_argument("--relation_as_nodes", action="store_true", help="将 relation 当作节点（bipartite：h—r—t）")
    ap.add_argument("--exclude_relation_in_degree", action="store_true", help="度分布统计时排除 relation 节点")
    ap.add_argument("--cluster_on_entities_only", action="store_true", help="Leiden/后续流程仅在实体节点诱导子图上运行（配合 --relation_as_nodes 推荐开启）")

    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden 模块度分辨率 γ（越大→更多簇）")
    ap.add_argument("--max_cluster_size", type=int, default=4000, help="最大簇大小（达到即递归细分）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--cache_dir", type=str, default=None, help="HuggingFace 缓存目录")
    ap.add_argument("--tree_out_dir", type=str, default="trees_out", help="原始 HL 树状关系 JSON 输出目录")
    ap.add_argument("--leaf_graph_out_prefix", type=str, default="leaf_graph", help="叶簇图 CSV 前缀")
    ap.add_argument("--top_k", type=int, default=10, help="打印度数 Top-K（叶簇图）")
    ap.add_argument("--gh_capacity_attr", type=str, default="weight_sum", choices=["weight_sum", "edge_count"], help="GH 树容量字段")
    ap.add_argument("--levels_out_dir", type=str, default="levels", help="每层映射与统计输出目录")

    # 度分布导出参数
    ap.add_argument("--degree_hist_png", type=str, default="degree_hist.png", help="节点度数分布图输出路径（PNG）")
    ap.add_argument("--degree_hist_csv", type=str, default="degree_hist.csv", help="节点度数分布 CSV（degree,count）")
    ap.add_argument("--degree_bins", type=int, default=50, help="直方图分箱数（默认50）")
    ap.add_argument("--degree_loglog", action="store_true", help="将度分布图使用 log–log（至少 y 轴 log）")
    ap.add_argument("--degree_log_bins", action="store_true", help="对数不等距分桶（logspace bins）")

    args = ap.parse_args()

    # 1) 加载图
    G_full = load_yago310_from_hf(cache_dir=args.cache_dir, relation_as_nodes=args.relation_as_nodes)

    # 若 relation_as_nodes + cluster_on_entities_only：在“实体节点诱导子图”上跑 Leiden/后续
    if args.relation_as_nodes and args.cluster_on_entities_only:
        entity_nodes = [n for n in G_full.nodes() if not is_relation_node(n)]
        print(f"[INFO] Induced subgraph on entities only: |V|={len(entity_nodes):,}")
        G = G_full.subgraph(entity_nodes).copy()
    else:
        G = G_full

    # 1.5) 度数统计 + 作图（可选排除 relation 节点）
    if args.exclude_relation_in_degree:
        pred = (lambda x: not is_relation_node(x))
    else:
        pred = None

    _ = compute_and_save_degree_stats(
        G_full,  # 注意：在原图上统计（但可通过 pred 排除 relation 节点）
        out_img_path=args.degree_hist_png,
        out_csv_path=args.degree_hist_csv,
        bins=args.degree_bins,
        loglog=args.degree_loglog,
        log_bins=args.degree_log_bins,
        node_include_pred=pred,
    )

    # 2) 连通分量概览（对将要进行 Leiden 的图 G）
    components = list(nx.connected_components(G))
    print(f"[INFO] Detected {len(components)} connected components (for Leiden graph).")
    for i, comp in enumerate(components):
        print(f"  Component #{i}: |V|={len(comp)}")

    # 3) 逐分量执行 HL（跳过过小分量）
    os.makedirs(args.tree_out_dir, exist_ok=True)
    global_part = {}
    trees_summary = []
    skipped = 0

    for i, comp in enumerate(components):
        if len(comp) < 3:
            skipped += 1
            continue
        subg = G.subgraph(comp).copy()
        part_i, tree_i = run_hierarchical_leiden_component(
            subg, comp_id=i,
            resolution=args.resolution,
            max_cluster_size=args.max_cluster_size,
            seed=args.seed,
        )
        global_part.update(part_i)

        # 导出原始 HL 树 JSON
        out_path = os.path.join(args.tree_out_dir, f"hier_tree_component_{i}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tree_i, f, ensure_ascii=False)
        trees_summary.append((i, out_path, sum(tree_i["levels"].values())))
        print(f"[INFO] Saved tree for component #{i} -> {out_path}")

    print(f"[INFO] Finished Leiden for all components. Skipped {skipped} tiny components (<3 nodes).")

    # 4) 统计并输出叶簇内部信息（Top-10）
    stats = analyze_clusters(G, global_part)
    stats_sorted = sorted(stats, key=lambda x: -x["num_nodes"])

    print("\n=== Top-10 clusters by node count (final leaves) ===")
    for rank, s in enumerate(stats_sorted[:10], 1):
        print(f"#{rank:>2}: Cluster {s['cid']:>12} | nodes={s['num_nodes']:>6} | edges={s['num_edges']:>7} | borders={s['num_borders']:>5}")

    total_clusters = len(stats)
    avg_borders = float(np.mean([s["num_borders"] for s in stats])) if stats else 0.0
    print("\n=== Summary ===")
    print(f"Total clusters (leaf level): {total_clusters}")
    print(f"Average border count per cluster: {avg_borders:.2f}")

    # 5) 树导出汇总（原始 HL）
    if trees_summary:
        print("\n=== HL Tree files ===")
        for comp_id, path, num_clusters in trees_summary:
            print(f"Component #{comp_id}: clusters={num_clusters} -> {path}")

    # 6) 叶簇收缩图 + 导出 + Top-K（注意：以 G 的 partition 为基础）
    print("\n[INFO] Building leaf-level contracted graph (LG) ...")
    LG = build_leaf_graph(G, global_part)
    save_leaf_graph_as_csv(LG, args.leaf_graph_out_prefix)

    top_rows = leaf_graph_topk_by_degree(LG, top_k=args.top_k)
    print(f"\n=== Leaf graph Top-{args.top_k} by degree ===")
    for r in top_rows:
        print(f"#{r['rank']:>2}: leaf={r['cluster_id']:>12} | degree={r['degree']:>6} | weighted_degree={r['weighted_degree']:.1f} | size={r['size']:>6}")

    # 7) 基于 LG 构建 GH 树 & 导出
    print("\n[INFO] Building Gomory–Hu tree on LG ...")
    T = build_gomory_hu_tree(LG, capacity_attr=args.gh_capacity_attr)
    os.makedirs(args.levels_out_dir, exist_ok=True)
    export_tree_json(T, os.path.join(args.levels_out_dir, "gh_tree.json"))

    # 8) GH 树 → 分层（按最小割从小到大切）并统计每层 border
    print("[INFO] Deriving hierarchy from GH tree (cutting min-cuts from small to large) ...")
    levels = gh_levels_by_cuts(T)

    # GH 层级汇总 CSV
    levels_summary_csv = os.path.join(args.levels_out_dir, "gh_levels_summary.csv")
    with open(levels_summary_csv, "w", newline="", encoding="utf-8") as fsum:
        wsum = csv.writer(fsum)
        wsum.writerow(["level", "cut_weight", "num_clusters", "border_pairs", "border_edge_count", "border_weight_sum"])

        for L in levels:
            lvl = L["level"]
            cut_w = L["cut_weight"]
            part = L["partition"]

            stats_level = compute_level_border_stats(LG, part)

            print(f"[LEVEL {lvl:>3}] cut_weight={cut_w} | clusters={L['num_clusters']:>5} | border_pairs={stats_level['border_pairs']:>7} | border_edge_count={stats_level['border_edge_count']:>10} | border_weight_sum={stats_level['border_weight_sum']:.1f}")

            wsum.writerow([lvl, cut_w, L["num_clusters"], stats_level["border_pairs"], stats_level["border_edge_count"], stats_level["border_weight_sum"]])

            level_clusters_csv = os.path.join(args.levels_out_dir, f"level_{lvl}_clusters.csv")
            with open(level_clusters_csv, "w", newline="", encoding="utf-8") as fc:
                wc = csv.writer(fc)
                wc.writerow(["leaf_cluster_id", "block_id"])
                for leaf_cid in LG.nodes():
                    wc.writerow([leaf_cid, part[leaf_cid]])

            level_blocks_csv = os.path.join(args.levels_out_dir, f"level_{lvl}_blocks.csv")
            with open(level_blocks_csv, "w", newline="", encoding="utf-8") as fb:
                wb = csv.writer(fb)
                wb.writerow(["block_id", "leaf_clusters", "entities", "border_edge_count", "border_weight_sum"])
                for row in stats_level["per_block"]:
                    wb.writerow([row["block_id"], row["leaf_clusters"], row["entities"], row["border_edge_count"], row["border_weight_sum"]])

    print(f"[INFO] Saved GH levels summary to: {levels_summary_csv}")
    print(f"[OK] Finished. Hierarchy derived from GH tree and per-level border counts computed.")


if __name__ == "__main__":
    main()
