#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出：
  - 原图度数分布
  - 随机抽 5 个叶簇的度数分布
  - 超图度数分布
全部图片保存到 ./pic/ 文件夹下

用法示例：
  python deepresearch.py --kg_file ./data/WN18.txt
  python deepresearch.py --dataset Cora
"""

from __future__ import annotations
import argparse
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

try:
    from torch_geometric.datasets import Planetoid
except Exception:
    Planetoid = None

from graspologic.partition import hierarchical_leiden


# ---------------------- 基础工具 ----------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_degree_distribution(G: nx.Graph):
    """返回度分布 dict: {degree -> count}"""
    degs = dict(G.degree())
    hist = defaultdict(int)
    for _, d in degs.items():
        hist[d] += 1
    return dict(sorted(hist.items(), key=lambda x: x[0]))


def plot_degree_distribution(dist: dict, title: str, save_path: str):
    """绘制并保存度数分布"""
    if not dist:
        print(f"[WARN] {title}: 空图，跳过保存")
        return
    degrees = list(dist.keys())
    counts = list(dist.values())
    plt.figure(figsize=(6, 4))
    plt.bar(degrees, counts, edgecolor="k")
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Count (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] {title} → {save_path}")


# ---------------------- 数据加载 ----------------------
def load_planetoid_graph(name="Cora", root=None) -> nx.Graph:
    if Planetoid is None:
        raise RuntimeError("未安装 torch_geometric，且未指定 --kg_file")
    root = root or os.path.abspath(f"./data/{name}")
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    edges = set()
    for u, v in zip(edge_index[0], edge_index[1]):
        if u == v:
            continue
        a, b = int(u), int(v)
        if a > b:
            a, b = b, a
        edges.add((a, b))
    for u, v in edges:
        G.add_edge(u, v, weight=1.0)
    return G


def load_wn18_graph_aggregated(path: str) -> nx.Graph:
    """
    读 WN18 文本，把所有 (h, t) 聚合成一条无向边，权重取最小 1.0
    允许第一行不是数字
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"WN18 file not found: {path}")
    pair2minw = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()

        def add(h, t, w=1.0):
            if h == t:
                return
            # WN18 里节点都是字符串，所以按字符串排序
            a, b = (h, t) if str(h) <= str(t) else (t, h)
            key = (a, b)
            cur = pair2minw.get(key)
            if cur is None or w < cur:
                pair2minw[key] = w

        # 可能第一行是三元组
        try:
            _ = int(first.strip())
        except Exception:
            parts = first.strip().split()
            if len(parts) >= 2:
                add(parts[0], parts[1])
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            h, t = parts[0], parts[1]
            # 第三个是关系，我们先不用
            add(h, t, 1.0)

    G = nx.Graph()
    for (h, t), w in pair2minw.items():
        G.add_edge(h, t, weight=float(w))
    return G


# ---------------------- Leiden 层次聚类 (leaf-only) ----------------------
def final_partition_from_hl(hl_result, G: nx.Graph) -> dict:
    node_to_lc = {}
    for h in hl_result:
        if getattr(h, "is_final_cluster", False):
            node_to_lc[h.node] = (h.level, h.cluster)
    for n in G.nodes():
        if n not in node_to_lc:
            node_to_lc[n] = (-1, f"iso_{n}")
    lc2cid, next_id = {}, 0
    part = {}
    for n, lc in node_to_lc.items():
        if lc not in lc2cid:
            lc2cid[lc] = next_id
            next_id += 1
        part[n] = lc2cid[lc]
    return part


def build_skeleton_leaves(
    G: nx.Graph,
    hl_resolution=0.3,
    hl_max_cluster_size=1200,
    random_seed=42,
):
    """
    返回:
      cluster_tree: {cid -> {level,parent,children,nodes}}
      node_to_leaf: {node -> leaf_cid}
    """
    cluster_tree = {}
    node_to_leaf = {}

    comps = list(nx.connected_components(G))
    comps_sorted = sorted(comps, key=lambda s: -len(s))
    for cc_id, nodes in enumerate(comps_sorted):
        subg = G.subgraph(nodes).copy()
        cc_cid = ("cc", cc_id)
        cluster_tree[cc_cid] = dict(level=-1, parent=None, children=[], nodes=set(nodes))

        hl = hierarchical_leiden(
            subg,
            max_cluster_size=hl_max_cluster_size,
            resolution=hl_resolution,
            use_modularity=True,
            random_seed=random_seed,
            check_directed=True,
        )
        part = final_partition_from_hl(hl, subg)
        cid2nodes = defaultdict(list)
        for n, local_cid in part.items():
            cid2nodes[local_cid].append(n)
        for k, nlist in cid2nodes.items():
            leaf_cid = ("leaf", cc_id, int(k))
            cluster_tree[leaf_cid] = dict(level=0, parent=cc_cid, children=[], nodes=set(nlist))
            cluster_tree[cc_cid]["children"].append(leaf_cid)
            for n in nlist:
                node_to_leaf[n] = leaf_cid
    return cluster_tree, node_to_leaf


# ---------------------- 超图（叶 → 点） ----------------------
def build_supergraph_from_node2leaf(G: nx.Graph, node_to_leaf: dict):
    """
    把每个叶簇抽象成一个点，只保留跨簇边
    返回:
      SG: 超图
      leaf2super: {leaf_cid -> super_id}
      super2leaf: {super_id -> leaf_cid}
    """
    leaf_set = set(node_to_leaf.values())
    # 排序是为了 super_id 稳定
    leaf_list = sorted(list(leaf_set), key=str)
    leaf2super = {leaf: i for i, leaf in enumerate(leaf_list)}
    super2leaf = {i: leaf for leaf, i in leaf2super.items()}

    SG = nx.Graph()
    SG.add_nodes_from(range(len(leaf_list)))

    for u, v, data in G.edges(data=True):
        lu = node_to_leaf.get(u)
        lv = node_to_leaf.get(v)
        if lu is None or lv is None:
            continue
        if lu == lv:
            continue  # 同簇内的边不加
        su = leaf2super[lu]
        sv = leaf2super[lv]
        w = float(data.get("weight", 1.0))
        if SG.has_edge(su, sv):
            # 我们取更小的权，防止太大
            if w < SG[su][sv]["weight"]:
                SG[su][sv]["weight"] = w
        else:
            SG.add_edge(su, sv, weight=w)

    return SG, leaf2super, super2leaf


# ---------------------- 主程序 ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None,
                    help="如果不用kg_file，可以用Cora/CiteSeer/Pubmed")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--kg_file", type=str, default=None,
                    help="例如 --kg_file ./data/WN18.txt")
    ap.add_argument("--resolution", type=float, default=0.3)
    ap.add_argument("--max_cluster_size", type=int, default=2000)
    ap.add_argument("--sample_clusters", type=int, default=5)
    args = ap.parse_args()

    # 1) 创建 pic 目录
    pic_dir = os.path.abspath("./pic")
    ensure_dir(pic_dir)

    # 2) 加载图
    if args.kg_file:
        G = load_wn18_graph_aggregated(args.kg_file)
        src_name = f"WN18({os.path.basename(args.kg_file)})"
    else:
        # 回退到 Planetoid
        dataset = args.dataset or "Cora"
        G = load_planetoid_graph(dataset, root=args.data_root)
        src_name = dataset
    print(f"[INFO] Graph loaded from {src_name}: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # 3) 原图度数分布
    dist_orig = compute_degree_distribution(G)
    plot_degree_distribution(
        dist_orig,
        "Original Graph Degree Distribution",
        os.path.join(pic_dir, "original_degree_distribution.png"),
    )

    # 4) Leiden 生成叶簇
    cluster_tree, node_to_leaf = build_skeleton_leaves(
        G,
        hl_resolution=args.resolution,
        hl_max_cluster_size=args.max_cluster_size,
        random_seed=42,
    )
    leaves = [cid for cid, meta in cluster_tree.items() if meta.get("level") == 0]
    print(f"[INFO] #leaves = {len(leaves)}")

    # 5) 抽样 5 个叶簇做“簇内诱导子图”度数分布
    if leaves:
        k = min(args.sample_clusters, len(leaves))
        sampled = random.sample(leaves, k)
        for i, leaf in enumerate(sampled):
            nodes = list(cluster_tree[leaf]["nodes"])
            sub = G.subgraph(nodes).copy()
            dist_sub = compute_degree_distribution(sub)
            plot_degree_distribution(
                dist_sub,
                f"Cluster {i} Degree Distribution (|V|={sub.number_of_nodes()})",
                os.path.join(pic_dir, f"cluster{i}_degree_distribution.png"),
            )
    else:
        print("[WARN] 没有叶簇，无法输出 cluster 度数分布")

    # 6) 构建超图并输出度数分布
    SG, leaf2super, super2leaf = build_supergraph_from_node2leaf(G, node_to_leaf)
    print(f"[INFO] Supergraph: |V|={SG.number_of_nodes()}, |E|={SG.number_of_edges()}")
    dist_super = compute_degree_distribution(SG)
    plot_degree_distribution(
        dist_super,
        "Supergraph Degree Distribution",
        os.path.join(pic_dir, "supergraph_degree_distribution.png"),
    )

    print(f"\n[OK] All degree plots saved to: {pic_dir}")


if __name__ == "__main__":
    main()
