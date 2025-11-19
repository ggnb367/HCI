#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict

class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        # 路径压缩
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 按秩合并
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

def parse_edges(path):
    """
    解析 WN18 格式：
      - 常见行为：第一行只有一个数字（计数），跳过
      - 之后每行通常三列：head tail rel
      - 若行只有两列，也按无向边处理
    返回 (dsu, nodes) 便于统计联通分量
    """
    dsu = DSU()
    nodes_seen = set()

    with open(path, "r", encoding="utf-8") as f:
        first = True
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 跳过仅一个整数的首行
            if first and len(parts) == 1 and parts[0].lstrip("-").isdigit():
                first = False
                continue
            first = False

            # 取前两列作为边两端
            if len(parts) >= 2:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    # 若不是纯数字，也当作字符串 ID 处理
                    u, v = parts[0], parts[1]

                dsu.add(u); dsu.add(v)
                dsu.union(u, v)
                nodes_seen.add(u); nodes_seen.add(v)

    return dsu, nodes_seen

def main():
    ap = argparse.ArgumentParser(description="Count connected components in a KG file.")
    ap.add_argument("--file", default="./data/WN18.txt", help="Path to KG file.")
    ap.add_argument("--show_sizes", action="store_true",
                    help="Also print sizes of components (top 10).")
    args = ap.parse_args()

    dsu, nodes = parse_edges(args.file)

    # 统计每个根的大小
    comp_size = defaultdict(int)
    for n in nodes:
        comp_size[dsu.find(n)] += 1

    num_components = len(comp_size)
    print(f"Connected components (undirected): {num_components}")

    if args.show_sizes:
        sizes = sorted(comp_size.values(), reverse=True)
        print("Top 10 component sizes:", sizes[:10])

if __name__ == "__main__":
    main()
