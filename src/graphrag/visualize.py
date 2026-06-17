"""【視覚化】ナレッジグラフ (ノードとエッジ) を HTML で可視化する。

使い方:
  uv run python -m graphrag.visualize [--input <graphml>] [--output graph.html]

既定では WORKING_DIR 内の GraphML を読み込み、インタラクティブな
HTML (pyvis) を出力する。ブラウザで開くとノードのドラッグ・拡大縮小・
ホバーで詳細表示ができる。
"""

from __future__ import annotations

import argparse
import os

import networkx as nx
from pyvis.network import Network

from . import config

# エンティティ種別ごとの色 (なければ既定色)
_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _attr(d: dict, *keys: str, default: str = "") -> str:
    for k in keys:
        if d.get(k):
            return str(d[k])
    return default


def visualize(input_path: str | None = None, output: str = "graph.html") -> str:
    graphml = input_path or os.path.join(config.WORKING_DIR, config.GRAPHML_FILENAME)
    if not os.path.exists(graphml):
        raise FileNotFoundError(
            f"GraphML が見つかりません: {graphml}\n"
            "先に `uv run python -m graphrag.build <text>` でグラフを構築してください。"
        )

    graph = nx.read_graphml(graphml)

    # エンティティ種別 -> 色 の割り当て
    types: dict[str, str] = {}
    for _, data in graph.nodes(data=True):
        t = _attr(data, "entity_type", default="UNKNOWN")
        if t not in types:
            types[t] = _COLORS[len(types) % len(_COLORS)]

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="in_line",
    )
    net.from_nx(graph)

    for node in net.nodes:
        data = graph.nodes[node["id"]]
        etype = _attr(data, "entity_type", default="UNKNOWN")
        desc = _attr(data, "description")
        node["label"] = _attr(data, "entity_id", default=str(node["id"]))
        node["color"] = types.get(etype, _COLORS[0])
        node["title"] = f"[{etype}] {node['label']}\n{desc}"[:600]
        node["shape"] = "dot"
        node["size"] = 16

    for edge in net.edges:
        data = graph.get_edge_data(edge["from"], edge["to"]) or {}
        desc = _attr(data, "description")
        keywords = _attr(data, "keywords")
        # エッジ上に関係を文字で表示する。短い keywords (例: founder, acquisition)
        # を優先し、無ければ description を短縮して使う。
        label = keywords or desc
        if len(label) > 40:
            label = label[:40] + "…"
        edge["label"] = label
        # 詳細はホバーで全文表示する。
        edge["title"] = (keywords + "\n" + desc).strip()[:400]
        edge["font"] = {"size": 12, "color": "#555555", "align": "middle",
                        "strokeWidth": 4, "strokeColor": "#ffffff"}
        edge["color"] = {"color": "#aaaaaa", "highlight": "#e15759"}

    net.toggle_physics(True)
    # pyvis の write_html は OS 既定エンコーディング (Windows では cp932) で
    # 書き込むため、© などを含むと失敗する。UTF-8 で自前出力する。
    html = net.generate_html(notebook=False)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(
        f"出力しました: {output} "
        f"(ノード {graph.number_of_nodes()} / エッジ {graph.number_of_edges()})"
    )
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="ナレッジグラフを HTML で視覚化する")
    ap.add_argument("--input", default=None, help="GraphML ファイル (既定: WORKING_DIR 内)")
    ap.add_argument("--output", default="graph.html", help="出力 HTML (既定 graph.html)")
    args = ap.parse_args()
    visualize(args.input, args.output)


if __name__ == "__main__":
    main()
