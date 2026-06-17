"""Minimal LightRAG-based GraphRAG toolkit.

3 つの処理をモジュールごとに分離している:
  - build      : テキストからナレッジグラフを構築する
  - query      : グラフに対してクエリを実行する
  - visualize  : ノードとエッジを HTML で視覚化する
共通設定とインスタンス生成は config / core に集約している。
"""

__all__ = ["config", "core"]
