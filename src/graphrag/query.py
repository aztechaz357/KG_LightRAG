"""【クエリ実行】ナレッジグラフに対して検索・質問を行う。

使い方:
  uv run python -m graphrag.query "質問文" [--mode hybrid]

mode:
  naive  : 通常のベクトル検索のみ
  local  : エンティティ近傍中心
  global : 関係グラフ中心
  hybrid : local + global (既定)
  mix    : グラフ + ベクトルの統合
"""

from __future__ import annotations

import argparse
import asyncio

from lightrag import QueryParam

from .core import create_rag

MODES = ["naive", "local", "global", "hybrid", "mix"]


async def run_query(question: str, mode: str = "hybrid") -> str:
    rag = await create_rag()
    try:
        # rerank モデルは未設定なので無効化 (有効だと警告が出る)
        return await rag.aquery(question, param=QueryParam(mode=mode, enable_rerank=False))
    finally:
        await rag.finalize_storages()


def main() -> None:
    ap = argparse.ArgumentParser(description="ナレッジグラフにクエリを実行する")
    ap.add_argument("question", help="質問文")
    ap.add_argument("--mode", default="hybrid", choices=MODES, help="検索モード (既定 hybrid)")
    args = ap.parse_args()
    answer = asyncio.run(run_query(args.question, args.mode))
    print(answer)


if __name__ == "__main__":
    main()
