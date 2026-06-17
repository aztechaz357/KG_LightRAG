"""【グラフ構築】テキストからナレッジグラフを作成する。

使い方:
  uv run python -m graphrag.build <file_or_dir> [<file_or_dir> ...]

ディレクトリを渡すと .txt / .md を再帰的に読み込む。
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import glob
import os

print = functools.partial(print, flush=True)  # noqa: A001  進捗を即時表示する

from . import config
from .core import create_rag

TEXT_EXTENSIONS = (".txt", ".md", ".markdown")


def _collect_files(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for ext in TEXT_EXTENSIONS:
                files.extend(glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"warning: パスが見つかりません: {p}")
    return sorted(set(files))


async def build(paths: list[str]) -> None:
    files = _collect_files(paths)
    if not files:
        print("入力テキストが見つかりませんでした。")
        return

    print(f"[{config.summary()}]")
    print(f"{len(files)} 件のファイルを読み込みます。")

    rag = await create_rag()
    try:
        for f in files:
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
            if not text.strip():
                continue
            print(f"  insert: {f}")
            await rag.ainsert(text, file_paths=f)
    finally:
        await rag.finalize_storages()

    graphml = os.path.join(config.WORKING_DIR, config.GRAPHML_FILENAME)
    print(f"完了。グラフを保存しました: {graphml}")


def main() -> None:
    ap = argparse.ArgumentParser(description="テキストからナレッジグラフを構築する")
    ap.add_argument("paths", nargs="+", help="テキスト/Markdown ファイルまたはディレクトリ")
    args = ap.parse_args()
    asyncio.run(build(args.paths))


if __name__ == "__main__":
    main()
