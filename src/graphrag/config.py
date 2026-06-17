"""設定と LLM / 埋め込みプロバイダの切り替え.

環境変数 (または .env) で制御する。基本はローカル LLM (Ollama)。

  LLM_PROVIDER    : local(既定) | openai | gemini | claude | copilot
  EMBED_PROVIDER  : local(既定) | openai | gemini | copilot  (claude は埋め込み非対応)
  LLM_MODEL       : LLM のモデル名 (プロバイダ毎に既定値あり)
  EMBED_MODEL     : 埋め込みモデル名 (同上)
  EMBED_DIM       : 埋め込み次元数 (同上)
  WORKING_DIR     : グラフ等の保存先 (既定 ./rag_storage)
  OLLAMA_HOST     : Ollama のホスト (既定 http://localhost:11434)

API キーは各プロバイダの標準環境変数で渡す:
  OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY
  copilot は COPILOT_TOKEN (詳細は providers.py を参照)
"""

from __future__ import annotations

import os
import sys
from functools import partial

from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc

# Windows のコンソール (cp932) で LightRAG のスピナー等が UnicodeEncodeError を
# 起こすのを避けるため、UTF-8 を強制する。
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

load_dotenv()

WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()

GRAPHML_FILENAME = "graph_chunk_entity_relation.graphml"


def get_llm():
    """(llm_model_func, llm_model_name, llm_model_kwargs) を返す。"""
    provider = LLM_PROVIDER
    if provider == "local":
        from lightrag.llm.ollama import ollama_model_complete

        model = os.getenv("LLM_MODEL", "qwen2.5:7b")
        return ollama_model_complete, model, {
            "host": OLLAMA_HOST,
            "options": {"num_ctx": 32768},
        }
    if provider == "openai":
        from lightrag.llm.openai import openai_complete

        return openai_complete, os.getenv("LLM_MODEL", "gpt-4o-mini"), {}
    if provider == "gemini":
        from lightrag.llm.gemini import gemini_model_complete

        return gemini_model_complete, os.getenv("LLM_MODEL", "gemini-2.0-flash"), {}
    if provider in ("claude", "anthropic"):
        from lightrag.llm.anthropic import anthropic_complete

        return anthropic_complete, os.getenv("LLM_MODEL", "claude-3-5-haiku-latest"), {}
    if provider == "copilot":
        from .providers import copilot_complete

        return copilot_complete, os.getenv("LLM_MODEL", "gpt-4o"), {}
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} (local|openai|gemini|claude|copilot)"
    )


# 自動検出に失敗した場合のフォールバック次元 (プロバイダ毎の代表値)
_DEFAULT_EMBED_DIM = {"local": 1024, "openai": 1536, "gemini": 1536, "copilot": 1536}


def _detect_embedding_dim(func) -> int:
    """埋め込み関数を 1 回実行して実際の次元数を測る。

    EMBED_DIM 未設定時に EMBED_MODEL の実際の次元へ自動追従させるための関数。
    create_rag() は実行中のイベントループ内から呼ばれるため、別スレッドで
    新しいループを起こして同期的に待つ。
    """
    import asyncio
    import concurrent.futures

    async def _probe() -> int:
        import numpy as np

        arr = np.asarray(await func(["dimension probe"]))
        return int(arr.shape[-1])

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 実行中のループが無い → そのまま実行
        return asyncio.run(_probe())
    # 実行中ループ内 → 別スレッドで実行
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(_probe())).result()


def get_embedding() -> EmbeddingFunc:
    """埋め込み関数 (EmbeddingFunc) を返す。

    次元数 (embedding_dim) は EMBED_DIM が設定されていればそれを使い、未設定なら
    EMBED_MODEL を実際に 1 回実行して自動検出する (モデル変更時に追従)。

    Claude は埋め込み API を持たないため、claude を使う場合でも埋め込みは
    EMBED_PROVIDER (既定 local) のものを使う。
    """
    # 各 *_embed は LightRAG 側で既に EmbeddingFunc にラップされており、
    # 固定次元 (例: ollama=1024) で次元検証を行う。ここで再ラップすると二重に
    # なり、実際の次元 (例: nomic=768) と食い違ってエラーになる。よって生の
    # 関数 (`.func`) を取り出し、こちら側の EmbeddingFunc で正しい次元を指定する。
    provider = EMBED_PROVIDER
    explicit_dim = os.getenv("EMBED_DIM")  # 未設定なら None → 自動検出

    if provider == "local":
        from lightrag.llm.ollama import ollama_embed

        model = os.getenv("EMBED_MODEL", "bge-m3:latest")
        func = partial(ollama_embed.func, embed_model=model, host=OLLAMA_HOST)
        max_token = 8192
    elif provider == "openai":
        from lightrag.llm.openai import openai_embed

        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        func = partial(openai_embed.func, model=model)
        max_token = 8192
    elif provider == "gemini":
        from lightrag.llm.gemini import gemini_embed

        model = os.getenv("EMBED_MODEL", "gemini-embedding-001")
        # Gemini は要求した次元で返すため、明示時のみ embedding_dim を渡す。
        # 未指定ならモデル既定の次元で返り、それを自動検出する。
        if explicit_dim is not None:
            func = partial(gemini_embed.func, model=model, embedding_dim=int(explicit_dim))
        else:
            func = partial(gemini_embed.func, model=model)
        max_token = 2048
    elif provider == "copilot":
        from .providers import copilot_embed

        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        func = partial(copilot_embed, model=model)
        max_token = 8192
    else:
        raise ValueError(
            f"Unknown EMBED_PROVIDER: {provider!r} (local|openai|gemini|copilot)"
        )

    if explicit_dim is not None:
        dim = int(explicit_dim)
    else:
        try:
            dim = _detect_embedding_dim(func)
            print(f"[embedding] {provider}/{model} の次元を自動検出: {dim}")
        except Exception as e:  # noqa: BLE001  検出失敗は致命的ではないので既定値で続行
            dim = _DEFAULT_EMBED_DIM[provider]
            print(
                f"[embedding] 次元の自動検出に失敗 ({e}); 既定値 {dim} を使用。"
                "正しい値を EMBED_DIM で明示してください。"
            )

    return EmbeddingFunc(embedding_dim=dim, max_token_size=max_token, func=func)


def summary() -> str:
    """現在の設定の概要文字列。"""
    return (
        f"LLM_PROVIDER={LLM_PROVIDER} EMBED_PROVIDER={EMBED_PROVIDER} "
        f"WORKING_DIR={WORKING_DIR}"
    )
