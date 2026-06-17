"""GitHub Copilot (OpenAI 互換 API) プロバイダ。

会社環境などで GitHub Copilot を LLM として使うための薄いラッパ。
LightRAG 同梱の OpenAI 実装は内部で `default_headers` を固定上書きするため、
Copilot 必須ヘッダ (Copilot-Integration-Id 等) を注入できない。よってここでは
OpenAI SDK を直接使い、LightRAG の `llm_model_func` / 埋め込み関数の規約に合わせる。

設定 (環境変数):
  COPILOT_TOKEN        : Copilot トークン (必須。COPILOT_API_KEY でも可)
  COPILOT_BASE_URL     : 既定 https://api.githubcopilot.com
                         (社内プロキシや GitHub Models を使う場合は変更)
  COPILOT_INTEGRATION_ID : 既定 vscode-chat
  COPILOT_EDITOR_VERSION : 既定 vscode/1.95.0

LLM:    LLM_PROVIDER=copilot   (モデルは LLM_MODEL、既定 gpt-4o)
埋め込み: EMBED_PROVIDER=copilot (モデルは EMBED_MODEL、既定 text-embedding-3-small)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _client():
    from openai import AsyncOpenAI

    base_url = os.getenv("COPILOT_BASE_URL", "https://api.githubcopilot.com")
    api_key = os.getenv("COPILOT_TOKEN") or os.getenv("COPILOT_API_KEY")
    if not api_key:
        raise ValueError(
            "Copilot を使うには COPILOT_TOKEN (または COPILOT_API_KEY) を設定してください。"
        )
    headers = {
        "Copilot-Integration-Id": os.getenv("COPILOT_INTEGRATION_ID", "vscode-chat"),
        "Editor-Version": os.getenv("COPILOT_EDITOR_VERSION", "vscode/1.95.0"),
    }
    return AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=headers)


async def copilot_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """LightRAG の llm_model_func 規約に合わせた補完関数。"""
    # 他プロバイダ同様、モデル名は LightRAG の global_config から取得する。
    hashing_kv = kwargs.get("hashing_kv")
    model = None
    if hashing_kv is not None:
        model = hashing_kv.global_config.get("llm_model_name")
    model = model or os.getenv("LLM_MODEL", "gpt-4o")

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    create_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    # エンティティ抽出時などに JSON 出力指定が来たら尊重する。
    if kwargs.get("response_format") is not None:
        create_kwargs["response_format"] = kwargs["response_format"]
    if "temperature" in kwargs:
        create_kwargs["temperature"] = kwargs["temperature"]

    client = _client()
    resp = await client.chat.completions.create(**create_kwargs)
    return resp.choices[0].message.content or ""


async def copilot_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    **kwargs: Any,
) -> np.ndarray:
    """LightRAG の埋め込み関数規約に合わせた埋め込み関数。"""
    client = _client()
    resp = await client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])
