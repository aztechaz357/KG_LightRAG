"""LightRAG インスタンスの生成 (build / query で共有)。"""

from __future__ import annotations

import os

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status

from . import config


async def create_rag() -> LightRAG:
    """設定に従って初期化済みの LightRAG インスタンスを返す。"""
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    llm_func, llm_name, llm_kwargs = config.get_llm()
    rag = LightRAG(
        working_dir=config.WORKING_DIR,
        llm_model_func=llm_func,
        llm_model_name=llm_name,
        llm_model_kwargs=llm_kwargs,
        embedding_func=config.get_embedding(),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag
