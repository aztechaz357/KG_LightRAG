# GraphRAG_002

LightRAG を使った最小構成の GraphRAG ツール。テキストからナレッジグラフを構築し、
クエリ検索とノード／エッジの視覚化ができる。解析に使う LLM は
**ローカル(Ollama) / OpenAI / Gemini / Claude / GitHub Copilot** を環境変数で
切り替えられる（既定はローカル）。

3 つの処理はソースを分離している:

| 処理 | ソース | コマンド |
|------|--------|----------|
| グラフ構築 | `src/graphrag/build.py` | `uv run python -m graphrag.build <file_or_dir>` |
| クエリ実行 | `src/graphrag/query.py` | `uv run python -m graphrag.query "質問"` |
| 視覚化 | `src/graphrag/visualize.py` | `uv run python -m graphrag.visualize` |

共通設定は `src/graphrag/config.py`（プロバイダ切替）、LightRAG の生成は
`src/graphrag/core.py` に集約。

## セットアップ

**uv を使う場合（推奨）:**

```bash
uv sync                # 依存を一括インストール
cp .env.example .env   # 必要に応じて編集
```

**pip を使う場合（uv がない環境）:**

```bash
pip install -e .
# または固定バージョンで揃えたい場合:
pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` は `uv export` で生成したハッシュ付き固定バージョンファイル。
依存を追加・更新した後は `uv export --format requirements-txt --no-dev --output-file requirements.txt` で再生成する。

### ローカル LLM (既定)

[Ollama](https://ollama.com/) を起動し、モデルを取得しておく:

```bash
ollama pull qwen2.5:7b      # 解析用 LLM (LLM_MODEL の既定)
ollama pull bge-m3          # 埋め込み用 (EMBED_MODEL の既定, 1024 次元)
```

リモートの Ollama を使う場合は `.env` の `OLLAMA_HOST`（既定 `http://localhost:11434`）を変更する。

### クラウド LLM を使う場合

`.env` で `LLM_PROVIDER` を `openai` / `gemini` / `claude` に変え、対応する
API キー（`OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY`）を設定する。
Claude は埋め込み API が無いため、埋め込みは `EMBED_PROVIDER`（既定 local）を使う。

### GitHub Copilot を使う場合（会社環境向け）

Copilot は OpenAI 互換 API として利用する。`.env` に以下を設定する:

```bash
LLM_PROVIDER=copilot
EMBED_PROVIDER=copilot          # Copilot で埋め込みも行う場合
LLM_MODEL=gpt-4o                # 既定
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
COPILOT_TOKEN=<Copilot トークン>
# 社内プロキシや GitHub Models 経由なら base_url を変更
# COPILOT_BASE_URL=https://api.githubcopilot.com
```

- 必須ヘッダ（`Copilot-Integration-Id` / `Editor-Version`）は自動付与される。
  既定値は `COPILOT_INTEGRATION_ID` / `COPILOT_EDITOR_VERSION` で上書き可能。
- 埋め込みは Copilot 以外（`local` / `openai`）に分けることもできる。
- 実装は `src/graphrag/providers.py`（OpenAI SDK 直叩きの薄いラッパ）。

### 環境変数

設定はすべて環境変数（`.env`）で行う。詳細とプロバイダ毎の例は `.env.example` を参照。

| 変数 | 既定 | 説明 |
|------|------|------|
| `LLM_PROVIDER` | `local` | `local` / `openai` / `gemini` / `claude` / `copilot` |
| `EMBED_PROVIDER` | `local` | `local` / `openai` / `gemini` / `copilot` |
| `LLM_MODEL` | プロバイダ毎 | 解析用 LLM のモデル名 |
| `EMBED_MODEL` | プロバイダ毎 | 埋め込みモデル名 |
| `EMBED_DIM` | 自動検出 | 埋め込み次元数（未設定なら `EMBED_MODEL` から自動検出） |
| `WORKING_DIR` | `./rag_storage` | グラフ等の保存先 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama のホスト |
| `COPILOT_TOKEN` | — | Copilot トークン（`copilot` 利用時のみ） |
| `COPILOT_BASE_URL` | `https://api.githubcopilot.com` | Copilot/互換エンドポイント |

> **`EMBED_DIM` は通常設定不要**。未設定の場合、起動時に `EMBED_MODEL` を 1 回実行して
> 実際の次元を**自動検出**するため、埋め込みモデルを変えても自動で追従する
> （例: `bge-m3`=1024, `nomic-embed-text`=768, OpenAI=1536）。
> 自動検出を避けたい・失敗する場合のみ `EMBED_DIM` で明示する。
> なお Gemini は要求した次元で返すため、特定の次元にしたい場合は `EMBED_DIM` を指定する。

## 使い方

```bash
# 1. グラフ構築（テキスト or ディレクトリを指定）
uv run python -m graphrag.build ./docs

# 2. クエリ
uv run python -m graphrag.query "登場人物の関係は？" --mode hybrid

# 3. 視覚化（graph.html を生成 → ブラウザで開く）
uv run python -m graphrag.visualize --output graph.html
```

クエリの `--mode` は `naive` / `local` / `global` / `hybrid`(既定) / `mix`。

視覚化した HTML では、ノードをエンティティ種別ごとに色分けし、エッジ上に
関係（例: founder, acquisition）を文字で表示する。ホバーで詳細説明を確認できる。
`--input` で GraphML を明示指定することも可能（既定は `WORKING_DIR` 内）。
