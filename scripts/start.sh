#!/bin/bash

# 玄界RAGシステム 起動スクリプト

set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 環境変数を読み込み
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# デフォルト値を設定
GENKAI_HOST=${GENKAI_HOST:-"0.0.0.0"}
GENKAI_PORT=${GENKAI_PORT:-8000}
GENKAI_CONFIG_PATH=${GENKAI_CONFIG_PATH:-"./config/production.yaml"}
GENKAI_LOG_LEVEL=${GENKAI_LOG_LEVEL:-"INFO"}

echo "=========================================="
echo "玄界RAGシステム 起動中..."
echo "=========================================="
echo "Host: $GENKAI_HOST"
echo "Port: $GENKAI_PORT"
echo "Config: $GENKAI_CONFIG_PATH"
echo "Log Level: $GENKAI_LOG_LEVEL"
echo "=========================================="

# 必要なディレクトリを作成
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/backups"

# Pythonパスを設定
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# 仮想環境をアクティベート（存在する場合）
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# 依存関係をチェック
echo "Checking dependencies..."
python -c "import llama_index, fastapi, chromadb" 2>/dev/null || {
    echo "Error: Required dependencies not found. Please install requirements:"
    echo "pip install -r requirements.txt"
    exit 1
}

# Ollamaの接続をチェック
echo "Checking Ollama connection..."
OLLAMA_URL=${GENKAI_OLLAMA_URL:-"http://localhost:11434"}
curl -s "$OLLAMA_URL/api/tags" > /dev/null || {
    echo "Warning: Cannot connect to Ollama at $OLLAMA_URL"
    echo "Please ensure Ollama is running or update GENKAI_OLLAMA_URL"
}

# システムを起動
echo "Starting Genkai RAG System..."
python main.py server \
    --config "$GENKAI_CONFIG_PATH" \
    --host "$GENKAI_HOST" \
    --port "$GENKAI_PORT" \
    --log-level "$GENKAI_LOG_LEVEL"