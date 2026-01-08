#!/bin/bash

# 玄界RAGシステム Docker起動スクリプト

set -e

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 環境変数を読み込み
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

echo "=========================================="
echo "玄界RAGシステム Docker起動中..."
echo "=========================================="

# プロジェクトルートに移動
cd "$PROJECT_ROOT"

# 必要なディレクトリを作成
mkdir -p data logs config backups

# .envファイルが存在しない場合は作成
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file with your settings before running again."
    exit 1
fi

# Dockerイメージをビルド
echo "Building Docker image..."
docker-compose build

# Ollamaモデルをダウンロード（初回のみ）
echo "Checking Ollama models..."
docker-compose up -d ollama

# Ollamaが起動するまで待機
echo "Waiting for Ollama to start..."
sleep 10

# 必要なモデルをダウンロード
MODELS=("llama3.2:3b" "gemma2:2b")
for model in "${MODELS[@]}"; do
    echo "Checking model: $model"
    docker-compose exec ollama ollama list | grep -q "$model" || {
        echo "Downloading model: $model"
        docker-compose exec ollama ollama pull "$model"
    }
done

# システムを起動
echo "Starting Genkai RAG System..."
docker-compose up -d genkai-rag

# ログを表示
echo "=========================================="
echo "System started successfully!"
echo "=========================================="
echo "Web interface: http://localhost:${GENKAI_PORT:-8000}"
echo "Ollama API: http://localhost:${OLLAMA_PORT:-11434}"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f genkai-rag"
echo ""
echo "To stop the system:"
echo "  docker-compose down"