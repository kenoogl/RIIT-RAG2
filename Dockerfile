# 玄界RAGシステム - Dockerファイル
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# データディレクトリを作成
RUN mkdir -p /app/data /app/logs /app/config /app/backups

# 非rootユーザーを作成
RUN useradd -m -u 1000 genkai && \
    chown -R genkai:genkai /app
USER genkai

# ポートを公開
EXPOSE 8000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# デフォルトコマンド
CMD ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"]