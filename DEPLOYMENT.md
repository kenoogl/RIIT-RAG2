# 玄界RAGシステム デプロイメントガイド

## 概要

このドキュメントでは、玄界RAGシステムをオンプレミス環境にデプロイする方法を説明します。

## デプロイメント方法

### 1. Dockerを使用したデプロイメント（推奨）

#### 前提条件
- Docker 20.10以降
- Docker Compose 2.0以降
- 最小8GB RAM、20GB以上の空きディスク容量

#### 手順

1. **リポジトリをクローン**
```bash
git clone <repository-url>
cd genkai-rag-system
```

2. **環境変数を設定**
```bash
cp .env.example .env
# .envファイルを編集して設定を調整
```

3. **システムを起動**
```bash
./scripts/docker-start.sh
```

4. **動作確認**
```bash
curl http://localhost:8000/api/health
```

### 2. ネイティブインストール

#### 前提条件
- Ubuntu 20.04以降 または CentOS 8以降
- Python 3.11以降
- 最小8GB RAM、20GB以上の空きディスク容量

#### 手順

1. **システムにインストール**
```bash
sudo ./scripts/install.sh
```

2. **設定を編集**
```bash
sudo nano /opt/genkai-rag-system/.env
```

3. **サービスを開始**
```bash
sudo systemctl start genkai-rag
sudo systemctl enable genkai-rag
```

4. **ステータス確認**
```bash
sudo systemctl status genkai-rag
```

## 設定

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `GENKAI_PORT` | 8000 | Webサーバーのポート |
| `GENKAI_HOST` | 0.0.0.0 | バインドするホストアドレス |
| `GENKAI_OLLAMA_URL` | http://localhost:11434 | OllamaのURL |
| `GENKAI_LOG_LEVEL` | INFO | ログレベル |
| `GENKAI_MAX_CONCURRENT_REQUESTS` | 10 | 最大同時リクエスト数 |

### 設定ファイル

#### プロダクション設定 (`config/production.yaml`)

主要な設定項目：

- **LLMモデル設定**: 使用するモデルとパラメータ
- **ベクトルデータベース**: Chromaの設定
- **API設定**: ポート、CORS、レート制限
- **セキュリティ**: 認証、暗号化設定
- **パフォーマンス**: キャッシュ、接続プール

## Ollamaの設定

### インストール

```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### モデルのダウンロード

```bash
# 日本語対応モデル
ollama pull llama3.2:3b
ollama pull gemma2:2b

# 軽量モデル
ollama pull llama3.2:1b
```

## 監視とログ

### ログファイル

- **アプリケーションログ**: `logs/genkai_rag.log`
- **システムログ**: `journalctl -u genkai-rag`
- **アクセスログ**: Nginxログ（リバースプロキシ使用時）

### ヘルスチェック

```bash
# システム状態確認
curl http://localhost:8000/api/health

# 詳細ステータス
python main.py status
```

### メトリクス

システムは以下のメトリクスを提供します：

- CPU使用率
- メモリ使用率
- ディスク使用率
- 応答時間
- エラー率

## セキュリティ

### ファイアウォール設定

```bash
# UFW (Ubuntu)
sudo ufw allow 8000/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# firewalld (CentOS)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

### SSL/TLS設定

Nginxリバースプロキシを使用してSSL/TLSを設定：

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        # その他のプロキシ設定...
    }
}
```

## バックアップ

### 自動バックアップ

システムは以下のデータを自動的にバックアップします：

- ベクトルデータベース (`data/chroma_db`)
- 設定ファイル (`config/`)
- 会話履歴 (`data/chat_history`)

### 手動バックアップ

```bash
# データディレクトリをバックアップ
tar -czf backup-$(date +%Y%m%d).tar.gz data/ config/ logs/

# リストア
tar -xzf backup-20240101.tar.gz
```

## トラブルシューティング

### よくある問題

1. **Ollamaに接続できない**
   - Ollamaが起動しているか確認
   - `GENKAI_OLLAMA_URL`が正しいか確認

2. **メモリ不足エラー**
   - より軽量なモデルに切り替え
   - `max_concurrent_requests`を減らす

3. **ポートが使用中**
   - `GENKAI_PORT`を変更
   - 他のプロセスがポートを使用していないか確認

### ログの確認

```bash
# アプリケーションログ
tail -f logs/genkai_rag.log

# システムログ
journalctl -u genkai-rag -f

# Dockerログ
docker-compose logs -f genkai-rag
```

## パフォーマンス最適化

### ハードウェア要件

| コンポーネント | 最小 | 推奨 |
|---------------|------|------|
| CPU | 4コア | 8コア以上 |
| RAM | 8GB | 16GB以上 |
| ストレージ | 20GB | 100GB以上 SSD |
| GPU | なし | NVIDIA GPU（オプション） |

### 設定の最適化

1. **同時リクエスト数の調整**
```yaml
api:
  max_concurrent_requests: 20  # CPUコア数の2-3倍
```

2. **キャッシュの有効化**
```yaml
performance:
  cache_enabled: true
  cache_size: 2000
```

3. **バッチ処理の最適化**
```yaml
document_processing:
  batch_processing: true
  max_workers: 8  # CPUコア数
```

## アップデート

### システムアップデート

```bash
# Dockerの場合
git pull
docker-compose build
docker-compose up -d

# ネイティブインストールの場合
git pull
sudo systemctl restart genkai-rag
```

### 設定の移行

新しいバージョンでは設定ファイルの形式が変更される場合があります。
アップデート前に設定ファイルをバックアップしてください。

## サポート

問題が発生した場合は、以下の情報を含めてサポートに連絡してください：

- システム情報（OS、Python版、Docker版）
- エラーログ
- 設定ファイル（機密情報を除く）
- 再現手順