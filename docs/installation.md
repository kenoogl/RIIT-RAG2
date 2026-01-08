# 玄界RAGシステム インストールガイド

## 概要

このドキュメントでは、玄界RAGシステムのインストール方法を説明します。**最も簡単で確実な方法は、LLMアシスタント（Kiro/Antigravity）を活用したインストールです。**

## 🚀 推奨インストール方法（最も簡単）

### LLMアシスタントを使用したインストール

この方法が**最も簡単で確実**です。LLMアシスタントがすべての手順を自動化し、エラーが発生した場合も自動的に解決します。

#### 対応LLMアシスタント

##### 🔧 コード実行機能付きLLMアシスタント（推奨）

**VSCodeベースのIDE統合型**
- **Kiro**: VSCode拡張として動作、ファイル操作とコード実行が可能
- **Antigravity**: VSCode統合、プロジェクト全体の理解とコード生成に優れる

**Webベースのコード実行対応**
- **ChatGPT Plus（Code Interpreter）**: ファイルアップロード・ダウンロード、Python実行環境
- **Claude 4.5（Artifacts）**: コード生成・実行、ファイル操作対応
- **Cursor**: AI統合エディタ、リアルタイムコード支援

##### 💬 チャット型LLMサービス（指示のみ）

**商用サービス**
- **ChatGPT（GPT-5.2）**: OpenAI - 最新モデル
- **Claude（Claude 4.5）**: Anthropic - 最新モデル
- **Gemini（Gemini 3）**: Google - 最新モデル
- **Copilot**: Microsoft

**オープンソース・ローカル**
- **Ollama**: llama3.2, gemma2, qwen2.5等
- **LM Studio**: ローカルLLM実行環境
- **Jan**: オープンソースChatGPT代替

#### インストール手順

##### 方法A: コード実行機能付きアシスタント（最も簡単）

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/kenoogl/RIIT-RAG2.git
   cd RIIT-RAG2
   ```

2. **LLMアシスタントの起動**
   - **Kiro/Antigravity**: プロジェクトディレクトリでVSCodeを開き、アシスタントを起動
   - **ChatGPT Plus**: Code Interpreterを有効化
   - **Claude**: Artifactsモードを使用
   - **Cursor**: AIチャットを開始

3. **インストール指示**
   ```
   玄界RAGシステムをインストールして起動してください。
   必要な依存関係のインストール、Ollamaのセットアップ、
   初期設定、動作確認まで全て行ってください。
   ```

4. **完了**
   アシスタントが以下を自動実行します：
   - Python環境の確認・セットアップ
   - 依存関係のインストール
   - Ollamaのインストール・設定
   - 初期LLMモデルのダウンロード
   - システムの起動
   - 動作確認テスト
   - エラーが発生した場合の自動修正

##### 方法B: チャット型LLMサービス（指示ベース）

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/kenoogl/RIIT-RAG2.git
   cd RIIT-RAG2
   ```

2. **LLMサービスにアクセス**
   - ChatGPT、Claude、Gemini等のWebインターフェイス
   - またはローカルOllama、LM Studio等

3. **段階的インストール指示**
   
   **ステップ1: 環境確認**
   ```
   玄界RAGシステムのインストールを開始します。
   まず、私のシステム環境（OS、Python、Git）を確認して、
   必要な前提条件のインストール手順を教えてください。
   ```

   **ステップ2: 依存関係インストール**
   ```
   Python仮想環境を作成し、requirements.txtから
   依存関係をインストールする手順を教えてください。
   エラーが発生した場合の対処法も含めてください。
   ```

   **ステップ3: Ollamaセットアップ**
   ```
   Ollamaをインストールし、玄界RAGシステムに必要な
   LLMモデル（llama3.2:1b等）をダウンロードする
   手順を教えてください。
   ```

   **ステップ4: システム起動**
   ```
   玄界RAGシステムを起動し、動作確認を行う
   手順を教えてください。問題が発生した場合の
   トラブルシューティング方法も含めてください。
   ```

4. **各ステップの実行**
   LLMの指示に従って各コマンドを実行し、エラーが発生した場合は
   エラーメッセージをLLMに報告して解決策を求める

#### この方法の利点

**方法A（コード実行機能付き）の利点**
- ✅ **完全自動化**: 手動作業が最小限
- ✅ **エラー自動修正**: 問題が発生しても自動的に解決
- ✅ **リアルタイム実行**: コマンドを直接実行して結果を確認
- ✅ **ファイル操作**: 設定ファイルの編集や作成が可能
- ✅ **環境適応**: システム環境を自動検出して最適化

**方法B（チャット型）の利点**
- ✅ **幅広い対応**: 多くのLLMサービスで利用可能
- ✅ **段階的学習**: 各ステップを理解しながら進行
- ✅ **カスタマイズ対応**: 特定の要件に柔軟に対応
- ✅ **トラブルシューティング**: エラー発生時の詳細な解決策
- ✅ **無料利用**: 多くのサービスで無料プランが利用可能

#### 推奨LLMサービス別の特徴

| サービス | タイプ | 自動実行 | ファイル操作 | 推奨度 | 備考 |
|----------|--------|----------|--------------|--------|------|
| **Kiro** | VSCode統合 | ✅ | ✅ | ⭐⭐⭐⭐⭐ | 最も統合的 |
| **Antigravity** | VSCode統合 | ✅ | ✅ | ⭐⭐⭐⭐⭐ | プロジェクト理解に優れる |
| **Cursor** | AI統合エディタ | ✅ | ✅ | ⭐⭐⭐⭐⭐ | リアルタイム支援 |
| **ChatGPT Plus** | Web + Code Interpreter | ✅ | 部分的 | ⭐⭐⭐⭐ | 高い理解力 |
| **Claude 4.5** | Web + Artifacts | ✅ | 部分的 | ⭐⭐⭐⭐ | 最新モデル、コード生成に優れる |
| **ChatGPT（GPT-5.2）** | チャット | ❌ | ❌ | ⭐⭐⭐⭐ | 最新モデル、高い理解力 |
| **Claude 4.5（無料）** | チャット | ❌ | ❌ | ⭐⭐⭐ | 最新モデル、詳細な説明 |
| **Gemini 3** | チャット | ❌ | ❌ | ⭐⭐⭐ | 最新モデル、Google統合 |
| **Ollama（ローカル）** | ローカル | ❌ | ❌ | ⭐⭐ | プライバシー重視 |

#### 対応環境

- **macOS**: Intel/Apple Silicon両対応
- **Linux**: Ubuntu、CentOS、RHEL等
- **Windows**: WSL2環境推奨

---

## 📋 従来のインストール方法

LLMアシスタントを使用できない場合や、手動でインストールしたい場合は、以下の詳細手順をご利用ください。

## 目次

1. [システム要件](#システム要件)
2. [事前準備](#事前準備)
3. [Dockerを使用したインストール](#dockerを使用したインストール)
4. [ネイティブインストール](#ネイティブインストール)
5. [設定](#設定)
6. [初期セットアップ](#初期セットアップ)
7. [動作確認](#動作確認)
8. [トラブルシューティング](#トラブルシューティング)

## システム要件

### ハードウェア要件

| 項目 | 最小構成 | 推奨構成 | 高負荷環境 |
|------|----------|----------|------------|
| CPU | 4コア | 8コア | 16コア以上 |
| メモリ | 8GB | 16GB | 32GB以上 |
| ストレージ | 20GB SSD | 100GB SSD | 500GB SSD以上 |
| ネットワーク | 10Mbps | 100Mbps | 1Gbps以上 |

### ソフトウェア要件

#### 基本要件
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **Python**: 3.11以上
- **Git**: 2.25以上

#### Docker環境（推奨）
- **Docker**: 20.10以上
- **Docker Compose**: 2.0以上

#### 外部依存関係
- **Ollama**: LLMモデル実行環境
- **Nginx**: リバースプロキシ（オプション）

## 事前準備

### 1. システム更新

#### Ubuntu/Debian
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential
```

#### CentOS/RHEL
```bash
sudo yum update -y
sudo yum groupinstall -y "Development Tools"
sudo yum install -y curl wget git
```

### 2. Python 3.11のインストール

#### Ubuntu 20.04/22.04
```bash
# Python 3.11のインストール
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# pipのインストール
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
```

#### CentOS 8/RHEL 8
```bash
# Python 3.11のビルドとインストール
sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel
cd /tmp
wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
tar xzf Python-3.11.7.tgz
cd Python-3.11.7
./configure --enable-optimizations
make altinstall
sudo ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11
```

### 3. Ollamaのインストール

```bash
# Ollamaのインストール
curl -fsSL https://ollama.ai/install.sh | sh

# サービスの開始
sudo systemctl start ollama
sudo systemctl enable ollama

# 動作確認
ollama --version
```

## Dockerを使用したインストール

### 1. Dockerのインストール

#### Ubuntu
```bash
# Docker公式リポジトリの追加
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Dockerのインストール
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER
newgrp docker
```

#### CentOS/RHEL
```bash
# Docker公式リポジトリの追加
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Dockerのインストール
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# サービスの開始
sudo systemctl start docker
sudo systemctl enable docker

# ユーザーをdockerグループに追加
sudo usermod -aG docker $USER
newgrp docker
```

### 2. プロジェクトのクローン

```bash
# プロジェクトのクローン
git clone https://github.com/your-org/genkai-rag-system.git
cd genkai-rag-system

# ブランチの確認（必要に応じて）
git checkout main
```

### 3. 環境設定

```bash
# 環境変数ファイルの作成
cp .env.example .env

# 環境変数の編集
nano .env
```

#### .envファイルの設定例

```bash
# 基本設定
GENKAI_HOST=0.0.0.0
GENKAI_PORT=8000
GENKAI_DEBUG=false

# Ollama設定
GENKAI_OLLAMA_URL=http://host.docker.internal:11434

# データベース設定
GENKAI_CHROMA_PERSIST_DIR=./data/chroma_db

# ログ設定
GENKAI_LOG_LEVEL=INFO
GENKAI_LOG_FILE=./logs/genkai_rag.log

# パフォーマンス設定
GENKAI_MAX_CONCURRENT_REQUESTS=10
GENKAI_CONCURRENCY_MAX_REQUESTS=20

# セキュリティ設定
GENKAI_SECRET_KEY=your-secret-key-here-change-in-production
GENKAI_CORS_ORIGINS=*
```

### 4. Dockerコンテナの起動

```bash
# コンテナのビルドと起動
docker-compose up -d

# ログの確認
docker-compose logs -f genkai-rag

# コンテナの状態確認
docker-compose ps
```

### 5. LLMモデルのダウンロード

```bash
# 推奨モデルのダウンロード
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull llama3.2:1b

# モデルの確認
ollama list
```

## ネイティブインストール

### 1. 自動インストールスクリプト（推奨）

```bash
# インストールスクリプトの実行
sudo ./scripts/install.sh

# インストール完了後の確認
sudo systemctl status genkai-rag
```

### 2. 手動インストール

#### プロジェクトのセットアップ

```bash
# プロジェクトディレクトリの作成
sudo mkdir -p /opt/genkai-rag-system
sudo chown $USER:$USER /opt/genkai-rag-system

# プロジェクトのクローン
cd /opt/genkai-rag-system
git clone https://github.com/your-org/genkai-rag-system.git .

# 仮想環境の作成
python3.11 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install --upgrade pip
pip install -r requirements.txt
```

#### システムユーザーの作成

```bash
# システムユーザーの作成
sudo useradd --system --home /opt/genkai-rag-system --shell /bin/bash genkai-rag
sudo chown -R genkai-rag:genkai-rag /opt/genkai-rag-system
```

#### ディレクトリ構造の作成

```bash
# 必要なディレクトリの作成
sudo -u genkai-rag mkdir -p /opt/genkai-rag-system/{data,logs,config/backups}
sudo -u genkai-rag mkdir -p /opt/genkai-rag-system/data/{chroma_db,chat_history,index}

# 権限の設定
sudo chmod 755 /opt/genkai-rag-system
sudo chmod 750 /opt/genkai-rag-system/{data,logs,config}
```

#### systemdサービスの設定

```bash
# サービスファイルのコピー
sudo cp scripts/genkai-rag.service /etc/systemd/system/

# サービスの有効化
sudo systemctl daemon-reload
sudo systemctl enable genkai-rag
```

#### 環境設定

```bash
# 環境変数ファイルの作成
sudo -u genkai-rag cp .env.example /opt/genkai-rag-system/.env
sudo -u genkai-rag nano /opt/genkai-rag-system/.env
```

#### 設定ファイルの配置

```bash
# プロダクション設定のコピー
sudo -u genkai-rag cp config/production.yaml /opt/genkai-rag-system/config/

# 設定の編集
sudo -u genkai-rag nano /opt/genkai-rag-system/config/production.yaml
```

## 設定

### 1. 基本設定

#### config/production.yaml の主要設定

```yaml
# LLMモデル設定
llm:
  default_model: "llama3.2:3b"
  ollama:
    base_url: "http://localhost:11434"
    timeout: 120

# API設定
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: "*"
  max_concurrent_requests: 10

# データベース設定
vector_store:
  type: "chroma"
  persist_directory: "./data/chroma_db"
  collection_name: "genkai_documents"

# 監視設定
monitoring:
  log_level: "INFO"
  log_file: "./logs/genkai_rag.log"
  metrics_enabled: true
```

### 2. セキュリティ設定

#### ファイアウォール設定

```bash
# UFW (Ubuntu)
sudo ufw allow 8000/tcp
sudo ufw allow ssh
sudo ufw enable

# firewalld (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

#### SSL/TLS設定（Nginx使用）

```bash
# Nginxのインストール
sudo apt install -y nginx  # Ubuntu
sudo yum install -y nginx  # CentOS

# 設定ファイルの作成
sudo nano /etc/nginx/sites-available/genkai-rag
```

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

```bash
# 設定の有効化
sudo ln -s /etc/nginx/sites-available/genkai-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 初期セットアップ

### 1. システムの起動

#### Docker環境
```bash
# コンテナの起動
docker-compose up -d

# 起動確認
docker-compose ps
```

#### ネイティブ環境
```bash
# サービスの起動
sudo systemctl start genkai-rag

# 起動確認
sudo systemctl status genkai-rag
```

### 2. 初期データの準備

```bash
# 文書の初期取得（オプション）
python main.py query "玄界システムについて教えてください" --url https://www.cc.kyushu-u.ac.jp/scp/

# インデックスの確認
curl http://localhost:8000/api/system/status
```

### 3. 管理者アカウントの設定（将来の拡張）

```bash
# 現在は認証機能なし
# 将来的に認証機能を追加する場合の準備
echo "認証機能は将来のバージョンで実装予定"
```

## 動作確認

### 1. 基本動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/api/health

# 詳細ステータス
curl http://localhost:8000/api/health/detailed

# 利用可能モデルの確認
curl http://localhost:8000/api/models
```

### 2. 機能テスト

```bash
# 質問応答テスト
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "玄界システムとは何ですか？",
    "session_id": "test-session",
    "max_sources": 3
  }'

# システム状態の確認
curl http://localhost:8000/api/system/status
```

### 3. Web UIの確認

```bash
# ブラウザでアクセス
# http://localhost:8000 または https://your-domain.com
```

### 4. ログの確認

```bash
# アプリケーションログ
tail -f /opt/genkai-rag-system/logs/genkai_rag.log

# システムログ（systemd）
sudo journalctl -u genkai-rag -f

# Dockerログ
docker-compose logs -f genkai-rag
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. ポートが使用中

**エラー**: `Address already in use`

**解決方法**:
```bash
# ポート使用状況の確認
sudo netstat -tlnp | grep :8000

# プロセスの終了
sudo kill -9 <PID>

# または設定でポートを変更
export GENKAI_PORT=8001
```

#### 2. Ollamaに接続できない

**エラー**: `Connection refused to Ollama`

**解決方法**:
```bash
# Ollamaの状態確認
systemctl status ollama

# Ollamaの再起動
sudo systemctl restart ollama

# 接続テスト
curl http://localhost:11434/api/tags

# Docker環境の場合
export GENKAI_OLLAMA_URL=http://host.docker.internal:11434
```

#### 3. 権限エラー

**エラー**: `Permission denied`

**解決方法**:
```bash
# ファイル権限の修正
sudo chown -R genkai-rag:genkai-rag /opt/genkai-rag-system
sudo chmod -R 755 /opt/genkai-rag-system

# SELinuxの確認（CentOS/RHEL）
sudo setsebool -P httpd_can_network_connect 1
```

#### 4. メモリ不足

**エラー**: `OutOfMemoryError`

**解決方法**:
```bash
# スワップファイルの作成
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 軽量モデルへの変更
export GENKAI_DEFAULT_MODEL=llama3.2:1b
```

#### 5. 依存関係エラー

**エラー**: `ModuleNotFoundError`

**解決方法**:
```bash
# 仮想環境の確認
source /opt/genkai-rag-system/venv/bin/activate

# 依存関係の再インストール
pip install --force-reinstall -r requirements.txt

# Pythonバージョンの確認
python --version  # 3.11以上であることを確認
```

#### 6. LlamaIndexの依存関係競合

**エラー**: `pip's dependency resolver does not currently take into account all the packages`

**解決方法**:
```bash
# 特定のバージョンを強制インストール
pip install llama-index-llms-ollama==0.1.2 --force-reinstall

# 依存関係の確認
pip check

# 必要に応じて仮想環境を再作成
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 7. OpenAI APIキーエラー

**エラー**: `No API key found for OpenAI`

**解決方法**:
```bash
# これは正常な動作です。システムはOllamaを使用するため、OpenAIキーは不要
# ログに表示されるエラーは無視して構いません

# 確認方法：
curl http://localhost:8000/api/models
# 利用可能なOllamaモデルが表示されれば正常
```

#### 8. asyncioイベントループエラー

**エラー**: `asyncio.run() cannot be called from a running event loop`

**解決方法**:
```bash
# main.pyを使用してサーバーを起動
python main.py server --port 8000

# または直接uvicornを使用
uvicorn genkai_rag.api.app:app --host 0.0.0.0 --port 8000
```

## macOS特有の注意事項

### 1. 環境準備

#### Homebrewのインストール（推奨）
```bash
# Homebrewのインストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# パスの設定
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### Python 3.12のインストール
```bash
# Homebrewを使用（推奨）
brew install python@3.12

# パスの確認
which python3.12
python3.12 --version
```

#### Ollamaのインストール
```bash
# Homebrewを使用（推奨）
brew install ollama

# サービスの開始
brew services start ollama

# 動作確認
ollama --version
ollama list
```

### 2. 依存関係のインストール

#### 必要なシステムライブラリ
```bash
# Xcodeコマンドラインツール
xcode-select --install

# 追加のライブラリ（必要に応じて）
brew install pkg-config libffi openssl
```

#### Python仮想環境の作成
```bash
# プロジェクトディレクトリに移動
cd /path/to/genkai-rag-system

# 仮想環境の作成
python3.12 -m venv venv

# 仮想環境の有効化
source venv/bin/activate

# pipのアップグレード
pip install --upgrade pip

# 依存関係のインストール
pip install -r requirements.txt
```

### 3. 初回起動時の注意点

#### モデルのダウンロード
```bash
# 軽量モデルから開始（推奨）
ollama pull llama3.2:1b

# モデルの確認
ollama list
```

#### サーバーの起動
```bash
# main.pyを使用（推奨）
python main.py server --port 8000

# または直接uvicornを使用
uvicorn genkai_rag.api.app:app --host 0.0.0.0 --port 8000
```

### 4. 動作確認手順

#### 1. 基本的なヘルスチェック
```bash
# サーバーが起動するまで待機（約15-30秒）
sleep 15

# ヘルスチェック
curl http://localhost:8000/api/health
```

#### 2. モデル管理の確認
```bash
# 利用可能なモデル一覧
curl http://localhost:8000/api/models

# モデルの切り替え
curl -X POST http://localhost:8000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3.2:1b"}'

# 現在のモデル確認
curl http://localhost:8000/api/models/current
```

#### 3. システム状態の確認
```bash
# システム状態
curl http://localhost:8000/api/system/status
```

#### 4. 質問応答のテスト
```bash
# 簡単な質問でテスト
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "玄界システムについて教えてください",
    "session_id": "test-session-001",
    "include_history": false,
    "max_sources": 3
  }'
```

### 5. macOS特有のトラブルシューティング

#### メモリ使用量の監視
```bash
# メモリ使用量の確認
top -l 1 | grep "PhysMem"

# プロセス別メモリ使用量
ps aux | grep python | head -10
```

#### ポート競合の解決
```bash
# ポート使用状況の確認（macOS）
lsof -i :8000

# プロセスの終了
kill -9 <PID>

# 別のポートを使用
python main.py server --port 8001
```

#### Homebrewサービスの管理
```bash
# Ollamaサービスの状態確認
brew services list | grep ollama

# サービスの再起動
brew services restart ollama

# サービスの停止
brew services stop ollama
```

### 6. パフォーマンス最適化

#### Apple Siliconでの最適化
```bash
# Apple Silicon（M1/M2）の場合
export PYTORCH_ENABLE_MPS_FALLBACK=1

# メモリ使用量の制限
export GENKAI_MAX_CONCURRENT_REQUESTS=5
```

#### メモリ不足時の対処
```bash
# 軽量モデルの使用
ollama pull llama3.2:1b

# システム設定での制限
export GENKAI_CHUNK_SIZE=500
export GENKAI_MAX_SOURCES=3
```

### 7. 開発環境での注意点

#### ログレベルの調整
```bash
# デバッグモードでの起動
python main.py server --log-level DEBUG --port 8000
```

#### 自動リロードの有効化
```bash
# 開発時の自動リロード
uvicorn genkai_rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 8. 本番環境への移行

#### 設定ファイルの調整
```bash
# 本番用設定の作成
cp config/default.yaml config/production.yaml

# 設定の編集
nano config/production.yaml
```

#### セキュリティ設定
```bash
# ファイアウォール設定（macOS）
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setallowsigned on
```

この手順に従うことで、macOS環境でも玄界RAGシステムを確実にインストール・運用できます。

### ログ分析

#### エラーログの確認

```bash
# エラーの抽出
grep -i error /opt/genkai-rag-system/logs/genkai_rag.log | tail -20

# 警告の確認
grep -i warning /opt/genkai-rag-system/logs/genkai_rag.log | tail -10

# 特定時間のログ
grep "2024-01-15 10:" /opt/genkai-rag-system/logs/genkai_rag.log
```

### パフォーマンス問題

#### リソース使用量の確認

```bash
# CPU使用率
top -p $(pgrep -f genkai-rag)

# メモリ使用量
ps aux | grep genkai-rag

# ディスク使用量
df -h /opt/genkai-rag-system

# ネットワーク接続
ss -tlnp | grep :8000
```

## アップグレード

### 1. バックアップの作成

```bash
# データのバックアップ
sudo -u genkai-rag tar -czf /tmp/genkai-rag-backup-$(date +%Y%m%d).tar.gz \
  -C /opt/genkai-rag-system data/ config/ logs/
```

### 2. システムの更新

```bash
# システム停止
sudo systemctl stop genkai-rag

# コードの更新
cd /opt/genkai-rag-system
sudo -u genkai-rag git pull origin main

# 依存関係の更新
sudo -u genkai-rag /opt/genkai-rag-system/venv/bin/pip install -r requirements.txt

# システム再起動
sudo systemctl start genkai-rag
```

### 3. 動作確認

```bash
# ヘルスチェック
curl http://localhost:8000/api/health

# ログの確認
sudo journalctl -u genkai-rag -f
```

## セキュリティ強化

### 1. ファイアウォール設定

```bash
# 必要最小限のポートのみ開放
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. 定期更新の設定

```bash
# 自動セキュリティ更新の有効化（Ubuntu）
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 3. ログ監視の設定

```bash
# fail2banのインストール
sudo apt install -y fail2ban

# 設定ファイルの作成
sudo nano /etc/fail2ban/jail.local
```

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
```

このインストールガイドに従って、玄界RAGシステムを安全かつ確実にインストールしてください。問題が発生した場合は、トラブルシューティングセクションを参照するか、開発チームにお問い合わせください。