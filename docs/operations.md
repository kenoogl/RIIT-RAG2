# 玄界RAGシステム 運用ガイド

## 概要

このドキュメントは、玄界RAGシステムの日常運用、監視、保守、トラブルシューティングに関する包括的なガイドです。システム管理者および運用担当者向けに作成されています。

## 目次

1. [日常運用](#日常運用)
2. [システム監視](#システム監視)
3. [ログ管理](#ログ管理)
4. [パフォーマンス管理](#パフォーマンス管理)
5. [バックアップ・復旧](#バックアップ復旧)
6. [セキュリティ運用](#セキュリティ運用)
7. [トラブルシューティング](#トラブルシューティング)
8. [定期メンテナンス](#定期メンテナンス)

## 日常運用

### システム起動・停止

#### 手動起動・停止

```bash
# システム起動
sudo systemctl start genkai-rag
sudo systemctl status genkai-rag

# システム停止
sudo systemctl stop genkai-rag

# 再起動
sudo systemctl restart genkai-rag

# 自動起動設定
sudo systemctl enable genkai-rag
```

#### Docker環境での起動・停止

```bash
# 起動
docker-compose up -d

# 停止
docker-compose down

# 再起動
docker-compose restart

# ログ確認
docker-compose logs -f genkai-rag
```

### 設定変更

#### 設定ファイルの編集

```bash
# プロダクション設定の編集
sudo nano /opt/genkai-rag-system/config/production.yaml

# 環境変数の編集
sudo nano /opt/genkai-rag-system/.env

# 設定の検証
python -c "from genkai_rag.core.config_manager import ConfigManager; cm = ConfigManager(); print('設定OK' if cm.validate_config() else '設定エラー')"
```

#### 設定変更の適用

```bash
# 設定リロード（ホットリロード対応項目）
curl -X POST http://localhost:8000/api/system/reload-config

# サービス再起動（全設定反映）
sudo systemctl restart genkai-rag
```

### LLMモデル管理

#### 利用可能モデルの確認

```bash
# APIで確認
curl http://localhost:8000/api/models

# Ollamaで直接確認
ollama list
```

#### モデルの追加・削除

```bash
# 新しいモデルのダウンロード
ollama pull llama3.2:7b

# モデルの削除
ollama rm llama3.2:1b

# モデルの切り替え（API経由）
curl -X POST http://localhost:8000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3.2:7b"}'

# 現在のモデル確認
curl http://localhost:8000/api/models/current
```

#### モデル最適化設定

```bash
# モデル固有の設定確認
curl http://localhost:8000/api/models/llama3.2:1b/config

# パフォーマンス設定の調整
curl -X PUT http://localhost:8000/api/models/llama3.2:1b/config \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0.7,
    "max_tokens": 2048,
    "context_length": 4096
  }'
```

### macOS環境での運用

#### Homebrewサービス管理

```bash
# Ollamaサービスの状態確認
brew services list | grep ollama

# サービスの開始
brew services start ollama

# サービスの停止
brew services stop ollama

# サービスの再起動
brew services restart ollama
```

#### プロセス管理

```bash
# 玄界RAGシステムのプロセス確認
ps aux | grep "python.*main.py"

# メモリ使用量の確認
top -l 1 | grep "python.*main.py"

# プロセスの終了
pkill -f "python main.py server"
```

#### 開発環境での起動

```bash
# 開発モードでの起動
python main.py server --log-level DEBUG --port 8000

# 自動リロード付きでの起動
uvicorn genkai_rag.api.app:app --host 0.0.0.0 --port 8000 --reload
```

# システムでのモデル切り替え
curl -X POST http://localhost:8000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "llama3.2:7b"}'
```

## システム監視

### ヘルスチェック

#### 基本ヘルスチェック

```bash
# システム全体の状態確認
curl http://localhost:8000/api/health

# 詳細ヘルスチェック
curl http://localhost:8000/api/health/detailed

# コマンドラインでの状態確認
python main.py status
```

#### 監視スクリプト例

```bash
#!/bin/bash
# health_check.sh - システムヘルスチェックスクリプト

HEALTH_URL="http://localhost:8000/api/health"
LOG_FILE="/var/log/genkai-rag/health_check.log"

# ヘルスチェック実行
response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$HEALTH_URL")
http_code="${response: -3}"

if [ "$http_code" -eq 200 ]; then
    echo "$(date): システム正常" >> "$LOG_FILE"
else
    echo "$(date): システム異常 (HTTP: $http_code)" >> "$LOG_FILE"
    # アラート送信（メール、Slack等）
    # send_alert "玄界RAGシステム異常検知"
fi
```

### リソース監視

#### システムリソースの確認

```bash
# CPU使用率
top -p $(pgrep -f genkai-rag)

# メモリ使用量
ps aux | grep genkai-rag | awk '{sum+=$6} END {print "Memory: " sum/1024 " MB"}'

# ディスク使用量
df -h /opt/genkai-rag-system/data

# ネットワーク接続
netstat -tlnp | grep :8000
```

#### パフォーマンスメトリクス

```bash
# API経由でメトリクス取得
curl http://localhost:8000/api/system/performance

# 応答時間統計
curl http://localhost:8000/api/system/performance?operation_type=query&hours=24

# システム状態の詳細
curl http://localhost:8000/api/system/status
```

### アラート設定

#### しきい値設定

```yaml
# config/production.yaml
monitoring:
  alert_thresholds:
    memory_usage_percent: 80
    disk_usage_percent: 90
    cpu_usage_percent: 85
    response_time_ms: 5000
    error_rate_percent: 5
```

#### アラート通知スクリプト

```bash
#!/bin/bash
# alert_monitor.sh - アラート監視スクリプト

check_metrics() {
    local metrics=$(curl -s http://localhost:8000/api/system/status)
    local memory_usage=$(echo "$metrics" | jq -r '.memory_usage_mb')
    local disk_usage=$(echo "$metrics" | jq -r '.disk_usage_mb')
    
    # メモリ使用量チェック
    if [ "$memory_usage" -gt 6400 ]; then  # 80% of 8GB
        send_alert "高メモリ使用量: ${memory_usage}MB"
    fi
    
    # ディスク使用量チェック
    if [ "$disk_usage" -gt 18000 ]; then  # 90% of 20GB
        send_alert "高ディスク使用量: ${disk_usage}MB"
    fi
}

send_alert() {
    local message="$1"
    echo "$(date): ALERT - $message" >> /var/log/genkai-rag/alerts.log
    # メール送信、Slack通知等
}

# 5分間隔で実行
while true; do
    check_metrics
    sleep 300
done
```

## ログ管理

### ログファイルの場所

```
/opt/genkai-rag-system/logs/
├── genkai_rag.log          # アプリケーションログ
├── genkai_rag.log.1        # ローテーション済みログ
├── system_status.json      # システム状態ログ
├── system_alerts.json      # アラートログ
└── access.log              # アクセスログ（Nginx使用時）
```

### ログレベル設定

```yaml
# config/production.yaml
monitoring:
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  structured_logging: true
  log_file: "./logs/genkai_rag.log"
  max_log_size_mb: 100
  backup_count: 10
```

### ログ監視コマンド

```bash
# リアルタイムログ監視
tail -f /opt/genkai-rag-system/logs/genkai_rag.log

# エラーログの抽出
grep -i error /opt/genkai-rag-system/logs/genkai_rag.log

# 特定期間のログ抽出
grep "2024-01-15" /opt/genkai-rag-system/logs/genkai_rag.log

# ログ統計
awk '/ERROR/ {error++} /WARNING/ {warning++} /INFO/ {info++} END {print "ERROR:", error, "WARNING:", warning, "INFO:", info}' /opt/genkai-rag-system/logs/genkai_rag.log
```

### ログローテーション設定

```bash
# /etc/logrotate.d/genkai-rag
/opt/genkai-rag-system/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 genkai-rag genkai-rag
    postrotate
        systemctl reload genkai-rag
    endscript
}
```

## パフォーマンス管理

### パフォーマンス監視

#### 応答時間監視

```bash
# 平均応答時間の確認
curl -s http://localhost:8000/api/system/performance | jq '.performance_stats.query.average_response_time'

# 応答時間分布の確認
curl -s http://localhost:8000/api/system/performance | jq '.performance_stats.query'
```

#### スループット監視

```bash
# 1時間あたりのリクエスト数
curl -s http://localhost:8000/api/system/performance?hours=1 | jq '.performance_stats.query.total_requests'

# 同時接続数の確認
curl -s http://localhost:8000/api/system/status | jq '.active_sessions'
```

### パフォーマンス最適化

#### 同時リクエスト数の調整

```yaml
# config/production.yaml
concurrency:
  max_concurrent_requests: 20  # CPUコア数の2-3倍
  max_queue_size: 200
  request_timeout: 60.0
```

#### キャッシュ設定の最適化

```yaml
# config/production.yaml
performance:
  cache_enabled: true
  cache_size: 2000  # エントリ数
  cache_ttl: 3600   # 秒
```

#### データベース最適化

```bash
# インデックスの最適化
python -c "
from genkai_rag.core.processor import DocumentProcessor
processor = DocumentProcessor()
processor.optimize_index()
print('インデックス最適化完了')
"

# 古いデータのクリーンアップ
python -c "
from genkai_rag.core.chat_manager import ChatManager
chat_manager = ChatManager()
cleaned = chat_manager.cleanup_old_sessions(days=30)
print(f'{cleaned}個の古いセッションを削除')
"
```

## バックアップ・復旧

### 自動バックアップ設定

#### バックアップスクリプト

```bash
#!/bin/bash
# backup.sh - 自動バックアップスクリプト

BACKUP_DIR="/opt/genkai-rag-system/backups"
DATA_DIR="/opt/genkai-rag-system/data"
CONFIG_DIR="/opt/genkai-rag-system/config"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="genkai_rag_backup_${DATE}.tar.gz"

# バックアップディレクトリ作成
mkdir -p "$BACKUP_DIR"

# データのバックアップ
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    -C /opt/genkai-rag-system \
    data/ config/ logs/

# 古いバックアップの削除（30日以上）
find "$BACKUP_DIR" -name "genkai_rag_backup_*.tar.gz" -mtime +30 -delete

echo "バックアップ完了: $BACKUP_FILE"
```

#### Cronジョブ設定

```bash
# crontabに追加
# 毎日午前2時にバックアップ実行
0 2 * * * /opt/genkai-rag-system/scripts/backup.sh >> /var/log/genkai-rag/backup.log 2>&1
```

### 復旧手順

#### データ復旧

```bash
# システム停止
sudo systemctl stop genkai-rag

# バックアップからの復旧
cd /opt/genkai-rag-system
tar -xzf backups/genkai_rag_backup_20240115_020000.tar.gz

# 権限の修正
chown -R genkai-rag:genkai-rag data/ config/ logs/

# システム再起動
sudo systemctl start genkai-rag
```

#### 設定復旧

```bash
# 設定ファイルのみ復旧
tar -xzf backups/genkai_rag_backup_20240115_020000.tar.gz config/

# 設定の検証
python -c "from genkai_rag.core.config_manager import ConfigManager; cm = ConfigManager(); print('設定OK' if cm.validate_config() else '設定エラー')"
```

## セキュリティ運用

### セキュリティ監視

#### アクセスログ監視

```bash
# 異常なアクセスパターンの検出
awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -nr | head -10

# 404エラーの監視
grep " 404 " /var/log/nginx/access.log | tail -20

# 大量リクエストの検出
awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | awk '$1 > 100 {print $2, $1}'
```

#### セキュリティ設定の確認

```bash
# ファイル権限の確認
find /opt/genkai-rag-system -type f -perm /o+w -ls

# 設定ファイルのセキュリティチェック
grep -i "secret\|password\|key" /opt/genkai-rag-system/config/*.yaml
```

### セキュリティ更新

#### 依存関係の更新

```bash
# Pythonパッケージの更新
pip list --outdated
pip install --upgrade -r requirements.txt

# セキュリティ脆弱性のチェック
pip-audit

# システムパッケージの更新
sudo apt update && sudo apt upgrade
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. システムが起動しない

**症状**: サービスが開始されない

**確認手順**:
```bash
# サービス状態の確認
sudo systemctl status genkai-rag

# ログの確認
sudo journalctl -u genkai-rag -f

# 設定ファイルの検証
python -c "from genkai_rag.core.config_manager import ConfigManager; cm = ConfigManager(); cm.validate_config()"
```

**解決方法**:
- 設定ファイルの構文エラー修正
- 必要なディレクトリの作成
- 権限の修正

#### 2. Ollamaに接続できない

**症状**: LLMモデルが応答しない

**確認手順**:
```bash
# Ollamaサービスの状態確認
systemctl status ollama

# Ollamaへの接続テスト
curl http://localhost:11434/api/tags

# モデルの確認
ollama list
```

**解決方法**:
```bash
# Ollamaの再起動
sudo systemctl restart ollama

# モデルの再ダウンロード
ollama pull llama3.2:3b

# 設定の確認・修正
nano config/production.yaml
```

#### 3. メモリ不足エラー

**症状**: OutOfMemoryError、システムが重い

**確認手順**:
```bash
# メモリ使用量の確認
free -h
ps aux --sort=-%mem | head -10

# システムリソースの確認
curl http://localhost:8000/api/system/status
```

**解決方法**:
```yaml
# config/production.yaml
concurrency:
  max_concurrent_requests: 5  # 削減
llm:
  default_model: "llama3.2:1b"  # 軽量モデルに変更
```

#### 4. 応答が遅い

**症状**: クエリの応答時間が長い

**確認手順**:
```bash
# パフォーマンスメトリクスの確認
curl http://localhost:8000/api/system/performance

# システムリソースの確認
top
iotop
```

**解決方法**:
- インデックスの最適化
- キャッシュの有効化
- 軽量モデルへの切り替え
- ハードウェアのアップグレード

### ログ分析によるトラブルシューティング

#### エラーパターンの分析

```bash
# エラーの種類別集計
grep -i error /opt/genkai-rag-system/logs/genkai_rag.log | \
awk -F': ' '{print $NF}' | sort | uniq -c | sort -nr

# 時間別エラー発生状況
grep -i error /opt/genkai-rag-system/logs/genkai_rag.log | \
awk '{print $1, $2}' | cut -d: -f1-2 | sort | uniq -c

# 特定エラーの詳細調査
grep -A 5 -B 5 "ConnectionError" /opt/genkai-rag-system/logs/genkai_rag.log
```

## 定期メンテナンス

### 日次メンテナンス

```bash
#!/bin/bash
# daily_maintenance.sh

# ログローテーション
logrotate -f /etc/logrotate.d/genkai-rag

# 古いセッションのクリーンアップ
python -c "
from genkai_rag.core.chat_manager import ChatManager
cm = ChatManager()
cleaned = cm.cleanup_old_sessions(days=7)
print(f'Cleaned {cleaned} old sessions')
"

# システム状態の記録
curl -s http://localhost:8000/api/system/status > /var/log/genkai-rag/daily_status_$(date +%Y%m%d).json
```

### 週次メンテナンス

```bash
#!/bin/bash
# weekly_maintenance.sh

# インデックスの最適化
python -c "
from genkai_rag.core.processor import DocumentProcessor
dp = DocumentProcessor()
dp.optimize_index()
print('Index optimization completed')
"

# パフォーマンスメトリクスのクリーンアップ
curl -X DELETE http://localhost:8000/api/system/performance

# ディスク使用量の確認とアラート
DISK_USAGE=$(df /opt/genkai-rag-system | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "Warning: Disk usage is ${DISK_USAGE}%" | mail -s "Genkai RAG Disk Alert" admin@example.com
fi
```

### 月次メンテナンス

```bash
#!/bin/bash
# monthly_maintenance.sh

# 完全バックアップ
/opt/genkai-rag-system/scripts/backup.sh

# 依存関係の更新チェック
pip list --outdated > /tmp/outdated_packages.txt

# セキュリティ監査
pip-audit --output json > /var/log/genkai-rag/security_audit_$(date +%Y%m).json

# システムリソース使用量レポート
python -c "
import json
from genkai_rag.core.system_monitor import SystemMonitor
sm = SystemMonitor()
stats = sm.get_performance_stats(hours=720)  # 30 days
with open(f'/var/log/genkai-rag/monthly_report_{$(date +%Y%m)}.json', 'w') as f:
    json.dump(stats, f, indent=2)
print('Monthly report generated')
"
```

### メンテナンススケジュール

```bash
# /etc/crontab に追加

# 日次メンテナンス（毎日午前3時）
0 3 * * * genkai-rag /opt/genkai-rag-system/scripts/daily_maintenance.sh

# 週次メンテナンス（毎週日曜日午前4時）
0 4 * * 0 genkai-rag /opt/genkai-rag-system/scripts/weekly_maintenance.sh

# 月次メンテナンス（毎月1日午前5時）
0 5 1 * * genkai-rag /opt/genkai-rag-system/scripts/monthly_maintenance.sh

# ヘルスチェック（5分間隔）
*/5 * * * * genkai-rag /opt/genkai-rag-system/scripts/health_check.sh
```

## 運用チェックリスト

### 日次チェック項目

- [ ] システムの稼働状況確認
- [ ] エラーログの確認
- [ ] リソース使用量の確認
- [ ] バックアップの実行確認
- [ ] 応答時間の確認

### 週次チェック項目

- [ ] パフォーマンスメトリクスの分析
- [ ] セキュリティログの確認
- [ ] ディスク使用量の確認
- [ ] インデックス最適化の実行
- [ ] 古いデータのクリーンアップ

### 月次チェック項目

- [ ] 依存関係の更新確認
- [ ] セキュリティ監査の実行
- [ ] 設定ファイルのレビュー
- [ ] 災害復旧テストの実行
- [ ] 運用レポートの作成

この運用ガイドに従って、玄界RAGシステムの安定した運用を実現してください。問題が発生した場合は、トラブルシューティングセクションを参照し、必要に応じて開発チームに連絡してください。