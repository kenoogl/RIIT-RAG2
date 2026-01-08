#!/bin/bash

# 玄界RAGシステム 停止スクリプト

set -e

echo "=========================================="
echo "玄界RAGシステム 停止中..."
echo "=========================================="

# プロセスIDファイルの場所
PID_FILE="/tmp/genkai_rag.pid"

# プロセスを検索して停止
PIDS=$(pgrep -f "main.py server" || true)

if [ -n "$PIDS" ]; then
    echo "Found running processes: $PIDS"
    
    # GRACEFULに停止を試行
    echo "Sending SIGTERM..."
    kill -TERM $PIDS
    
    # 10秒待機
    sleep 10
    
    # まだ動いているかチェック
    REMAINING_PIDS=$(pgrep -f "main.py server" || true)
    
    if [ -n "$REMAINING_PIDS" ]; then
        echo "Processes still running, sending SIGKILL..."
        kill -KILL $REMAINING_PIDS
        sleep 2
    fi
    
    echo "Genkai RAG System stopped successfully."
else
    echo "No running Genkai RAG System processes found."
fi

# PIDファイルを削除
if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
fi

echo "=========================================="