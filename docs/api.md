# 玄界RAGシステム API ドキュメント

## 概要

玄界RAGシステムは、RESTful APIを通じて質問応答、モデル管理、システム監視などの機能を提供します。このドキュメントでは、すべてのAPIエンドポイント、リクエスト・レスポンス形式、エラーハンドリングについて詳細に説明します。

## 目次

1. [基本情報](#基本情報)
2. [認証](#認証)
3. [質問応答API](#質問応答api)
4. [モデル管理API](#モデル管理api)
5. [会話履歴API](#会話履歴api)
6. [システム管理API](#システム管理api)
7. [ヘルスチェックAPI](#ヘルスチェックapi)
8. [エラーハンドリング](#エラーハンドリング)
9. [レート制限](#レート制限)
10. [SDKとサンプルコード](#sdkとサンプルコード)

## 基本情報

### ベースURL
```
http://localhost:8000/api
```

### Content-Type
すべてのリクエストとレスポンスは `application/json` 形式です。

### APIバージョン
現在のAPIバージョン: `v1`

### OpenAPI仕様書
システム起動後、以下のURLでOpenAPI仕様書を確認できます：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## 認証

現在のバージョンでは認証機能は実装されていません。将来のバージョンでJWT認証を実装予定です。

## 質問応答API

### POST /api/query

文書に対する質問応答を実行します。

#### リクエスト

```http
POST /api/query
Content-Type: application/json

{
  "question": "玄界システムとは何ですか？",
  "session_id": "user-session-123",
  "model_name": "llama3.2:3b",
  "max_sources": 5,
  "include_history": true
}
```

#### リクエストパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|----|----|------|
| `question` | string | ✓ | 質問文（1-1000文字） |
| `session_id` | string | ✓ | セッションID（会話履歴管理用） |
| `model_name` | string | | 使用するLLMモデル名（省略時はデフォルト） |
| `max_sources` | integer | | 最大出典数（1-10、デフォルト: 3） |
| `include_history` | boolean | | 会話履歴を含めるか（デフォルト: true） |

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "answer": "玄界システムは、九州大学情報基盤研究開発センターが運用するスーパーコンピュータシステムです...",
  "sources": [
    {
      "title": "玄界システム概要",
      "url": "https://www.cc.kyushu-u.ac.jp/scp/system/",
      "section": "システム概要",
      "relevance_score": 0.95,
      "content_preview": "玄界システムは..."
    }
  ],
  "processing_time": 2.34,
  "model_used": "llama3.2:3b",
  "session_id": "user-session-123",
  "metadata": {
    "total_documents_searched": 150,
    "reranking_applied": true,
    "context_length": 2048
  }
}
```

#### レスポンスフィールド

| フィールド | 型 | 説明 |
|------------|----|----|
| `answer` | string | 生成された回答 |
| `sources` | array | 出典情報の配列 |
| `processing_time` | number | 処理時間（秒） |
| `model_used` | string | 使用されたモデル名 |
| `session_id` | string | セッションID |
| `metadata` | object | 追加のメタデータ |

#### エラーレスポンス

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": "Validation Error",
  "message": "Question cannot be empty",
  "details": {
    "field": "question",
    "code": "EMPTY_FIELD"
  }
}
```

## モデル管理API

### GET /api/models

利用可能なLLMモデル一覧を取得します。

#### リクエスト

```http
GET /api/models
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "models": [
    {
      "name": "llama3.2:3b",
      "display_name": "Llama 3.2 3B",
      "description": "高性能汎用モデル",
      "is_available": true,
      "is_default": true,
      "parameters": {
        "size": "3B",
        "context_window": 8192,
        "languages": ["ja", "en"]
      }
    },
    {
      "name": "gemma2:2b",
      "display_name": "Gemma 2 2B",
      "description": "バランス型モデル",
      "is_available": true,
      "is_default": false,
      "parameters": {
        "size": "2B",
        "context_window": 4096,
        "languages": ["ja", "en"]
      }
    }
  ],
  "current_model": "llama3.2:3b"
}
```

### POST /api/models/switch

使用するLLMモデルを切り替えます。

#### リクエスト

```http
POST /api/models/switch
Content-Type: application/json

{
  "model_name": "gemma2:2b",
  "force": false
}
```

#### リクエストパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|----|----|------|
| `model_name` | string | ✓ | 切り替え先のモデル名 |
| `force` | boolean | | 強制切り替え（デフォルト: false） |

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "success": true,
  "message": "Model switched to gemma2:2b",
  "data": {
    "model_name": "gemma2:2b",
    "previous_model": "llama3.2:3b",
    "switch_time": "2024-01-15T10:30:00Z"
  }
}
```

### GET /api/models/current

現在使用中のモデルを取得します。

#### リクエスト

```http
GET /api/models/current
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "current_model": "llama3.2:3b"
}
```

## 会話履歴API

### GET /api/chat/history

指定されたセッションの会話履歴を取得します。

#### リクエスト

```http
GET /api/chat/history?session_id=user-session-123&limit=10&include_sources=true
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|----|----|------|
| `session_id` | string | ✓ | セッションID |
| `limit` | integer | | 取得するメッセージ数（デフォルト: 10） |
| `include_sources` | boolean | | 出典情報を含めるか（デフォルト: true） |

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "session_id": "user-session-123",
  "messages": [
    {
      "id": "msg-001",
      "role": "user",
      "content": "玄界システムとは何ですか？",
      "timestamp": "2024-01-15T10:00:00Z",
      "metadata": {}
    },
    {
      "id": "msg-002",
      "role": "assistant",
      "content": "玄界システムは、九州大学情報基盤研究開発センターが運用するスーパーコンピュータシステムです...",
      "timestamp": "2024-01-15T10:00:05Z",
      "sources": ["https://www.cc.kyushu-u.ac.jp/scp/system/"],
      "metadata": {
        "model_used": "llama3.2:3b",
        "processing_time": 2.34
      }
    }
  ],
  "total_count": 24,
  "has_more": true
}
```

### DELETE /api/chat/history/{session_id}

指定されたセッションの会話履歴をクリアします。

#### リクエスト

```http
DELETE /api/chat/history/user-session-123
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "success": true,
  "message": "Chat history cleared for session user-session-123",
  "data": {
    "session_id": "user-session-123",
    "cleared_messages": 24
  }
}
```

### GET /api/chat/sessions

アクティブなチャットセッション一覧を取得します。

#### リクエスト

```http
GET /api/chat/sessions
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "sessions": [
    {
      "session_id": "user-session-123",
      "created_at": "2024-01-15T09:00:00Z",
      "last_activity": "2024-01-15T10:30:00Z",
      "message_count": 24
    },
    {
      "session_id": "user-session-456",
      "created_at": "2024-01-15T10:00:00Z",
      "last_activity": "2024-01-15T10:15:00Z",
      "message_count": 6
    }
  ],
  "total_count": 2
}
```

## システム管理API

### GET /api/system/status

システムの詳細ステータスを取得します。

#### リクエスト

```http
GET /api/system/status
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 86400,
  "memory_usage_mb": 1024.5,
  "disk_usage_mb": 5120.0,
  "active_sessions": 15,
  "total_queries": 1250,
  "current_model": "llama3.2:3b",
  "concurrency_metrics": {
    "active_requests": 3,
    "queued_requests": 0,
    "total_requests": 1250,
    "average_response_time": 2.1,
    "error_rate": 0.02
  },
  "performance_stats": {
    "query": {
      "total_requests": 1200,
      "average_response_time": 2.1,
      "p95_response_time": 4.5,
      "error_count": 24,
      "success_rate": 0.98
    }
  }
}
```

### POST /api/system/health-check

詳細なヘルスチェックを実行します。

#### リクエスト

```http
POST /api/system/health-check
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "overall": "healthy",
  "components": {
    "llm_manager": "healthy",
    "database": "healthy",
    "storage": "healthy"
  },
  "details": {
    "llm_health": true,
    "database_connections": 5,
    "storage_available_gb": 45.2
  }
}
```

### GET /api/system/performance

パフォーマンスメトリクスを取得します。

#### リクエスト

```http
GET /api/system/performance?operation_type=query&hours=24
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|----|----|------|
| `operation_type` | string | | 操作タイプ（query, model_switch等） |
| `hours` | integer | | 統計期間（時間、デフォルト: 24） |

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "performance_stats": {
    "query": {
      "total_requests": 1200,
      "average_response_time": 2.1,
      "p50_response_time": 1.8,
      "p95_response_time": 4.5,
      "p99_response_time": 8.2,
      "error_count": 24,
      "success_rate": 0.98,
      "throughput_per_hour": 50.0
    }
  },
  "response_time_history": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "operation_type": "query",
      "response_time": 2.1,
      "success": true
    }
  ],
  "total_metrics": 1200,
  "time_range_hours": 24,
  "operation_type_filter": "query"
}
```

### DELETE /api/system/performance

パフォーマンスメトリクスをクリアします。

#### リクエスト

```http
DELETE /api/system/performance?operation_type=query
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "success": true,
  "message": "Cleared 1200 performance metrics",
  "data": {
    "cleared_count": 1200,
    "operation_type": "query"
  }
}
```

## ヘルスチェックAPI

### GET /api/health

シンプルなヘルスチェックを実行します。

#### リクエスト

```http
GET /api/health
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "timestamp": 1705312800.123,
  "service": "genkai-rag-system",
  "version": "1.0.0"
}
```

### GET /api/health/detailed

詳細なヘルスチェックを実行します。

#### リクエスト

```http
GET /api/health/detailed
```

#### レスポンス

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "status": "healthy",
  "timestamp": 1705312800.123,
  "service": "genkai-rag-system",
  "version": "1.0.0",
  "components": {
    "system_monitor": "healthy",
    "llm_manager": "healthy",
    "chat_manager": "healthy",
    "database": "healthy"
  },
  "metrics": {
    "memory_usage_percent": 65.2,
    "disk_usage_percent": 25.6,
    "active_sessions": 15,
    "uptime_seconds": 86400
  },
  "warnings": []
}
```

## エラーハンドリング

### HTTPステータスコード

| コード | 説明 |
|--------|------|
| 200 | 成功 |
| 400 | リクエストエラー（バリデーション失敗等） |
| 401 | 認証エラー（将来実装予定） |
| 403 | 認可エラー（将来実装予定） |
| 404 | リソースが見つからない |
| 429 | レート制限に達した |
| 500 | 内部サーバーエラー |
| 503 | サービス利用不可 |

### エラーレスポンス形式

```http
HTTP/1.1 400 Bad Request
Content-Type: application/json

{
  "error": "Validation Error",
  "message": "Question cannot be empty",
  "details": {
    "field": "question",
    "code": "EMPTY_FIELD",
    "value": ""
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req-123456"
}
```

### エラーコード一覧

| コード | 説明 |
|--------|------|
| `EMPTY_FIELD` | 必須フィールドが空 |
| `INVALID_FORMAT` | フィールド形式が不正 |
| `VALUE_TOO_LONG` | 値が長すぎる |
| `VALUE_TOO_SHORT` | 値が短すぎる |
| `INVALID_RANGE` | 値が範囲外 |
| `MODEL_NOT_FOUND` | モデルが見つからない |
| `SESSION_NOT_FOUND` | セッションが見つからない |
| `RATE_LIMIT_EXCEEDED` | レート制限超過 |
| `INTERNAL_ERROR` | 内部エラー |

## レート制限

### 制限値

| エンドポイント | 制限 | 期間 |
|----------------|------|------|
| `/api/query` | 60リクエスト | 1分 |
| `/api/models/*` | 30リクエスト | 1分 |
| `/api/chat/*` | 100リクエスト | 1分 |
| `/api/system/*` | 20リクエスト | 1分 |
| `/api/health` | 制限なし | - |

### レート制限ヘッダー

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705312860
```

### レート制限エラー

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 60

{
  "error": "Rate Limit Exceeded",
  "message": "Too many requests. Please try again later.",
  "details": {
    "limit": 60,
    "remaining": 0,
    "reset_time": "2024-01-15T10:31:00Z"
  }
}
```

## SDKとサンプルコード

### Python SDK

```python
import requests
import json

class GenkaiRAGClient:
    def __init__(self, base_url="http://localhost:8000/api"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def query(self, question, session_id, model_name=None, max_sources=3):
        """質問応答を実行"""
        payload = {
            "question": question,
            "session_id": session_id,
            "max_sources": max_sources
        }
        if model_name:
            payload["model_name"] = model_name
        
        response = self.session.post(
            f"{self.base_url}/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_models(self):
        """利用可能なモデル一覧を取得"""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def switch_model(self, model_name):
        """モデルを切り替え"""
        payload = {"model_name": model_name}
        response = self.session.post(
            f"{self.base_url}/models/switch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_chat_history(self, session_id, limit=10):
        """会話履歴を取得"""
        params = {"session_id": session_id, "limit": limit}
        response = self.session.get(
            f"{self.base_url}/chat/history",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """ヘルスチェック"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# 使用例
client = GenkaiRAGClient()

# 質問応答
result = client.query(
    question="玄界システムとは何ですか？",
    session_id="my-session"
)
print(f"回答: {result['answer']}")

# モデル一覧取得
models = client.get_models()
print(f"利用可能なモデル: {[m['name'] for m in models['models']]}")

# 会話履歴取得
history = client.get_chat_history("my-session")
print(f"履歴件数: {len(history['messages'])}")
```

### JavaScript SDK

```javascript
class GenkaiRAGClient {
    constructor(baseUrl = 'http://localhost:8000/api') {
        this.baseUrl = baseUrl;
    }

    async query(question, sessionId, options = {}) {
        const payload = {
            question,
            session_id: sessionId,
            max_sources: options.maxSources || 3,
            ...options
        };

        const response = await fetch(`${this.baseUrl}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    async getModels() {
        const response = await fetch(`${this.baseUrl}/models`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    }

    async switchModel(modelName) {
        const response = await fetch(`${this.baseUrl}/models/switch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: modelName })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    async getChatHistory(sessionId, limit = 10) {
        const params = new URLSearchParams({
            session_id: sessionId,
            limit: limit.toString()
        });

        const response = await fetch(`${this.baseUrl}/chat/history?${params}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return await response.json();
    }
}

// 使用例
const client = new GenkaiRAGClient();

// 質問応答
client.query('玄界システムとは何ですか？', 'my-session')
    .then(result => {
        console.log('回答:', result.answer);
        console.log('出典:', result.sources);
    })
    .catch(error => {
        console.error('エラー:', error.message);
    });

// モデル一覧取得
client.getModels()
    .then(models => {
        console.log('利用可能なモデル:', models.models.map(m => m.name));
    });
```

### cURL サンプル

```bash
# 質問応答
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "玄界システムとは何ですか？",
    "session_id": "test-session",
    "max_sources": 3
  }'

# モデル一覧取得
curl http://localhost:8000/api/models

# モデル切り替え
curl -X POST http://localhost:8000/api/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma2:2b"}'

# 会話履歴取得
curl "http://localhost:8000/api/chat/history?session_id=test-session&limit=10"

# システム状態確認
curl http://localhost:8000/api/system/status

# ヘルスチェック
curl http://localhost:8000/api/health
```

## WebSocket API（将来実装予定）

将来のバージョンでは、リアルタイム通信のためのWebSocket APIを実装予定です。

```javascript
// 将来の実装例
const ws = new WebSocket('ws://localhost:8000/ws/query');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'query',
        question: '玄界システムとは何ですか？',
        session_id: 'my-session'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'partial_answer') {
        // ストリーミング回答の表示
        console.log('部分回答:', data.content);
    } else if (data.type === 'final_answer') {
        // 最終回答の表示
        console.log('最終回答:', data.answer);
    }
};
```

このAPIドキュメントは、玄界RAGシステムのすべてのAPIエンドポイントと使用方法を網羅しています。開発者は、このドキュメントを参考にしてクライアントアプリケーションを構築できます。