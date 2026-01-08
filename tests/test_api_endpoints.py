"""
API エンドポイントのテスト

FastAPI エンドポイントの動作を検証するテストスイート
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime

from genkai_rag.api.app import create_app
from genkai_rag.models.api import QueryRequest, QueryResponse
from genkai_rag.models.document import DocumentSource
from genkai_rag.models.chat import Message, create_user_message, create_assistant_message


class MockRAGEngine:
    """RAGエンジンのモッククラス"""
    
    def query(self, question, chat_history=None, model_name=None):
        from genkai_rag.core.rag_engine import RAGResponse
        from genkai_rag.models.document import DocumentSource
        
        return RAGResponse(
            answer="テスト回答",
            sources=[
                DocumentSource(
                    url="https://example.com/doc1",
                    title="テスト文書1",
                    section="概要",
                    relevance_score=0.95
                )
            ],
            processing_time=0.1,
            model_used="test-model",
            retrieval_score=0.9,
            confidence_score=0.95
        )


class MockLLMManager:
    """LLMマネージャーのモッククラス"""
    
    def get_current_model(self):
        return "test-model"
    
    async def list_available_models(self):
        return {
            "test-model": {
                "display_name": "テストモデル",
                "description": "テスト用モデル",
                "is_available": True,
                "parameters": {}
            }
        }
    
    async def switch_model(self, model_name, force=False):
        return True
    
    async def check_model_health(self):
        return True


class MockChatManager:
    """チャットマネージャーのモッククラス"""
    
    def get_chat_history(self, session_id, limit=10):
        return []
    
    def get_session_info(self, session_id):
        class SessionInfo:
            message_count = 0
        return SessionInfo()
    
    def list_sessions(self):
        return ["session1", "session2"]
    
    def save_message(self, session_id, message):
        pass
    
    def clear_history(self, session_id):
        pass


class MockSystemMonitor:
    """システムモニターのモッククラス"""
    
    def get_system_status(self):
        class Status:
            timestamp = datetime.now()
            uptime_seconds = 3600.0
            memory_usage_mb = 512.0
            disk_usage_mb = 1024.0
            memory_usage_percent = 50.0
            memory_available_gb = 4.0
            memory_total_gb = 8.0
            disk_usage_percent = 25.0
            disk_available_gb = 100.0
            disk_total_gb = 200.0
            cpu_usage_percent = 25.5
            process_count = 150
        return Status()
    
    def get_performance_stats(self, operation_type=None, hours=24):
        """パフォーマンス統計のモック"""
        from genkai_rag.core.system_monitor import PerformanceStats
        
        if operation_type:
            return {
                operation_type: PerformanceStats(
                    operation_type=operation_type,
                    total_requests=10,
                    successful_requests=9,
                    failed_requests=1,
                    avg_response_time_ms=150.0,
                    min_response_time_ms=50.0,
                    max_response_time_ms=300.0,
                    p50_response_time_ms=120.0,
                    p95_response_time_ms=250.0,
                    p99_response_time_ms=290.0,
                    error_rate_percent=10.0,
                    requests_per_minute=0.4
                )
            }
        else:
            return {
                "query": PerformanceStats(
                    operation_type="query",
                    total_requests=10,
                    successful_requests=9,
                    failed_requests=1,
                    avg_response_time_ms=150.0,
                    min_response_time_ms=50.0,
                    max_response_time_ms=300.0,
                    p50_response_time_ms=120.0,
                    p95_response_time_ms=250.0,
                    p99_response_time_ms=290.0,
                    error_rate_percent=10.0,
                    requests_per_minute=0.4
                )
            }


class MockConfigManager:
    """設定マネージャーのモッククラス"""
    
    def load_config(self):
        return {
            "web": {
                "cors_origins": ["*"],
                "allowed_hosts": ["*"]
            },
            "llm": {
                "ollama_url": "http://localhost:11434"
            },
            "chat": {
                "max_history_size": 50
            }
        }


class MockDocumentProcessor:
    """文書プロセッサーのモッククラス"""
    pass


class MockConcurrencyManager:
    """同時アクセス管理のモッククラス"""
    
    async def execute_with_concurrency_control(self, handler, *args, **kwargs):
        # request_idとclient_idを除外（実際のハンドラーが受け取らないため）
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['request_id', 'client_id']}
        
        # ハンドラーが非同期かどうかを確認
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            return await handler(*args, **filtered_kwargs)
        else:
            # 同期ハンドラーの場合は直接実行
            return handler(*args, **filtered_kwargs)
    
    def get_metrics(self):
        return {
            "total_requests": 10,
            "completed_requests": 9,
            "failed_requests": 1,
            "success_rate": 0.9,
            "average_processing_time": 0.5,
            "queue_size": 0,
            "active_connections": 2
        }


@pytest.fixture
def client():
    """テスト用FastAPIクライアントを作成"""
    # 依存性注入用のコンポーネントを作成
    dependencies = {
        "rag_engine": MockRAGEngine(),
        "llm_manager": MockLLMManager(),
        "chat_manager": MockChatManager(),
        "system_monitor": MockSystemMonitor(),
        "config_manager": MockConfigManager(),
        "document_processor": MockDocumentProcessor(),
        "concurrency_manager": MockConcurrencyManager()
    }
    
    config = {
        "cors_origins": ["*"],
        "allowed_hosts": ["*"],
        "debug": True
    }
    
    app = create_app(dependencies=dependencies, config=config)
    
    # 依存性注入のオーバーライド
    from genkai_rag.api.routes import (
        get_rag_engine, get_llm_manager, get_chat_manager, get_system_monitor, get_concurrency_manager
    )
    
    app.dependency_overrides[get_rag_engine] = lambda: dependencies["rag_engine"]
    app.dependency_overrides[get_llm_manager] = lambda: dependencies["llm_manager"]
    app.dependency_overrides[get_chat_manager] = lambda: dependencies["chat_manager"]
    app.dependency_overrides[get_system_monitor] = lambda: dependencies["system_monitor"]
    app.dependency_overrides[get_concurrency_manager] = lambda: dependencies["concurrency_manager"]
    
    return TestClient(app)


class TestQueryEndpoint:
    """質問応答エンドポイントのテスト"""
    
    def test_query_endpoint_basic(self, client):
        """基本的な質問応答のテスト"""
        request_data = {
            "question": "玄界システムについて教えてください",
            "session_id": "test_session",
            "model_name": "test-model",
            "max_sources": 5,
            "include_history": True
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # レスポンス構造の検証
        assert "answer" in data
        assert "sources" in data
        assert "processing_time" in data
        assert "model_used" in data
        assert "session_id" in data
        assert "timestamp" in data
        
        # 出典情報の検証
        assert isinstance(data["sources"], list)
        if data["sources"]:
            source = data["sources"][0]
            assert "url" in source
            assert "title" in source
            assert "section" in source
    
    def test_query_endpoint_validation_error(self, client):
        """バリデーションエラーのテスト"""
        # 空の質問
        request_data = {
            "question": "",
            "session_id": "test_session"
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 422  # Validation Error
    
    def test_query_endpoint_missing_fields(self, client):
        """必須フィールド不足のテスト"""
        request_data = {
            "question": "テスト質問"
            # session_id が不足
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 422  # Validation Error
    
    @given(
        question=st.text(min_size=1, max_size=100),
        session_id=st.text(min_size=1, max_size=50),
        max_sources=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_query_endpoint_property_source_inclusion(self, client, question, session_id, max_sources):
        """
        プロパティ 7: 出典情報の包含
        
        質問応答の結果には必ず出典情報が含まれることを検証
        """
        request_data = {
            "question": question,
            "session_id": session_id,
            "max_sources": max_sources,
            "include_history": True
        }
        
        response = client.post("/api/query", json=request_data)
        
        # リクエストが成功した場合
        if response.status_code == 200:
            data = response.json()
            
            # 出典情報が含まれていることを検証
            assert "sources" in data, "レスポンスに出典情報が含まれていません"
            assert isinstance(data["sources"], list), "出典情報がリスト形式ではありません"
            
            # 各出典が適切な構造を持つことを検証
            for source in data["sources"]:
                assert "url" in source, "出典にURLが含まれていません"
                assert "title" in source, "出典にタイトルが含まれていません"
                assert "section" in source, "出典にセクションが含まれていません"
                assert isinstance(source["url"], str), "出典のURLが文字列ではありません"
                assert isinstance(source["title"], str), "出典のタイトルが文字列ではありません"
            
            # 出典数が指定された最大数を超えないことを検証
            assert len(data["sources"]) <= max_sources, f"出典数が最大値 {max_sources} を超えています"


class TestModelEndpoints:
    """モデル管理エンドポイントのテスト"""
    
    def test_list_models_endpoint(self, client):
        """モデル一覧取得のテスト"""
        response = client.get("/api/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "models" in data
        assert "current_model" in data
        assert "timestamp" in data
        
        # モデル情報の構造検証
        for model in data["models"]:
            assert "name" in model
            assert "display_name" in model
            assert "is_available" in model
            assert "is_default" in model
    
    def test_switch_model_endpoint(self, client):
        """モデル切り替えのテスト"""
        request_data = {
            "model_name": "test-model",
            "force": False
        }
        
        response = client.post("/api/models/switch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
    
    def test_get_current_model_endpoint(self, client):
        """現在のモデル取得のテスト"""
        response = client.get("/api/models/current")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "current_model" in data
        assert isinstance(data["current_model"], str)


class TestChatEndpoints:
    """チャット履歴エンドポイントのテスト"""
    
    def test_get_chat_history_endpoint(self, client):
        """チャット履歴取得のテスト"""
        response = client.get("/api/chat/history?session_id=test_session&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "messages" in data
        assert "total_count" in data
        assert "has_more" in data
        assert "timestamp" in data
    
    def test_clear_chat_history_endpoint(self, client):
        """チャット履歴クリアのテスト"""
        response = client.delete("/api/chat/history/test_session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
    
    def test_list_chat_sessions_endpoint(self, client):
        """チャットセッション一覧のテスト"""
        response = client.get("/api/chat/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "total_count" in data
        assert isinstance(data["sessions"], list)


class TestSystemEndpoints:
    """システム管理エンドポイントのテスト"""
    
    def test_system_status_endpoint(self, client):
        """システムステータス取得のテスト"""
        response = client.get("/api/system/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "memory_usage_mb" in data
        assert "disk_usage_mb" in data
        assert "active_sessions" in data
        assert "total_queries" in data
        assert "current_model" in data
        assert "timestamp" in data
    
    def test_health_check_endpoint(self, client):
        """ヘルスチェックのテスト"""
        response = client.post("/api/system/health-check")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall" in data
        assert "components" in data
        assert isinstance(data["components"], dict)


class TestWebInterface:
    """Webインターフェイスのテスト"""
    
    def test_root_endpoint(self, client):
        """ルートエンドポイント（Webインターフェイス）のテスト"""
        response = client.get("/")
        
        # HTMLレスポンスが返されることを確認
        assert response.status_code == 200
        assert "html" in response.headers.get("content-type", "").lower()
        
        # HTMLコンテンツの基本的な検証
        html_content = response.text
        assert "<html" in html_content
        assert "玄界RAGシステム" in html_content
    
    def test_health_endpoint(self, client):
        """ヘルスエンドポイントのテスト"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert isinstance(data["components"], dict)


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_validation_error_handling(self, client):
        """バリデーションエラーハンドリングのテスト"""
        # 不正なデータ型
        request_data = {
            "question": 123,  # 文字列であるべき
            "session_id": "test_session"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """非同期エンドポイントのテスト"""
    
    async def test_async_query_processing(self):
        """非同期質問処理のテスト"""
        from genkai_rag.api.routes import query_documents
        from genkai_rag.models.api import QueryRequest
        from fastapi import BackgroundTasks
        
        # リクエストを作成
        request = QueryRequest(
            question="テスト質問",
            session_id="test_session"
        )
        
        background_tasks = BackgroundTasks()
        
        # モックコンポーネントを作成
        rag_engine = MockRAGEngine()
        chat_manager = MockChatManager()
        concurrency_manager = MockConcurrencyManager()
        
        # 非同期処理を実行
        response = await query_documents(
            request=request,
            background_tasks=background_tasks,
            rag_engine=rag_engine,
            chat_manager=chat_manager,
            concurrency_manager=concurrency_manager
        )
        
        # レスポンスの検証
        assert isinstance(response, QueryResponse)
        assert response.answer == "テスト回答"
        assert response.session_id == "test_session"
        assert len(response.sources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])