"""
API エンドポイントのテスト

FastAPI エンドポイントの動作を検証するテストスイート
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime

from genkai_rag.api.app import create_app
from genkai_rag.models.api import QueryRequest, QueryResponse
from genkai_rag.models.document import DocumentSource
from genkai_rag.models.chat import Message, create_user_message, create_assistant_message


@pytest.fixture
def mock_app_state():
    """モックされたアプリケーション状態を作成"""
    with patch('genkai_rag.api.app.app_state') as mock_state:
        # RAGエンジンのモック
        mock_rag_engine = AsyncMock()
        mock_rag_engine.query.return_value = {
            "answer": "テスト回答",
            "sources": [
                DocumentSource(
                    url="https://example.com/doc1",
                    title="テスト文書1",
                    section="概要",
                    relevance_score=0.95
                )
            ],
            "model_used": "test-model",
            "metadata": {"confidence": 0.95}
        }
        
        # LLMマネージャーのモック
        mock_llm_manager = Mock()
        mock_llm_manager.get_current_model.return_value = "test-model"
        
        # 非同期メソッドのモック
        async def mock_list_available_models():
            return {
                "test-model": {
                    "display_name": "テストモデル",
                    "description": "テスト用モデル",
                    "is_available": True,
                    "parameters": {}
                }
            }
        
        async def mock_switch_model(model_name, force=False):
            return True
            
        async def mock_check_model_health():
            return True
        
        mock_llm_manager.list_available_models = mock_list_available_models
        mock_llm_manager.switch_model = mock_switch_model
        mock_llm_manager.check_model_health = mock_check_model_health
        
        # チャットマネージャーのモック
        mock_chat_manager = Mock()
        mock_chat_manager.get_chat_history.return_value = []
        mock_session_info = Mock()
        mock_session_info.message_count = 0
        mock_chat_manager.get_session_info.return_value = mock_session_info
        mock_chat_manager.list_sessions.return_value = ["session1", "session2"]
        mock_chat_manager.save_message = Mock()
        mock_chat_manager.clear_history = Mock()
        
        # システムモニターのモック
        mock_system_monitor = Mock()
        mock_status = Mock()
        mock_status.timestamp = datetime.now()
        mock_status.uptime_seconds = 3600.0
        mock_status.memory_usage_mb = 512.0
        mock_status.disk_usage_mb = 1024.0
        mock_system_monitor.get_system_status.return_value = mock_status
        
        # 設定マネージャーのモック
        mock_config_manager = Mock()
        mock_config_manager.load_config.return_value = {
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
        
        # 文書プロセッサーのモック
        mock_document_processor = Mock()
        
        # テンプレートのモック
        mock_templates = Mock()
        
        # モック状態を設定
        mock_state.rag_engine = mock_rag_engine
        mock_state.llm_manager = mock_llm_manager
        mock_state.chat_manager = mock_chat_manager
        mock_state.system_monitor = mock_system_monitor
        mock_state.config_manager = mock_config_manager
        mock_state.document_processor = mock_document_processor
        mock_state.templates = mock_templates
        
        yield mock_state


@pytest.fixture
def client(mock_app_state):
    """テスト用FastAPIクライアントを作成"""
    # 依存性注入用の辞書を作成
    dependencies = {
        "rag_engine": mock_app_state.rag_engine,
        "llm_manager": mock_app_state.llm_manager,
        "chat_manager": mock_app_state.chat_manager,
        "system_monitor": mock_app_state.system_monitor,
        "config_manager": mock_app_state.config_manager,
        "document_processor": mock_app_state.document_processor
    }
    
    config = {
        "cors_origins": ["*"],
        "allowed_hosts": ["*"],
        "debug": True
    }
    
    app = create_app(dependencies=dependencies, config=config)
    return TestClient(app)


class TestQueryEndpoint:
    """質問応答エンドポイントのテスト"""
    
    def test_query_endpoint_basic(self, client, mock_app_state):
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
    
    def test_query_endpoint_validation_error(self, client, mock_app_state):
        """バリデーションエラーのテスト"""
        # 空の質問
        request_data = {
            "question": "",
            "session_id": "test_session"
        }
        
        response = client.post("/api/query", json=request_data)
        assert response.status_code == 422  # Validation Error
    
    def test_query_endpoint_missing_fields(self, client, mock_app_state):
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
    def test_query_endpoint_property_source_inclusion(self, question, session_id, max_sources):
        """
        プロパティ 7: 出典情報の包含
        
        質問応答の結果には必ず出典情報が含まれることを検証
        """
        # テスト用クライアントを作成
        app = create_app()
        client = TestClient(app)
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
    
    def test_list_models_endpoint(self, client, mock_app_state):
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
    
    def test_switch_model_endpoint(self, client, mock_app_state):
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
    
    def test_get_current_model_endpoint(self, client, mock_app_state):
        """現在のモデル取得のテスト"""
        response = client.get("/api/models/current")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "current_model" in data
        assert isinstance(data["current_model"], str)


class TestChatEndpoints:
    """チャット履歴エンドポイントのテスト"""
    
    def test_get_chat_history_endpoint(self, client, mock_app_state):
        """チャット履歴取得のテスト"""
        response = client.get("/api/chat/history?session_id=test_session&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "session_id" in data
        assert "messages" in data
        assert "total_count" in data
        assert "has_more" in data
        assert "timestamp" in data
    
    def test_clear_chat_history_endpoint(self, client, mock_app_state):
        """チャット履歴クリアのテスト"""
        response = client.delete("/api/chat/history/test_session")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "message" in data
        assert data["success"] is True
    
    def test_list_chat_sessions_endpoint(self, client, mock_app_state):
        """チャットセッション一覧のテスト"""
        response = client.get("/api/chat/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "sessions" in data
        assert "total_count" in data
        assert isinstance(data["sessions"], list)


class TestSystemEndpoints:
    """システム管理エンドポイントのテスト"""
    
    def test_system_status_endpoint(self, client, mock_app_state):
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
    
    def test_health_check_endpoint(self, client, mock_app_state):
        """ヘルスチェックのテスト"""
        response = client.post("/api/system/health-check")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall" in data
        assert "components" in data
        assert isinstance(data["components"], dict)


class TestWebInterface:
    """Webインターフェイスのテスト"""
    
    def test_root_endpoint(self, client, mock_app_state):
        """ルートエンドポイント（Webインターフェイス）のテスト"""
        # テンプレートレスポンスのモック
        mock_app_state.templates.TemplateResponse.return_value = Mock()
        mock_app_state.templates.TemplateResponse.return_value.status_code = 200
        
        response = client.get("/")
        
        # テンプレートが呼び出されることを確認
        mock_app_state.templates.TemplateResponse.assert_called_once()
    
    def test_health_endpoint(self, client, mock_app_state):
        """ヘルスエンドポイントのテスト"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data


class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_internal_server_error(self, client, mock_app_state):
        """内部サーバーエラーのテスト"""
        # RAGエンジンでエラーを発生させる
        mock_app_state.rag_engine.query.side_effect = Exception("テストエラー")
        
        request_data = {
            "question": "テスト質問",
            "session_id": "test_session"
        }
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
    
    def test_validation_error_handling(self, client, mock_app_state):
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
    
    async def test_async_query_processing(self, mock_app_state):
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
        
        # 非同期処理を実行
        response = await query_documents(
            request=request,
            background_tasks=background_tasks,
            rag_engine=mock_app_state.rag_engine,
            chat_manager=mock_app_state.chat_manager
        )
        
        # レスポンスの検証
        assert isinstance(response, QueryResponse)
        assert response.answer == "テスト回答"
        assert response.session_id == "test_session"
        assert len(response.sources) > 0
        
        # RAGエンジンが呼び出されたことを確認
        mock_app_state.rag_engine.query.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])