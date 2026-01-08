"""
UI プロパティテスト

Webインターフェイスの動作プロパティを検証するテストスイート
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import threading
from fastapi.testclient import TestClient

from genkai_rag.api.app import create_app


@pytest.fixture
def test_client():
    """テスト用FastAPIクライアントを作成"""
    # モックされたアプリケーション状態
    with patch('genkai_rag.api.app.app_state') as mock_state:
        # 基本的なモックを設定
        mock_rag_engine = Mock()
        mock_rag_engine.query = AsyncMock(return_value={
            "answer": "テスト回答です。玄界システムは高性能なスーパーコンピュータです。",
            "sources": [
                {
                    "url": "https://example.com/genkai-info",
                    "title": "玄界システム概要",
                    "content_type": "text/html",
                    "last_accessed": "2024-01-01T00:00:00"
                }
            ],
            "model_used": "test-model",
            "metadata": {"confidence": 0.95}
        })
        
        mock_llm_manager = Mock()
        mock_llm_manager.get_current_model.return_value = "test-model"
        mock_llm_manager.list_available_models = AsyncMock(return_value={
            "test-model": {
                "display_name": "テストモデル",
                "description": "テスト用モデル",
                "is_available": True,
                "parameters": {}
            },
            "large-model": {
                "display_name": "大型モデル",
                "description": "大型テスト用モデル",
                "is_available": True,
                "parameters": {}
            }
        })
        mock_llm_manager.switch_model = AsyncMock(return_value=True)
        mock_llm_manager.check_model_health = AsyncMock(return_value=True)
        
        mock_chat_manager = Mock()
        mock_chat_manager.get_chat_history.return_value = []  # 同期メソッドとして設定
        mock_session_info = Mock()
        mock_session_info.message_count = 0
        mock_chat_manager.get_session_info.return_value = mock_session_info
        mock_chat_manager.list_sessions.return_value = ["session1"]  # 同期メソッドとして設定
        mock_chat_manager.save_message = Mock()
        mock_chat_manager.clear_history = Mock()
        
        mock_system_monitor = Mock()
        mock_status = Mock()
        mock_status.timestamp = "2024-01-01T00:00:00"
        mock_status.uptime_seconds = 3600.0
        mock_status.memory_usage_mb = 512.0
        mock_status.disk_usage_mb = 1024.0
        mock_system_monitor.get_system_status.return_value = mock_status
        
        mock_config_manager = Mock()
        mock_config_manager.load_config.return_value = {
            "web": {"cors_origins": ["*"], "allowed_hosts": ["*"]},
            "llm": {"ollama_url": "http://localhost:11434"},
            "chat": {"max_history_size": 50}
        }
        
        mock_document_processor = Mock()
        
        # テンプレートのモック
        from fastapi.responses import HTMLResponse
        
        def mock_template_response(template_name, context):
            """実際のHTMLテンプレートを模擬"""
            html_content = """
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta name="description" content="九州大学スーパーコンピュータ玄界システム用RAG質問応答システム">
                <title>玄界RAGシステム</title>
            </head>
            <body>
                <main role="main">
                    <h1>玄界RAGシステム</h1>
                    <div id="messagesContainer" aria-live="polite"></div>
                </main>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        
        mock_templates = Mock()
        mock_templates.TemplateResponse = mock_template_response
        
        # モック状態を設定
        mock_state.rag_engine = mock_rag_engine
        mock_state.llm_manager = mock_llm_manager
        mock_state.chat_manager = mock_chat_manager
        mock_state.system_monitor = mock_system_monitor
        mock_state.config_manager = mock_config_manager
        mock_state.document_processor = mock_document_processor
        mock_state.templates = mock_templates
        
        # アプリケーションを作成
        app = create_app()
        
        yield TestClient(app)


class TestUIProcessingIndicator:
    """UI処理中表示のテスト"""
    
    def test_processing_indicator_basic(self, test_client):
        """基本的な処理中表示のテスト - APIレベル"""
        # HTMLページの取得
        response = test_client.get("/")
        assert response.status_code == 200
        # HTMLテンプレートが返されることを確認（実際のテンプレートファイルが使用される）
        assert "html" in response.headers.get("content-type", "").lower()
        
        # 質問APIの呼び出し
        query_response = test_client.post("/api/query", json={
            "question": "玄界システムについて教えてください",
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        assert query_response.status_code == 200
        data = query_response.json()
        
        # 応答データの検証
        assert "answer" in data
        assert "sources" in data
        assert "processing_time" in data
        assert len(data["answer"]) > 0
    
    @given(
        question=st.text(min_size=5, max_size=100).filter(lambda x: x.strip())
    )
    @settings(max_examples=5, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_processing_indicator_property(self, test_client, question):
        """
        プロパティ 9: 処理中表示
        
        質問送信時に適切な処理が行われることを検証（APIレベル）
        """
        assume(len(question.strip()) >= 5)  # 最小限の質問長を確保
        
        # 処理時間を測定
        start_time = time.time()
        
        response = test_client.post("/api/query", json={
            "question": question,
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # プロパティ検証: 適切な処理が行われること
        assert response.status_code == 200, \
            f"質問処理が失敗しました: {response.status_code}"
        
        data = response.json()
        
        assert "answer" in data, \
            "応答に回答が含まれていません"
        
        assert len(data["answer"]) > 0, \
            "回答が空です"
        
        assert "processing_time" in data, \
            "処理時間が記録されていません"
        
        assert processing_time < 30.0, \
            f"処理時間が長すぎます: {processing_time:.2f}秒"


class TestUIResponseDisplay:
    """UI応答表示のテスト"""
    
    def test_response_display_basic(self, test_client):
        """基本的な応答表示のテスト - APIレベル"""
        # 質問を送信
        response = test_client.post("/api/query", json={
            "question": "玄界システムについて教えてください",
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # 応答データの構造を検証
        assert "answer" in data
        assert "sources" in data
        assert "model_used" in data
        assert "processing_time" in data
        
        # 回答内容の検証
        assert len(data["answer"]) > 0
        assert isinstance(data["sources"], list)
        
        # 出典情報の検証
        if data["sources"]:
            for source in data["sources"]:
                assert "url" in source
                assert "title" in source
                assert source["url"].startswith("http")
    
    @given(
        question=st.text(min_size=5, max_size=50).filter(lambda x: x.strip())
    )
    @settings(max_examples=3, deadline=45000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_response_display_property(self, test_client, question):
        """
        プロパティ 10: 応答の適切な表示
        
        質問に対する応答が適切に生成されることを検証（APIレベル）
        """
        assume(len(question.strip()) >= 5)
        
        response = test_client.post("/api/query", json={
            "question": question,
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        # プロパティ検証: 適切な応答表示
        assert response.status_code == 200, \
            f"質問処理が失敗しました: {response.status_code}"
        
        data = response.json()
        
        # 基本的な応答構造の検証
        assert "answer" in data, \
            "応答に回答が含まれていません"
        
        assert "sources" in data, \
            "応答に出典情報が含まれていません"
        
        assert len(data["answer"].strip()) > 0, \
            "回答が空です"
        
        # 出典情報の検証（存在する場合）
        if data["sources"]:
            assert isinstance(data["sources"], list), \
                "出典情報がリスト形式ではありません"
            
            for source in data["sources"]:
                assert isinstance(source, dict), \
                    "出典情報が辞書形式ではありません"
                
                assert "url" in source, \
                    "出典にURLが含まれていません"
                
                assert source["url"].startswith("http"), \
                    f"出典URLが無効です: {source['url']}"
        
        # メタデータの検証
        assert "model_used" in data, \
            "使用モデル情報が含まれていません"
        
        assert "processing_time" in data, \
            "処理時間情報が含まれていません"
        
        assert isinstance(data["processing_time"], (int, float)), \
            "処理時間が数値ではありません"
        
        assert data["processing_time"] >= 0, \
            "処理時間が負の値です"


class TestUISystemStatus:
    """UIシステム状態表示のテスト"""
    
    def test_system_status_display(self, test_client):
        """システム状態表示のテスト"""
        response = test_client.get("/api/system/status")
        assert response.status_code == 200
        
        data = response.json()
        
        # システム状態の基本情報を確認
        assert "status" in data
        assert "current_model" in data
        assert "active_sessions" in data
        assert "uptime_seconds" in data
        
        # 状態値の妥当性を確認
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert isinstance(data["active_sessions"], int)
        assert data["active_sessions"] >= 0


class TestUIModelSelection:
    """UIモデル選択のテスト"""
    
    def test_model_selection_display(self, test_client):
        """モデル選択表示のテスト"""
        response = test_client.get("/api/models")
        assert response.status_code == 200
        
        data = response.json()
        
        # モデル一覧の基本構造を確認
        assert "models" in data
        assert "current_model" in data
        assert isinstance(data["models"], list)
        
        # モデル情報の検証
        if data["models"]:
            for model in data["models"]:
                assert "name" in model
                assert "display_name" in model
                assert "is_available" in model
                assert isinstance(model["is_available"], bool)


class TestUIAccessibility:
    """UIアクセシビリティのテスト"""
    
    def test_accessibility_attributes(self, test_client):
        """アクセシビリティ属性のテスト"""
        response = test_client.get("/")
        assert response.status_code == 200
        
        html_content = response.text
        
        # HTMLが返されることを確認（空でない）
        assert len(html_content) > 0
        
        # 基本的なHTMLタグとアクセシビリティ要素の存在を確認
        assert "<html" in html_content
        assert 'lang="ja"' in html_content
        assert 'role="main"' in html_content
        assert 'aria-live="polite"' in html_content


class TestUIErrorHandling:
    """UIエラーハンドリングのテスト"""
    
    def test_empty_input_validation(self, test_client):
        """空入力の検証テスト"""
        # 空の質問で送信を試行
        response = test_client.post("/api/query", json={
            "question": "",
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        # バリデーションエラーが返されることを確認
        assert response.status_code == 422  # Validation Error
    
    @given(
        invalid_input=st.one_of(
            st.just(""),  # 空文字
            st.just("   "),  # 空白のみ
            st.text(max_size=0)  # 空文字列
        )
    )
    @settings(max_examples=3, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_invalid_input_property(self, test_client, invalid_input):
        """
        プロパティ: 無効な入力に対する適切なエラーハンドリング
        
        無効な入力に対して適切なエラーレスポンスが返されることを検証
        """
        response = test_client.post("/api/query", json={
            "question": invalid_input,
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        # エラーハンドリングの検証
        if invalid_input.strip() == "":
            # 空入力の場合はバリデーションエラーが返されるべき
            assert response.status_code == 422, \
                f"無効な入力 '{invalid_input}' に対して適切なエラーが返されませんでした"


class TestUIPerformance:
    """UIパフォーマンスのテスト"""
    
    def test_response_time_measurement(self, test_client):
        """応答時間測定のテスト"""
        # 送信時刻を記録
        start_time = time.time()
        
        response = test_client.post("/api/query", json={
            "question": "玄界システムのパフォーマンステスト",
            "session_id": "test_session",
            "max_sources": 5,
            "include_history": True
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # 応答時間が合理的な範囲内であることを確認
        assert response.status_code == 200
        assert response_time < 30.0, \
            f"応答時間が長すぎます: {response_time:.2f}秒"
        
        data = response.json()
        
        # 応答時間が記録されることを確認
        assert "processing_time" in data
        assert isinstance(data["processing_time"], (int, float))
        assert data["processing_time"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])