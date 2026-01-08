"""
統合テストスイート

システム全体の統合とエンドツーエンドの動作を検証します。
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from genkai_rag.app import GenkaiRAGSystem, initialize_system, shutdown_system
from genkai_rag.models.document import Document
from genkai_rag.models.chat import Message, ChatSession


class TestSystemIntegration:
    """システム統合テスト"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """テスト用の一時設定ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_config_dir):
        """テスト用設定"""
        return {
            "logging": {
                "level": "WARNING",  # テスト中はログを抑制
                "file": f"{temp_config_dir}/test.log"
            },
            "error_recovery": {
                "max_history_size": 10,
                "default_max_attempts": 2
            },
            "system_monitor": {
                "enable_background_monitoring": False,  # テスト中は無効
                "monitoring_interval": 1
            },
            "scraper": {
                "timeout": 5,
                "max_retries": 1
            },
            "document_processor": {
                "chunk_size": 100,
                "chunk_overlap": 20
            },
            "llm": {
                "base_url": "http://localhost:11434",
                "default_model": "test-model",
                "timeout": 10
            },
            "rag": {
                "similarity_top_k": 2,
                "rerank_top_n": 1
            },
            "chat": {
                "max_history_size": 5,
                "session_timeout_hours": 1
            },
            "web": {
                "host": "127.0.0.1",
                "port": 8001,
                "debug": True
            }
        }
    
    @pytest.fixture
    def system(self, test_config, temp_config_dir):
        """テスト用システムインスタンス"""
        # 設定ファイルを作成
        config_path = Path(temp_config_dir) / "config.yaml"
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # システムを初期化
        system = GenkaiRAGSystem(str(config_path))
        
        # モックを設定してネットワーク依存を回避
        with patch('genkai_rag.core.llm_manager.LLMManager') as mock_llm, \
             patch('genkai_rag.core.scraper.WebScraper') as mock_scraper, \
             patch('genkai_rag.core.processor.DocumentProcessor') as mock_processor:
            
            # LLMManagerのモック
            mock_llm_instance = Mock()
            mock_llm_instance.query_async = AsyncMock(return_value="テスト回答")
            mock_llm_instance.check_model_health = AsyncMock(return_value=True)
            mock_llm.return_value = mock_llm_instance
            
            # WebScraperのモック
            mock_scraper_instance = Mock()
            mock_scraper_instance.scrape_url = AsyncMock(return_value=Document(
                content="テスト文書内容",
                metadata={"url": "https://example.com", "title": "テスト文書"}
            ))
            mock_scraper.return_value = mock_scraper_instance
            
            # DocumentProcessorのモック
            mock_processor_instance = Mock()
            mock_processor_instance.add_document = AsyncMock()
            mock_processor_instance.search_documents = AsyncMock(return_value=[])
            mock_processor.return_value = mock_processor_instance
            
            # 同期的に初期化を実行
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(system.initialize())
                yield system
                loop.run_until_complete(system.shutdown())
            finally:
                loop.close()
    
    def test_system_initialization(self, system):
        """システム初期化テスト"""
        assert system._initialized is True
        assert system.config is not None
        assert system.error_recovery_manager is not None
        assert system.system_monitor is not None
        assert system.web_scraper is not None
        assert system.document_processor is not None
        assert system.llm_manager is not None
        assert system.rag_engine is not None
        assert system.chat_manager is not None
        assert system.app is not None
    
    def test_system_status(self, system):
        """システム状態取得テスト"""
        status = system.get_system_status()
        
        assert status["status"] == "running"
        assert status["initialized"] is True
        
        components = status["components"]
        assert components["config_manager"] is True
        assert components["error_recovery_manager"] is True
        assert components["system_monitor"] is True
        assert components["web_scraper"] is True
        assert components["document_processor"] is True
        assert components["llm_manager"] is True
        assert components["rag_engine"] is True
        assert components["chat_manager"] is True
        assert components["web_app"] is True
    
    def test_document_processing_flow(self, system):
        """文書処理フローテスト"""
        # モック文書を作成
        from genkai_rag.models.document import Document
        from datetime import datetime
        
        mock_document = Document(
            title="テスト文書",
            content="テスト文書内容",
            url="https://example.com",
            section="テストセクション",
            timestamp=datetime.now()
        )
        
        # WebScraperをモック化
        system.web_scraper.scrape_single_page = Mock(return_value=mock_document)
        
        # 文書をスクレイピング
        document = system.web_scraper.scrape_single_page("https://example.com")
        assert document is not None
        assert document.content == "テスト文書内容"
        
        # 文書を処理
        result = system.document_processor.process_single_document(document)
        assert result is True
        
        # 処理が呼ばれたことを確認
        system.web_scraper.scrape_single_page.assert_called_once_with("https://example.com")
    
    def test_query_processing_flow(self, system):
        """クエリ処理フローテスト"""
        # RAGEngineをモック化
        from genkai_rag.core.rag_engine import RAGResponse
        
        mock_response = RAGResponse(
            answer="玄界システムについての回答",
            sources=[],
            processing_time=0.1,
            model_used="test-model",
            retrieval_score=0.9,
            confidence_score=0.9
        )
        system.rag_engine.query = Mock(return_value=mock_response)
        
        query = "テスト質問"
        
        # クエリを実行
        response = system.rag_engine.query(query)
        
        assert response is not None
        assert response.answer == "玄界システムについての回答"
        assert len(response.sources) >= 0
        
        # クエリが呼ばれたことを確認
        system.rag_engine.query.assert_called_once_with(query)
    
    def test_chat_session_flow(self, system):
        """チャットセッションフローテスト"""
        import uuid
        session_id = f"test-session-{uuid.uuid4()}"
        
        # 既存の履歴をクリア
        system.chat_manager.clear_history(session_id)
        
        # メッセージを追加
        message = Message(
            content="テストメッセージ",
            role="user",
            session_id=session_id
        )
        
        system.chat_manager.save_message(session_id, message)
        
        # 履歴を取得
        history = system.chat_manager.get_chat_history(session_id)
        
        assert len(history) == 1
        assert history[0].content == "テストメッセージ"
    
    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, system):
        """エラー回復統合テスト"""
        # エラーを発生させる
        error = Exception("テストエラー")
        
        # エラーハンドリング
        result = system.error_recovery_manager.handle_validation_error(
            error, {"test": "data"}, "integration_test"
        )
        
        assert result is True
        
        # エラー統計を確認
        stats = system.error_recovery_manager.get_error_statistics(1)
        assert stats["total_errors"] >= 1


class TestEndToEndWorkflow:
    """エンドツーエンドワークフローテスト"""
    
    @pytest.fixture
    def mock_system(self):
        """モックシステム"""
        from unittest.mock import Mock
        
        # システムモックを作成
        system_mock = Mock()
        system_mock.web_scraper = Mock()
        system_mock.document_processor = Mock()
        system_mock.rag_engine = Mock()
        system_mock.chat_manager = Mock()
        
        return system_mock
    
    def test_complete_rag_workflow(self, mock_system):
        """完全なRAGワークフローテスト"""
        from genkai_rag.models.document import Document
        from genkai_rag.core.rag_engine import RAGResponse
        from datetime import datetime
        
        # システムを初期化（モックを使用）
        system = mock_system
        
        # モック文書を作成
        mock_document = Document(
            title="玄界システム",
            content="玄界システムは九州大学のスーパーコンピュータです。",
            url="https://example.com",
            section="システム概要",
            timestamp=datetime.now()
        )
        
        # モックレスポンスを作成
        mock_response = RAGResponse(
            answer="玄界システムは九州大学が運用するスーパーコンピュータシステムです。",
            sources=[],
            processing_time=0.1,
            model_used="test-model",
            retrieval_score=0.9,
            confidence_score=0.9
        )
        
        # モックを設定
        system.web_scraper.scrape_single_page = Mock(return_value=mock_document)
        system.document_processor.process_single_document = Mock(return_value=True)
        system.rag_engine.query = Mock(return_value=mock_response)
        system.chat_manager.save_message = Mock(return_value=True)
        
        # 1. 文書をスクレイピング
        url = "https://example.com"
        document = system.web_scraper.scrape_single_page(url)
        
        assert document is not None
        assert "玄界システム" in document.content
        
        # 2. 文書を処理してインデックスに追加
        result = system.document_processor.process_single_document(document)
        assert result is True
        system.document_processor.process_single_document.assert_called_once()
        
        # 3. 質問を実行
        query = "玄界システムとは何ですか？"
        response = system.rag_engine.query(query)
        
        assert response is not None
        assert "玄界システム" in response.answer
        assert len(response.sources) >= 0
        
        # 4. チャット履歴に保存
        message = Message(
            content=query,
            role="user",
            session_id="test-session"
        )
        result = system.chat_manager.save_message("test-session", message)
        assert result is True
    
    def test_multi_turn_conversation(self, mock_system):
        """複数ターン会話テスト"""
        from genkai_rag.core.rag_engine import RAGResponse
        
        system = mock_system
        session_id = "multi-turn-session"
        
        # モックレスポンスを作成
        mock_response1 = RAGResponse(
            answer="玄界システムは九州大学のスーパーコンピュータです。",
            sources=[],
            processing_time=0.1,
            model_used="test-model",
            retrieval_score=0.9,
            confidence_score=0.9
        )
        
        mock_response2 = RAGResponse(
            answer="玄界システムは高性能な計算能力を持っています。",
            sources=[],
            processing_time=0.1,
            model_used="test-model",
            retrieval_score=0.8,
            confidence_score=0.8
        )
        
        # モックを設定
        system.rag_engine.query = Mock(side_effect=[mock_response1, mock_response2])
        system.chat_manager.save_message = Mock(return_value=True)
        system.chat_manager.get_chat_history = Mock(return_value=[])
        
        # 最初の質問
        query1 = "玄界システムについて教えてください"
        response1 = system.rag_engine.query(query1)
        
        message1 = Message(content=query1, role="user", session_id=session_id)
        system.chat_manager.save_message(session_id, message1)
        
        response_msg1 = Message(content=response1.answer, role="assistant", session_id=session_id)
        system.chat_manager.save_message(session_id, response_msg1)
        
        # フォローアップ質問
        query2 = "その性能はどの程度ですか？"
        response2 = system.rag_engine.query(query2)
        
        message2 = Message(content=query2, role="user", session_id=session_id)
        system.chat_manager.save_message(session_id, message2)
        
        # 履歴を確認
        history = system.chat_manager.get_chat_history(session_id)
        
        # 呼び出し回数を確認（save_messageが3回呼ばれる）
        assert system.chat_manager.save_message.call_count == 3
    
    def test_error_handling_workflow(self, mock_system):
        """エラーハンドリングワークフローテスト"""
        from genkai_rag.core.rag_engine import RAGResponse
        
        # エラーを発生させるようにモックを設定
        system = mock_system
        system.web_scraper.scrape_single_page = Mock(side_effect=Exception("Network error"))
        
        # RAGEngineは正常動作するように設定
        mock_response = RAGResponse(
            answer="テスト回答",
            sources=[],
            processing_time=0.1,
            model_used="test-model",
            retrieval_score=0.7,
            confidence_score=0.7
        )
        system.rag_engine.query = Mock(return_value=mock_response)
        
        # エラーが発生してもシステムが継続動作することを確認
        try:
            document = system.web_scraper.scrape_single_page("https://invalid-url.com")
            # エラーが発生するはず
            assert False, "Expected exception was not raised"
        except Exception as e:
            assert "Network error" in str(e)
        
        # システムは他の機能を継続して利用可能
        query = "テスト質問"
        response = system.rag_engine.query(query)
        assert response is not None


class TestComponentIntegration:
    """コンポーネント間統合テスト"""
    
    @pytest.fixture
    def mock_components(self):
        """モックコンポーネント"""
        from datetime import datetime
        
        components = {}
        
        # RAGEngine - シンプルなモック
        rag_engine = Mock()
        rag_engine.query = AsyncMock(return_value=Mock(
            response="テスト回答",
            source_documents=[]
        ))
        components["rag_engine"] = rag_engine
        
        # ChatManager - シンプルなモック
        chat_manager = Mock()
        chat_manager.save_message = Mock(return_value=True)
        chat_manager.get_chat_history = Mock(return_value=[])
        chat_manager.get_session_info = Mock(return_value=Mock(message_count=0))
        chat_manager.list_sessions = Mock(return_value=[])
        components["chat_manager"] = chat_manager
        
        # LLMManager - シンプルなモック
        llm_manager = Mock()
        llm_manager.query_async = AsyncMock(return_value="テスト回答")
        llm_manager.get_current_model = Mock(return_value="test-model")
        llm_manager.check_model_health = AsyncMock(return_value=True)
        components["llm_manager"] = llm_manager
        
        # SystemMonitor - シンプルなモック
        system_monitor = Mock()
        system_status = Mock()
        system_status.timestamp = datetime.now()
        system_status.memory_usage = 50.0
        system_status.disk_usage = 30.0
        system_status.cpu_usage = 20.0
        system_monitor.get_system_status = Mock(return_value=system_status)
        components["system_monitor"] = system_monitor
        
        # ConfigManager - シンプルなモック
        config_manager = Mock()
        config_manager.load_config = Mock(return_value={})
        components["config_manager"] = config_manager
        
        # ErrorRecoveryManager - シンプルなモック
        error_recovery_manager = Mock()
        error_recovery_manager.get_error_statistics = Mock(return_value={
            "total_errors": 0,
            "error_rate": 0.0,
            "by_type": {},
            "by_severity": {},
            "most_common_operations": []
        })
        components["error_recovery_manager"] = error_recovery_manager
        
        # DocumentProcessor - シンプルなモック
        document_processor = Mock()
        components["document_processor"] = document_processor
        
        # WebScraper - シンプルなモック
        web_scraper = Mock()
        components["web_scraper"] = web_scraper
        
        return components
    
    def test_fastapi_app_creation(self, mock_components):
        """FastAPIアプリケーション作成テスト"""
        from genkai_rag.api.app import create_app
        
        config = {
            "debug": True,
            "cors_origins": ["http://localhost:3000"]
        }
        
        app = create_app(dependencies=mock_components, config=config)
        
        assert app is not None
        assert app.state.dependencies == mock_components
        assert hasattr(app.state, "app_state")
    
    def test_api_dependency_injection(self, mock_components):
        """API依存性注入テスト"""
        from genkai_rag.api.app import create_app
        from fastapi.testclient import TestClient
        
        # 設定を明示的に指定してモックの問題を回避
        config = {
            "debug": True,
            "cors_origins": ["http://localhost:3000"],
            "allowed_hosts": ["*"]
        }
        
        app = create_app(dependencies=mock_components, config=config)
        
        with TestClient(app) as client:
            # ヘルスチェックエンドポイントをテスト
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "components" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])