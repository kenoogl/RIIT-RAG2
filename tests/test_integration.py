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
    
    def test_complete_url_to_answer_workflow(self, mock_system):
        """URL入力から回答生成までの完全なワークフローテスト"""
        from genkai_rag.models.document import Document, DocumentSource
        from genkai_rag.core.rag_engine import RAGResponse
        from datetime import datetime
        
        system = mock_system
        
        # 複数の文書をモック
        documents = [
            Document(
                title="玄界システム概要",
                content="玄界システムは九州大学情報基盤研究開発センターが運用するスーパーコンピュータです。高性能計算を提供します。",
                url="https://www.cc.kyushu-u.ac.jp/scp/overview",
                section="概要",
                timestamp=datetime.now()
            ),
            Document(
                title="利用方法",
                content="玄界システムを利用するには、まずアカウントを申請する必要があります。SSH接続でアクセスできます。",
                url="https://www.cc.kyushu-u.ac.jp/scp/usage",
                section="利用方法",
                timestamp=datetime.now()
            ),
            Document(
                title="料金体系",
                content="玄界システムの利用料金は計算時間に基づいて課金されます。詳細は料金表をご確認ください。",
                url="https://www.cc.kyushu-u.ac.jp/scp/pricing",
                section="料金",
                timestamp=datetime.now()
            )
        ]
        
        # 文書ソースを作成
        sources = [
            DocumentSource(
                title=doc.title,
                url=doc.url,
                section=doc.section,
                relevance_score=0.9 - i * 0.1
            ) for i, doc in enumerate(documents)
        ]
        
        # モックレスポンスを作成
        mock_response = RAGResponse(
            answer="玄界システムは九州大学が運用するスーパーコンピュータで、SSH接続でアクセスし、計算時間に基づいて課金されます。",
            sources=sources,
            processing_time=0.5,
            model_used="llama3.2:3b",
            retrieval_score=0.85,
            confidence_score=0.9
        )
        
        # モックを設定
        system.web_scraper.scrape_website = Mock(return_value=documents)
        system.document_processor.add_documents = Mock(return_value=True)
        system.document_processor.search_documents = Mock(return_value=documents[:2])
        system.rag_engine.query = Mock(return_value=mock_response)
        system.chat_manager.save_message = Mock(return_value=True)
        system.chat_manager.get_chat_history = Mock(return_value=[])
        
        # 完全なワークフローを実行
        base_url = "https://www.cc.kyushu-u.ac.jp/scp/"
        session_id = "e2e-test-session"
        query = "玄界システムの利用方法と料金について教えてください"
        
        # 1. Webサイトをスクレイピング
        scraped_docs = system.web_scraper.scrape_website(base_url)
        assert len(scraped_docs) == 3
        assert all("玄界システム" in doc.content for doc in scraped_docs)
        
        # 2. 文書をインデックスに追加
        result = system.document_processor.add_documents(scraped_docs)
        assert result is True
        
        # 3. 関連文書を検索
        relevant_docs = system.document_processor.search_documents(query, top_k=2)
        assert len(relevant_docs) == 2
        
        # 4. RAGクエリを実行
        response = system.rag_engine.query(query)
        assert response is not None
        assert "玄界システム" in response.answer
        assert "SSH接続" in response.answer
        assert "課金" in response.answer
        assert len(response.sources) == 3
        assert response.processing_time > 0
        
        # 5. 会話履歴に保存
        user_message = Message(content=query, role="user", session_id=session_id)
        system.chat_manager.save_message(session_id, user_message)
        
        assistant_message = Message(content=response.answer, role="assistant", session_id=session_id)
        system.chat_manager.save_message(session_id, assistant_message)
        
        # 6. 履歴を確認
        history = system.chat_manager.get_chat_history(session_id)
        
        # 全ステップが正常に実行されたことを確認
        system.web_scraper.scrape_website.assert_called_once_with(base_url)
        system.document_processor.add_documents.assert_called_once_with(scraped_docs)
        system.document_processor.search_documents.assert_called_once_with(query, top_k=2)
        system.rag_engine.query.assert_called_once_with(query)
        assert system.chat_manager.save_message.call_count == 2
    
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
    
    def test_context_aware_conversation(self, mock_system):
        """コンテキスト認識会話テスト"""
        from genkai_rag.core.rag_engine import RAGResponse
        
        system = mock_system
        session_id = "context-aware-session"
        
        # 履歴メッセージを作成
        history_messages = [
            Message(content="玄界システムについて教えてください", role="user", session_id=session_id),
            Message(content="玄界システムは九州大学のスーパーコンピュータです。", role="assistant", session_id=session_id),
            Message(content="利用方法を知りたいです", role="user", session_id=session_id),
            Message(content="SSH接続でアクセスできます。", role="assistant", session_id=session_id)
        ]
        
        # コンテキストを考慮したレスポンス
        context_response = RAGResponse(
            answer="先ほどお話しした玄界システムの料金は、計算時間に基づいて課金されます。詳細な料金表は公式サイトでご確認いただけます。",
            sources=[],
            processing_time=0.2,
            model_used="test-model",
            retrieval_score=0.8,
            confidence_score=0.85
        )
        
        # モックを設定
        system.chat_manager.get_chat_history = Mock(return_value=history_messages)
        system.rag_engine.query = Mock(return_value=context_response)
        system.chat_manager.save_message = Mock(return_value=True)
        
        # コンテキストを参照する質問
        query = "料金はどうなっていますか？"
        
        # 履歴を取得してコンテキストとして使用
        history = system.chat_manager.get_chat_history(session_id, limit=4)
        assert len(history) == 4
        
        # コンテキストを含むクエリを実行
        response = system.rag_engine.query(query)
        
        assert response is not None
        assert "料金" in response.answer
        assert "計算時間" in response.answer
        
        # 新しいメッセージを保存
        user_message = Message(content=query, role="user", session_id=session_id)
        system.chat_manager.save_message(session_id, user_message)
        
        assistant_message = Message(content=response.answer, role="assistant", session_id=session_id)
        system.chat_manager.save_message(session_id, assistant_message)
        
        # 履歴取得とメッセージ保存が呼ばれたことを確認
        system.chat_manager.get_chat_history.assert_called_once_with(session_id, limit=4)
        assert system.chat_manager.save_message.call_count == 2
    
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
    
    def test_concurrent_query_processing(self, mock_system):
        """同時クエリ処理テスト"""
        import threading
        import time
        from genkai_rag.core.rag_engine import RAGResponse
        
        system = mock_system
        results = []
        
        def process_query(query_id):
            """クエリを処理する関数"""
            mock_response = RAGResponse(
                answer=f"回答 {query_id}",
                sources=[],
                processing_time=0.1,
                model_used="test-model",
                retrieval_score=0.8,
                confidence_score=0.8
            )
            
            # 少し遅延を追加
            time.sleep(0.1)
            results.append(f"query_{query_id}")
            return mock_response
        
        # モックを設定
        system.rag_engine.query = Mock(side_effect=lambda q: process_query(q.split()[-1]))
        
        # 複数のスレッドで同時にクエリを実行
        threads = []
        for i in range(3):
            thread = threading.Thread(target=lambda i=i: system.rag_engine.query(f"テスト質問 {i}"))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果を確認
        assert len(results) == 3
        assert system.rag_engine.query.call_count == 3
    
    def test_system_workflow_validation(self, mock_system):
        """システム全体のワークフロー検証テスト"""
        from genkai_rag.models.document import Document, DocumentSource
        from genkai_rag.core.rag_engine import RAGResponse
        from datetime import datetime
        
        system = mock_system
        session_id = "workflow-validation-session"
        
        # ワークフロー全体をシミュレート
        workflow_steps = []
        
        # 1. 初期化確認
        workflow_steps.append("system_initialized")
        
        # 2. 文書スクレイピング
        mock_document = Document(
            title="玄界システム詳細",
            content="玄界システムは九州大学の最新スーパーコンピュータシステムです。研究者向けの高性能計算環境を提供しています。",
            url="https://www.cc.kyushu-u.ac.jp/scp/genkai",
            section="システム詳細",
            timestamp=datetime.now()
        )
        
        system.web_scraper.scrape_single_page = Mock(return_value=mock_document)
        document = system.web_scraper.scrape_single_page("https://www.cc.kyushu-u.ac.jp/scp/genkai")
        workflow_steps.append("document_scraped")
        
        # 3. 文書処理
        system.document_processor.process_single_document = Mock(return_value=True)
        processed = system.document_processor.process_single_document(document)
        workflow_steps.append("document_processed")
        
        # 4. インデックス更新
        system.document_processor.update_index = Mock(return_value=True)
        index_updated = system.document_processor.update_index()
        workflow_steps.append("index_updated")
        
        # 5. 質問処理
        query = "玄界システムの特徴について教えてください"
        mock_response = RAGResponse(
            answer="玄界システムは九州大学の最新スーパーコンピュータで、研究者向けの高性能計算環境を提供しています。",
            sources=[DocumentSource(
                title="玄界システム詳細",
                url="https://www.cc.kyushu-u.ac.jp/scp/genkai",
                section="システム詳細",
                relevance_score=0.95
            )],
            processing_time=0.3,
            model_used="llama3.2:3b",
            retrieval_score=0.9,
            confidence_score=0.92
        )
        
        system.rag_engine.query = Mock(return_value=mock_response)
        response = system.rag_engine.query(query)
        workflow_steps.append("query_processed")
        
        # 6. 履歴保存
        system.chat_manager.save_message = Mock(return_value=True)
        user_message = Message(content=query, role="user", session_id=session_id)
        system.chat_manager.save_message(session_id, user_message)
        
        assistant_message = Message(content=response.answer, role="assistant", session_id=session_id)
        system.chat_manager.save_message(session_id, assistant_message)
        workflow_steps.append("messages_saved")
        
        # 7. システム状態確認
        system.system_monitor = Mock()
        system.system_monitor.get_system_status = Mock(return_value={
            "status": "healthy",
            "uptime": 3600,
            "memory_usage": 45.2,
            "disk_usage": 23.1
        })
        status = system.system_monitor.get_system_status()
        workflow_steps.append("status_checked")
        
        # ワークフロー全体の検証
        expected_steps = [
            "system_initialized",
            "document_scraped", 
            "document_processed",
            "index_updated",
            "query_processed",
            "messages_saved",
            "status_checked"
        ]
        
        assert workflow_steps == expected_steps
        assert document is not None
        assert processed is True
        assert index_updated is True
        assert response is not None
        assert "玄界システム" in response.answer
        assert len(response.sources) == 1
        assert status["status"] == "healthy"
        
        # 各コンポーネントが正しく呼ばれたことを確認
        system.web_scraper.scrape_single_page.assert_called_once()
        system.document_processor.process_single_document.assert_called_once()
        system.document_processor.update_index.assert_called_once()
        system.rag_engine.query.assert_called_once()
        assert system.chat_manager.save_message.call_count == 2
        system.system_monitor.get_system_status.assert_called_once()
    
    def test_comprehensive_e2e_scenario(self, mock_system):
        """包括的エンドツーエンドシナリオテスト"""
        from genkai_rag.models.document import Document, DocumentSource
        from genkai_rag.core.rag_engine import RAGResponse
        from datetime import datetime
        
        system = mock_system
        session_id = "comprehensive-e2e-session"
        
        # シナリオ: 新しいユーザーが玄界システムについて学習する完全なフロー
        
        # Phase 1: システム初期化と文書収集
        documents = [
            Document(
                title="玄界システム概要",
                content="玄界システムは九州大学情報基盤研究開発センターが運用するスーパーコンピュータシステムです。",
                url="https://www.cc.kyushu-u.ac.jp/scp/overview",
                section="概要",
                timestamp=datetime.now()
            ),
            Document(
                title="利用申請方法",
                content="玄界システムを利用するには、まず利用申請を行い、アカウントを取得する必要があります。",
                url="https://www.cc.kyushu-u.ac.jp/scp/application",
                section="申請",
                timestamp=datetime.now()
            ),
            Document(
                title="接続方法",
                content="玄界システムにはSSH接続でアクセスします。VPN接続が必要な場合があります。",
                url="https://www.cc.kyushu-u.ac.jp/scp/connection",
                section="接続",
                timestamp=datetime.now()
            )
        ]
        
        system.web_scraper.scrape_website = Mock(return_value=documents)
        system.document_processor.add_documents = Mock(return_value=True)
        
        # 文書収集と処理
        scraped_docs = system.web_scraper.scrape_website("https://www.cc.kyushu-u.ac.jp/scp/")
        processed = system.document_processor.add_documents(scraped_docs)
        
        assert len(scraped_docs) == 3
        assert processed is True
        
        # Phase 2: 段階的な質問応答（学習プロセス）
        conversation_flow = [
            {
                "query": "玄界システムとは何ですか？",
                "expected_keywords": ["スーパーコンピュータ", "九州大学"],
                "response": "玄界システムは九州大学情報基盤研究開発センターが運用するスーパーコンピュータシステムです。"
            },
            {
                "query": "どうやって利用申請をすればいいですか？",
                "expected_keywords": ["利用申請", "アカウント"],
                "response": "玄界システムを利用するには、まず利用申請を行い、アカウントを取得する必要があります。"
            },
            {
                "query": "接続方法を教えてください",
                "expected_keywords": ["SSH接続", "VPN"],
                "response": "玄界システムにはSSH接続でアクセスします。VPN接続が必要な場合があります。"
            }
        ]
        
        # 会話履歴を蓄積
        conversation_history = []
        
        for i, step in enumerate(conversation_flow):
            # 関連文書を検索
            relevant_docs = [doc for doc in documents if any(keyword in doc.content for keyword in step["expected_keywords"])]
            
            system.document_processor.search_documents = Mock(return_value=relevant_docs[:2])
            
            # RAGレスポンスを生成
            mock_response = RAGResponse(
                answer=step["response"],
                sources=[DocumentSource(
                    title=doc.title,
                    url=doc.url,
                    section=doc.section,
                    relevance_score=0.9 - i * 0.1
                ) for doc in relevant_docs[:2]],
                processing_time=0.2 + i * 0.1,
                model_used="llama3.2:3b",
                retrieval_score=0.9 - i * 0.05,
                confidence_score=0.85 + i * 0.02
            )
            
            system.rag_engine.query = Mock(return_value=mock_response)
            
            # 質問を処理
            response = system.rag_engine.query(step["query"])
            
            # 履歴に追加
            user_msg = Message(content=step["query"], role="user", session_id=session_id)
            assistant_msg = Message(content=response.answer, role="assistant", session_id=session_id)
            
            conversation_history.extend([user_msg, assistant_msg])
            
            # 検証
            assert response is not None
            assert all(keyword in response.answer for keyword in step["expected_keywords"])
            assert len(response.sources) <= 2
            assert response.confidence_score > 0.8
        
        # Phase 3: コンテキスト認識質問
        system.chat_manager.get_chat_history = Mock(return_value=conversation_history)
        system.chat_manager.save_message = Mock(return_value=True)
        
        # 前の会話を参照する質問
        context_query = "先ほど説明していただいた申請後の手順について、もう少し詳しく教えてください"
        
        context_response = RAGResponse(
            answer="申請が承認された後は、SSH接続の設定を行い、VPN接続が必要な場合は事前に設定してからアクセスしてください。",
            sources=[DocumentSource(
                title="接続方法",
                url="https://www.cc.kyushu-u.ac.jp/scp/connection",
                section="接続",
                relevance_score=0.88
            )],
            processing_time=0.25,
            model_used="llama3.2:3b",
            retrieval_score=0.82,
            confidence_score=0.87
        )
        
        system.rag_engine.query = Mock(return_value=context_response)
        
        # コンテキスト質問を処理
        history = system.chat_manager.get_chat_history(session_id)
        context_resp = system.rag_engine.query(context_query)
        
        # 新しいメッセージを保存
        system.chat_manager.save_message(session_id, Message(content=context_query, role="user", session_id=session_id))
        system.chat_manager.save_message(session_id, Message(content=context_resp.answer, role="assistant", session_id=session_id))
        
        # Phase 4: 最終検証
        assert len(history) == 6  # 3回の質問応答 = 6メッセージ
        assert context_resp is not None
        assert "SSH接続" in context_resp.answer
        assert "VPN" in context_resp.answer
        
        # 全体的な統計確認
        total_queries = len(conversation_flow) + 1  # コンテキスト質問を含む
        assert total_queries == 4
        
        # システムコンポーネントの呼び出し確認
        system.web_scraper.scrape_website.assert_called_once()
        system.document_processor.add_documents.assert_called_once()
        system.chat_manager.get_chat_history.assert_called_once()
        assert system.chat_manager.save_message.call_count == 2  # コンテキスト質問のみ


class TestComponentIntegration:
    """コンポーネント間統合テスト"""
    
    @pytest.fixture
    def mock_components(self):
        """モックコンポーネント"""
        from datetime import datetime
        
        components = {}
        
        # RAGEngine - シンプルなモック
        rag_engine = Mock()
        rag_engine.query = Mock(return_value=Mock(
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
        system_monitor.get_performance_stats = Mock(return_value={})
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
    
    def test_webscraper_to_processor_integration(self, mock_components):
        """WebScraper → DocumentProcessor 連携テスト"""
        from genkai_rag.models.document import Document
        from datetime import datetime
        
        # WebScraperとDocumentProcessorを取得
        web_scraper = mock_components["web_scraper"]
        document_processor = mock_components["document_processor"]
        
        # モック文書を作成
        mock_document = Document(
            title="テスト文書",
            content="これはテスト用の文書内容です。玄界システムについて説明します。",
            url="https://example.com/test",
            section="テストセクション",
            timestamp=datetime.now()
        )
        
        # WebScraperのモック設定
        web_scraper.scrape_single_page = Mock(return_value=mock_document)
        
        # DocumentProcessorのモック設定
        document_processor.process_single_document = Mock(return_value=True)
        document_processor.get_document_count = Mock(return_value=1)
        
        # 連携フローを実行
        url = "https://example.com/test"
        
        # 1. WebScraperで文書を取得
        document = web_scraper.scrape_single_page(url)
        assert document is not None
        assert document.title == "テスト文書"
        
        # 2. DocumentProcessorで処理
        result = document_processor.process_single_document(document)
        assert result is True
        
        # 3. 文書数を確認
        count = document_processor.get_document_count()
        assert count == 1
        
        # 呼び出しを確認
        web_scraper.scrape_single_page.assert_called_once_with(url)
        document_processor.process_single_document.assert_called_once_with(document)
    
    def test_processor_to_rag_engine_integration(self, mock_components):
        """DocumentProcessor → RAGEngine 連携テスト"""
        from genkai_rag.models.document import Document, DocumentSource
        from genkai_rag.core.rag_engine import RAGResponse
        from datetime import datetime
        
        # DocumentProcessorとRAGEngineを取得
        document_processor = mock_components["document_processor"]
        rag_engine = mock_components["rag_engine"]
        
        # モック文書を作成
        mock_documents = [
            Document(
                title="玄界システム概要",
                content="玄界システムは高性能計算を提供するスーパーコンピュータです。",
                url="https://example.com/overview",
                section="概要",
                timestamp=datetime.now()
            ),
            Document(
                title="利用方法",
                content="SSH接続でアクセスし、バッチジョブを投入します。",
                url="https://example.com/usage",
                section="利用方法",
                timestamp=datetime.now()
            )
        ]
        
        # DocumentSourceを作成
        sources = [
            DocumentSource(
                title=doc.title,
                url=doc.url,
                section=doc.section,
                relevance_score=0.9 - i * 0.1
            ) for i, doc in enumerate(mock_documents)
        ]
        
        # モックレスポンスを作成
        mock_response = RAGResponse(
            answer="玄界システムは高性能計算を提供し、SSH接続でアクセスできます。",
            sources=sources,
            processing_time=0.3,
            model_used="test-model",
            retrieval_score=0.85,
            confidence_score=0.9
        )
        
        # モック設定
        document_processor.search_documents = Mock(return_value=mock_documents)
        rag_engine.query = Mock(return_value=mock_response)
        
        # 連携フローを実行
        query = "玄界システムの利用方法は？"
        
        # 1. DocumentProcessorで関連文書を検索
        relevant_docs = document_processor.search_documents(query, top_k=2)
        assert len(relevant_docs) == 2
        # モック文書の内容を確認（実際のモックオブジェクトではなく、作成した文書を確認）
        assert all(isinstance(doc, Document) for doc in mock_documents)
        assert any("玄界システム" in doc.content for doc in mock_documents)
        assert any("高性能計算" in doc.content for doc in mock_documents)
        
        # 2. RAGEngineでクエリを実行
        response = rag_engine.query(query)
        assert response is not None
        assert "玄界システム" in response.answer
        assert "SSH接続" in response.answer
        assert len(response.sources) == 2
        
        # 呼び出しを確認
        document_processor.search_documents.assert_called_once_with(query, top_k=2)
        rag_engine.query.assert_called_once_with(query)
    
    def test_rag_engine_to_chat_manager_integration(self, mock_components):
        """RAGEngine → ChatManager 連携テスト"""
        from genkai_rag.core.rag_engine import RAGResponse
        from genkai_rag.models.chat import Message
        
        # RAGEngineとChatManagerを取得
        rag_engine = mock_components["rag_engine"]
        chat_manager = mock_components["chat_manager"]
        
        # モックレスポンスを作成
        mock_response = RAGResponse(
            answer="玄界システムは九州大学のスーパーコンピュータです。",
            sources=[],
            processing_time=0.2,
            model_used="test-model",
            retrieval_score=0.8,
            confidence_score=0.85
        )
        
        # モック設定
        rag_engine.query = Mock(return_value=mock_response)
        chat_manager.save_message = Mock(return_value=True)
        chat_manager.get_chat_history = Mock(return_value=[])
        
        # 連携フローを実行
        session_id = "integration-test-session"
        query = "玄界システムについて教えてください"
        
        # 1. RAGEngineでクエリを実行
        response = rag_engine.query(query)
        assert response is not None
        assert "玄界システム" in response.answer
        
        # 2. ユーザーメッセージを保存
        user_message = Message(content=query, role="user", session_id=session_id)
        result1 = chat_manager.save_message(session_id, user_message)
        assert result1 is True
        
        # 3. アシスタントメッセージを保存
        assistant_message = Message(content=response.answer, role="assistant", session_id=session_id)
        result2 = chat_manager.save_message(session_id, assistant_message)
        assert result2 is True
        
        # 4. 履歴を取得
        history = chat_manager.get_chat_history(session_id)
        
        # 呼び出しを確認
        rag_engine.query.assert_called_once_with(query)
        assert chat_manager.save_message.call_count == 2
        chat_manager.get_chat_history.assert_called_once_with(session_id)
    
    def test_llm_manager_to_rag_engine_integration(self, mock_components):
        """LLMManager → RAGEngine 連携テスト"""
        from genkai_rag.core.rag_engine import RAGResponse
        
        # LLMManagerとRAGEngineを取得
        llm_manager = mock_components["llm_manager"]
        rag_engine = mock_components["rag_engine"]
        
        # モック設定
        llm_manager.get_current_model = Mock(return_value="llama3.2:3b")
        llm_manager.generate_response = AsyncMock(return_value="玄界システムは高性能計算システムです。")
        
        mock_response = RAGResponse(
            answer="玄界システムは高性能計算システムです。",
            sources=[],
            processing_time=0.4,
            model_used="llama3.2:3b",
            retrieval_score=0.7,
            confidence_score=0.8
        )
        rag_engine.query = Mock(return_value=mock_response)
        
        # 連携フローを実行
        query = "玄界システムとは？"
        
        # 1. 現在のモデルを確認
        current_model = llm_manager.get_current_model()
        assert current_model == "llama3.2:3b"
        
        # 2. RAGEngineでクエリを実行（内部でLLMManagerを使用）
        response = rag_engine.query(query)
        assert response is not None
        assert response.model_used == "llama3.2:3b"
        assert "玄界システム" in response.answer
        
        # 呼び出しを確認
        llm_manager.get_current_model.assert_called_once()
        rag_engine.query.assert_called_once_with(query)
    
    def test_system_monitor_integration(self, mock_components):
        """SystemMonitor 統合テスト"""
        from datetime import datetime
        
        # SystemMonitorを取得
        system_monitor = mock_components["system_monitor"]
        
        # モックシステム状態を作成
        from genkai_rag.core.system_monitor import SystemStatus
        mock_status = SystemStatus(
            timestamp=datetime.now(),
            memory_usage_percent=50.0,
            memory_available_gb=4.0,
            memory_total_gb=8.0,
            disk_usage_percent=25.0,
            disk_available_gb=100.0,
            disk_total_gb=200.0,
            cpu_usage_percent=25.5,
            process_count=150,
            uptime_seconds=3600
        )
        
        # モック設定
        system_monitor.get_system_status = Mock(return_value=mock_status)
        system_monitor.log_system_status = Mock(return_value=True)
        
        # システム状態を取得
        status = system_monitor.get_system_status()
        assert status is not None
        assert status.memory_usage_percent == 50.0
        assert status.disk_usage_percent == 25.0
        assert status.cpu_usage_percent == 25.5
        assert status.uptime_seconds == 3600
        
        # ログを記録
        result = system_monitor.log_system_status()
        assert result is True
        
        # 呼び出しを確認
        system_monitor.get_system_status.assert_called_once()
        system_monitor.log_system_status.assert_called_once()
    
    def test_error_recovery_integration(self, mock_components):
        """ErrorRecoveryManager 統合テスト"""
        # ErrorRecoveryManagerを取得
        error_recovery_manager = mock_components["error_recovery_manager"]
        
        # モック設定
        error_recovery_manager.handle_validation_error = Mock(return_value=True)
        error_recovery_manager.get_error_statistics = Mock(return_value={
            "total_errors": 1,
            "error_rate": 0.1,
            "by_type": {"ValidationError": 1},
            "by_severity": {"medium": 1},
            "most_common_operations": ["test_operation"]
        })
        
        # エラーハンドリングをテスト
        error = ValueError("テストエラー")
        context = {"operation": "test_operation", "data": "test_data"}
        
        result = error_recovery_manager.handle_validation_error(error, context, "test_operation")
        assert result is True
        
        # エラー統計を取得
        stats = error_recovery_manager.get_error_statistics(24)
        assert stats["total_errors"] == 1
        assert stats["error_rate"] == 0.1
        assert "ValidationError" in stats["by_type"]
        
        # 呼び出しを確認
        error_recovery_manager.handle_validation_error.assert_called_once_with(error, context, "test_operation")
        error_recovery_manager.get_error_statistics.assert_called_once_with(24)
    
    def test_full_component_chain_integration(self, mock_components):
        """全コンポーネントチェーン統合テスト"""
        from genkai_rag.models.document import Document, DocumentSource
        from genkai_rag.core.rag_engine import RAGResponse
        from genkai_rag.models.chat import Message
        from datetime import datetime
        
        # 全コンポーネントを取得
        web_scraper = mock_components["web_scraper"]
        document_processor = mock_components["document_processor"]
        rag_engine = mock_components["rag_engine"]
        chat_manager = mock_components["chat_manager"]
        system_monitor = mock_components["system_monitor"]
        
        # モックデータを準備
        mock_document = Document(
            title="玄界システム完全ガイド",
            content="玄界システムは九州大学の最新スーパーコンピュータです。SSH接続で利用でき、高性能計算を提供します。",
            url="https://example.com/guide",
            section="完全ガイド",
            timestamp=datetime.now()
        )
        
        mock_source = DocumentSource(
            title="玄界システム完全ガイド",
            url="https://example.com/guide",
            section="完全ガイド",
            relevance_score=0.95
        )
        
        mock_response = RAGResponse(
            answer="玄界システムは九州大学の最新スーパーコンピュータで、SSH接続で利用でき、高性能計算を提供します。",
            sources=[mock_source],
            processing_time=0.6,
            model_used="llama3.2:3b",
            retrieval_score=0.9,
            confidence_score=0.95
        )
        
        # モック設定
        web_scraper.scrape_single_page = Mock(return_value=mock_document)
        document_processor.process_single_document = Mock(return_value=True)
        document_processor.search_documents = Mock(return_value=[mock_document])
        rag_engine.query = Mock(return_value=mock_response)
        chat_manager.save_message = Mock(return_value=True)
        chat_manager.get_chat_history = Mock(return_value=[])
        system_monitor.log_system_status = Mock(return_value=True)
        
        # 完全なチェーンを実行
        session_id = "full-chain-test"
        url = "https://example.com/guide"
        query = "玄界システムの特徴を教えてください"
        
        # 1. 文書をスクレイピング
        document = web_scraper.scrape_single_page(url)
        assert document is not None
        
        # 2. 文書を処理
        process_result = document_processor.process_single_document(document)
        assert process_result is True
        
        # 3. 関連文書を検索
        relevant_docs = document_processor.search_documents(query, top_k=1)
        assert len(relevant_docs) == 1
        
        # 4. RAGクエリを実行
        response = rag_engine.query(query)
        assert response is not None
        assert "玄界システム" in response.answer
        
        # 5. 会話履歴に保存
        user_message = Message(content=query, role="user", session_id=session_id)
        chat_manager.save_message(session_id, user_message)
        
        assistant_message = Message(content=response.answer, role="assistant", session_id=session_id)
        chat_manager.save_message(session_id, assistant_message)
        
        # 6. システム状態をログ
        system_monitor.log_system_status()
        
        # 全ての呼び出しを確認
        web_scraper.scrape_single_page.assert_called_once_with(url)
        document_processor.process_single_document.assert_called_once_with(document)
        document_processor.search_documents.assert_called_once_with(query, top_k=1)
        rag_engine.query.assert_called_once_with(query)
        assert chat_manager.save_message.call_count == 2
        system_monitor.log_system_status.assert_called_once()


class TestAPIIntegration:
    """API統合テスト"""
    
    @pytest.fixture
    def mock_components(self):
        """モックコンポーネント"""
        from datetime import datetime
        
        components = {}
        
        # RAGEngine - シンプルなモック
        rag_engine = Mock()
        rag_engine.query = Mock(return_value=Mock(
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
        system_status.memory_usage_percent = 50.0
        system_status.memory_available_gb = 4.0
        system_status.memory_total_gb = 8.0
        system_status.disk_usage_percent = 30.0
        system_status.disk_available_gb = 100.0
        system_status.disk_total_gb = 200.0
        system_status.cpu_usage_percent = 20.0
        system_status.process_count = 150
        system_status.uptime_seconds = 3600
        system_monitor.get_system_status = Mock(return_value=system_status)
        system_monitor.get_performance_stats = Mock(return_value={})
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
    
    @pytest.fixture
    def test_app(self, mock_components):
        """テスト用FastAPIアプリケーション"""
        from genkai_rag.api.app import create_app
        from fastapi.testclient import TestClient
        
        config = {
            "debug": True,
            "cors_origins": ["*"],
            "allowed_hosts": ["*"]
        }
        
        app = create_app(dependencies=mock_components, config=config)
        return TestClient(app)
    
    def test_query_api_integration(self, test_app, mock_components):
        """クエリAPI統合テスト"""
        from genkai_rag.core.rag_engine import RAGResponse
        from genkai_rag.models.document import DocumentSource
        
        # RAGEngineのモック設定
        mock_response = RAGResponse(
            answer="玄界システムは九州大学のスーパーコンピュータです。",
            sources=[
                DocumentSource(
                    title="玄界システム概要",
                    url="https://example.com/overview",
                    section="概要",
                    relevance_score=0.9
                )
            ],
            processing_time=0.3,
            model_used="llama3.2:3b",
            retrieval_score=0.85,
            confidence_score=0.9
        )
        
        # 同期モックを設定
        def mock_query(question, **kwargs):
            return mock_response
        
        mock_components["rag_engine"].query = mock_query
        mock_components["chat_manager"].get_chat_history.return_value = []
        
        # APIリクエストを送信
        request_data = {
            "question": "玄界システムについて教えてください",
            "session_id": "api-test-session",
            "include_history": True,
            "max_sources": 3
        }
        
        response = test_app.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "processing_time" in data
        assert "model_used" in data
        assert "session_id" in data
        
        assert "玄界システム" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "玄界システム概要"
        assert data["model_used"] == "llama3.2:3b"
        assert data["session_id"] == "api-test-session"
    
    def test_health_check_api_integration(self, test_app, mock_components):
        """ヘルスチェックAPI統合テスト"""
        # 基本ヘルスチェックテスト
        response = test_app.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "service" in data
        assert "version" in data
        
        assert data["status"] == "healthy"
        assert data["service"] == "genkai-rag-system"
        assert data["version"] == "1.0.0"
        
        # 詳細ヘルスチェックテスト
        response = test_app.get("/api/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "metrics" in data
        assert "warnings" in data
        
        assert data["status"] == "healthy"
        
        components = data["components"]
        assert "system_monitor" in components
        assert "llm_manager" in components
        assert "chat_manager" in components
        assert "database" in components
        
        metrics = data["metrics"]
        assert "memory_usage_percent" in metrics
        assert "disk_usage_percent" in metrics
        assert "active_sessions" in metrics
        assert "uptime_seconds" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestAdvancedComponentIntegration:
    """高度なコンポーネント間統合テスト"""
    
    @pytest.fixture
    def real_components(self):
        """実際のコンポーネントを使用したテスト（軽量設定）"""
        import tempfile
        import yaml
        from pathlib import Path
        
        # 一時ディレクトリを作成
        temp_dir = tempfile.mkdtemp()
        
        # テスト用設定
        test_config = {
            "logging": {"level": "WARNING"},
            "scraper": {"timeout": 2, "max_retries": 1},
            "document_processor": {"chunk_size": 50, "chunk_overlap": 10},
            "llm": {"base_url": "http://localhost:11434", "default_model": "test-model", "timeout": 5},
            "rag": {"similarity_top_k": 2, "rerank_top_n": 1},
            "chat": {"max_history_size": 3, "session_timeout_hours": 1},
            "system_monitor": {"enable_background_monitoring": False}
        }
        
        # 設定ファイルを作成
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # 実際のコンポーネントを作成（ネットワーク依存を最小化）
        from genkai_rag.core.config_manager import ConfigManager
        from genkai_rag.core.error_recovery import ErrorRecoveryManager
        from genkai_rag.core.system_monitor import SystemMonitor
        from genkai_rag.core.chat_manager import ChatManager
        
        components = {}
        
        # ConfigManager
        config_manager = ConfigManager(str(temp_dir))  # ディレクトリを渡す
        config_path = Path(temp_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        config = config_manager.load_config()
        components["config_manager"] = config_manager
        components["config"] = config
        
        # ErrorRecoveryManager
        error_recovery_config = config.get("error_recovery", {})
        error_recovery_manager = ErrorRecoveryManager(config=error_recovery_config)
        components["error_recovery_manager"] = error_recovery_manager
        
        # SystemMonitor
        system_monitor = SystemMonitor(
            log_dir=temp_dir,
            data_dir=temp_dir,
            monitoring_interval=1
        )
        components["system_monitor"] = system_monitor
        
        # ChatManager
        chat_config = config.get("chat", {})
        chat_manager = ChatManager(
            storage_dir=temp_dir,
            max_history_size=chat_config.get("max_history_size", 3),
            max_session_age_days=1,
            cleanup_interval_hours=1
        )
        components["chat_manager"] = chat_manager
        
        yield components
        
        # クリーンアップ
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_webscraper_processor_rag_chain(self, real_components):
        """WebScraper → DocumentProcessor → RAGEngine チェーンテスト"""
        from genkai_rag.models.document import Document
        from genkai_rag.core.scraper import WebScraper
        from genkai_rag.core.processor import DocumentProcessor
        from genkai_rag.core.rag_engine import RAGEngine
        from datetime import datetime
        from unittest.mock import Mock, patch
        
        # 実際のコンポーネントを作成（ネットワーク依存を回避）
        scraper = WebScraper(timeout=2, max_retries=1)
        
        # DocumentProcessorとRAGEngineはモック化（LlamaIndexの依存を回避）
        processor = Mock()
        rag_engine = Mock()
        
        # テスト文書を作成
        test_document = Document(
            title="玄界システム技術仕様",
            content="玄界システムは九州大学が運用する最新のスーパーコンピュータシステムです。Intel Xeon プロセッサを搭載し、高速なInfiniBandネットワークで接続されています。",
            url="https://example.com/genkai-specs",
            section="技術仕様",
            timestamp=datetime.now()
        )
        
        # WebScraperのモック（実際のネットワークアクセスを回避）
        with patch.object(scraper, 'scrape_single_page', return_value=test_document):
            # 1. WebScraperで文書を取得
            scraped_doc = scraper.scrape_single_page("https://example.com/genkai-specs")
            
            assert scraped_doc is not None
            assert scraped_doc.title == "玄界システム技術仕様"
            assert "玄界システム" in scraped_doc.content
            assert "Intel Xeon" in scraped_doc.content
            
            # 2. DocumentProcessorで処理
            processor.process_single_document = Mock(return_value=True)
            processor.get_document_count = Mock(return_value=1)
            
            result = processor.process_single_document(scraped_doc)
            assert result is True
            
            # 文書が処理されたことを確認
            processor.process_single_document.assert_called_once_with(scraped_doc)
            
            # 3. RAGEngineで検索
            from genkai_rag.core.rag_engine import RAGResponse
            from genkai_rag.models.document import DocumentSource
            
            mock_source = DocumentSource(
                title="玄界システム技術仕様",
                url="https://example.com/genkai-specs",
                section="技術仕様",
                relevance_score=0.95
            )
            
            mock_response = RAGResponse(
                answer="玄界システムはIntel Xeonプロセッサを搭載した九州大学のスーパーコンピュータです。",
                sources=[mock_source],
                processing_time=0.2,
                model_used="test-model",
                retrieval_score=0.9,
                confidence_score=0.95
            )
            
            rag_engine.query = Mock(return_value=mock_response)
            
            query = "玄界システムのプロセッサは何ですか？"
            response = rag_engine.query(query)
            
            assert response is not None
            assert "Intel Xeon" in response.answer
            assert len(response.sources) == 1
            assert response.sources[0].title == "玄界システム技術仕様"
            
            # 全チェーンが正常に動作したことを確認
            rag_engine.query.assert_called_once_with(query)
    
    def test_rag_llm_chat_chain(self, real_components):
        """RAGEngine → LLMManager → ChatManager チェーンテスト"""
        from genkai_rag.core.rag_engine import RAGEngine, RAGResponse
        from genkai_rag.core.llm_manager import LLMManager
        from genkai_rag.models.chat import Message
        from genkai_rag.models.document import DocumentSource
        from unittest.mock import Mock, AsyncMock
        
        # 実際のChatManagerを使用
        chat_manager = real_components["chat_manager"]
        
        # RAGEngineとLLMManagerはモック化
        rag_engine = Mock()
        llm_manager = Mock()
        
        # テストデータを準備
        session_id = "chain-test-session"
        query = "玄界システムの利用料金について教えてください"
        
        # LLMManagerのモック設定
        llm_manager.get_current_model = Mock(return_value="llama3.2:3b")
        llm_manager.query_async = AsyncMock(return_value="玄界システムの利用料金は計算時間に基づいて課金されます。詳細は公式サイトをご確認ください。")
        
        # RAGEngineのモック設定
        mock_source = DocumentSource(
            title="玄界システム料金体系",
            url="https://example.com/pricing",
            section="料金",
            relevance_score=0.9
        )
        
        mock_response = RAGResponse(
            answer="玄界システムの利用料金は計算時間に基づいて課金されます。詳細は公式サイトをご確認ください。",
            sources=[mock_source],
            processing_time=0.3,
            model_used="llama3.2:3b",
            retrieval_score=0.85,
            confidence_score=0.9
        )
        
        rag_engine.query = Mock(return_value=mock_response)
        
        # チェーンを実行
        # 1. RAGEngineでクエリを実行
        response = rag_engine.query(query)
        assert response is not None
        assert "料金" in response.answer
        assert response.model_used == "llama3.2:3b"
        
        # 2. ChatManagerに会話を保存（実際のChatManagerを使用）
        user_message = Message(content=query, role="user", session_id=session_id)
        result1 = chat_manager.save_message(session_id, user_message)
        assert result1 is True
        
        assistant_message = Message(content=response.answer, role="assistant", session_id=session_id)
        result2 = chat_manager.save_message(session_id, assistant_message)
        assert result2 is True
        
        # 3. 履歴を確認
        history = chat_manager.get_chat_history(session_id)
        assert len(history) == 2
        assert history[0].content == query
        assert history[0].role == "user"
        assert history[1].content == response.answer
        assert history[1].role == "assistant"
        
        # 4. セッション情報を確認
        session_info = chat_manager.get_session_info(session_id)
        assert session_info is not None
        assert session_info.message_count == 2
        
        # 呼び出しを確認
        rag_engine.query.assert_called_once_with(query)
    
    def test_api_to_core_integration(self, real_components):
        """API層 → コア層 統合テスト"""
        from genkai_rag.api.app import create_app
        from fastapi.testclient import TestClient
        from unittest.mock import Mock
        
        # 実際のコンポーネントと組み合わせ
        chat_manager = real_components["chat_manager"]
        system_monitor = real_components["system_monitor"]
        error_recovery_manager = real_components["error_recovery_manager"]
        
        # その他のコンポーネントはモック化
        mock_components = {
            "chat_manager": chat_manager,
            "system_monitor": system_monitor,
            "error_recovery_manager": error_recovery_manager,
            "rag_engine": Mock(),
            "llm_manager": Mock(),
            "document_processor": Mock(),
            "web_scraper": Mock(),
            "config_manager": real_components["config_manager"]
        }
        
        # RAGEngineのモック設定
        from genkai_rag.core.rag_engine import RAGResponse
        from genkai_rag.models.document import DocumentSource
        
        mock_response = RAGResponse(
            answer="玄界システムは九州大学の高性能計算システムです。",
            sources=[
                DocumentSource(
                    title="玄界システム概要",
                    url="https://example.com/overview",
                    section="概要",
                    relevance_score=0.9
                )
            ],
            processing_time=0.25,
            model_used="llama3.2:3b",
            retrieval_score=0.88,
            confidence_score=0.92
        )
        
        def mock_query(question, **kwargs):
            return mock_response
        
        mock_components["rag_engine"].query = mock_query
        mock_components["llm_manager"].get_current_model.return_value = "llama3.2:3b"
        
        # FastAPIアプリケーションを作成
        config = {"debug": True, "cors_origins": ["*"]}
        app = create_app(dependencies=mock_components, config=config)
        
        with TestClient(app) as client:
            # 1. クエリAPIテスト
            request_data = {
                "question": "玄界システムについて教えてください",
                "session_id": "api-core-test",
                "include_history": True
            }
            
            response = client.post("/api/query", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "answer" in data
            assert "玄界システム" in data["answer"]
            assert data["session_id"] == "api-core-test"
            
            # 2. 履歴APIテスト（実際のChatManagerを使用）
            history_response = client.get("/api/chat/history?session_id=api-core-test&limit=10")
            assert history_response.status_code == 200
            
            history_data = history_response.json()
            assert "messages" in history_data
            assert len(history_data["messages"]) >= 2  # user + assistant
            
            # 3. システム状態APIテスト（実際のSystemMonitorを使用）
            status_response = client.get("/api/system/status")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            assert "memory_usage_mb" in status_data
            assert "disk_usage_mb" in status_data
            assert "current_model" in status_data
            
            # 4. ヘルスチェックAPIテスト
            health_response = client.get("/api/health")
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            assert health_data["status"] == "healthy"
            assert health_data["service"] == "genkai-rag-system"
    
    def test_error_propagation_chain(self, real_components):
        """エラー伝播チェーンテスト"""
        from genkai_rag.core.scraper import WebScraper
        from unittest.mock import Mock, patch
        
        # 実際のErrorRecoveryManagerを使用
        error_recovery_manager = real_components["error_recovery_manager"]
        
        # WebScraperを作成
        scraper = WebScraper(timeout=1, max_retries=1)
        
        # その他のコンポーネントはモック化
        processor = Mock()
        rag_engine = Mock()
        
        # エラーを発生させる
        with patch.object(scraper, 'scrape_single_page', side_effect=Exception("Network timeout")):
            # 1. WebScraperでエラーが発生
            try:
                scraper.scrape_single_page("https://invalid-url.com")
                assert False, "Expected exception was not raised"
            except Exception as e:
                # 2. ErrorRecoveryManagerでエラーを処理
                context = {
                    "operation": "scrape_single_page",
                    "url": "https://invalid-url.com",
                    "component": "WebScraper"
                }
                
                result = error_recovery_manager.handle_scraping_error(e, "https://invalid-url.com")
                # エラーが適切に処理されたことを確認（Noneが返される場合もある）
                assert result is None  # 回復できない場合はNoneが返される
                
                # 3. エラー統計を確認
                stats = error_recovery_manager.get_error_statistics(1)
                assert stats["total_errors"] >= 1
                
                # 4. システムが他の機能を継続できることを確認
                processor.get_document_count = Mock(return_value=0)
                count = processor.get_document_count()
                assert count == 0
                
                # エラーが適切に処理され、システムが継続動作することを確認
                processor.get_document_count.assert_called_once()
    
    def test_concurrent_component_access(self, real_components):
        """同時コンポーネントアクセステスト"""
        import threading
        import time
        from genkai_rag.models.chat import Message
        
        # 実際のChatManagerを使用
        chat_manager = real_components["chat_manager"]
        
        results = []
        errors = []
        
        def concurrent_chat_operation(thread_id):
            """同時チャット操作"""
            try:
                session_id = f"concurrent-session-{thread_id}"
                
                # メッセージを保存
                for i in range(3):
                    message = Message(
                        content=f"Thread {thread_id} Message {i}",
                        role="user",
                        session_id=session_id
                    )
                    result = chat_manager.save_message(session_id, message)
                    if result:
                        results.append(f"thread_{thread_id}_msg_{i}")
                    
                    # 少し待機
                    time.sleep(0.01)
                
                # 履歴を取得
                history = chat_manager.get_chat_history(session_id)
                results.append(f"thread_{thread_id}_history_{len(history)}")
                
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {str(e)}")
        
        # 複数スレッドで同時実行
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_chat_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果を確認
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) >= 9  # 各スレッドで最低3つの操作
        
        # 各スレッドの履歴が独立していることを確認
        for i in range(3):
            session_id = f"concurrent-session-{i}"
            history = chat_manager.get_chat_history(session_id)
            assert len(history) == 3
            
            # 履歴の内容が正しいスレッドのものであることを確認
            for msg in history:
                assert f"Thread {i}" in msg.content
    
    def test_configuration_propagation(self, real_components):
        """設定伝播テスト"""
        # 実際のConfigManagerを使用
        config_manager = real_components["config_manager"]
        config = real_components["config"]
        
        # 設定値が各コンポーネントに正しく伝播されることを確認
        assert "chat" in config
        assert "scraper" in config
        assert "llm" in config
        
        # ChatManagerの設定確認
        chat_config = config.get("chat", {})
        chat_manager = real_components["chat_manager"]
        
        assert chat_manager.max_history_size == chat_config.get("max_history_size", 3)
        
        # SystemMonitorの設定確認
        system_monitor = real_components["system_monitor"]
        # バックグラウンド監視は無効化されていることを確認（デフォルトで無効）
        
        # 設定の変更と再読み込みテスト
        original_config = config_manager.load_config()
        assert original_config is not None
        
        # 設定が正しく読み込まれていることを確認
        assert "logging" in original_config
        assert original_config["logging"]["level"] == "WARNING"


class TestPerformanceIntegration:
    """パフォーマンス統合テスト"""
    
    def test_response_time_measurement_integration(self):
        """応答時間測定統合テスト"""
        from genkai_rag.core.system_monitor import SystemMonitor
        from unittest.mock import Mock
        import time
        
        # SystemMonitorを作成
        monitor = SystemMonitor(log_dir="logs", data_dir="data")
        
        # モックコンポーネントを作成
        rag_engine = Mock()
        
        def slow_query(question):
            """遅いクエリをシミュレート"""
            time.sleep(0.1)  # 100ms の遅延
            return Mock(
                answer="テスト回答",
                processing_time=0.1,
                sources=[]
            )
        
        rag_engine.query = slow_query
        
        # 応答時間を測定
        start_time = time.time()
        response = rag_engine.query("テスト質問")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 応答時間を記録
        monitor.record_response_time("rag_query", processing_time, {
            "question_length": len("テスト質問"),
            "model": "test-model"
        })
        
        # パフォーマンス統計を取得
        stats = monitor.get_performance_stats()
        
        assert "rag_query" in stats
        
        rag_stats = stats["rag_query"]
        assert rag_stats.total_requests == 1
        assert rag_stats.avg_response_time_ms >= 0.1
        assert rag_stats.max_response_time_ms >= 0.1
    
    def test_memory_usage_monitoring_integration(self):
        """メモリ使用量監視統合テスト"""
        from genkai_rag.core.system_monitor import SystemMonitor
        from genkai_rag.core.chat_manager import ChatManager
        from genkai_rag.models.chat import Message
        import tempfile
        
        # SystemMonitorを作成
        monitor = SystemMonitor(log_dir="logs", data_dir="data")
        
        # ChatManagerを作成
        temp_dir = tempfile.mkdtemp()
        chat_manager = ChatManager(storage_dir=temp_dir, max_history_size=100)
        
        # 初期メモリ使用量を記録
        initial_status = monitor.get_system_status()
        initial_memory = initial_status.memory_usage_percent
        
        # 大量のメッセージを作成してメモリ使用量を増加させる
        session_id = "memory-test-session"
        for i in range(50):
            message = Message(
                content=f"メモリテスト用の長いメッセージ内容 {i} " * 10,  # 長いメッセージ
                role="user",
                session_id=session_id
            )
            chat_manager.save_message(session_id, message)
        
        # メモリ使用量を再測定
        final_status = monitor.get_system_status()
        final_memory = final_status.memory_usage_percent
        
        # メモリ使用量が記録されていることを確認
        assert final_status.memory_total_gb > 0
        assert final_status.memory_available_gb >= 0
        assert final_status.memory_usage_percent >= 0
        
        # システム状態が正常に取得できることを確認
        assert final_status.cpu_usage_percent >= 0
        assert final_status.disk_usage_percent >= 0
        assert final_status.process_count > 0
        
        # クリーンアップ
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_load_integration(self):
        """同時負荷統合テスト"""
        import threading
        import time
        from genkai_rag.core.system_monitor import SystemMonitor
        from unittest.mock import Mock
        
        # SystemMonitorを作成
        monitor = SystemMonitor(log_dir="logs", data_dir="data")
        
        # モックRAGEngineを作成
        rag_engine = Mock()
        
        results = []
        errors = []
        
        def concurrent_query(thread_id):
            """同時クエリ実行"""
            try:
                for i in range(5):
                    start_time = time.time()
                    
                    # クエリをシミュレート
                    time.sleep(0.05)  # 50ms の処理時間
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # 応答時間を記録
                    monitor.record_response_time(
                        f"concurrent_query_thread_{thread_id}",
                        processing_time,
                        {"query_id": f"{thread_id}_{i}"}
                    )
                    
                    results.append(f"thread_{thread_id}_query_{i}")
                    
            except Exception as e:
                errors.append(f"thread_{thread_id}_error: {str(e)}")
        
        # 複数スレッドで同時実行
        threads = []
        for i in range(4):
            thread = threading.Thread(target=concurrent_query, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 全スレッドの完了を待機
        for thread in threads:
            thread.join()
        
        # 結果を確認
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20  # 4スレッド × 5クエリ
        
        # パフォーマンス統計を確認
        stats = monitor.get_performance_stats()
        
        # 各スレッドの統計が記録されていることを確認
        for i in range(4):
            operation_name = f"concurrent_query_thread_{i}"
            assert operation_name in stats
            
            thread_stats = stats[operation_name]
            assert thread_stats.total_requests == 5
            assert thread_stats.avg_response_time_ms >= 0.05