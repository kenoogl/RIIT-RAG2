"""
RAGEngineクラスのテスト

このモジュールは、RAGEngineクラスの機能をテストします。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume
import hypothesis
import time

from genkai_rag.core.rag_engine import RAGEngine, RAGResponse
from genkai_rag.core.llm_manager import LLMManager
from genkai_rag.core.processor import DocumentProcessor
from genkai_rag.models.document import DocumentSource
from genkai_rag.models.chat import Message


class TestRAGEngine:
    """RAGEngineクラスの基本機能テスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.mock_llm_manager = Mock(spec=LLMManager)
        self.mock_document_processor = Mock(spec=DocumentProcessor)
        
        # DocumentProcessorのモック設定
        self.mock_document_processor.get_index = Mock(return_value=None)
        
        # LLMManagerのモック設定
        self.mock_llm_manager.get_current_model = Mock(return_value="test-model")
    
    def test_rag_engine_initialization(self):
        """RAGEngineの初期化テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor,
            similarity_threshold=0.8,
            max_retrieved_docs=15,
            max_context_docs=7
        )
        
        assert engine.llm_manager == self.mock_llm_manager
        assert engine.document_processor == self.mock_document_processor
        assert engine.similarity_threshold == 0.8
        assert engine.max_retrieved_docs == 15
        assert engine.max_context_docs == 7
        assert engine.query_engine is None  # インデックスがないため
    
    def test_rag_engine_default_parameters(self):
        """RAGEngineのデフォルトパラメータテスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        assert engine.similarity_threshold == 0.7
        assert engine.max_retrieved_docs == 10
        assert engine.max_context_docs == 5
    
    @patch('genkai_rag.core.rag_engine.VectorIndexRetriever')
    @patch('genkai_rag.core.rag_engine.SimilarityPostprocessor')
    @patch('genkai_rag.core.rag_engine.RetrieverQueryEngine')
    def test_initialize_query_engine_success(self, mock_query_engine, mock_postprocessor, mock_retriever):
        """クエリエンジン初期化の成功テスト"""
        # モックインデックスを設定
        mock_index = Mock()
        self.mock_document_processor.get_index.return_value = mock_index
        
        # モックオブジェクトを設定
        mock_retriever_instance = Mock()
        mock_retriever.return_value = mock_retriever_instance
        
        mock_postprocessor_instance = Mock()
        mock_postprocessor.return_value = mock_postprocessor_instance
        
        mock_query_engine_instance = Mock()
        mock_query_engine.return_value = mock_query_engine_instance
        
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # クエリエンジンが初期化されている
        assert engine.query_engine == mock_query_engine_instance
        assert engine.retriever == mock_retriever_instance
        assert engine.reranker == mock_postprocessor_instance
        
        # 適切なパラメータで初期化されている
        mock_retriever.assert_called_once_with(
            index=mock_index,
            similarity_top_k=10
        )
        mock_postprocessor.assert_called_once_with(
            similarity_cutoff=0.7
        )
    
    def test_query_empty_question(self):
        """空の質問に対するテスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            engine.query("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            engine.query("   ")
    
    def test_query_no_query_engine(self):
        """クエリエンジンが初期化されていない場合のテスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # インデックスがないため、クエリエンジンは初期化されない
        assert engine.query_engine is None
        
        with pytest.raises(RuntimeError, match="Query engine is not initialized"):
            engine.query("Test question")
    
    @patch('genkai_rag.core.rag_engine.RAGEngine._initialize_query_engine')
    def test_query_success_flow(self, mock_init):
        """クエリ実行の成功フローテスト"""
        # クエリエンジンをモック
        mock_query_engine = Mock()
        
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        engine.query_engine = mock_query_engine
        
        # メソッドをモック
        engine.retrieve_documents = Mock(return_value=[])
        engine.rerank_documents = Mock(return_value=[])
        engine.generate_response = Mock(return_value="Test answer")
        engine._build_contextual_query = Mock(return_value="Enhanced query")
        engine._convert_to_document_sources = Mock(return_value=[])
        engine._calculate_retrieval_score = Mock(return_value=0.8)
        engine._calculate_confidence_score = Mock(return_value=0.9)
        
        result = engine.query("Test question")
        
        assert isinstance(result, RAGResponse)
        assert result.answer == "Test answer"
        assert result.model_used == "test-model"
        assert result.retrieval_score == 0.8
        assert result.confidence_score == 0.9
        assert result.processing_time > 0
    
    def test_retrieve_documents_no_retriever(self):
        """リトリーバーがない場合の文書検索テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        result = engine.retrieve_documents("test query")
        
        assert result == []
    
    def test_rerank_documents_empty_list(self):
        """空の文書リストのrerankingテスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        result = engine.rerank_documents("test query", [])
        
        assert result == []
    
    def test_rerank_documents_no_reranker(self):
        """rerankingが利用できない場合のテスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        engine.reranker = None
        
        # モック文書ノードを作成
        mock_nodes = [Mock() for _ in range(7)]
        
        result = engine.rerank_documents("test query", mock_nodes)
        
        # 最大コンテキスト文書数に制限される
        assert len(result) == 5  # max_context_docs のデフォルト値
        assert result == mock_nodes[:5]
    
    def test_generate_response_success(self):
        """回答生成の成功テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # LLMManagerのモック設定
        self.mock_llm_manager.generate_response.return_value = "  Generated answer  "
        
        # モック文書ノードを作成
        mock_node = Mock()
        mock_node.node.text = "Document content"
        mock_node.node.metadata = {"title": "Test Doc", "url": "http://test.com"}
        mock_nodes = [mock_node]
        
        result = engine.generate_response("Test question", mock_nodes)
        
        assert result == "Generated answer"  # 前後の空白が除去される
        self.mock_llm_manager.generate_response.assert_called_once()
    
    def test_generate_response_error(self):
        """回答生成時のエラーテスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # LLMManagerでエラーが発生
        self.mock_llm_manager.generate_response.side_effect = Exception("Generation failed")
        
        result = engine.generate_response("Test question", [])
        
        assert "エラーが発生しました" in result
        assert "Generation failed" in result
    
    def test_build_contextual_query_no_history(self):
        """履歴なしでのコンテキストクエリ構築テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        result = engine._build_contextual_query("Test question", None)
        
        assert result == "Test question"
    
    def test_build_contextual_query_with_history(self):
        """履歴ありでのコンテキストクエリ構築テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # モック履歴を作成
        messages = [
            Message(
                id="1", session_id="test", role="user", 
                content="Previous question", timestamp=datetime.now(), sources=[]
            ),
            Message(
                id="2", session_id="test", role="assistant", 
                content="Previous answer", timestamp=datetime.now(), sources=[]
            )
        ]
        
        result = engine._build_contextual_query("Current question", messages)
        
        assert "Previous question" in result
        assert "Previous answer" in result
        assert "Current question" in result
        assert len(result) > len("Current question")
    
    def test_convert_to_document_sources(self):
        """文書ソース変換テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # モック文書ノードを作成
        mock_node1 = Mock()
        mock_node1.node.metadata = {
            "title": "Test Doc 1",
            "url": "http://test1.com",
            "section": "Section 1"
        }
        mock_node1.score = 0.9
        
        mock_node2 = Mock()
        mock_node2.node.metadata = {
            "title": "Test Doc 2",
            "url": "http://test2.com"
        }
        mock_node2.score = 0.7
        
        mock_nodes = [mock_node1, mock_node2]
        
        result = engine._convert_to_document_sources(mock_nodes)
        
        assert len(result) == 2
        assert isinstance(result[0], DocumentSource)
        assert result[0].title == "Test Doc 1"
        assert result[0].url == "http://test1.com"
        assert result[0].section == "Section 1"
        assert result[0].relevance_score == 0.9
        
        assert result[1].title == "Test Doc 2"
        assert result[1].section == ""  # デフォルト値
    
    def test_calculate_retrieval_score_empty(self):
        """空の文書リストでの検索スコア計算テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        result = engine._calculate_retrieval_score([])
        
        assert result == 0.0
    
    def test_calculate_retrieval_score_with_scores(self):
        """スコア付き文書での検索スコア計算テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # モック文書ノードを作成
        mock_nodes = []
        scores = [0.9, 0.8, 0.7]
        for score in scores:
            mock_node = Mock()
            mock_node.score = score
            mock_nodes.append(mock_node)
        
        result = engine._calculate_retrieval_score(mock_nodes)
        
        # 重み付き平均が計算される
        assert 0.0 < result <= 1.0
        assert result > 0.7  # 上位スコアにより高い重みが付与される
    
    def test_calculate_confidence_score(self):
        """信頼度スコア計算テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # モック文書ノードを作成
        mock_node = Mock()
        mock_node.score = 0.8
        mock_nodes = [mock_node]
        
        # 通常の回答
        result = engine._calculate_confidence_score(mock_nodes, "This is a normal answer.")
        assert 0.0 < result <= 1.0
        
        # エラーメッセージを含む回答
        error_result = engine._calculate_confidence_score(mock_nodes, "申し訳ございませんが、エラーが発生しました。")
        assert error_result < result  # エラーメッセージは信頼度を下げる
    
    def test_update_configuration(self):
        """設定更新テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # 設定を更新
        result = engine.update_configuration(
            similarity_threshold=0.8,
            max_retrieved_docs=15,
            max_context_docs=7
        )
        
        assert result is True
        assert engine.similarity_threshold == 0.8
        assert engine.max_retrieved_docs == 15
        assert engine.max_context_docs == 7
    
    def test_update_configuration_partial(self):
        """部分的な設定更新テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        original_threshold = engine.similarity_threshold
        
        # 一部の設定のみ更新
        result = engine.update_configuration(max_retrieved_docs=20)
        
        assert result is True
        assert engine.similarity_threshold == original_threshold  # 変更されない
        assert engine.max_retrieved_docs == 20  # 変更される
    
    def test_get_engine_stats(self):
        """エンジン統計情報取得テスト"""
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # DocumentProcessorのモック設定
        self.mock_document_processor.get_index_statistics.return_value = {
            "document_count": 100,
            "chunk_count": 500
        }
        
        stats = engine.get_engine_stats()
        
        assert "similarity_threshold" in stats
        assert "max_retrieved_docs" in stats
        assert "max_context_docs" in stats
        assert "query_engine_initialized" in stats
        assert "current_model" in stats
        assert stats["document_count"] == 100
        assert stats["chunk_count"] == 500


class TestRAGEngineProperties:
    """RAGEngineのプロパティベーステスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.mock_llm_manager = Mock(spec=LLMManager)
        self.mock_document_processor = Mock(spec=DocumentProcessor)
        
        # デフォルトのモック設定
        self.mock_document_processor.get_index = Mock(return_value=None)
        self.mock_llm_manager.get_current_model = Mock(return_value="test-model")
    
    @given(
        similarity_threshold=st.floats(min_value=0.0, max_value=1.0),
        max_retrieved_docs=st.integers(min_value=1, max_value=50),
        max_context_docs=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=30, deadline=None)
    def test_rag_engine_initialization_properties(self, similarity_threshold, max_retrieved_docs, max_context_docs):
        """
        Feature: genkai-rag-system, Property 6: 文書検索機能
        任意の設定パラメータに対して、RAGEngineが適切に初期化される
        """
        assume(max_context_docs <= max_retrieved_docs)  # 論理的制約
        
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor,
            similarity_threshold=similarity_threshold,
            max_retrieved_docs=max_retrieved_docs,
            max_context_docs=max_context_docs
        )
        
        # 設定が正しく保存されている
        assert engine.similarity_threshold == similarity_threshold
        assert engine.max_retrieved_docs == max_retrieved_docs
        assert engine.max_context_docs == max_context_docs
        
        # 依存関係が正しく設定されている
        assert engine.llm_manager == self.mock_llm_manager
        assert engine.document_processor == self.mock_document_processor
    
    @given(
        question=st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        model_name=st.one_of(st.none(), st.text(min_size=1, max_size=50))
    )
    @settings(max_examples=10, deadline=None)
    def test_query_error_handling_properties(self, question, model_name):
        """
        Feature: genkai-rag-system, Property 6: 文書検索機能
        任意の質問に対して、エラーが発生してもRAGResponseが返される
        """
        assume(question.strip())  # 空白のみの質問を除外
        
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # クエリエンジンが初期化されていない状態でもエラーハンドリングされる
        try:
            result = engine.query(question, model_name=model_name)
            
            # 結果が適切な型である
            assert isinstance(result, RAGResponse)
            assert isinstance(result.answer, str)
            assert isinstance(result.sources, list)
            assert isinstance(result.processing_time, float)
            assert isinstance(result.model_used, str)
            assert isinstance(result.retrieval_score, float)
            assert isinstance(result.confidence_score, float)
            
            # スコアが妥当な範囲内
            assert 0.0 <= result.retrieval_score <= 1.0
            assert 0.0 <= result.confidence_score <= 1.0
            assert result.processing_time >= 0.0
            
        except (ValueError, RuntimeError) as e:
            # 空の質問やクエリエンジン未初期化などの妥当なエラーは許容
            assert any(keyword in str(e).lower() for keyword in ["empty", "not initialized"])
        except Exception:
            # その他の予期しないエラーは許容しない
            pytest.fail("Unexpected exception occurred")
    
    @given(
        doc_count=st.integers(min_value=0, max_value=20),
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0), 
            min_size=0, max_size=20
        )
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[hypothesis.HealthCheck.filter_too_much])
    def test_reranking_properties(self, doc_count, scores):
        """
        Feature: genkai-rag-system, Property 8: Rerankingによる順序付け
        任意の文書リストに対して、rerankingが適切に実行される
        """
        # スコア数と文書数を一致させる
        if len(scores) != doc_count:
            scores = scores[:doc_count] + [0.5] * max(0, doc_count - len(scores))
        
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor,
            max_context_docs=min(10, doc_count + 1)  # 制限を適切に設定
        )
        
        # モック文書ノードを作成
        mock_nodes = []
        for i, score in enumerate(scores):
            mock_node = Mock()
            mock_node.score = score
            mock_node.node = Mock()
            mock_node.node.text = f"Document {i}"
            mock_node.node.metadata = {"title": f"Doc {i}"}
            mock_nodes.append(mock_node)
        
        result = engine.rerank_documents("test query", mock_nodes)
        
        # 結果が適切な型とサイズ
        assert isinstance(result, list)
        assert len(result) <= engine.max_context_docs
        assert len(result) <= len(mock_nodes)
        
        # 元の文書のサブセット
        for doc in result:
            assert doc in mock_nodes
        
        # 空のリストの場合は空のリストが返される
        if not mock_nodes:
            assert result == []
    
    @given(
        answer_length=st.integers(min_value=0, max_value=3000),
        doc_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=0, max_size=10
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_confidence_calculation_properties(self, answer_length, doc_scores):
        """
        Feature: genkai-rag-system, Property 6: 文書検索機能
        任意の回答と文書スコアに対して、信頼度が適切に計算される
        """
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # モック文書ノードを作成
        mock_nodes = []
        for score in doc_scores:
            mock_node = Mock()
            mock_node.score = score
            mock_nodes.append(mock_node)
        
        # テスト用の回答を生成
        answer = "A" * answer_length
        
        confidence = engine._calculate_confidence_score(mock_nodes, answer)
        
        # 信頼度が妥当な範囲内
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        # 空の入力に対しては0.0
        if not mock_nodes and not answer:
            assert confidence == 0.0
    
    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0),
        max_retrieved=st.integers(min_value=1, max_value=100),
        max_context=st.integers(min_value=1, max_value=50)
    )
    @settings(max_examples=20, deadline=None)
    def test_configuration_update_properties(self, threshold, max_retrieved, max_context):
        """
        Feature: genkai-rag-system, Property 6: 文書検索機能
        任意の設定値に対して、設定更新が適切に実行される
        """
        assume(max_context <= max_retrieved)  # 論理的制約
        
        engine = RAGEngine(
            llm_manager=self.mock_llm_manager,
            document_processor=self.mock_document_processor
        )
        
        # 設定を更新
        result = engine.update_configuration(
            similarity_threshold=threshold,
            max_retrieved_docs=max_retrieved,
            max_context_docs=max_context
        )
        
        # 更新が成功
        assert result is True
        
        # 設定が正しく適用されている
        assert engine.similarity_threshold == threshold
        assert engine.max_retrieved_docs == max_retrieved
        assert engine.max_context_docs == max_context
        
        # 統計情報に反映されている
        stats = engine.get_engine_stats()
        assert stats["similarity_threshold"] == threshold
        assert stats["max_retrieved_docs"] == max_retrieved
        assert stats["max_context_docs"] == max_context
    
    @given(
        questions=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            min_size=1, max_size=5
        )
    )
    @settings(max_examples=5, deadline=10000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_concurrent_operation_during_update_property(self, questions):
        """
        プロパティ 20: 更新中の継続動作
        任意のインデックス更新中の質問に対して、システムは既存のインデックスを使用して回答を生成し、更新処理と並行して動作する
        
        Feature: genkai-rag-system, Property 20: 更新中の継続動作
        **検証: 要件 6.4**
        """
        # モックインデックスを設定（既存のインデックス）
        mock_index = Mock()
        self.mock_document_processor.get_index.return_value = mock_index
        
        # LLMManagerのモック設定
        self.mock_llm_manager.generate_response.return_value = "Test answer"
        
        # RAGEngineを初期化（既存のインデックスあり）
        with patch('genkai_rag.core.rag_engine.VectorIndexRetriever'), \
             patch('genkai_rag.core.rag_engine.SimilarityPostprocessor'), \
             patch('genkai_rag.core.rag_engine.RetrieverQueryEngine') as mock_query_engine:
            
            mock_query_engine_instance = Mock()
            mock_query_engine.return_value = mock_query_engine_instance
            
            engine = RAGEngine(
                llm_manager=self.mock_llm_manager,
                document_processor=self.mock_document_processor
            )
            
            # プロパティ1: 既存のインデックスでクエリエンジンが初期化される
            assert engine.query_engine is not None
            assert engine.query_engine == mock_query_engine_instance
            
            # プロパティ2: インデックス更新中でも質問に回答できる
            # インデックス更新をシミュレート（時間のかかる処理）
            update_in_progress = True
            
            # 更新中フラグを設定
            engine._index_updating = update_in_progress
            
            # 各質問に対して回答を生成
            for question in questions:
                assume(question.strip())  # 空白のみの質問を除外
                
                # モック設定：既存のインデックスを使用した検索結果
                mock_node = Mock()
                mock_node.node.text = f"Content for: {question}"
                mock_node.node.metadata = {"title": "Test Doc", "url": "http://test.com"}
                mock_node.score = 0.8
                
                engine.retrieve_documents = Mock(return_value=[mock_node])
                engine.rerank_documents = Mock(return_value=[mock_node])
                
                try:
                    # プロパティ3: 更新中でも回答が生成される
                    result = engine.query(question)
                    
                    assert isinstance(result, RAGResponse)
                    assert isinstance(result.answer, str)
                    assert len(result.answer) > 0
                    assert result.processing_time >= 0.0
                    assert 0.0 <= result.confidence_score <= 1.0
                    assert 0.0 <= result.retrieval_score <= 1.0
                    
                    # プロパティ4: 既存のインデックスが使用されている
                    # （新しいインデックスではなく、既存のものを使用）
                    assert engine.query_engine == mock_query_engine_instance
                    
                except (ValueError, RuntimeError) as e:
                    # 妥当なエラー（空の質問など）は許容
                    assert any(keyword in str(e).lower() for keyword in ["empty", "not initialized"])
            
            # プロパティ5: 更新処理と並行して動作する
            # （更新中フラグがあっても、クエリエンジンは動作し続ける）
            assert engine.query_engine is not None
            
            # プロパティ6: システム状態が一貫している
            stats = engine.get_engine_stats()
            assert stats["query_engine_initialized"] is True
            assert isinstance(stats["similarity_threshold"], float)
            assert isinstance(stats["max_retrieved_docs"], int)
            assert isinstance(stats["max_context_docs"], int)
            
            # 更新完了をシミュレート
            engine._index_updating = False
            
            # プロパティ7: 更新完了後も正常に動作する
            if questions:  # 質問がある場合のみテスト
                final_question = questions[0]
                try:
                    final_result = engine.query(final_question)
                    assert isinstance(final_result, RAGResponse)
                    assert isinstance(final_result.answer, str)
                except (ValueError, RuntimeError):
                    # 妥当なエラーは許容
                    pass