"""
RAGEngine: LlamaIndexを使用したRAG推論エンジン

このモジュールは、文書検索、reranking、回答生成を統合したRAGエンジンを提供します。
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from llama_index.core import VectorStoreIndex, Document as LlamaDocument, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama

from ..models.document import Document, DocumentSourceInfo
from ..models.chat import ChatMessage
from .llm_manager import LLMManager
from .processor import DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG推論結果"""
    answer: str
    sources: List[DocumentSourceInfo]
    processing_time: float
    model_used: str
    retrieval_score: float
    confidence_score: float


class RAGEngine:
    """
    RAG推論エンジンクラス
    
    機能:
    - LlamaIndexを使用したクエリエンジンの構築
    - 文書検索とreranking
    - LLMを使用した回答生成
    - 会話履歴を考慮したコンテキスト管理
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        document_processor: DocumentProcessor,
        similarity_threshold: float = 0.6,  # 閾値を下げて、より多くの関連文書を取得
        max_retrieved_docs: int = 15,  # 検索文書数を増加
        max_context_docs: int = 5,
        system_monitor: Optional[Any] = None
    ):
        """
        RAGEngineを初期化
        
        Args:
            llm_manager: LLM管理インスタンス
            document_processor: 文書処理インスタンス
            similarity_threshold: 類似度閾値
            max_retrieved_docs: 最大検索文書数
            max_context_docs: コンテキストに含める最大文書数
            system_monitor: システムモニター（レスポンス時間測定用）
        """
        self.llm_manager = llm_manager
        self.document_processor = document_processor
        self.similarity_threshold = similarity_threshold
        self.max_retrieved_docs = max_retrieved_docs
        self.max_context_docs = max_context_docs
        self.system_monitor = system_monitor
        
        # クエリエンジンの初期化
        self.query_engine = None
        self.retriever = None
        self.reranker = None
        self.ollama_llm = None
        
        # OllamaのLLMを設定
        self._setup_ollama_llm()
        
        self._initialize_query_engine()
        
        logger.info(f"RAGEngine initialized with similarity_threshold={similarity_threshold}")
    
    def _setup_ollama_llm(self) -> None:
        """OllamaのLLMを設定"""
        try:
            # LLMManagerからOllamaの設定を取得
            ollama_url = self.llm_manager.ollama_base_url
            
            # デフォルトモデルを設定（利用可能なモデルから選択）
            try:
                available_models = self.llm_manager.get_available_models()
                if available_models:
                    default_model = available_models[0].name
                else:
                    default_model = "llama3.2:1b"  # フォールバック
            except Exception:
                default_model = "llama3.2:1b"  # フォールバック
            
            # OllamaのLLMインスタンスを作成
            self.ollama_llm = Ollama(
                model=default_model,
                base_url=ollama_url,
                request_timeout=120.0
            )
            
            # LlamaIndexのグローバル設定にOllamaを設定
            Settings.llm = self.ollama_llm
            
            logger.info(f"Ollama LLM configured with model: {default_model}, URL: {ollama_url}")
            
        except Exception as e:
            logger.error(f"Failed to setup Ollama LLM: {e}")
            # フォールバック: LLMManagerを直接使用
            self.ollama_llm = None
    
    def _initialize_query_engine(self) -> None:
        """クエリエンジンを初期化"""
        try:
            # DocumentProcessorからインデックスを取得
            index = self.document_processor.get_index()
            if index is None:
                logger.warning("No index available, query engine will be initialized later")
                return
            
            # リトリーバーを設定
            self.retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=self.max_retrieved_docs
            )
            
            # Rerankingポストプロセッサーを設定
            self.reranker = SimilarityPostprocessor(
                similarity_cutoff=self.similarity_threshold
            )
            
            # クエリエンジンを構築
            self.query_engine = RetrieverQueryEngine(
                retriever=self.retriever,
                node_postprocessors=[self.reranker]
            )
            
            logger.info("Query engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            self.query_engine = None
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[ChatMessage]] = None,
        model_name: Optional[str] = None
    ) -> RAGResponse:
        """
        質問に対してRAG推論を実行
        
        Args:
            question: ユーザーの質問
            chat_history: 会話履歴
            model_name: 使用するLLMモデル名
            
        Returns:
            RAG推論結果
            
        Raises:
            ValueError: 質問が空の場合
            RuntimeError: クエリエンジンが初期化されていない場合
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        if self.query_engine is None:
            self._initialize_query_engine()
            if self.query_engine is None:
                raise RuntimeError("Query engine is not initialized")
        
        start_time = time.time()
        
        try:
            # 会話履歴を考慮したクエリを構築
            enhanced_query = self._build_contextual_query(question, chat_history)
            
            # 文書検索を実行
            retrieved_docs = self.retrieve_documents(enhanced_query)
            
            # 文書をrerankingで並び替え
            reranked_docs = self.rerank_documents(question, retrieved_docs)
            
            # LLMで回答を生成
            answer = self.generate_response(question, reranked_docs, chat_history, model_name)
            
            # 処理時間を計算
            processing_time = time.time() - start_time
            
            # レスポンス時間を記録
            if self.system_monitor:
                self.system_monitor.record_response_time(
                    operation_type="rag_query",
                    response_time_ms=processing_time * 1000,
                    success=True,
                    metadata={
                        "model_name": model_name or self.llm_manager.get_current_model(),
                        "sources_count": len(reranked_docs),
                        "question_length": len(question)
                    }
                )
            
            # 結果を構築
            response = RAGResponse(
                answer=answer,
                sources=self._convert_to_document_sources(reranked_docs),
                processing_time=processing_time,
                model_used=model_name or self.llm_manager.get_current_model() or "unknown",
                retrieval_score=self._calculate_retrieval_score(reranked_docs),
                confidence_score=self._calculate_confidence_score(reranked_docs, answer)
            )
            
            logger.info(f"RAG query completed in {processing_time:.2f}s with {len(reranked_docs)} sources")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # エラー時のレスポンス時間を記録
            if self.system_monitor:
                self.system_monitor.record_response_time(
                    operation_type="rag_query",
                    response_time_ms=processing_time * 1000,
                    success=False,
                    error_message=str(e),
                    metadata={
                        "model_name": model_name or "unknown",
                        "question_length": len(question)
                    }
                )
            
            logger.error(f"RAG query failed: {e}")
            # フォールバック応答
            return RAGResponse(
                answer=f"申し訳ございませんが、質問の処理中にエラーが発生しました: {str(e)}",
                sources=[],
                processing_time=time.time() - start_time,
                model_used=model_name or "unknown",
                retrieval_score=0.0,
                confidence_score=0.0
            )
    
    def retrieve_documents(self, query: str) -> List[NodeWithScore]:
        """
        クエリに関連する文書を検索
        
        Args:
            query: 検索クエリ
            
        Returns:
            検索された文書ノードのリスト
        """
        if self.retriever is None:
            logger.warning("Retriever not initialized, returning empty results")
            return []
        
        try:
            retrieved_nodes = self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(retrieved_nodes)} documents for query")
            return retrieved_nodes
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def rerank_documents(self, query: str, documents: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        検索された文書をrerankingで並び替え
        
        Args:
            query: 元のクエリ
            documents: 検索された文書ノード
            
        Returns:
            Rerankingされた文書ノードのリスト
        """
        if not documents:
            return []
        
        try:
            if self.reranker is None:
                # Rerankingが利用できない場合は元の順序を維持
                logger.warning("Reranker not available, maintaining original order")
                return documents[:self.max_context_docs]
            
            # Rerankingを実行
            reranked_nodes = self.reranker.postprocess_nodes(documents, query_str=query)
            
            # 最大文書数に制限
            limited_nodes = reranked_nodes[:self.max_context_docs]
            
            logger.info(f"Reranked {len(documents)} documents to {len(limited_nodes)} top results")
            return limited_nodes
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            # フォールバック: 元の順序で制限
            return documents[:self.max_context_docs]
    
    def generate_response(
        self,
        question: str,
        context_docs: List[NodeWithScore],
        chat_history: Optional[List[ChatMessage]] = None,
        model_name: Optional[str] = None
    ) -> str:
        """
        コンテキスト文書を使用して回答を生成
        
        Args:
            question: ユーザーの質問
            context_docs: コンテキスト文書
            chat_history: 会話履歴
            model_name: 使用するLLMモデル名
            
        Returns:
            生成された回答
        """
        try:
            # プロンプトを構築
            prompt = self._build_generation_prompt(question, context_docs, chat_history)
            
            # 質問の種類に応じてパラメータを調整
            question_lower = question.lower()
            
            # 技術的な質問や手順説明の場合は温度を下げて正確性を重視
            if any(word in question_lower for word in ["方法", "手順", "コマンド", "設定", "エラー", "インストール"]):
                temperature = 0.3
                max_tokens = 3072
            # 概念説明や一般的な質問の場合は少し温度を上げて自然な回答を生成
            elif any(word in question_lower for word in ["とは", "について", "説明", "概要"]):
                temperature = 0.5
                max_tokens = 2048
            else:
                temperature = 0.4
                max_tokens = 2048
            
            # LLMで回答を生成
            response = self.llm_manager.generate_response(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 回答の後処理
            cleaned_response = self._post_process_response(response.strip())
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return f"申し訳ございませんが、回答の生成中にエラーが発生しました。もう一度お試しください。"
    
    def _post_process_response(self, response: str) -> str:
        """
        生成された回答の後処理
        
        Args:
            response: 生成された回答
            
        Returns:
            後処理された回答
        """
        # 不要な前置きを削除
        prefixes_to_remove = [
            "以下の参考文書の情報に基づいて回答します：",
            "参考文書の内容から、",
            "提供された文書に基づいて、",
            "文書の情報によると、"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # 改行の正規化
        response = response.replace('\n\n\n', '\n\n')
        
        # 空の行を削除
        lines = response.split('\n')
        cleaned_lines = [line for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines)
    
    def _build_contextual_query(self, question: str, chat_history: Optional[List[ChatMessage]]) -> str:
        """
        会話履歴を考慮したクエリを構築
        
        Args:
            question: 現在の質問
            chat_history: 会話履歴
            
        Returns:
            拡張されたクエリ
        """
        if not chat_history:
            return question
        
        # 最近の履歴から関連するコンテキストを抽出
        recent_messages = chat_history[-4:]  # 最新4メッセージ
        context_parts = []
        
        for message in recent_messages:
            if message.role == "user":
                context_parts.append(f"前の質問: {message.content}")
            elif message.role == "assistant":
                # 回答の要約を含める
                summary = message.content[:100] + "..." if len(message.content) > 100 else message.content
                context_parts.append(f"前の回答: {summary}")
        
        if context_parts:
            context_str = "\n".join(context_parts)
            enhanced_query = f"{context_str}\n\n現在の質問: {question}"
            logger.debug(f"Enhanced query with context: {len(enhanced_query)} characters")
            return enhanced_query
        
        return question
    
    def _build_generation_prompt(
        self,
        question: str,
        context_docs: List[NodeWithScore],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        回答生成用のプロンプトを構築
        
        Args:
            question: ユーザーの質問
            context_docs: コンテキスト文書
            chat_history: 会話履歴
            
        Returns:
            構築されたプロンプト
        """
        # 改善されたシステムプロンプト
        system_prompt = """あなたは九州大学情報基盤研究開発センターのスーパーコンピュータ「玄界システム」の専門サポートアシスタントです。

【あなたの役割】
- 玄界システムの利用者に対して、正確で実用的な情報を提供する
- 技術的な質問に対して、具体的で分かりやすい回答をする
- システムの利用方法、設定、トラブルシューティングをサポートする

【回答の指針】
1. **正確性**: 提供された文書の情報のみを使用し、推測や憶測は避ける
2. **具体性**: 手順やコマンドは具体的に示し、実際に使える形で提供する
3. **分かりやすさ**: 技術用語は必要に応じて説明を加える
4. **実用性**: ユーザーが実際に行動できるような情報を含める
5. **構造化**: 情報を整理して、読みやすい形で提示する

【回答形式】
- 重要なポイントは箇条書きで整理する
- コマンドやファイル名は明確に区別する
- 手順がある場合は番号付きリストを使用する
- 文書に記載されていない情報については「文書に記載されていません」と明記する"""
        
        # コンテキスト文書を整理
        context_parts = []
        for i, doc_node in enumerate(context_docs, 1):
            doc_content = doc_node.node.text
            doc_metadata = doc_node.node.metadata
            
            # メタデータ情報を整理
            metadata_info = []
            if "title" in doc_metadata and doc_metadata["title"]:
                metadata_info.append(f"タイトル: {doc_metadata['title']}")
            if "section" in doc_metadata and doc_metadata["section"]:
                metadata_info.append(f"セクション: {doc_metadata['section']}")
            if "url" in doc_metadata and doc_metadata["url"]:
                metadata_info.append(f"URL: {doc_metadata['url']}")
            
            metadata_str = " | ".join(metadata_info) if metadata_info else "メタデータなし"
            
            context_parts.append(f"【文書{i}】({metadata_str})\n{doc_content.strip()}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "関連する文書が見つかりませんでした。一般的な知識で回答してください。"
        
        # 会話履歴を含める（より詳細に）
        history_str = ""
        if chat_history and len(chat_history) > 0:
            recent_history = chat_history[-4:]  # 最新4つのメッセージ
            history_parts = []
            for message in recent_history:
                role_label = "ユーザー" if message.role.value == "user" else "アシスタント"
                # メッセージを短縮（長すぎる場合）
                content = message.content
                if len(content) > 200:
                    content = content[:200] + "..."
                history_parts.append(f"{role_label}: {content}")
            
            if history_parts:
                history_str = f"\n\n【会話履歴（参考）】\n" + "\n".join(history_parts)
        
        # 質問の種類を分析してヒントを追加
        question_hints = ""
        question_lower = question.lower()
        if any(word in question_lower for word in ["方法", "やり方", "手順", "how"]):
            question_hints = "\n※ 手順や方法について質問されています。具体的なステップを示してください。"
        elif any(word in question_lower for word in ["エラー", "問題", "トラブル", "error"]):
            question_hints = "\n※ トラブルシューティングについて質問されています。原因と解決策を示してください。"
        elif any(word in question_lower for word in ["設定", "config", "configuration"]):
            question_hints = "\n※ 設定について質問されています。具体的な設定値や設定ファイルを示してください。"
        elif any(word in question_lower for word in ["とは", "what", "について"]):
            question_hints = "\n※ 概念や定義について質問されています。分かりやすく説明してください。"
        
        # 最終プロンプト
        prompt = f"""{system_prompt}

【参考文書】
{context_str}{history_str}

【質問】
{question}{question_hints}

【回答】
以下の参考文書の情報に基づいて回答します：

"""
        
        return prompt
    
    def _convert_to_document_sources(self, doc_nodes: List[NodeWithScore]) -> List[DocumentSourceInfo]:
        """
        文書ノードをDocumentSourceInfoに変換
        
        Args:
            doc_nodes: 文書ノードのリスト
            
        Returns:
            DocumentSourceInfoのリスト
        """
        sources = []
        
        for doc_node in doc_nodes:
            metadata = doc_node.node.metadata
            
            source = DocumentSourceInfo(
                title=metadata.get("title", "Unknown Title"),
                url=metadata.get("url", ""),
                section=metadata.get("section", ""),
                relevance_score=doc_node.score or 0.0
            )
            sources.append(source)
        
        return sources
    
    def _calculate_retrieval_score(self, doc_nodes: List[NodeWithScore]) -> float:
        """
        検索結果の総合スコアを計算
        
        Args:
            doc_nodes: 文書ノードのリスト
            
        Returns:
            総合検索スコア
        """
        if not doc_nodes:
            return 0.0
        
        scores = [node.score for node in doc_nodes if node.score is not None]
        if not scores:
            return 0.0
        
        # 重み付き平均（上位の文書により高い重みを付与）
        weighted_sum = sum(score * (len(scores) - i) for i, score in enumerate(scores))
        weight_sum = sum(len(scores) - i for i in range(len(scores)))
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _calculate_confidence_score(self, doc_nodes: List[NodeWithScore], answer: str) -> float:
        """
        回答の信頼度スコアを計算
        
        Args:
            doc_nodes: 使用された文書ノード
            answer: 生成された回答
            
        Returns:
            信頼度スコア
        """
        if not doc_nodes or not answer:
            return 0.0
        
        # 基本スコア（検索結果の品質）
        retrieval_score = self._calculate_retrieval_score(doc_nodes)
        
        # 回答の長さによる調整（適度な長さが望ましい）
        answer_length = len(answer)
        length_factor = 1.0
        if answer_length < 50:
            length_factor = 0.7  # 短すぎる回答
        elif answer_length > 2000:
            length_factor = 0.8  # 長すぎる回答
        
        # エラーメッセージの検出
        error_indicators = ["エラーが発生", "申し訳ございません", "処理できません"]
        error_factor = 0.3 if any(indicator in answer for indicator in error_indicators) else 1.0
        
        confidence = retrieval_score * length_factor * error_factor
        return min(confidence, 1.0)  # 最大値を1.0に制限
    
    def update_configuration(
        self,
        similarity_threshold: Optional[float] = None,
        max_retrieved_docs: Optional[int] = None,
        max_context_docs: Optional[int] = None
    ) -> bool:
        """
        RAGエンジンの設定を更新
        
        Args:
            similarity_threshold: 新しい類似度閾値
            max_retrieved_docs: 新しい最大検索文書数
            max_context_docs: 新しい最大コンテキスト文書数
            
        Returns:
            更新が成功した場合True
        """
        try:
            updated = False
            
            if similarity_threshold is not None:
                self.similarity_threshold = similarity_threshold
                updated = True
            
            if max_retrieved_docs is not None:
                self.max_retrieved_docs = max_retrieved_docs
                updated = True
            
            if max_context_docs is not None:
                self.max_context_docs = max_context_docs
                updated = True
            
            if updated:
                # クエリエンジンを再初期化
                self._initialize_query_engine()
                logger.info("RAG engine configuration updated successfully")
            
            return updated
            
        except Exception as e:
            logger.error(f"Failed to update RAG engine configuration: {e}")
            return False
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """
        RAGエンジンの統計情報を取得
        
        Returns:
            エンジン統計情報
        """
        stats = {
            "similarity_threshold": self.similarity_threshold,
            "max_retrieved_docs": self.max_retrieved_docs,
            "max_context_docs": self.max_context_docs,
            "query_engine_initialized": self.query_engine is not None,
            "retriever_initialized": self.retriever is not None,
            "reranker_initialized": self.reranker is not None,
            "current_model": self.llm_manager.get_current_model()
        }
        
        # インデックス統計を追加
        try:
            index_stats = self.document_processor.get_index_statistics()
            stats.update(index_stats)
        except Exception as e:
            logger.warning(f"Failed to get index statistics: {e}")
        
        return stats