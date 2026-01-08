"""文書処理機能

LlamaIndexを使用した文書のインデックス化と検索機能
"""

from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import pickle
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex, 
    Document as LlamaDocument,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from ..models.document import Document, DocumentChunk, create_chunks_from_document
from ..utils.logging import get_logger


class DocumentProcessor:
    """
    文書処理クラス
    
    LlamaIndexを使用して文書のチャンク分割、エンベディング生成、
    インデックス作成を行うクラス
    """
    
    def __init__(
        self,
        index_dir: str = "./data/index",
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        embedding_model: str = "intfloat/multilingual-e5-small"
    ):
        """
        DocumentProcessorを初期化
        
        Args:
            index_dir: インデックス保存ディレクトリ
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ（文字数）
            embedding_model: 使用するエンベディングモデル
        """
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        self.logger = get_logger("processor")
        
        # ディレクトリを作成
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # エンベディングモデルを初期化
        try:
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                trust_remote_code=True
            )
            self.logger.info(f"エンベディングモデルを初期化: {embedding_model}")
        except Exception as e:
            self.logger.warning(f"エンベディングモデルの初期化に失敗: {e}")
            self.embedding_model = None
        
        # ノードパーサーを初期化
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[。！？]"  # 日本語の文区切り
        )
        
        # インデックスとメタデータ
        self.index: Optional[VectorStoreIndex] = None
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        self.chunks_metadata: Dict[str, List[DocumentChunk]] = {}
        
        # インデックスを読み込み
        self._load_index()
    
    def process_documents(self, documents: List[Document]) -> bool:
        """
        複数の文書を処理してインデックスに追加
        
        Args:
            documents: 処理する文書のリスト
            
        Returns:
            処理が成功した場合True
        """
        if not documents:
            self.logger.warning("処理する文書がありません")
            return False
        
        self.logger.info(f"{len(documents)}個の文書の処理を開始")
        
        try:
            # 文書をチャンクに分割
            all_chunks = []
            for document in documents:
                if not document.is_valid():
                    self.logger.warning(f"無効な文書をスキップ: {document.id}")
                    continue
                
                chunks = self._process_single_document(document)
                all_chunks.extend(chunks)
                
                # メタデータを保存
                self.document_metadata[document.id] = document.to_dict()
                self.chunks_metadata[document.id] = chunks
            
            if not all_chunks:
                self.logger.warning("有効なチャンクが生成されませんでした")
                return False
            
            # LlamaIndexドキュメントに変換
            llama_documents = []
            for chunk in all_chunks:
                llama_doc = LlamaDocument(
                    text=chunk.content,
                    doc_id=chunk.id,
                    metadata={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "original_title": chunk.get_metadata("original_title", ""),
                        "original_url": chunk.get_metadata("original_url", ""),
                        "original_section": chunk.get_metadata("original_section", "")
                    }
                )
                llama_documents.append(llama_doc)
            
            # インデックスを更新
            self._update_index(llama_documents)
            
            # インデックスを保存
            self._save_index()
            
            self.logger.info(f"文書処理完了: {len(documents)}個の文書、{len(all_chunks)}個のチャンク")
            return True
            
        except Exception as e:
            self.logger.error(f"文書処理エラー: {e}")
            return False
    
    def process_single_document(self, document: Document) -> bool:
        """
        単一の文書を処理してインデックスに追加
        
        Args:
            document: 処理する文書
            
        Returns:
            処理が成功した場合True
        """
        return self.process_documents([document])
    
    def _process_single_document(self, document: Document) -> List[DocumentChunk]:
        """
        単一文書をチャンクに分割
        
        Args:
            document: 処理する文書
            
        Returns:
            生成されたチャンクのリスト
        """
        try:
            # 基本的なチャンク分割
            chunks = create_chunks_from_document(
                document, 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            self.logger.debug(f"文書 {document.id} を {len(chunks)}個のチャンクに分割")
            return chunks
            
        except Exception as e:
            self.logger.error(f"文書チャンク分割エラー ({document.id}): {e}")
            return []
    
    def _update_index(self, llama_documents: List[LlamaDocument]) -> None:
        """
        インデックスを更新
        
        Args:
            llama_documents: 追加するLlamaIndexドキュメント
        """
        try:
            if self.index is None:
                # 新しいインデックスを作成
                self.logger.info("新しいインデックスを作成")
                
                if self.embedding_model:
                    self.index = VectorStoreIndex.from_documents(
                        llama_documents,
                        embed_model=self.embedding_model
                    )
                else:
                    # エンベディングモデルが利用できない場合はデフォルトを使用
                    self.index = VectorStoreIndex.from_documents(llama_documents)
            else:
                # 既存のインデックスに追加
                self.logger.info(f"既存のインデックスに {len(llama_documents)}個のドキュメントを追加")
                
                for doc in llama_documents:
                    self.index.insert(doc)
            
            self.logger.info("インデックス更新完了")
            
        except Exception as e:
            self.logger.error(f"インデックス更新エラー: {e}")
            raise
    
    def _save_index(self) -> None:
        """
        インデックスをディスクに保存
        """
        try:
            if self.index is None:
                self.logger.warning("保存するインデックスがありません")
                return
            
            # LlamaIndexの標準的な保存方法
            self.index.storage_context.persist(persist_dir=str(self.index_dir))
            
            # メタデータを別途保存
            metadata_file = self.index_dir / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'document_metadata': self.document_metadata,
                    'chunks_metadata': self.chunks_metadata,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'embedding_model': self.embedding_model_name,
                    'last_updated': datetime.now().isoformat()
                }, f)
            
            self.logger.info(f"インデックスを保存: {self.index_dir}")
            
        except Exception as e:
            self.logger.error(f"インデックス保存エラー: {e}")
            raise
    
    def _load_index(self) -> None:
        """
        ディスクからインデックスを読み込み
        """
        try:
            # LlamaIndexの標準的な読み込み方法
            if (self.index_dir / "index_store.json").exists():
                self.logger.info(f"既存のインデックスを読み込み: {self.index_dir}")
                
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.index_dir)
                )
                
                if self.embedding_model:
                    self.index = load_index_from_storage(
                        storage_context,
                        embed_model=self.embedding_model
                    )
                else:
                    self.index = load_index_from_storage(storage_context)
                
                # メタデータを読み込み
                metadata_file = self.index_dir / "metadata.pkl"
                if metadata_file.exists():
                    with open(metadata_file, 'rb') as f:
                        metadata = pickle.load(f)
                        self.document_metadata = metadata.get('document_metadata', {})
                        self.chunks_metadata = metadata.get('chunks_metadata', {})
                
                self.logger.info("インデックス読み込み完了")
            else:
                self.logger.info("既存のインデックスが見つかりません。新規作成します。")
                self.index = None
                
        except Exception as e:
            self.logger.warning(f"インデックス読み込みエラー: {e}")
            self.index = None
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        現在のインデックスを取得
        
        Returns:
            VectorStoreIndexオブジェクト（存在しない場合はNone）
        """
        return self.index
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        インデックスの統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        stats = {
            'index_exists': self.index is not None,
            'document_count': len(self.document_metadata),
            'total_chunks': sum(len(chunks) for chunks in self.chunks_metadata.values()),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model_name,
            'index_dir': str(self.index_dir)
        }
        
        if self.index is not None:
            try:
                # インデックス内のドキュメント数を取得
                docstore = self.index.storage_context.docstore
                stats['indexed_documents'] = len(docstore.docs)
            except Exception as e:
                self.logger.warning(f"インデックス統計取得エラー: {e}")
                stats['indexed_documents'] = 0
        else:
            stats['indexed_documents'] = 0
        
        return stats
    
    def search_documents(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        文書を検索
        
        Args:
            query: 検索クエリ
            top_k: 返す結果の最大数
            similarity_threshold: 類似度の閾値
            
        Returns:
            検索結果のリスト
        """
        if self.index is None:
            self.logger.warning("インデックスが存在しません")
            return []
        
        try:
            # クエリエンジンを作成
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="no_text"  # テキスト生成なし、検索のみ
            )
            
            # 検索を実行
            response = query_engine.query(query)
            
            results = []
            for node in response.source_nodes:
                score = getattr(node, 'score', 0.0)
                
                if score >= similarity_threshold:
                    result = {
                        'chunk_id': node.node.node_id,
                        'content': node.node.text,
                        'score': score,
                        'metadata': node.node.metadata,
                        'document_id': node.node.metadata.get('document_id', ''),
                        'chunk_index': node.node.metadata.get('chunk_index', 0)
                    }
                    results.append(result)
            
            self.logger.info(f"検索完了: クエリ='{query}', 結果数={len(results)}")
            return results
            
        except Exception as e:
            self.logger.error(f"文書検索エラー: {e}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        IDで文書を取得
        
        Args:
            document_id: 文書ID
            
        Returns:
            文書オブジェクト（見つからない場合はNone）
        """
        if document_id in self.document_metadata:
            return Document.from_dict(self.document_metadata[document_id])
        return None
    
    def get_chunks_by_document_id(self, document_id: str) -> List[DocumentChunk]:
        """
        文書IDでチャンクを取得
        
        Args:
            document_id: 文書ID
            
        Returns:
            チャンクのリスト
        """
        return self.chunks_metadata.get(document_id, [])
    
    def remove_document(self, document_id: str) -> bool:
        """
        文書をインデックスから削除
        
        Args:
            document_id: 削除する文書のID
            
        Returns:
            削除が成功した場合True
        """
        try:
            if document_id not in self.document_metadata:
                self.logger.warning(f"文書が見つかりません: {document_id}")
                return False
            
            # チャンクIDを取得
            chunks = self.chunks_metadata.get(document_id, [])
            chunk_ids = [chunk.id for chunk in chunks]
            
            # インデックスからチャンクを削除
            if self.index is not None:
                for chunk_id in chunk_ids:
                    try:
                        self.index.delete_ref_doc(chunk_id, delete_from_docstore=True)
                    except Exception as e:
                        self.logger.warning(f"チャンク削除エラー ({chunk_id}): {e}")
            
            # メタデータから削除
            del self.document_metadata[document_id]
            if document_id in self.chunks_metadata:
                del self.chunks_metadata[document_id]
            
            # インデックスを保存
            self._save_index()
            
            self.logger.info(f"文書を削除: {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"文書削除エラー ({document_id}): {e}")
            return False
    
    def clear_index(self) -> bool:
        """
        インデックスを完全にクリア
        
        Returns:
            クリアが成功した場合True
        """
        try:
            # インデックスをリセット
            self.index = None
            self.document_metadata.clear()
            self.chunks_metadata.clear()
            
            # ディスクからファイルを削除
            if self.index_dir.exists():
                import shutil
                shutil.rmtree(self.index_dir)
                self.index_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("インデックスをクリアしました")
            return True
            
        except Exception as e:
            self.logger.error(f"インデックスクリアエラー: {e}")
            return False
    
    def rebuild_index(self) -> bool:
        """
        既存の文書からインデックスを再構築
        
        Returns:
            再構築が成功した場合True
        """
        try:
            if not self.document_metadata:
                self.logger.warning("再構築する文書がありません")
                return False
            
            # 既存の文書を取得
            documents = []
            for doc_data in self.document_metadata.values():
                documents.append(Document.from_dict(doc_data))
            
            # インデックスをクリア
            self.clear_index()
            
            # 文書を再処理
            return self.process_documents(documents)
            
        except Exception as e:
            self.logger.error(f"インデックス再構築エラー: {e}")
            return False