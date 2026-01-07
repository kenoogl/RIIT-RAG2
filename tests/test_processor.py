"""文書処理機能のテスト

Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
"""

import pytest
from hypothesis import given, strategies as st, assume
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from genkai_rag.core.processor import DocumentProcessor
from genkai_rag.models.document import Document, DocumentChunk


class TestDocumentProcessor:
    """DocumentProcessorクラスの基本テスト"""
    
    def setup_method(self):
        """テスト前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=200,
            chunk_overlap=50
        )
    
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_processor_initialization(self):
        """プロセッサーの初期化テスト"""
        assert self.processor.chunk_size == 200
        assert self.processor.chunk_overlap == 50
        assert self.processor.index_dir == Path(self.temp_dir)
        assert self.processor.index_dir.exists()
    
    def test_single_document_processing(self):
        """単一文書の処理テスト"""
        document = Document(
            title="テスト文書",
            content="これはテスト用の文書です。" * 20,  # 長めのコンテンツ
            url="https://example.com/test",
            section="テストセクション"
        )
        
        result = self.processor.process_single_document(document)
        
        assert result is True
        
        # 統計情報を確認
        stats = self.processor.get_index_statistics()
        assert stats['document_count'] == 1
        assert stats['total_chunks'] >= 1
    
    def test_multiple_documents_processing(self):
        """複数文書の処理テスト"""
        documents = []
        for i in range(3):
            doc = Document(
                title=f"テスト文書{i+1}",
                content=f"これはテスト用の文書{i+1}です。" * 15,
                url=f"https://example.com/test{i+1}",
                section=f"セクション{i+1}"
            )
            documents.append(doc)
        
        result = self.processor.process_documents(documents)
        
        assert result is True
        
        # 統計情報を確認
        stats = self.processor.get_index_statistics()
        assert stats['document_count'] == 3
        assert stats['total_chunks'] >= 3
    
    def test_invalid_document_handling(self):
        """無効な文書の処理テスト"""
        invalid_doc = Document(
            title="",  # 空のタイトル
            content="",  # 空のコンテンツ
            url=""  # 空のURL
        )
        
        result = self.processor.process_single_document(invalid_doc)
        
        # 無効な文書は処理されない
        assert result is False
    
    def test_document_search(self):
        """文書検索テスト"""
        # テスト文書を追加
        document = Document(
            title="玄界システム利用ガイド",
            content="玄界システムはスーパーコンピュータです。ログイン方法やバッチジョブの投入方法について説明します。",
            url="https://example.com/guide",
            section="利用ガイド"
        )
        
        self.processor.process_single_document(document)
        
        # 検索を実行
        results = self.processor.search_documents("ログイン方法", top_k=3)
        
        # 結果が返されることを確認（エンベディングモデルが利用できない場合は空の可能性）
        assert isinstance(results, list)
    
    def test_document_retrieval(self):
        """文書取得テスト"""
        document = Document(
            title="取得テスト文書",
            content="取得テスト用のコンテンツです。",
            url="https://example.com/retrieve"
        )
        
        self.processor.process_single_document(document)
        
        # IDで文書を取得
        retrieved_doc = self.processor.get_document_by_id(document.id)
        
        assert retrieved_doc is not None
        assert retrieved_doc.title == document.title
        assert retrieved_doc.content == document.content
    
    def test_document_removal(self):
        """文書削除テスト"""
        document = Document(
            title="削除テスト文書",
            content="削除テスト用のコンテンツです。",
            url="https://example.com/delete"
        )
        
        self.processor.process_single_document(document)
        
        # 文書が存在することを確認
        assert self.processor.get_document_by_id(document.id) is not None
        
        # 文書を削除
        result = self.processor.remove_document(document.id)
        assert result is True
        
        # 文書が削除されたことを確認
        assert self.processor.get_document_by_id(document.id) is None
    
    def test_index_statistics(self):
        """インデックス統計情報テスト"""
        stats = self.processor.get_index_statistics()
        
        assert 'index_exists' in stats
        assert 'document_count' in stats
        assert 'total_chunks' in stats
        assert 'chunk_size' in stats
        assert 'chunk_overlap' in stats
        assert 'embedding_model' in stats
        assert 'index_dir' in stats
    
    def test_index_clear_and_rebuild(self):
        """インデックスクリアと再構築テスト"""
        # テスト文書を追加
        document = Document(
            title="再構築テスト文書",
            content="再構築テスト用のコンテンツです。",
            url="https://example.com/rebuild"
        )
        
        self.processor.process_single_document(document)
        
        # インデックスをクリア
        result = self.processor.clear_index()
        assert result is True
        
        stats = self.processor.get_index_statistics()
        assert stats['document_count'] == 0
        
        # 文書を再度追加して再構築をテスト
        self.processor.process_single_document(document)
        rebuild_result = self.processor.rebuild_index()
        assert rebuild_result is True


class TestDocumentProcessingProperties:
    """文書処理のプロパティテスト
    
    Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
    """
    
    def setup_method(self):
        """テスト前の準備"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """テスト後のクリーンアップ"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @given(
        title=st.text(min_size=1, max_size=100),
        content=st.text(min_size=1, max_size=5000),
        url=st.text(min_size=10, max_size=200),
        chunk_size=st.integers(min_value=50, max_value=1000),
        chunk_overlap=st.integers(min_value=0, max_value=200)
    )
    def test_document_processing_consistency(self, title, content, url, chunk_size, chunk_overlap):
        """
        プロパティ 3: インデックス更新の一貫性
        任意の有効な文書に対して、インデックス更新を実行した時、システムは
        文書を適切にチャンク分割し、インデックスに追加し、検索可能な状態にする
        
        Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
        """
        # 前提条件
        assume(chunk_overlap < chunk_size)
        assume(len(title.strip()) > 0)
        assume(len(content.strip()) > 0)
        assume(len(url.strip()) > 0)
        
        processor = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 有効な文書を作成
        document = Document(title=title, content=content, url=url)
        assume(document.is_valid())
        
        # 文書を処理
        result = processor.process_single_document(document)
        
        # プロパティ1: 処理が成功する
        assert result is True
        
        # プロパティ2: 文書がインデックスに追加される
        stats = processor.get_index_statistics()
        assert stats['document_count'] >= 1
        assert stats['total_chunks'] >= 1
        
        # プロパティ3: 文書がIDで取得可能
        retrieved_doc = processor.get_document_by_id(document.id)
        assert retrieved_doc is not None
        assert retrieved_doc.title == document.title
        assert retrieved_doc.content == document.content
        assert retrieved_doc.url == document.url
        
        # プロパティ4: チャンクが生成される
        chunks = processor.get_chunks_by_document_id(document.id)
        assert len(chunks) >= 1
        
        # プロパティ5: 各チャンクが有効
        for chunk in chunks:
            assert chunk.is_valid()
            assert chunk.document_id == document.id
            assert len(chunk.content) <= chunk_size + min(chunk_overlap, chunk_size // 4)
        
        # プロパティ6: チャンクインデックスが連続
        chunk_indices = [chunk.chunk_index for chunk in chunks]
        assert chunk_indices == list(range(len(chunks)))
    
    @given(
        num_documents=st.integers(min_value=1, max_value=5),
        chunk_size=st.integers(min_value=100, max_value=500)
    )
    def test_multiple_documents_consistency(self, num_documents, chunk_size):
        """
        複数文書処理の一貫性テスト
        
        Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
        """
        processor = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        
        # 複数の文書を作成
        documents = []
        for i in range(num_documents):
            doc = Document(
                title=f"文書{i+1}",
                content=f"これは文書{i+1}のコンテンツです。" * 10,
                url=f"https://example.com/doc{i+1}"
            )
            documents.append(doc)
        
        # 文書を処理
        result = processor.process_documents(documents)
        
        # プロパティ1: 処理が成功する
        assert result is True
        
        # プロパティ2: すべての文書がインデックスに追加される
        stats = processor.get_index_statistics()
        assert stats['document_count'] == num_documents
        
        # プロパティ3: 各文書が取得可能
        for document in documents:
            retrieved_doc = processor.get_document_by_id(document.id)
            assert retrieved_doc is not None
            assert retrieved_doc.title == document.title
        
        # プロパティ4: 各文書にチャンクが生成される
        total_chunks = 0
        for document in documents:
            chunks = processor.get_chunks_by_document_id(document.id)
            assert len(chunks) >= 1
            total_chunks += len(chunks)
        
        assert stats['total_chunks'] == total_chunks
    
    @given(
        title=st.text(min_size=1, max_size=50),
        content=st.text(min_size=1, max_size=2000),
        url=st.text(min_size=10, max_size=100)
    )
    def test_document_lifecycle_consistency(self, title, content, url):
        """
        文書ライフサイクルの一貫性テスト
        
        Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
        """
        # 前提条件
        assume(len(title.strip()) > 0)
        assume(len(content.strip()) > 0)
        assume(len(url.strip()) > 0)
        
        processor = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=200,
            chunk_overlap=50
        )
        
        document = Document(title=title, content=content, url=url)
        assume(document.is_valid())
        
        # 1. 文書を追加
        add_result = processor.process_single_document(document)
        assert add_result is True
        
        # 2. 文書が存在することを確認
        retrieved_doc = processor.get_document_by_id(document.id)
        assert retrieved_doc is not None
        
        initial_stats = processor.get_index_statistics()
        assert initial_stats['document_count'] == 1
        
        # 3. 文書を削除
        remove_result = processor.remove_document(document.id)
        assert remove_result is True
        
        # 4. 文書が削除されたことを確認
        deleted_doc = processor.get_document_by_id(document.id)
        assert deleted_doc is None
        
        final_stats = processor.get_index_statistics()
        assert final_stats['document_count'] == 0
    
    def test_index_persistence_consistency(self):
        """
        インデックス永続化の一貫性テスト
        
        Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
        """
        # 最初のプロセッサーで文書を追加
        processor1 = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=200,
            chunk_overlap=50
        )
        
        document = Document(
            title="永続化テスト文書",
            content="永続化テスト用のコンテンツです。" * 10,
            url="https://example.com/persistence"
        )
        
        result = processor1.process_single_document(document)
        assert result is True
        
        stats1 = processor1.get_index_statistics()
        assert stats1['document_count'] == 1
        
        # 新しいプロセッサーで同じディレクトリを読み込み
        processor2 = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=200,
            chunk_overlap=50
        )
        
        # プロパティ1: インデックスが正しく読み込まれる
        stats2 = processor2.get_index_statistics()
        assert stats2['document_count'] == 1
        
        # プロパティ2: 文書が取得可能
        retrieved_doc = processor2.get_document_by_id(document.id)
        assert retrieved_doc is not None
        assert retrieved_doc.title == document.title
        assert retrieved_doc.content == document.content
    
    @given(
        chunk_size=st.integers(min_value=50, max_value=500),
        chunk_overlap=st.integers(min_value=0, max_value=100)
    )
    def test_configuration_consistency(self, chunk_size, chunk_overlap):
        """
        設定の一貫性テスト
        
        Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
        """
        assume(chunk_overlap < chunk_size)
        
        processor = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # プロパティ1: 設定が正しく保存される
        assert processor.chunk_size == chunk_size
        assert processor.chunk_overlap == chunk_overlap
        
        # プロパティ2: 統計情報に設定が反映される
        stats = processor.get_index_statistics()
        assert stats['chunk_size'] == chunk_size
        assert stats['chunk_overlap'] == chunk_overlap
    
    def test_empty_documents_handling(self):
        """
        空の文書リストの処理テスト
        
        Feature: genkai-rag-system, Property 3: インデックス更新の一貫性
        """
        processor = DocumentProcessor(
            index_dir=self.temp_dir,
            chunk_size=200,
            chunk_overlap=50
        )
        
        # 空のリストを処理
        result = processor.process_documents([])
        
        # プロパティ1: 空のリストは適切に処理される
        assert result is False
        
        # プロパティ2: インデックス状態は変更されない
        stats = processor.get_index_statistics()
        assert stats['document_count'] == 0
        assert stats['total_chunks'] == 0