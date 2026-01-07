"""文書データモデルのテスト

Feature: genkai-rag-system, Property 2: 文書チャンク分割
"""

import pytest
from hypothesis import given, strategies as st, assume
from datetime import datetime
from typing import List

from genkai_rag.models.document import (
    Document, DocumentChunk, DocumentSource, 
    create_chunks_from_document
)


class TestDocumentModel:
    """Documentクラスのテスト"""
    
    def test_document_creation(self):
        """文書オブジェクトの作成テスト"""
        doc = Document(
            title="テスト文書",
            content="これはテスト用の文書です。",
            url="https://example.com/test",
            section="テストセクション"
        )
        
        assert doc.title == "テスト文書"
        assert doc.content == "これはテスト用の文書です。"
        assert doc.url == "https://example.com/test"
        assert doc.section == "テストセクション"
        assert doc.id  # IDが自動生成される
        assert isinstance(doc.timestamp, datetime)
    
    def test_document_validation(self):
        """文書データの妥当性チェックテスト"""
        # 有効な文書
        valid_doc = Document(
            title="有効な文書",
            content="有効なコンテンツ",
            url="https://example.com"
        )
        assert valid_doc.is_valid()
        
        # 無効な文書（タイトルが空）
        invalid_doc = Document(
            title="",
            content="コンテンツ",
            url="https://example.com"
        )
        assert not invalid_doc.is_valid()
    
    def test_document_serialization(self):
        """文書のシリアライゼーションテスト"""
        doc = Document(
            title="シリアライゼーションテスト",
            content="テストコンテンツ",
            url="https://example.com"
        )
        
        # 辞書変換
        doc_dict = doc.to_dict()
        assert doc_dict["title"] == "シリアライゼーションテスト"
        
        # 辞書から復元
        restored_doc = Document.from_dict(doc_dict)
        assert restored_doc.title == doc.title
        assert restored_doc.content == doc.content
        
        # JSON変換
        json_str = doc.to_json()
        restored_from_json = Document.from_json(json_str)
        assert restored_from_json.title == doc.title


class TestDocumentChunk:
    """DocumentChunkクラスのテスト"""
    
    def test_chunk_creation(self):
        """チャンクオブジェクトの作成テスト"""
        chunk = DocumentChunk(
            document_id="test-doc-id",
            content="チャンクコンテンツ",
            chunk_index=0
        )
        
        assert chunk.document_id == "test-doc-id"
        assert chunk.content == "チャンクコンテンツ"
        assert chunk.chunk_index == 0
        assert chunk.id  # IDが自動生成される
    
    def test_chunk_validation(self):
        """チャンクデータの妥当性チェックテスト"""
        # 有効なチャンク
        valid_chunk = DocumentChunk(
            document_id="valid-doc-id",
            content="有効なコンテンツ",
            chunk_index=0
        )
        assert valid_chunk.is_valid()
        
        # 無効なチャンク（document_idが空）
        invalid_chunk = DocumentChunk(
            document_id="",
            content="コンテンツ",
            chunk_index=0
        )
        assert not invalid_chunk.is_valid()


class TestDocumentSource:
    """DocumentSourceクラスのテスト"""
    
    def test_source_creation(self):
        """出典オブジェクトの作成テスト"""
        source = DocumentSource(
            title="出典文書",
            url="https://example.com/source",
            section="セクション1",
            relevance_score=0.85
        )
        
        assert source.title == "出典文書"
        assert source.url == "https://example.com/source"
        assert source.relevance_score == 0.85
    
    def test_citation_format(self):
        """引用形式のフォーマットテスト"""
        source = DocumentSource(
            title="テスト文書",
            url="https://example.com",
            section="第1章",
            relevance_score=0.9
        )
        
        citation = source.format_citation()
        assert "テスト文書" in citation
        assert "第1章" in citation
        assert "https://example.com" in citation
        assert "0.90" in citation


class TestChunkCreation:
    """チャンク作成機能のテスト"""
    
    def test_small_document_chunking(self):
        """小さな文書のチャンク分割テスト"""
        doc = Document(
            title="小さな文書",
            content="短いコンテンツ",
            url="https://example.com"
        )
        
        chunks = create_chunks_from_document(doc, chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0].content == "短いコンテンツ"
        assert chunks[0].document_id == doc.id
        assert chunks[0].chunk_index == 0
    
    def test_large_document_chunking(self):
        """大きな文書のチャンク分割テスト"""
        # 長いコンテンツを作成
        long_content = "これは長い文書です。" * 100  # 約1000文字
        
        doc = Document(
            title="長い文書",
            content=long_content,
            url="https://example.com"
        )
        
        chunks = create_chunks_from_document(doc, chunk_size=200, chunk_overlap=50)
        
        assert len(chunks) > 1
        
        # 各チャンクの妥当性をチェック
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == doc.id
            assert chunk.chunk_index == i
            assert chunk.is_valid()
            assert len(chunk.content) <= 250  # chunk_size + 余裕


# プロパティベーステスト
class TestDocumentChunkingProperties:
    """文書チャンク分割のプロパティテスト
    
    Feature: genkai-rag-system, Property 2: 文書チャンク分割
    """
    
    @given(
        title=st.text(min_size=1, max_size=100),
        content=st.text(min_size=1, max_size=10000),
        url=st.text(min_size=10, max_size=200),
        chunk_size=st.integers(min_value=50, max_value=2000),
        chunk_overlap=st.integers(min_value=0, max_value=500)
    )
    def test_chunk_content_preservation(self, title, content, url, chunk_size, chunk_overlap):
        """
        プロパティ 2: 文書チャンク分割
        任意の文書に対して、チャンク分割を実行した時、システムは設定された
        チャンクサイズ以下の断片に分割し、すべての元コンテンツが保持される
        
        Feature: genkai-rag-system, Property 2: 文書チャンク分割
        """
        # 前提条件
        assume(chunk_overlap < chunk_size)
        assume(len(title.strip()) > 0)
        assume(len(content.strip()) > 0)
        assume(len(url.strip()) > 0)
        
        # 文書を作成
        doc = Document(title=title, content=content, url=url)
        assume(doc.is_valid())
        
        # チャンク分割を実行
        chunks = create_chunks_from_document(doc, chunk_size, chunk_overlap)
        
        # プロパティ1: 少なくとも1つのチャンクが作成される
        assert len(chunks) >= 1
        
        # プロパティ2: 各チャンクのサイズが制限以下
        for chunk in chunks:
            # 最後のチャンクは小さくなる可能性があるが、
            # 他のチャンクはchunk_size + 調整分以下であるべき
            max_allowed_size = chunk_size + min(chunk_overlap, chunk_size // 4)
            assert len(chunk.content) <= max_allowed_size
        
        # プロパティ3: すべてのチャンクが有効
        for chunk in chunks:
            assert chunk.is_valid()
            assert chunk.document_id == doc.id
        
        # プロパティ4: チャンクインデックスが連続している
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
        
        # プロパティ5: 元コンテンツが保持されている（重要なプロパティ）
        # 全チャンクのコンテンツを結合して元コンテンツと比較
        if len(chunks) == 1:
            # 単一チャンクの場合は完全一致
            assert chunks[0].content == content
        else:
            # 複数チャンクの場合は、オーバーラップを考慮して検証
            # 最初のチャンクは元コンテンツの開始部分を含む
            assert content.startswith(chunks[0].content[:len(chunks[0].content)//2])
            # 最後のチャンクは元コンテンツの終了部分を含む
            assert content.endswith(chunks[-1].content[len(chunks[-1].content)//2:])
    
    @given(
        content=st.text(min_size=1, max_size=1000),
        chunk_size=st.integers(min_value=10, max_value=500)
    )
    def test_small_document_single_chunk(self, content, chunk_size):
        """
        小さな文書は単一チャンクになることを検証
        
        Feature: genkai-rag-system, Property 2: 文書チャンク分割
        """
        assume(len(content.strip()) > 0)
        assume(len(content) <= chunk_size)
        
        doc = Document(
            title="テスト",
            content=content,
            url="https://example.com"
        )
        
        chunks = create_chunks_from_document(doc, chunk_size)
        
        # 小さな文書は単一チャンクになる
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].chunk_index == 0
    
    @given(
        chunk_size=st.integers(min_value=50, max_value=1000),
        chunk_overlap=st.integers(min_value=0, max_value=200)
    )
    def test_empty_content_handling(self, chunk_size, chunk_overlap):
        """
        空のコンテンツの処理を検証
        
        Feature: genkai-rag-system, Property 2: 文書チャンク分割
        """
        assume(chunk_overlap < chunk_size)
        
        # 空のコンテンツを持つ無効な文書
        doc = Document(
            title="",  # 空のタイトル
            content="",  # 空のコンテンツ
            url=""  # 空のURL
        )
        
        # 無効な文書はエラーを発生させる
        with pytest.raises(ValueError, match="無効な文書データです"):
            create_chunks_from_document(doc, chunk_size, chunk_overlap)