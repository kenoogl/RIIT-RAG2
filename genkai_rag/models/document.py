"""
文書関連のデータモデル
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class DocumentSource(Enum):
    """文書ソース"""
    WEB = "web"
    FILE = "file"
    API = "api"
    DATABASE = "database"
    MANUAL = "manual"


@dataclass
class DocumentSourceInfo:
    """文書ソース情報（RAG結果用）"""
    title: str
    url: str = ""
    section: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "title": self.title,
            "url": self.url,
            "section": self.section,
            "relevance_score": self.relevance_score
        }


@dataclass
class DocumentMetadata:
    """文書メタデータ"""
    title: str
    url: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    language: str = "ja"
    content_type: str = "text/html"
    size: Optional[int] = None
    checksum: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class Document:
    """文書データモデル"""
    content: str
    metadata: DocumentMetadata
    id: Optional[str] = None
    
    def __post_init__(self):
        # IDが指定されていない場合は自動生成
        if self.id is None:
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"doc_{content_hash}"
        
        # コンテンツサイズを自動設定
        if self.metadata.size is None:
            self.metadata.size = len(self.content)
    
    def is_valid(self) -> bool:
        """文書が有効かどうかを判定"""
        return (
            bool(self.content and self.content.strip()) and
            bool(self.metadata.title and self.metadata.title.strip())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": {
                "title": self.metadata.title,
                "url": self.metadata.url,
                "source": self.metadata.source,
                "created_at": self.metadata.created_at.isoformat() if self.metadata.created_at else None,
                "updated_at": self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                "language": self.metadata.language,
                "content_type": self.metadata.content_type,
                "size": self.metadata.size,
                "checksum": self.metadata.checksum,
                "tags": self.metadata.tags
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """辞書から文書オブジェクトを作成"""
        metadata_dict = data.get("metadata", {})
        
        # 日時文字列をdatetimeオブジェクトに変換
        created_at = None
        if metadata_dict.get("created_at"):
            created_at = datetime.fromisoformat(metadata_dict["created_at"])
        
        updated_at = None
        if metadata_dict.get("updated_at"):
            updated_at = datetime.fromisoformat(metadata_dict["updated_at"])
        
        metadata = DocumentMetadata(
            title=metadata_dict.get("title", ""),
            url=metadata_dict.get("url"),
            source=metadata_dict.get("source"),
            created_at=created_at,
            updated_at=updated_at,
            language=metadata_dict.get("language", "ja"),
            content_type=metadata_dict.get("content_type", "text/html"),
            size=metadata_dict.get("size"),
            checksum=metadata_dict.get("checksum"),
            tags=metadata_dict.get("tags", [])
        )
        
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            metadata=metadata
        )


@dataclass
class DocumentChunk:
    """文書チャンク（分割された文書の一部）"""
    content: str
    metadata: DocumentMetadata
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    id: Optional[str] = None
    document_id: Optional[str] = None
    
    def __post_init__(self):
        # IDが指定されていない場合は自動生成
        if self.id is None:
            import hashlib
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.id = f"chunk_{self.chunk_index}_{content_hash}"
    
    def get_metadata(self, key: str = None, default: Any = None) -> Any:
        """メタデータを取得"""
        if key is None:
            # キーが指定されていない場合は全メタデータを辞書形式で返す
            return {
                "id": self.id,
                "document_id": self.document_id,
                "chunk_index": self.chunk_index,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "title": self.metadata.title,
                "url": self.metadata.url,
                "source": self.metadata.source,
                "created_at": self.metadata.created_at.isoformat() if self.metadata.created_at else None,
                "updated_at": self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                "language": self.metadata.language,
                "content_type": self.metadata.content_type,
                "size": self.metadata.size,
                "checksum": self.metadata.checksum,
                "tags": self.metadata.tags
            }
        else:
            # 特定のキーの値を返す
            metadata_dict = {
                "id": self.id,
                "document_id": self.document_id,
                "chunk_index": self.chunk_index,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "original_title": self.metadata.title,
                "original_url": self.metadata.url,
                "original_section": "",
                "title": self.metadata.title,
                "url": self.metadata.url,
                "source": self.metadata.source,
                "created_at": self.metadata.created_at.isoformat() if self.metadata.created_at else None,
                "updated_at": self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                "language": self.metadata.language,
                "content_type": self.metadata.content_type,
                "size": self.metadata.size,
                "checksum": self.metadata.checksum,
                "tags": self.metadata.tags
            }
            return metadata_dict.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata.__dict__,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """辞書からチャンクオブジェクトを作成"""
        metadata_dict = data.get("metadata", {})
        
        # 日時文字列をdatetimeオブジェクトに変換
        created_at = None
        if metadata_dict.get("created_at"):
            if isinstance(metadata_dict["created_at"], str):
                created_at = datetime.fromisoformat(metadata_dict["created_at"])
            else:
                created_at = metadata_dict["created_at"]
        
        updated_at = None
        if metadata_dict.get("updated_at"):
            if isinstance(metadata_dict["updated_at"], str):
                updated_at = datetime.fromisoformat(metadata_dict["updated_at"])
            else:
                updated_at = metadata_dict["updated_at"]
        
        metadata = DocumentMetadata(
            title=metadata_dict.get("title", ""),
            url=metadata_dict.get("url"),
            source=metadata_dict.get("source"),
            created_at=created_at,
            updated_at=updated_at,
            language=metadata_dict.get("language", "ja"),
            content_type=metadata_dict.get("content_type", "text/html"),
            size=metadata_dict.get("size"),
            checksum=metadata_dict.get("checksum"),
            tags=metadata_dict.get("tags", [])
        )
        
        return cls(
            id=data.get("id"),
            document_id=data.get("document_id"),
            content=data.get("content", ""),
            metadata=metadata,
            chunk_index=data.get("chunk_index", 0),
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0)
        )


def create_chunks_from_document(
    document: Document, 
    chunk_size: int = 1024, 
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    文書をチャンクに分割
    
    Args:
        document: 分割する文書
        chunk_size: チャンクサイズ
        chunk_overlap: チャンク間のオーバーラップ
        
    Returns:
        文書チャンクのリスト
    """
    content = document.content
    chunks = []
    
    if len(content) <= chunk_size:
        # 文書が小さい場合はそのまま1つのチャンクとして返す
        chunk = DocumentChunk(
            content=content,
            metadata=document.metadata,
            chunk_index=0,
            start_char=0,
            end_char=len(content),
            document_id=document.id
        )
        chunks.append(chunk)
        return chunks
    
    # 文書を分割
    start = 0
    chunk_index = 0
    
    while start < len(content):
        end = min(start + chunk_size, len(content))
        
        # チャンクの内容を取得
        chunk_content = content[start:end]
        
        # チャンクを作成
        chunk = DocumentChunk(
            content=chunk_content,
            metadata=document.metadata,
            chunk_index=chunk_index,
            start_char=start,
            end_char=end,
            document_id=document.id
        )
        chunks.append(chunk)
        
        # 次のチャンクの開始位置を計算（オーバーラップを考慮）
        start = end - chunk_overlap
        chunk_index += 1
        
        # 最後のチャンクの場合は終了
        if end >= len(content):
            break
    
    return chunks