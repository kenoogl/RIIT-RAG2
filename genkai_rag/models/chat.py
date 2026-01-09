"""
チャット関連のデータモデル
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .document import Document, DocumentSourceInfo


class MessageRole(Enum):
    """メッセージの役割"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """チャットメッセージ"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """辞書からメッセージオブジェクトを作成"""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


# エイリアス
Message = ChatMessage


@dataclass
class ChatSession:
    """チャットセッション"""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: ChatMessage):
        """メッセージを追加"""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """最近のメッセージを取得"""
        return self.messages[-limit:] if limit > 0 else self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """辞書からセッションオブジェクトを作成"""
        messages = [ChatMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]
        
        return cls(
            session_id=data["session_id"],
            messages=messages,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {})
        )


# エイリアス
ChatHistory = ChatSession


@dataclass
class QueryRequest:
    """クエリリクエスト"""
    question: str
    session_id: Optional[str] = None
    include_history: bool = True
    max_sources: int = 3
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "question": self.question,
            "session_id": self.session_id,
            "include_history": self.include_history,
            "max_sources": self.max_sources,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_name": self.model_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryRequest":
        """辞書からリクエストオブジェクトを作成"""
        return cls(
            question=data["question"],
            session_id=data.get("session_id"),
            include_history=data.get("include_history", True),
            max_sources=data.get("max_sources", 3),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            model_name=data.get("model_name")
        )


@dataclass
class QueryResponse:
    """クエリレスポンス"""
    response: str
    source_documents: List[DocumentSourceInfo] = field(default_factory=list)
    session_id: Optional[str] = None
    processing_time: Optional[float] = None
    model_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "response": self.response,
            "source_documents": [doc.to_dict() for doc in self.source_documents],
            "session_id": self.session_id,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResponse":
        """辞書からレスポンスオブジェクトを作成"""
        source_docs = [DocumentSourceInfo(**doc_data) for doc_data in data.get("source_documents", [])]
        
        return cls(
            response=data["response"],
            source_documents=source_docs,
            session_id=data.get("session_id"),
            processing_time=data.get("processing_time"),
            model_used=data.get("model_used"),
            metadata=data.get("metadata", {})
        )


def create_user_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """ユーザーメッセージを作成"""
    return ChatMessage(
        role=MessageRole.USER,
        content=content,
        metadata=metadata or {}
    )


def create_assistant_message(content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
    """アシスタントメッセージを作成"""
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        metadata=metadata or {}
    )