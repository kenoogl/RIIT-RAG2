"""
ChatManager: 会話履歴管理機能

このモジュールは、セッション単位の会話履歴の保存、取得、管理機能を提供します。
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading
from dataclasses import asdict

from ..models.chat import Message, ChatSession, ChatHistory, create_user_message, create_assistant_message
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class ChatManager:
    """
    会話履歴管理クラス
    
    機能:
    - セッション単位の履歴保存と取得
    - 履歴サイズ制限と古い履歴の管理
    - プライバシー保護のための自動削除
    - 並行アクセス対応
    """
    
    def __init__(
        self,
        storage_dir: str = "data/chat_history",
        max_history_size: int = 100,
        max_session_age_days: int = 30,
        cleanup_interval_hours: int = 24
    ):
        """
        ChatManagerを初期化
        
        Args:
            storage_dir: 履歴保存ディレクトリ
            max_history_size: セッション当たりの最大履歴数
            max_session_age_days: セッションの最大保存日数
            cleanup_interval_hours: 自動クリーンアップ間隔（時間）
        """
        self.storage_dir = Path(storage_dir)
        self.max_history_size = max_history_size
        self.max_session_age_days = max_session_age_days
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # スレッドセーフティのためのロック
        self._session_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        
        # セッション情報のキャッシュ
        self._session_cache: Dict[str, ChatSession] = {}
        self._cache_lock = threading.Lock()
        
        # 最後のクリーンアップ時刻
        self._last_cleanup = datetime.now()
        
        # 設定管理
        self.config_manager = ConfigManager()
        
        # ストレージディレクトリを作成
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChatManager initialized with storage_dir={storage_dir}, max_history_size={max_history_size}")
    
    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """セッション固有のロックを取得"""
        with self._locks_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """セッションファイルのパスを取得"""
        return self.storage_dir / f"{session_id}.json"
    
    def _load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """セッションデータをファイルから読み込み"""
        file_path = self._get_session_file_path(session_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load session data for {session_id}: {e}")
            return None
    
    def _save_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """セッションデータをファイルに保存"""
        file_path = self._get_session_file_path(session_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            return True
        except IOError as e:
            logger.error(f"Failed to save session data for {session_id}: {e}")
            return False
    
    def get_or_create_session(self, session_id: str) -> ChatSession:
        """
        セッションを取得または新規作成
        
        Args:
            session_id: セッションID
            
        Returns:
            ChatSessionオブジェクト
        """
        # ファイルから読み込み（キャッシュは使わない - 常に最新状態を取得）
        session_lock = self._get_session_lock(session_id)
        with session_lock:
            data = self._load_session_data(session_id)
            
            if data and 'session' in data:
                session_data = data['session']
                session = ChatSession(
                    session_id=session_data['session_id'],
                    created_at=datetime.fromisoformat(session_data['created_at']),
                    last_activity=datetime.now(),  # 現在時刻に更新
                    message_count=session_data.get('message_count', 0)
                )
            else:
                # 新規セッション作成
                session = ChatSession(
                    session_id=session_id,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    message_count=0
                )
            
            # キャッシュに保存
            with self._cache_lock:
                self._session_cache[session_id] = session
            
            return session
    
    def save_message(self, session_id: str, message: Message) -> bool:
        """
        メッセージを保存
        
        Args:
            session_id: セッションID
            message: 保存するメッセージ
            
        Returns:
            保存が成功した場合True
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            # 既存データを読み込み
            data = self._load_session_data(session_id) or {'session': {}, 'messages': []}
            
            # セッション情報を取得または作成（ロック内で直接処理）
            if 'session' in data and data['session']:
                session_data = data['session']
                session = ChatSession(
                    session_id=session_data['session_id'],
                    created_at=datetime.fromisoformat(session_data['created_at']),
                    last_activity=datetime.now(),  # 現在時刻に更新
                    message_count=session_data.get('message_count', 0)
                )
            else:
                # 新規セッション作成 - キャッシュから取得を試行
                with self._cache_lock:
                    if session_id in self._session_cache:
                        cached_session = self._session_cache[session_id]
                        session = ChatSession(
                            session_id=cached_session.session_id,
                            created_at=cached_session.created_at,  # キャッシュの作成時刻を保持
                            last_activity=datetime.now(),
                            message_count=0
                        )
                    else:
                        session = ChatSession(
                            session_id=session_id,
                            created_at=datetime.now(),
                            last_activity=datetime.now(),
                            message_count=0
                        )
            
            # メッセージを追加
            message_dict = asdict(message)
            data['messages'].append(message_dict)
            
            # 履歴サイズ制限を適用
            if len(data['messages']) > self.max_history_size:
                # 古いメッセージを削除
                removed_count = len(data['messages']) - self.max_history_size
                data['messages'] = data['messages'][removed_count:]
                logger.info(f"Removed {removed_count} old messages from session {session_id}")
            
            # セッション情報を更新
            session.message_count = len(data['messages'])
            session.last_activity = datetime.now()
            
            data['session'] = asdict(session)
            
            # ファイルに保存
            success = self._save_session_data(session_id, data)
            
            if success:
                # キャッシュを更新
                with self._cache_lock:
                    self._session_cache[session_id] = session
                
                logger.debug(f"Saved message to session {session_id}: {message.role}")
            
            return success
    
    def get_chat_history(self, session_id: str, limit: int = 10) -> List[Message]:
        """
        会話履歴を取得
        
        Args:
            session_id: セッションID
            limit: 取得する最大メッセージ数
            
        Returns:
            メッセージのリスト（時系列順）
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            data = self._load_session_data(session_id)
            
            if not data or 'messages' not in data:
                return []
            
            messages = []
            message_dicts = data['messages'][-limit:] if limit > 0 else data['messages']
            
            for msg_dict in message_dicts:
                try:
                    message = Message(
                        id=msg_dict['id'],
                        session_id=msg_dict['session_id'],
                        role=msg_dict['role'],
                        content=msg_dict['content'],
                        timestamp=datetime.fromisoformat(msg_dict['timestamp']),
                        sources=msg_dict.get('sources', [])
                    )
                    messages.append(message)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse message in session {session_id}: {e}")
                    continue
            
            return messages
    
    def clear_history(self, session_id: str) -> bool:
        """
        セッションの履歴をクリア
        
        Args:
            session_id: セッションID
            
        Returns:
            クリアが成功した場合True
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            file_path = self._get_session_file_path(session_id)
            
            try:
                if file_path.exists():
                    file_path.unlink()
                
                # キャッシュからも削除
                with self._cache_lock:
                    self._session_cache.pop(session_id, None)
                
                logger.info(f"Cleared history for session {session_id}")
                return True
                
            except OSError as e:
                logger.error(f"Failed to clear history for session {session_id}: {e}")
                return False
    
    def manage_history_size(self, session_id: str, max_size: int) -> bool:
        """
        履歴サイズを管理（指定サイズまで削減）
        
        Args:
            session_id: セッションID
            max_size: 最大履歴サイズ
            
        Returns:
            管理が成功した場合True
        """
        session_lock = self._get_session_lock(session_id)
        
        with session_lock:
            data = self._load_session_data(session_id)
            
            if not data or 'messages' not in data:
                return True
            
            messages = data['messages']
            if len(messages) <= max_size:
                return True
            
            # 古いメッセージを削除
            removed_count = len(messages) - max_size
            data['messages'] = messages[removed_count:]
            
            # セッション情報を更新
            if 'session' in data:
                session_data = data['session']
                session_data['message_count'] = len(data['messages'])
                session_data['last_activity'] = datetime.now().isoformat()
            
            success = self._save_session_data(session_id, data)
            
            if success:
                logger.info(f"Managed history size for session {session_id}: removed {removed_count} messages")
            
            return success
    
    def get_session_info(self, session_id: str) -> Optional[ChatSession]:
        """
        セッション情報を取得
        
        Args:
            session_id: セッションID
            
        Returns:
            ChatSessionオブジェクト、存在しない場合はNone
        """
        # キャッシュから確認
        with self._cache_lock:
            if session_id in self._session_cache:
                return self._session_cache[session_id]
        
        # ファイルから読み込み
        data = self._load_session_data(session_id)
        
        if not data or 'session' not in data:
            return None
        
        session_data = data['session']
        session = ChatSession(
            session_id=session_data['session_id'],
            created_at=datetime.fromisoformat(session_data['created_at']),
            last_activity=datetime.fromisoformat(session_data['last_activity']),
            message_count=session_data.get('message_count', 0)
        )
        
        return session
    
    def list_sessions(self, active_only: bool = False) -> List[ChatSession]:
        """
        セッション一覧を取得
        
        Args:
            active_only: アクティブなセッションのみを取得するか
            
        Returns:
            ChatSessionのリスト
        """
        sessions = []
        cutoff_time = datetime.now() - timedelta(days=self.max_session_age_days)
        
        for file_path in self.storage_dir.glob("*.json"):
            session_id = file_path.stem
            session = self.get_session_info(session_id)
            
            if session:
                if active_only and session.last_activity < cutoff_time:
                    continue
                sessions.append(session)
        
        # 最終活動時刻でソート
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions
    
    def cleanup_old_sessions(self) -> int:
        """
        古いセッションをクリーンアップ
        
        Returns:
            削除されたセッション数
        """
        cutoff_time = datetime.now() - timedelta(days=self.max_session_age_days)
        deleted_count = 0
        
        for file_path in self.storage_dir.glob("*.json"):
            session_id = file_path.stem
            session = self.get_session_info(session_id)
            
            if session and session.last_activity < cutoff_time:
                if self.clear_history(session_id):
                    deleted_count += 1
        
        self._last_cleanup = datetime.now()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old sessions")
        
        return deleted_count
    
    def auto_cleanup_if_needed(self) -> bool:
        """
        必要に応じて自動クリーンアップを実行
        
        Returns:
            クリーンアップが実行された場合True
        """
        time_since_cleanup = datetime.now() - self._last_cleanup
        
        if time_since_cleanup.total_seconds() >= self.cleanup_interval_hours * 3600:
            self.cleanup_old_sessions()
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        sessions = self.list_sessions()
        active_sessions = self.list_sessions(active_only=True)
        
        total_messages = 0
        for session in sessions:
            total_messages += session.message_count
        
        stats = {
            "total_sessions": len(sessions),
            "active_sessions": len(active_sessions),
            "total_messages": total_messages,
            "storage_dir": str(self.storage_dir),
            "max_history_size": self.max_history_size,
            "max_session_age_days": self.max_session_age_days,
            "last_cleanup": self._last_cleanup.isoformat(),
            "cache_size": len(self._session_cache)
        }
        
        return stats
    
    def create_user_message(self, session_id: str, content: str, sources: Optional[List[str]] = None) -> Message:
        """
        ユーザーメッセージを作成
        
        Args:
            session_id: セッションID
            content: メッセージ内容
            sources: 出典情報
            
        Returns:
            作成されたMessageオブジェクト
        """
        message = create_user_message(session_id, content)
        if sources:
            message.sources = sources
        return message
    
    def create_assistant_message(self, session_id: str, content: str, sources: Optional[List[str]] = None) -> Message:
        """
        アシスタントメッセージを作成
        
        Args:
            session_id: セッションID
            content: メッセージ内容
            sources: 出典情報
            
        Returns:
            作成されたMessageオブジェクト
        """
        return create_assistant_message(session_id, content, sources or [])
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        セッションデータをエクスポート
        
        Args:
            session_id: セッションID
            
        Returns:
            セッションデータの辞書、存在しない場合はNone
        """
        data = self._load_session_data(session_id)
        
        if not data:
            return None
        
        # 出力用にフォーマット
        export_data = {
            "session_id": session_id,
            "session_info": data.get('session', {}),
            "messages": data.get('messages', []),
            "exported_at": datetime.now().isoformat()
        }
        
        return export_data
    
    def import_session(self, session_data: Dict[str, Any]) -> bool:
        """
        セッションデータをインポート
        
        Args:
            session_data: インポートするセッションデータ
            
        Returns:
            インポートが成功した場合True
        """
        try:
            session_id = session_data['session_id']
            
            # データを変換
            import_data = {
                'session': session_data.get('session_info', {}),
                'messages': session_data.get('messages', [])
            }
            
            # セッションロックを取得して保存
            session_lock = self._get_session_lock(session_id)
            with session_lock:
                success = self._save_session_data(session_id, import_data)
                
                if success:
                    # キャッシュをクリア（次回アクセス時に再読み込み）
                    with self._cache_lock:
                        self._session_cache.pop(session_id, None)
                    
                    logger.info(f"Imported session data for {session_id}")
                
                return success
                
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to import session data: {e}")
            return False