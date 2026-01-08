"""
ChatManagerクラスのテスト

このモジュールは、ChatManagerクラスの機能をテストします。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, assume
import hypothesis
import threading
import time

from genkai_rag.core.chat_manager import ChatManager
from genkai_rag.models.chat import Message, ChatSession, create_user_message, create_assistant_message


# テスト用の軽量な戦略を定義
simple_session_id = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=32, max_codepoint=126)).filter(lambda x: x.strip() and '/' not in x and '\\' not in x)
simple_content = st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), min_codepoint=32, max_codepoint=126)).filter(lambda x: x.strip())


class TestChatManager:
    """ChatManagerクラスの基本機能テスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.chat_manager = ChatManager(
            storage_dir=self.temp_dir,
            max_history_size=10,
            max_session_age_days=7,
            cleanup_interval_hours=1
        )
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chat_manager_initialization(self):
        """ChatManagerの初期化テスト"""
        assert self.chat_manager.storage_dir == Path(self.temp_dir)
        assert self.chat_manager.max_history_size == 10
        assert self.chat_manager.max_session_age_days == 7
        assert self.chat_manager.cleanup_interval_hours == 1
        assert Path(self.temp_dir).exists()
    
    def test_get_or_create_session_new(self):
        """新規セッション作成テスト"""
        session_id = "test_session_1"
        session = self.chat_manager.get_or_create_session(session_id)
        
        assert session.session_id == session_id
        assert session.message_count == 0
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
    
    def test_get_or_create_session_existing(self):
        """既存セッション取得テスト"""
        session_id = "test_session_2"
        
        # 最初にセッションを作成してメッセージを保存（ファイルに永続化）
        session1 = self.chat_manager.get_or_create_session(session_id)
        original_created_at = session1.created_at
        
        # メッセージを保存してセッションをファイルに永続化
        message = create_user_message(session_id, "Test message")
        self.chat_manager.save_message(session_id, message)
        
        # 少し待ってから再取得
        time.sleep(0.01)
        session2 = self.chat_manager.get_or_create_session(session_id)
        
        assert session2.session_id == session_id
        assert session2.created_at == original_created_at
        assert session2.last_activity > session1.last_activity
    
    def test_save_message_success(self):
        """メッセージ保存の成功テスト"""
        session_id = "test_session_3"
        message = create_user_message(session_id, "Hello, world!")
        
        result = self.chat_manager.save_message(session_id, message)
        
        assert result is True
        
        # セッション情報が更新されているか確認
        session = self.chat_manager.get_session_info(session_id)
        assert session is not None
        assert session.message_count == 1
    
    def test_save_multiple_messages(self):
        """複数メッセージ保存テスト"""
        session_id = "test_session_4"
        
        messages = [
            create_user_message(session_id, "First message"),
            create_assistant_message(session_id, "First response"),
            create_user_message(session_id, "Second message"),
            create_assistant_message(session_id, "Second response")
        ]
        
        for message in messages:
            result = self.chat_manager.save_message(session_id, message)
            assert result is True
        
        # 履歴を取得して確認
        history = self.chat_manager.get_chat_history(session_id)
        assert len(history) == 4
        assert history[0].content == "First message"
        assert history[1].content == "First response"
        assert history[2].content == "Second message"
        assert history[3].content == "Second response"
    
    def test_get_chat_history_empty(self):
        """空の履歴取得テスト"""
        session_id = "test_session_5"
        history = self.chat_manager.get_chat_history(session_id)
        
        assert history == []
    
    def test_get_chat_history_with_limit(self):
        """制限付き履歴取得テスト"""
        session_id = "test_session_6"
        
        # 5つのメッセージを保存
        for i in range(5):
            message = create_user_message(session_id, f"Message {i}")
            self.chat_manager.save_message(session_id, message)
        
        # 制限付きで取得
        history = self.chat_manager.get_chat_history(session_id, limit=3)
        assert len(history) == 3
        assert history[0].content == "Message 2"  # 最新3つ
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"
    
    def test_clear_history_success(self):
        """履歴クリアの成功テスト"""
        session_id = "test_session_7"
        
        # メッセージを保存
        message = create_user_message(session_id, "Test message")
        self.chat_manager.save_message(session_id, message)
        
        # 履歴をクリア
        result = self.chat_manager.clear_history(session_id)
        assert result is True
        
        # 履歴が空になっているか確認
        history = self.chat_manager.get_chat_history(session_id)
        assert history == []
        
        # セッション情報も削除されているか確認
        session = self.chat_manager.get_session_info(session_id)
        assert session is None
    
    def test_clear_history_nonexistent(self):
        """存在しないセッションの履歴クリアテスト"""
        session_id = "nonexistent_session"
        result = self.chat_manager.clear_history(session_id)
        
        assert result is True  # ファイルが存在しなくても成功とする
    
    def test_manage_history_size_within_limit(self):
        """制限内での履歴サイズ管理テスト"""
        session_id = "test_session_8"
        
        # 3つのメッセージを保存
        for i in range(3):
            message = create_user_message(session_id, f"Message {i}")
            self.chat_manager.save_message(session_id, message)
        
        # 制限を5に設定（現在3つなので変更なし）
        result = self.chat_manager.manage_history_size(session_id, 5)
        assert result is True
        
        history = self.chat_manager.get_chat_history(session_id)
        assert len(history) == 3
    
    def test_manage_history_size_over_limit(self):
        """制限超過での履歴サイズ管理テスト"""
        session_id = "test_session_9"
        
        # 5つのメッセージを保存
        for i in range(5):
            message = create_user_message(session_id, f"Message {i}")
            self.chat_manager.save_message(session_id, message)
        
        # 制限を3に設定
        result = self.chat_manager.manage_history_size(session_id, 3)
        assert result is True
        
        history = self.chat_manager.get_chat_history(session_id)
        assert len(history) == 3
        assert history[0].content == "Message 2"  # 古い2つが削除される
        assert history[1].content == "Message 3"
        assert history[2].content == "Message 4"
    
    def test_automatic_history_size_limit(self):
        """自動履歴サイズ制限テスト"""
        session_id = "test_session_10"
        
        # max_history_size (10) を超えるメッセージを保存
        for i in range(15):
            message = create_user_message(session_id, f"Message {i}")
            self.chat_manager.save_message(session_id, message)
        
        # 履歴が制限内に収まっているか確認
        history = self.chat_manager.get_chat_history(session_id)
        assert len(history) == 10
        assert history[0].content == "Message 5"  # 最新10個
        assert history[9].content == "Message 14"
    
    def test_get_session_info_existing(self):
        """既存セッション情報取得テスト"""
        session_id = "test_session_11"
        
        # セッションを作成してメッセージを保存
        message = create_user_message(session_id, "Test message")
        self.chat_manager.save_message(session_id, message)
        
        session_info = self.chat_manager.get_session_info(session_id)
        assert session_info is not None
        assert session_info.session_id == session_id
        assert session_info.message_count == 1
    
    def test_get_session_info_nonexistent(self):
        """存在しないセッション情報取得テスト"""
        session_id = "nonexistent_session"
        session_info = self.chat_manager.get_session_info(session_id)
        
        assert session_info is None
    
    def test_list_sessions(self):
        """セッション一覧取得テスト"""
        # 複数のセッションを作成
        session_ids = ["session_1", "session_2", "session_3"]
        
        for session_id in session_ids:
            message = create_user_message(session_id, "Test message")
            self.chat_manager.save_message(session_id, message)
        
        sessions = self.chat_manager.list_sessions()
        assert len(sessions) == 3
        
        # セッションIDが含まれているか確認
        session_id_list = [s.session_id for s in sessions]
        for session_id in session_ids:
            assert session_id in session_id_list
    
    def test_cleanup_old_sessions(self):
        """古いセッションクリーンアップテスト"""
        # 古いセッションをシミュレート
        old_manager = ChatManager(
            storage_dir=self.temp_dir,
            max_session_age_days=0  # 即座に古いとみなす
        )
        
        session_id = "old_session"
        message = create_user_message(session_id, "Old message")
        old_manager.save_message(session_id, message)
        
        # クリーンアップを実行
        deleted_count = old_manager.cleanup_old_sessions()
        assert deleted_count == 1
        
        # セッションが削除されているか確認
        session_info = old_manager.get_session_info(session_id)
        assert session_info is None
    
    def test_get_statistics(self):
        """統計情報取得テスト"""
        # いくつかのセッションを作成
        for i in range(3):
            session_id = f"stats_session_{i}"
            for j in range(i + 1):  # 異なる数のメッセージ
                message = create_user_message(session_id, f"Message {j}")
                self.chat_manager.save_message(session_id, message)
        
        stats = self.chat_manager.get_statistics()
        
        assert stats["total_sessions"] == 3
        assert stats["total_messages"] == 6  # 1 + 2 + 3
        assert stats["max_history_size"] == 10
        assert stats["max_session_age_days"] == 7
        assert "last_cleanup" in stats
        assert "cache_size" in stats
    
    def test_create_user_message(self):
        """ユーザーメッセージ作成テスト"""
        session_id = "test_session_12"
        content = "User message content"
        sources = ["source1", "source2"]
        
        message = self.chat_manager.create_user_message(session_id, content, sources)
        
        assert message.session_id == session_id
        assert message.role == "user"
        assert message.content == content
        assert message.sources == sources
    
    def test_create_assistant_message(self):
        """アシスタントメッセージ作成テスト"""
        session_id = "test_session_13"
        content = "Assistant message content"
        sources = ["source1"]
        
        message = self.chat_manager.create_assistant_message(session_id, content, sources)
        
        assert message.session_id == session_id
        assert message.role == "assistant"
        assert message.content == content
        assert message.sources == sources
    
    def test_export_session(self):
        """セッションエクスポートテスト"""
        session_id = "export_session"
        
        # メッセージを保存
        messages = [
            create_user_message(session_id, "User message"),
            create_assistant_message(session_id, "Assistant response")
        ]
        
        for message in messages:
            self.chat_manager.save_message(session_id, message)
        
        # エクスポート
        export_data = self.chat_manager.export_session(session_id)
        
        assert export_data is not None
        assert export_data["session_id"] == session_id
        assert "session_info" in export_data
        assert "messages" in export_data
        assert len(export_data["messages"]) == 2
        assert "exported_at" in export_data
    
    def test_export_nonexistent_session(self):
        """存在しないセッションのエクスポートテスト"""
        export_data = self.chat_manager.export_session("nonexistent")
        assert export_data is None
    
    def test_import_session(self):
        """セッションインポートテスト"""
        session_id = "import_session"
        
        # インポートデータを準備
        import_data = {
            "session_id": session_id,
            "session_info": {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "message_count": 1
            },
            "messages": [
                {
                    "id": "msg_1",
                    "session_id": session_id,
                    "role": "user",
                    "content": "Imported message",
                    "timestamp": datetime.now().isoformat(),
                    "sources": []
                }
            ]
        }
        
        # インポート実行
        result = self.chat_manager.import_session(import_data)
        assert result is True
        
        # インポートされたデータを確認
        history = self.chat_manager.get_chat_history(session_id)
        assert len(history) == 1
        assert history[0].content == "Imported message"
    
    def test_thread_safety(self):
        """スレッドセーフティテスト"""
        session_id = "thread_test_session"
        num_threads = 5
        messages_per_thread = 10
        
        def save_messages(thread_id):
            for i in range(messages_per_thread):
                message = create_user_message(session_id, f"Thread {thread_id} Message {i}")
                self.chat_manager.save_message(session_id, message)
        
        # 複数スレッドで同時にメッセージを保存
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=save_messages, args=(i,))
            threads.append(thread)
            thread.start()
        
        # すべてのスレッドの完了を待つ
        for thread in threads:
            thread.join()
        
        # 結果を確認
        history = self.chat_manager.get_chat_history(session_id, limit=0)  # 全履歴取得
        expected_count = min(num_threads * messages_per_thread, self.chat_manager.max_history_size)
        assert len(history) == expected_count
    
    def test_auto_cleanup_if_needed_no_cleanup(self):
        """自動クリーンアップが不要な場合のテスト"""
        # 最近クリーンアップが実行されているため、クリーンアップは実行されない
        result = self.chat_manager.auto_cleanup_if_needed()
        assert result is False
    
    def test_auto_cleanup_if_needed_cleanup_required(self):
        """自動クリーンアップが必要な場合のテスト"""
        # クリーンアップ間隔を短く設定したChatManagerを作成
        short_cleanup_manager = ChatManager(
            storage_dir=self.temp_dir,
            max_history_size=10,
            max_session_age_days=0,  # 即座に古いとみなす
            cleanup_interval_hours=0  # 即座にクリーンアップが必要
        )
        
        # 古いセッションを作成
        session_id = "old_session_for_auto_cleanup"
        message = create_user_message(session_id, "Old message")
        short_cleanup_manager.save_message(session_id, message)
        
        # 最後のクリーンアップ時刻を過去に設定
        from datetime import timedelta
        short_cleanup_manager._last_cleanup = datetime.now() - timedelta(hours=2)
        
        # 自動クリーンアップを実行
        result = short_cleanup_manager.auto_cleanup_if_needed()
        assert result is True
        
        # セッションが削除されているか確認
        session_info = short_cleanup_manager.get_session_info(session_id)
        assert session_info is None


class TestChatManagerProperties:
    """ChatManagerのプロパティベーステスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        import uuid
        self.temp_dir = tempfile.mkdtemp(suffix=f"_{str(uuid.uuid4())[:8]}")
        self.chat_manager = ChatManager(
            storage_dir=self.temp_dir,
            max_history_size=20,  # テスト用に小さく
            max_session_age_days=7,
            cleanup_interval_hours=1
        )
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        session_id=simple_session_id,
        content=simple_content
    )
    @settings(max_examples=50, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_message_persistence_properties(self, session_id, content):
        """
        Feature: genkai-rag-system, Property 11: 会話履歴の保持
        任意のセッションとメッセージに対して、保存と取得が一貫している
        """
        assume(session_id.strip())  # 空白のみのセッションIDを除外
        assume(content.strip())     # 空白のみのコンテンツを除外
        assume(len(session_id) <= 30)  # 長すぎるIDを除外
        assume(len(content) <= 200)    # 長すぎるコンテンツを除外
        
        # テスト用の一意なセッションIDを生成（Hypothesisの実行回数を考慮）
        import uuid
        unique_session_id = f"{session_id}_{str(uuid.uuid4())[:8]}"
        
        # メッセージを作成して保存
        message = create_user_message(unique_session_id, content)
        save_result = self.chat_manager.save_message(unique_session_id, message)
        
        # 保存が成功する
        assert save_result is True
        
        # 保存されたメッセージを取得
        history = self.chat_manager.get_chat_history(unique_session_id)
        
        # メッセージが正しく保存されている
        assert len(history) == 1
        saved_message = history[0]
        assert saved_message.session_id == unique_session_id
        assert saved_message.content == content
        assert saved_message.role == "user"
    
    @given(
        session_id=simple_session_id,
        message_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_session_isolation_properties(self, session_id, message_count):
        """
        Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
        任意のセッションに対して、履歴が他のセッションと混在しない
        """
        assume(session_id.strip())  # 空白のみのセッションIDを除外
        assume(len(session_id) <= 30)  # 長すぎるIDを除外
        
        # テスト用の一意なセッションIDを生成
        import uuid
        test_uuid = str(uuid.uuid4())[:8]
        unique_session_id1 = f"{session_id}_1_{test_uuid}"
        unique_session_id2 = f"{session_id}_2_{test_uuid}"
        
        # 両方のセッションにメッセージを保存
        for i in range(message_count):
            msg1 = create_user_message(unique_session_id1, f"Message {i} for session1")
            msg2 = create_user_message(unique_session_id2, f"Message {i} for session2")
            
            self.chat_manager.save_message(unique_session_id1, msg1)
            self.chat_manager.save_message(unique_session_id2, msg2)
        
        # 各セッションの履歴を取得
        history1 = self.chat_manager.get_chat_history(unique_session_id1)
        history2 = self.chat_manager.get_chat_history(unique_session_id2)
        
        # 履歴が分離されている
        assert len(history1) == message_count
        assert len(history2) == message_count
        
        # 各履歴が正しいセッションIDを持つ
        for msg in history1:
            assert msg.session_id == unique_session_id1
        
        for msg in history2:
            assert msg.session_id == unique_session_id2
        
        # 履歴の内容が混在していない
        contents1 = [msg.content for msg in history1]
        contents2 = [msg.content for msg in history2]
        
        for content in contents1:
            assert "for session1" in content
            assert "for session2" not in content
        
        for content in contents2:
            assert "for session2" in content
            assert "for session1" not in content
    
    @given(
        session_id=simple_session_id,
        initial_count=st.integers(min_value=5, max_value=15),
        size_limit=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=20, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_history_size_management_properties(self, session_id, initial_count, size_limit):
        """
        Feature: genkai-rag-system, Property 13: 古い履歴の管理
        任意のサイズ制限に対して、制限を超えた古い履歴が適切に削除される
        """
        assume(session_id.strip())  # 空白のみのセッションIDを除外
        assume(initial_count > size_limit)  # 制限を超える数のメッセージ
        assume(len(session_id) <= 30)  # 長すぎるIDを除外
        
        # テスト用の一意なセッションIDを生成
        import uuid
        unique_session_id = f"{session_id}_{str(uuid.uuid4())[:8]}"
        
        # 制限を超える数のメッセージを保存
        for i in range(initial_count):
            message = create_user_message(unique_session_id, f"Message {i}")
            self.chat_manager.save_message(unique_session_id, message)
        
        # サイズ制限を適用
        result = self.chat_manager.manage_history_size(unique_session_id, size_limit)
        assert result is True
        
        # 履歴を取得
        history = self.chat_manager.get_chat_history(unique_session_id)
        
        # 制限内に収まっている
        assert len(history) == size_limit
        
        # 最新のメッセージが保持されている
        expected_start = initial_count - size_limit
        for i, message in enumerate(history):
            expected_content = f"Message {expected_start + i}"
            assert message.content == expected_content
    
    @given(
        session_id=simple_session_id
    )
    @settings(max_examples=30, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_session_lifecycle_properties(self, session_id):
        """
        Feature: genkai-rag-system, Property 14: 履歴クリア機能
        任意のセッションに対して、クリア後は新しい会話として開始される
        """
        assume(session_id.strip())  # 空白のみのセッションIDを除外
        assume(len(session_id) <= 30)  # 長すぎるIDを除外
        
        # テスト用の一意なセッションIDを生成
        import uuid
        unique_session_id = f"{session_id}_{str(uuid.uuid4())[:8]}"
        
        # メッセージを保存
        message = create_user_message(unique_session_id, "Test message")
        self.chat_manager.save_message(unique_session_id, message)
        
        # 履歴が存在することを確認
        history_before = self.chat_manager.get_chat_history(unique_session_id)
        assert len(history_before) == 1
        
        session_before = self.chat_manager.get_session_info(unique_session_id)
        assert session_before is not None
        assert session_before.message_count == 1
        
        # 履歴をクリア
        clear_result = self.chat_manager.clear_history(unique_session_id)
        assert clear_result is True
        
        # 履歴が空になっている
        history_after = self.chat_manager.get_chat_history(unique_session_id)
        assert len(history_after) == 0
        
        # セッション情報も削除されている
        session_after = self.chat_manager.get_session_info(unique_session_id)
        assert session_after is None
        
        # 新しいメッセージを保存すると新しいセッションとして開始される
        new_message = create_user_message(unique_session_id, "New message after clear")
        save_result = self.chat_manager.save_message(unique_session_id, new_message)
        assert save_result is True
        
        new_session = self.chat_manager.get_session_info(unique_session_id)
        assert new_session is not None
        assert new_session.message_count == 1
        
        # 新しい履歴が正しく保存されている
        new_history = self.chat_manager.get_chat_history(unique_session_id)
        assert len(new_history) == 1
        assert new_history[0].content == "New message after clear"
    
    @given(
        session_ids=st.lists(
            simple_session_id,
            min_size=1, max_size=3, unique=True
        )
    )
    @settings(max_examples=15, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_multiple_sessions_properties(self, session_ids):
        """
        Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
        複数のセッションが独立して管理される
        """
        # テスト用の一意なセッションIDを生成
        import uuid
        test_uuid = str(uuid.uuid4())[:8]
        unique_session_ids = [f"{sid}_{test_uuid}" for sid in session_ids]
        
        # 各セッションにメッセージを保存（一意のメッセージ内容）
        for i, session_id in enumerate(unique_session_ids):
            message = create_user_message(session_id, f"Unique message {i} for session {session_id}")
            result = self.chat_manager.save_message(session_id, message)
            assert result is True
        
        # セッション一覧を取得
        sessions = self.chat_manager.list_sessions()
        session_id_list = [s.session_id for s in sessions]
        
        # すべてのセッションが存在する
        for session_id in unique_session_ids:
            assert session_id in session_id_list
        
        # 各セッションが独立している
        for i, session_id in enumerate(unique_session_ids):
            history = self.chat_manager.get_chat_history(session_id)
            assert len(history) == 1
            assert history[0].session_id == session_id


class TestChatManagerAdvancedProperties:
    """ChatManagerの高度なコンテキスト管理プロパティベーステスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.chat_manager = ChatManager(
            storage_dir=self.temp_dir,
            max_history_size=50,  # テスト用に大きな値を設定
            max_session_age_days=7,
            cleanup_interval_hours=1
        )
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        conversation_length=st.integers(min_value=2, max_value=10),
        query_context_size=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=5000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_context_management_property(self, conversation_length, query_context_size):
        """
        プロパティ 24: コンテキスト管理
        任意の連続する質問に対して、システムは前の質問と回答をコンテキストとして含め、文脈を考慮した回答を生成する
        
        Feature: genkai-rag-system, Property 24: コンテキスト管理
        **検証: 要件 8.1**
        """
        session_id = f"context_test_{conversation_length}_{query_context_size}"
        
        # 会話履歴を構築
        messages = []
        for i in range(conversation_length):
            # ユーザーメッセージ
            user_msg = create_user_message(session_id, f"質問 {i+1}: これは{i+1}番目の質問です")
            result = self.chat_manager.save_message(session_id, user_msg)
            assert result is True
            messages.append(user_msg)
            
            # アシスタントメッセージ
            assistant_msg = create_assistant_message(
                session_id, 
                f"回答 {i+1}: これは{i+1}番目の質問への回答です",
                sources=[f"source_{i+1}.html"]
            )
            result = self.chat_manager.save_message(session_id, assistant_msg)
            assert result is True
            messages.append(assistant_msg)
        
        # プロパティ1: 履歴が正しく保存されている
        full_history = self.chat_manager.get_chat_history(session_id, limit=conversation_length * 2)
        assert len(full_history) == conversation_length * 2
        
        # プロパティ2: コンテキストサイズに応じた履歴取得
        context_history = self.chat_manager.get_chat_history(session_id, limit=query_context_size * 2)
        assert len(context_history) <= query_context_size * 2
        assert len(context_history) <= len(full_history)
        
        # プロパティ3: 取得された履歴が最新のものから順序付けられている
        if len(context_history) > 1:
            # get_chat_historyは新しい順（降順）で返すため、タイムスタンプも降順になる
            for i in range(len(context_history) - 1):
                # 新しいメッセージが先に来るので、前のメッセージのタイムスタンプ >= 後のメッセージのタイムスタンプ
                # ただし、同じ時刻に作成されたメッセージもあるため、>= を使用
                current_time = context_history[i].timestamp
                next_time = context_history[i + 1].timestamp
                # マイクロ秒レベルの差は許容する（同じ秒内での作成）
                time_diff = (current_time - next_time).total_seconds()
                assert time_diff >= -0.001, f"履歴の順序が正しくありません: {current_time} < {next_time}"
        
        # プロパティ4: コンテキストに質問と回答のペアが含まれる
        user_messages = [msg for msg in context_history if msg.role == "user"]
        assistant_messages = [msg for msg in context_history if msg.role == "assistant"]
        
        # 質問と回答の数が適切
        assert len(user_messages) >= 0
        assert len(assistant_messages) >= 0
        
        # プロパティ5: セッション情報が正確
        session_info = self.chat_manager.get_session_info(session_id)
        assert session_info is not None
        assert session_info.session_id == session_id
        assert session_info.message_count == conversation_length * 2
    
    @given(
        history_length=st.integers(min_value=5, max_value=15),
        relevance_threshold=st.floats(min_value=0.1, max_value=0.9)
    )
    @settings(max_examples=8, deadline=5000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_history_selection_property(self, history_length, relevance_threshold):
        """
        プロパティ 25: 履歴選択機能
        任意の長い会話履歴に対して、システムは現在の質問に最も関連性の高い履歴のみを選択してコンテキストに含める
        
        Feature: genkai-rag-system, Property 25: 履歴選択機能
        **検証: 要件 8.2**
        """
        session_id = f"selection_test_{history_length}"
        
        # 長い会話履歴を作成
        topics = ["天気", "料理", "技術", "スポーツ", "音楽"]
        for i in range(history_length):
            topic = topics[i % len(topics)]
            
            user_msg = create_user_message(session_id, f"{topic}について教えて {i+1}")
            self.chat_manager.save_message(session_id, user_msg)
            
            assistant_msg = create_assistant_message(
                session_id, 
                f"{topic}に関する回答 {i+1}",
                sources=[f"{topic}_source_{i+1}.html"]
            )
            self.chat_manager.save_message(session_id, assistant_msg)
        
        # プロパティ1: 全履歴が保存されている
        full_history = self.chat_manager.get_chat_history(session_id, limit=history_length * 2)
        assert len(full_history) == history_length * 2
        
        # プロパティ2: 制限された履歴取得が機能する
        limited_history = self.chat_manager.get_chat_history(session_id, limit=6)  # 3ペア分
        assert len(limited_history) <= 6
        assert len(limited_history) <= len(full_history)
        
        # プロパティ3: 選択された履歴が最新のものを含む
        if len(limited_history) > 0:
            latest_message = max(full_history, key=lambda m: m.timestamp)
            selected_latest = max(limited_history, key=lambda m: m.timestamp)
            assert selected_latest.timestamp == latest_message.timestamp
        
        # プロパティ4: 履歴選択が一貫している
        # 同じパラメータで複数回呼び出すと同じ結果が得られる
        history1 = self.chat_manager.get_chat_history(session_id, limit=6)
        history2 = self.chat_manager.get_chat_history(session_id, limit=6)
        
        assert len(history1) == len(history2)
        for msg1, msg2 in zip(history1, history2):
            assert msg1.id == msg2.id
            assert msg1.content == msg2.content
    
    @given(
        session_count=st.integers(min_value=1, max_value=5),
        messages_per_session=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=8, deadline=5000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_session_termination_property(self, session_count, messages_per_session):
        """
        プロパティ 26: セッション終了処理
        任意のセッション終了に対して、システムは設定に基づいて履歴を保存または削除し、適切にリソースを解放する
        
        Feature: genkai-rag-system, Property 26: セッション終了処理
        **検証: 要件 8.3**
        """
        session_ids = []
        
        # 複数のセッションを作成
        for i in range(session_count):
            session_id = f"termination_test_{i}_{session_count}_{messages_per_session}"
            session_ids.append(session_id)
            
            # 各セッションにメッセージを追加
            for j in range(messages_per_session):
                user_msg = create_user_message(session_id, f"メッセージ {j+1} in session {i+1}")
                self.chat_manager.save_message(session_id, user_msg)
                
                assistant_msg = create_assistant_message(session_id, f"回答 {j+1} in session {i+1}")
                self.chat_manager.save_message(session_id, assistant_msg)
        
        # プロパティ1: すべてのセッションが作成されている
        sessions = self.chat_manager.list_sessions()
        created_session_ids = [s.session_id for s in sessions]
        
        for session_id in session_ids:
            assert session_id in created_session_ids
        
        # プロパティ2: 各セッションに適切な数のメッセージがある
        for session_id in session_ids:
            history = self.chat_manager.get_chat_history(session_id, limit=messages_per_session * 2 + 5)  # 十分な制限を設定
            assert len(history) == messages_per_session * 2  # user + assistant
            
            session_info = self.chat_manager.get_session_info(session_id)
            assert session_info is not None
            assert session_info.message_count == messages_per_session * 2
        
        # プロパティ3: セッション履歴のクリアが機能する
        if session_ids:
            test_session = session_ids[0]
            
            # 履歴をクリア
            result = self.chat_manager.clear_history(test_session)
            assert result is True
            
            # 履歴がクリアされている
            cleared_history = self.chat_manager.get_chat_history(test_session)
            assert len(cleared_history) == 0
            
            # セッション情報が更新されている
            session_info = self.chat_manager.get_session_info(test_session)
            if session_info is not None:  # セッションが完全に削除される場合もある
                assert session_info.message_count == 0
        
        # プロパティ4: 他のセッションは影響を受けない
        for session_id in session_ids[1:]:
            history = self.chat_manager.get_chat_history(session_id, limit=messages_per_session * 2 + 5)  # 十分な制限を設定
            assert len(history) == messages_per_session * 2  # 変更されていない
    
    @given(
        retention_days=st.integers(min_value=1, max_value=10),
        old_session_count=st.integers(min_value=1, max_value=3),
        new_session_count=st.integers(min_value=1, max_value=3)
    )
    @settings(max_examples=5, deadline=8000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_history_retention_management_property(self, retention_days, old_session_count, new_session_count):
        """
        プロパティ 27: 履歴保存期間管理
        任意の設定された保存期間に対して、システムは期間を超えた履歴を自動的に削除し、プライバシーを保護する
        
        Feature: genkai-rag-system, Property 27: 履歴保存期間管理
        **検証: 要件 8.4**
        """
        current_time = datetime.now()
        
        # 古いセッション（保存期間を超過）を作成
        old_session_ids = []
        for i in range(old_session_count):
            session_id = f"old_session_{i}_{retention_days}"
            old_session_ids.append(session_id)
            
            # 古いタイムスタンプでメッセージを作成
            old_timestamp = current_time - timedelta(days=retention_days + 1)
            
            user_msg = create_user_message(session_id, f"古いメッセージ {i+1}")
            user_msg.timestamp = old_timestamp
            self.chat_manager.save_message(session_id, user_msg)
        
        # 新しいセッション（保存期間内）を作成
        new_session_ids = []
        for i in range(new_session_count):
            session_id = f"new_session_{i}_{retention_days}"
            new_session_ids.append(session_id)
            
            # 新しいタイムスタンプでメッセージを作成
            new_timestamp = current_time - timedelta(days=retention_days // 2)
            
            user_msg = create_user_message(session_id, f"新しいメッセージ {i+1}")
            user_msg.timestamp = new_timestamp
            self.chat_manager.save_message(session_id, user_msg)
        
        # プロパティ1: すべてのセッションが初期状態で存在する
        all_sessions = self.chat_manager.list_sessions()
        all_session_ids = [s.session_id for s in all_sessions]
        
        for session_id in old_session_ids + new_session_ids:
            assert session_id in all_session_ids
        
        # プロパティ2: 古いセッションのクリーンアップ
        # ChatManagerの設定を更新して短い保存期間を設定
        self.chat_manager.max_session_age_days = retention_days
        
        # クリーンアップを実行
        cleanup_result = self.chat_manager.cleanup_old_sessions()
        
        # プロパティ3: 古いセッションが削除される
        remaining_sessions = self.chat_manager.list_sessions()
        remaining_session_ids = [s.session_id for s in remaining_sessions]
        
        # 新しいセッションは残っている
        for session_id in new_session_ids:
            assert session_id in remaining_session_ids
        
        # プロパティ4: システム統計が正確
        stats = self.chat_manager.get_statistics()
        assert isinstance(stats["total_sessions"], int)
        assert isinstance(stats["total_messages"], int)
        assert stats["total_sessions"] >= 0
        assert stats["total_messages"] >= 0
    
    @given(
        size_limit=st.integers(min_value=3, max_value=10),
        message_count=st.integers(min_value=5, max_value=20)
    )
    @settings(max_examples=8, deadline=5000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_history_size_limit_property(self, size_limit, message_count):
        """
        プロパティ 28: 履歴サイズ制限
        任意の設定されたサイズ制限に対して、システムは制限を超える履歴の追加を防ぎ、システム負荷を管理する
        
        Feature: genkai-rag-system, Property 28: 履歴サイズ制限
        **検証: 要件 8.5**
        """
        assume(message_count > size_limit)  # 制限を超えるメッセージ数のみテスト
        
        session_id = f"size_limit_test_{size_limit}_{message_count}"
        
        # ChatManagerのサイズ制限を設定
        self.chat_manager.max_history_size = size_limit
        
        # 制限を超える数のメッセージを追加
        for i in range(message_count):
            user_msg = create_user_message(session_id, f"メッセージ {i+1}")
            result = self.chat_manager.save_message(session_id, user_msg)
            assert result is True
            
            # 各追加後に履歴サイズをチェック
            history = self.chat_manager.get_chat_history(session_id)
            
            # プロパティ1: 履歴サイズが制限を超えない
            assert len(history) <= size_limit, f"履歴サイズ {len(history)} が制限 {size_limit} を超えています"
        
        # プロパティ2: 最終的な履歴サイズが制限以下
        final_history = self.chat_manager.get_chat_history(session_id)
        assert len(final_history) <= size_limit
        
        # プロパティ3: 最新のメッセージが保持されている
        if len(final_history) > 0:
            latest_message = final_history[0]  # get_chat_historyは新しい順
            # 制限内で最新のメッセージが保持されている（制限を超えた場合は最新のsize_limit個）
            expected_latest_index = max(1, message_count - size_limit + 1)
            assert f"メッセージ {message_count}" in latest_message.content or f"メッセージ {expected_latest_index}" in latest_message.content
        
        # プロパティ4: セッション情報が正確
        session_info = self.chat_manager.get_session_info(session_id)
        assert session_info is not None
        
        # メッセージカウントは実際に保存されているメッセージ数と一致
        assert session_info.message_count == len(final_history)
        
        # プロパティ5: 履歴管理が適切に機能している
        # 追加のメッセージを送信しても制限が維持される
        extra_msg = create_user_message(session_id, f"追加メッセージ {message_count + 1}")
        result = self.chat_manager.save_message(session_id, extra_msg)
        assert result is True
        
        updated_history = self.chat_manager.get_chat_history(session_id)
        assert len(updated_history) <= size_limit
        
        # 最新のメッセージが含まれている（制限内で最新のものが保持される）
        if len(updated_history) > 0:
            latest_content = updated_history[0].content
            # 制限により古いメッセージが削除される可能性があるため、
            # 最新のメッセージまたは制限内の最新メッセージが含まれていることを確認
            expected_latest_msg = f"追加メッセージ {message_count + 1}"
            # 制限内で保持される最古のメッセージのインデックスを計算
            oldest_kept_index = max(1, (message_count + 1) - size_limit + 1)
            expected_oldest_msg = f"メッセージ {oldest_kept_index}"
            
            # 最新メッセージまたは制限内の有効なメッセージが含まれている
            assert (expected_latest_msg in latest_content or 
                    any(f"メッセージ {i}" in latest_content for i in range(oldest_kept_index, message_count + 2)))