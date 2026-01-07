"""会話データモデルのテスト

Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
"""

import pytest
from hypothesis import given, strategies as st, assume
from datetime import datetime, timedelta
from typing import List

from genkai_rag.models.chat import (
    Message, ChatSession, ChatHistory,
    create_user_message, create_assistant_message
)


class TestMessage:
    """Messageクラスのテスト"""
    
    def test_message_creation(self):
        """メッセージオブジェクトの作成テスト"""
        msg = Message(
            session_id="test-session",
            role="user",
            content="テストメッセージ"
        )
        
        assert msg.session_id == "test-session"
        assert msg.role == "user"
        assert msg.content == "テストメッセージ"
        assert msg.id  # IDが自動生成される
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_validation(self):
        """メッセージデータの妥当性チェックテスト"""
        # 有効なメッセージ
        valid_msg = Message(
            session_id="valid-session",
            role="user",
            content="有効なメッセージ"
        )
        assert valid_msg.is_valid()
        
        # 無効なメッセージ（roleが不正）
        with pytest.raises(ValueError, match="無効なrole"):
            Message(
                session_id="session",
                role="invalid",
                content="メッセージ"
            )
    
    def test_message_role_check(self):
        """メッセージロールのチェックテスト"""
        user_msg = Message(role="user", content="ユーザーメッセージ")
        assert user_msg.is_user_message()
        assert not user_msg.is_assistant_message()
        
        assistant_msg = Message(role="assistant", content="アシスタントメッセージ")
        assert assistant_msg.is_assistant_message()
        assert not assistant_msg.is_user_message()
    
    def test_message_serialization(self):
        """メッセージのシリアライゼーションテスト"""
        msg = Message(
            session_id="test-session",
            role="user",
            content="シリアライゼーションテスト"
        )
        
        # 辞書変換
        msg_dict = msg.to_dict()
        assert msg_dict["content"] == "シリアライゼーションテスト"
        
        # 辞書から復元
        restored_msg = Message.from_dict(msg_dict)
        assert restored_msg.content == msg.content
        assert restored_msg.role == msg.role


class TestChatSession:
    """ChatSessionクラスのテスト"""
    
    def test_session_creation(self):
        """セッションオブジェクトの作成テスト"""
        session = ChatSession()
        
        assert session.session_id  # IDが自動生成される
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_activity, datetime)
        assert session.message_count == 0
    
    def test_session_activity_update(self):
        """セッション活動時刻の更新テスト"""
        session = ChatSession()
        original_time = session.last_activity
        
        # 少し待ってから更新
        import time
        time.sleep(0.01)
        session.update_activity()
        
        assert session.last_activity > original_time
    
    def test_session_expiry(self):
        """セッション期限切れのテスト"""
        # 古いセッション
        old_session = ChatSession()
        old_session.last_activity = datetime.now() - timedelta(hours=25)
        
        assert old_session.is_expired(timeout_hours=24)
        
        # 新しいセッション
        new_session = ChatSession()
        assert not new_session.is_expired(timeout_hours=24)


class TestChatHistory:
    """ChatHistoryクラスのテスト"""
    
    def test_history_creation(self):
        """履歴オブジェクトの作成テスト"""
        session = ChatSession()
        history = ChatHistory(session, max_size=10)
        
        assert history.session == session
        assert history.max_size == 10
        assert history.get_message_count() == 0
    
    def test_message_addition(self):
        """メッセージ追加のテスト"""
        session = ChatSession()
        history = ChatHistory(session)
        
        msg = create_user_message(session.session_id, "テストメッセージ")
        history.add_message(msg)
        
        assert history.get_message_count() == 1
        assert session.message_count == 1
        
        messages = history.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "テストメッセージ"
    
    def test_history_size_limit(self):
        """履歴サイズ制限のテスト"""
        session = ChatSession()
        history = ChatHistory(session, max_size=3)
        
        # 5つのメッセージを追加
        for i in range(5):
            msg = create_user_message(session.session_id, f"メッセージ{i}")
            history.add_message(msg)
        
        # 最大サイズ（3）に制限される
        assert history.get_message_count() == 3
        
        # 最新の3つのメッセージが保持される
        messages = history.get_messages()
        assert messages[0].content == "メッセージ2"
        assert messages[1].content == "メッセージ3"
        assert messages[2].content == "メッセージ4"
    
    def test_history_clear(self):
        """履歴クリアのテスト"""
        session = ChatSession()
        history = ChatHistory(session)
        
        # メッセージを追加
        msg = create_user_message(session.session_id, "テストメッセージ")
        history.add_message(msg)
        
        assert history.get_message_count() == 1
        
        # 履歴をクリア
        history.clear()
        
        assert history.get_message_count() == 0
        assert session.message_count == 0


class TestMessageCreationHelpers:
    """メッセージ作成ヘルパー関数のテスト"""
    
    def test_create_user_message(self):
        """ユーザーメッセージ作成のテスト"""
        msg = create_user_message("test-session", "ユーザーメッセージ")
        
        assert msg.session_id == "test-session"
        assert msg.role == "user"
        assert msg.content == "ユーザーメッセージ"
        assert msg.is_user_message()
    
    def test_create_assistant_message(self):
        """アシスタントメッセージ作成のテスト"""
        sources = ["https://example.com/doc1", "https://example.com/doc2"]
        msg = create_assistant_message("test-session", "アシスタント回答", sources)
        
        assert msg.session_id == "test-session"
        assert msg.role == "assistant"
        assert msg.content == "アシスタント回答"
        assert msg.sources == sources
        assert msg.is_assistant_message()


# プロパティベーステスト
class TestChatHistoryProperties:
    """チャット履歴管理のプロパティテスト
    
    Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
    """
    
    @given(
        session_id=st.text(min_size=1, max_size=100).filter(lambda x: len(x.strip()) > 0),
        messages=st.lists(
            st.text(min_size=1, max_size=1000).filter(lambda x: len(x.strip()) > 0),
            min_size=1,
            max_size=20
        ),
        max_size=st.integers(min_value=1, max_value=50)
    )
    def test_session_isolation(self, session_id, messages, max_size):
        """
        プロパティ 12: セッション単位の履歴管理
        任意のセッションに対して、システムは履歴をセッションIDで管理し、
        異なるセッション間で履歴が混在しない
        
        Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
        """
        assume(len(session_id.strip()) > 0)
        
        # 2つの異なるセッションを作成
        session1 = ChatSession(session_id=session_id + "_1")
        session2 = ChatSession(session_id=session_id + "_2")
        
        history1 = ChatHistory(session1, max_size=max_size)
        history2 = ChatHistory(session2, max_size=max_size)
        
        # セッション1にメッセージを追加
        for i, content in enumerate(messages):
            if i % 2 == 0:  # 偶数インデックスはユーザーメッセージ
                msg = create_user_message(session1.session_id, content)
            else:  # 奇数インデックスはアシスタントメッセージ
                msg = create_assistant_message(session1.session_id, content)
            history1.add_message(msg)
        
        # セッション2にも異なるメッセージを追加
        for i, content in enumerate(messages):
            modified_content = f"session2_{content}"  # プレフィックスを付けて区別
            if i % 2 == 0:
                msg = create_user_message(session2.session_id, modified_content)
            else:
                msg = create_assistant_message(session2.session_id, modified_content)
            history2.add_message(msg)
        
        # プロパティ1: 各履歴のメッセージは正しいセッションIDを持つ
        for msg in history1.get_messages():
            assert msg.session_id == session1.session_id
        
        for msg in history2.get_messages():
            assert msg.session_id == session2.session_id
        
        # プロパティ2: セッション間で履歴が混在しない
        history1_contents = [msg.content for msg in history1.get_messages()]
        history2_contents = [msg.content for msg in history2.get_messages()]
        
        # 内容が異なることを確認（プレフィックスを付けているため）
        for content1, content2 in zip(history1_contents, history2_contents):
            assert content1 != content2
            # セッション2のメッセージにはプレフィックスが付いている
            assert content2.startswith("session2_")
        
        # プロパティ3: セッションIDが異なる
        assert session1.session_id != session2.session_id
        
        # プロパティ4: 各セッションのメッセージ数が正しく管理される
        expected_count1 = min(len(messages), max_size)
        expected_count2 = min(len(messages), max_size)
        
        assert history1.get_message_count() == expected_count1
        assert history2.get_message_count() == expected_count2
    
    @given(
        content_list=st.lists(
            st.text(min_size=1, max_size=500).filter(lambda x: len(x.strip()) > 0),
            min_size=1,
            max_size=30
        ),
        max_size=st.integers(min_value=1, max_value=20)
    )
    def test_message_order_preservation(self, content_list, max_size):
        """
        メッセージの順序が保持されることを検証
        
        Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
        """
        session = ChatSession()
        history = ChatHistory(session, max_size=max_size)
        
        # メッセージを順番に追加
        for i, content in enumerate(content_list):
            msg = create_user_message(session.session_id, f"{i}:{content}")
            history.add_message(msg)
        
        # 取得したメッセージの順序をチェック
        messages = history.get_messages()
        
        # サイズ制限により、最新のメッセージのみが保持される
        expected_start = max(0, len(content_list) - max_size)
        
        for i, msg in enumerate(messages):
            expected_index = expected_start + i
            expected_content = f"{expected_index}:{content_list[expected_index]}"
            assert msg.content == expected_content
    
    @given(
        user_messages=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: len(x.strip()) > 0),
            min_size=0, 
            max_size=10
        ),
        assistant_messages=st.lists(
            st.text(min_size=1, max_size=200).filter(lambda x: len(x.strip()) > 0),
            min_size=0, 
            max_size=10
        )
    )
    def test_message_role_filtering(self, user_messages, assistant_messages):
        """
        メッセージロールによるフィルタリングを検証
        
        Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
        """
        session = ChatSession()
        history = ChatHistory(session)
        
        # ユーザーメッセージを追加
        for content in user_messages:
            msg = create_user_message(session.session_id, content)
            history.add_message(msg)
        
        # アシスタントメッセージを追加
        for content in assistant_messages:
            msg = create_assistant_message(session.session_id, content)
            history.add_message(msg)
        
        # ロール別にメッセージを取得
        user_msgs = history.get_user_messages()
        assistant_msgs = history.get_assistant_messages()
        
        # プロパティ1: ユーザーメッセージ数が正しい
        assert len(user_msgs) == len(user_messages)
        
        # プロパティ2: アシスタントメッセージ数が正しい
        assert len(assistant_msgs) == len(assistant_messages)
        
        # プロパティ3: 各メッセージのロールが正しい
        for msg in user_msgs:
            assert msg.is_user_message()
            assert not msg.is_assistant_message()
        
        for msg in assistant_msgs:
            assert msg.is_assistant_message()
            assert not msg.is_user_message()
        
        # プロパティ4: 全メッセージ数が合計と一致
        total_messages = len(user_messages) + len(assistant_messages)
        assert history.get_message_count() == total_messages
    
    @given(
        max_size=st.integers(min_value=1, max_value=10),
        message_count=st.integers(min_value=1, max_value=20)
    )
    def test_size_limit_enforcement(self, max_size, message_count):
        """
        サイズ制限の適用を検証
        
        Feature: genkai-rag-system, Property 12: セッション単位の履歴管理
        """
        session = ChatSession()
        history = ChatHistory(session, max_size=max_size)
        
        # 指定された数のメッセージを追加
        for i in range(message_count):
            msg = create_user_message(session.session_id, f"メッセージ{i}")
            history.add_message(msg)
        
        # プロパティ1: 履歴サイズが制限以下
        assert history.get_message_count() <= max_size
        
        # プロパティ2: メッセージ数が多い場合は制限値と一致
        if message_count > max_size:
            assert history.get_message_count() == max_size
        else:
            assert history.get_message_count() == message_count
        
        # プロパティ3: 最新のメッセージが保持される
        messages = history.get_messages()
        if message_count > max_size:
            # 最新のmax_size個のメッセージが保持される
            expected_start = message_count - max_size
            for i, msg in enumerate(messages):
                expected_content = f"メッセージ{expected_start + i}"
                assert msg.content == expected_content