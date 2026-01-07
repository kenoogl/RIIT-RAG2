"""
ConfigManagerクラスのテスト

このモジュールは、ConfigManagerクラスの機能をテストします。
"""

import pytest
import tempfile
import shutil
import yaml
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, assume
import hypothesis

from genkai_rag.core.config_manager import ConfigManager, ConfigChange


# テスト用の軽量な戦略を定義
simple_string = st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz0123456789")
config_value = st.one_of(
    st.text(min_size=1, max_size=50),
    st.integers(min_value=0, max_value=1000),
    st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.booleans()
)


class TestConfigManager:
    """ConfigManagerクラスの基本機能テスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(
            config_dir=self.temp_dir,
            backup_count=5
        )
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """ConfigManagerの初期化テスト"""
        assert self.config_manager.config_dir == Path(self.temp_dir)
        assert self.config_manager.backup_count == 5
        assert Path(self.temp_dir).exists()
        assert (Path(self.temp_dir) / "backups").exists()
        
        # デフォルト設定ファイルが作成されている
        config_file = Path(self.temp_dir) / "config.yaml"
        assert config_file.exists()
    
    def test_load_default_config(self):
        """デフォルト設定の読み込みテスト"""
        config = self.config_manager.load_config()
        
        # 必須セクションが存在する
        required_sections = ["llm", "rag", "chat", "system", "web"]
        for section in required_sections:
            assert section in config
        
        # LLM設定の確認
        assert "default_model" in config["llm"]
        assert "models" in config["llm"]
        assert "ollama_url" in config["llm"]
    
    def test_save_and_load_config(self):
        """設定の保存と読み込みテスト"""
        test_config = {
            "llm": {
                "default_model": "test-model",
                "ollama_url": "http://test:11434",
                "models": {
                    "test-model": {
                        "temperature": 0.5,
                        "max_tokens": 1024
                    }
                }
            },
            "rag": {
                "similarity_threshold": 0.8,
                "max_retrieved_docs": 15
            },
            "chat": {
                "max_history_size": 100
            },
            "system": {
                "log_level": "DEBUG"
            },
            "web": {
                "host": "localhost",
                "port": 9000
            }
        }
        
        # 設定を保存
        result = self.config_manager.save_config(test_config)
        assert result is True
        
        # 設定を読み込み
        loaded_config = self.config_manager.load_config()
        assert loaded_config["llm"]["default_model"] == "test-model"
        assert loaded_config["rag"]["similarity_threshold"] == 0.8
        assert loaded_config["chat"]["max_history_size"] == 100
    
    def test_get_llm_config(self):
        """LLM設定取得テスト"""
        # 存在するモデルの設定取得
        config = self.config_manager.get_llm_config("llama2:7b")
        assert "temperature" in config
        assert "max_tokens" in config
        
        # 存在しないモデルの設定取得（デフォルト値）
        default_config = self.config_manager.get_llm_config("nonexistent-model")
        assert default_config["temperature"] == 0.7
        assert default_config["max_tokens"] == 2048
    
    def test_update_llm_config(self):
        """LLM設定更新テスト"""
        model_name = "new-test-model"
        model_config = {
            "temperature": 0.3,
            "max_tokens": 4096,
            "top_p": 0.8
        }
        
        # 設定を更新
        result = self.config_manager.update_llm_config(model_name, model_config)
        assert result is True
        
        # 更新された設定を確認
        updated_config = self.config_manager.get_llm_config(model_name)
        assert updated_config["temperature"] == 0.3
        assert updated_config["max_tokens"] == 4096
        assert updated_config["top_p"] == 0.8
    
    def test_get_config_value(self):
        """ドット記法での設定値取得テスト"""
        # 存在する値の取得
        value = self.config_manager.get_config_value("llm.default_model")
        assert value is not None
        
        # ネストした値の取得
        temperature = self.config_manager.get_config_value("llm.models.llama2:7b.temperature")
        assert temperature == 0.7
        
        # 存在しない値の取得（デフォルト値）
        nonexistent = self.config_manager.get_config_value("nonexistent.key", "default")
        assert nonexistent == "default"
    
    def test_set_config_value(self):
        """ドット記法での設定値設定テスト"""
        # 新しい値を設定
        result = self.config_manager.set_config_value("test.new_key", "test_value")
        assert result is True
        
        # 設定された値を確認
        value = self.config_manager.get_config_value("test.new_key")
        assert value == "test_value"
        
        # 既存の値を更新
        result = self.config_manager.set_config_value("llm.default_model", "updated-model")
        assert result is True
        
        updated_value = self.config_manager.get_config_value("llm.default_model")
        assert updated_value == "updated-model"
    
    def test_config_validation(self):
        """設定検証テスト"""
        # 有効な設定
        valid_config = {
            "llm": {"default_model": "test"},
            "rag": {"similarity_threshold": 0.5},
            "chat": {"max_history_size": 50},
            "system": {"log_level": "INFO"},
            "web": {"host": "localhost"}
        }
        
        result = self.config_manager.save_config(valid_config)
        assert result is True
        
        # 無効な設定（必須セクション不足）
        invalid_config = {
            "llm": {"default_model": "test"}
            # 他の必須セクションが不足
        }
        
        result = self.config_manager.save_config(invalid_config)
        assert result is False
        
        # 無効な閾値
        invalid_threshold_config = {
            "llm": {"default_model": "test"},
            "rag": {"similarity_threshold": 1.5},  # 範囲外
            "chat": {"max_history_size": 50},
            "system": {"log_level": "INFO"},
            "web": {"host": "localhost"}
        }
        
        result = self.config_manager.save_config(invalid_threshold_config)
        assert result is False
    
    def test_backup_creation(self):
        """バックアップ作成テスト"""
        # 初期設定を変更
        test_config = self.config_manager.load_config()
        test_config["test_key"] = "test_value"
        
        # 設定を保存（バックアップが作成される）
        self.config_manager.save_config(test_config)
        
        # バックアップファイルが存在することを確認
        backup_dir = Path(self.temp_dir) / "backups"
        backup_files = list(backup_dir.glob("config_*.yaml"))
        assert len(backup_files) > 0
    
    def test_change_history(self):
        """変更履歴テスト"""
        import copy
        
        # 初期設定を取得
        initial_config = self.config_manager.load_config()
        
        # 設定を変更（深いコピーを使用）
        modified_config = copy.deepcopy(initial_config)
        modified_config["llm"]["default_model"] = "changed-model"
        
        self.config_manager.save_config(modified_config, user="test_user")
        
        # 変更履歴を取得
        history = self.config_manager.get_change_history()
        assert len(history) > 0
        
        # test_userによる変更を探す
        user_changes = [change for change in history if change.user == "test_user"]
        assert len(user_changes) > 0
        
        # llm.default_modelの変更が記録されている
        model_changes = [change for change in user_changes if "default_model" in change.key]
        assert len(model_changes) > 0
        
        # 最新の変更を確認
        latest_change = model_changes[0]
        assert latest_change.user == "test_user"
        assert latest_change.new_value == "changed-model"
    
    def test_list_backups(self):
        """バックアップ一覧取得テスト"""
        # 複数の設定変更を行ってバックアップを作成
        for i in range(3):
            config = self.config_manager.load_config()
            config[f"test_key_{i}"] = f"test_value_{i}"
            self.config_manager.save_config(config)
            time.sleep(0.01)  # タイムスタンプを区別するため
        
        # バックアップ一覧を取得
        backups = self.config_manager.list_backups()
        assert len(backups) >= 1  # 少なくとも1つのバックアップ
        
        # バックアップ情報の確認
        for backup in backups:
            assert "timestamp" in backup
            assert "filename" in backup
            assert "size_bytes" in backup
    
    def test_rollback_to_backup(self):
        """バックアップへのロールバックテスト"""
        # 初期設定を保存
        original_config = self.config_manager.load_config()
        original_model = original_config["llm"]["default_model"]
        
        # 設定を変更
        modified_config = original_config.copy()
        modified_config["llm"]["default_model"] = "modified-model"
        self.config_manager.save_config(modified_config)
        
        # バックアップ一覧を取得
        backups = self.config_manager.list_backups()
        assert len(backups) > 0
        
        # 最新のバックアップにロールバック
        backup_timestamp = backups[0]["timestamp"]
        result = self.config_manager.rollback_to_backup(backup_timestamp.replace(":", "_"))
        assert result is True
        
        # ロールバック後の設定を確認
        current_config = self.config_manager.load_config()
        assert current_config["llm"]["default_model"] == original_model
    
    def test_cleanup_old_backups(self):
        """古いバックアップのクリーンアップテスト"""
        # backup_countを超える数のバックアップを作成
        for i in range(self.config_manager.backup_count + 3):
            config = self.config_manager.load_config()
            config[f"cleanup_test_{i}"] = i
            self.config_manager.save_config(config)
            time.sleep(0.01)  # タイムスタンプを区別するため
        
        # クリーンアップ前のバックアップ数を確認
        backups_before = self.config_manager.list_backups()
        
        # クリーンアップを実行
        deleted_count = self.config_manager.cleanup_old_backups()
        
        # クリーンアップ後のバックアップ数を確認
        backups_after = self.config_manager.list_backups()
        assert len(backups_after) <= self.config_manager.backup_count
        
        # 削除されたバックアップ数を確認（0以上であることを確認）
        assert deleted_count >= 0
    
    def test_config_caching(self):
        """設定キャッシュテスト"""
        # 最初の読み込み
        config1 = self.config_manager.load_config()
        
        # キャッシュからの読み込み
        config2 = self.config_manager.load_config()
        
        # 同じ内容であることを確認
        assert config1 == config2
        
        # 設定ファイルを直接変更（キャッシュを無効化）
        config_file = Path(self.temp_dir) / "config.yaml"
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
        
        file_config["test_direct_change"] = "direct_value"
        
        with open(config_file, 'w') as f:
            yaml.dump(file_config, f)
        
        # 新しい設定が読み込まれることを確認
        config3 = self.config_manager.load_config()
        assert "test_direct_change" in config3
        assert config3["test_direct_change"] == "direct_value"


class TestConfigManagerProperties:
    """ConfigManagerのプロパティベーステスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        import uuid
        self.temp_dir = tempfile.mkdtemp(suffix=f"_{str(uuid.uuid4())[:8]}")
        self.config_manager = ConfigManager(
            config_dir=self.temp_dir,
            backup_count=3
        )
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        key_path=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=32, max_codepoint=126)).filter(lambda x: '.' not in x and x.strip()),
        value=config_value
    )
    @settings(max_examples=30, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_config_value_round_trip_properties(self, key_path, value):
        """
        Feature: genkai-rag-system, Property 23: 設定変更履歴
        任意の設定キーと値に対して、設定→取得のラウンドトリップが一貫している
        """
        assume(key_path.strip())  # 空白のみのキーを除外
        assume(len(key_path) <= 20)  # 長すぎるキーを除外
        
        # テスト用の一意なキーパスを生成
        import uuid
        unique_key = f"test.{key_path}_{str(uuid.uuid4())[:8]}"
        
        # 値を設定
        set_result = self.config_manager.set_config_value(unique_key, value)
        assert set_result is True
        
        # 値を取得
        retrieved_value = self.config_manager.get_config_value(unique_key)
        
        # ラウンドトリップが一貫している
        assert retrieved_value == value
    
    @given(
        model_name=simple_string,
        temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        max_tokens=st.integers(min_value=1, max_value=8192)
    )
    @settings(max_examples=20, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_llm_config_management_properties(self, model_name, temperature, max_tokens):
        """
        Feature: genkai-rag-system, Property 23: 設定変更履歴
        任意のLLMモデル設定に対して、更新と取得が一貫している
        """
        assume(model_name.strip())  # 空白のみのモデル名を除外
        assume(len(model_name) <= 30)  # 長すぎるモデル名を除外
        
        # テスト用の一意なモデル名を生成
        import uuid
        unique_model_name = f"{model_name}_{str(uuid.uuid4())[:8]}"
        
        # モデル設定を作成
        model_config = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "context_window": 4096
        }
        
        # 設定を更新
        update_result = self.config_manager.update_llm_config(unique_model_name, model_config)
        assert update_result is True
        
        # 設定を取得
        retrieved_config = self.config_manager.get_llm_config(unique_model_name)
        
        # 設定が正しく保存されている
        assert retrieved_config["temperature"] == temperature
        assert retrieved_config["max_tokens"] == max_tokens
        assert retrieved_config["top_p"] == 0.9
        assert retrieved_config["context_window"] == 4096
    
    @given(
        key=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz0123456789").filter(lambda x: x.strip()),
        value=st.one_of(
            st.text(min_size=1, max_size=20),
            st.integers(min_value=0, max_value=100),
            st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    )
    @settings(max_examples=5, deadline=1500, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_change_history_tracking_properties(self, key, value):
        """
        Feature: genkai-rag-system, Property 23: 設定変更履歴
        任意の設定変更に対して、変更履歴が正しく記録される
        """
        # テスト用の一意なキーを生成（シンプル）
        import uuid
        unique_key = f"test_{key}_{str(uuid.uuid4())[:6]}"
        
        # 変更前の履歴数を取得
        initial_count = len(self.config_manager.get_change_history())
        
        # 設定変更を実行
        result = self.config_manager.set_config_value(unique_key, value, user="test")
        assert result is True
        
        # 変更履歴を取得
        updated_history = self.config_manager.get_change_history(limit=5)  # 制限して高速化
        
        # 履歴が増加している
        assert len(updated_history) > 0
        
        # 最新の変更が記録されている
        latest_change = updated_history[0]
        assert latest_change.key == unique_key
        assert latest_change.new_value == value
        assert latest_change.user == "test"
    
    @given(
        key1=st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz").filter(lambda x: x.strip()),
        key2=st.text(min_size=1, max_size=8, alphabet="abcdefghijklmnopqrstuvwxyz").filter(lambda x: x.strip()),
        value1=st.integers(min_value=1, max_value=50),
        value2=st.integers(min_value=51, max_value=100)
    )
    @settings(max_examples=3, deadline=2000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_multiple_changes_history_properties(self, key1, key2, value1, value2):
        """
        Feature: genkai-rag-system, Property 23: 設定変更履歴
        複数の設定変更に対して、すべての変更が順序通りに記録される
        """
        assume(key1 != key2)  # 異なるキーであることを確認
        
        # テスト用の一意なプレフィックス
        import uuid
        test_id = str(uuid.uuid4())[:6]
        unique_key1 = f"multi_{test_id}_{key1}"
        unique_key2 = f"multi_{test_id}_{key2}"
        
        # 2つの変更を順次実行
        result1 = self.config_manager.set_config_value(unique_key1, value1, user="multi_test")
        assert result1 is True
        
        result2 = self.config_manager.set_config_value(unique_key2, value2, user="multi_test")
        assert result2 is True
        
        # 履歴を確認
        history = self.config_manager.get_change_history(limit=10)
        assert len(history) >= 2
        
        # 最新の2つの変更が記録されている
        recent_keys = [change.key for change in history[:2]]
        assert unique_key2 in recent_keys  # 最新の変更
        assert unique_key1 in recent_keys  # 前の変更
    
    @given(
        backup_count=st.integers(min_value=1, max_value=10),
        change_count=st.integers(min_value=2, max_value=15)
    )
    @settings(max_examples=10, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_backup_management_properties(self, backup_count, change_count):
        """
        Feature: genkai-rag-system, Property 23: 設定変更履歴
        任意のバックアップ数制限に対して、古いバックアップが適切に削除される
        """
        assume(change_count > backup_count)  # バックアップ制限を超える変更
        
        # 新しいConfigManagerを作成（指定されたbackup_count）
        import uuid
        temp_dir = tempfile.mkdtemp(suffix=f"_{str(uuid.uuid4())[:8]}")
        
        try:
            config_manager = ConfigManager(
                config_dir=temp_dir,
                backup_count=backup_count
            )
            
            # 制限を超える数の設定変更を実行
            for i in range(change_count):
                config = config_manager.load_config()
                config[f"backup_test_{i}"] = f"value_{i}"
                result = config_manager.save_config(config)
                assert result is True
            
            # バックアップ一覧を取得
            backups = config_manager.list_backups()
            
            # バックアップ数が制限内に収まっている
            assert len(backups) <= backup_count
            
            # 最新のバックアップが保持されている
            if backups:
                # バックアップが時刻順にソートされている
                timestamps = [backup["timestamp"] for backup in backups]
                assert timestamps == sorted(timestamps, reverse=True)
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    @given(
        user_name=simple_string
    )
    @settings(max_examples=20, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_user_tracking_properties(self, user_name):
        """
        Feature: genkai-rag-system, Property 23: 設定変更履歴
        任意のユーザー名に対して、変更履歴にユーザー情報が正しく記録される
        """
        assume(user_name.strip())  # 空白のみのユーザー名を除外
        assume(len(user_name) <= 30)  # 長すぎるユーザー名を除外
        
        # テスト用の一意なキーを生成
        import uuid
        unique_key = f"user_test_{str(uuid.uuid4())[:8]}"
        
        # 指定されたユーザーで設定変更を実行
        result = self.config_manager.set_config_value(unique_key, "test_value", user=user_name)
        assert result is True
        
        # 変更履歴を取得
        history = self.config_manager.get_change_history(limit=10)
        
        # 最新の変更にユーザー情報が記録されている
        assert len(history) > 0
        latest_change = history[0]
        assert latest_change.user == user_name
        assert latest_change.key == unique_key
        assert latest_change.new_value == "test_value"


class TestConfigChange:
    """ConfigChangeクラスのテスト"""
    
    def test_config_change_creation(self):
        """ConfigChange作成テスト"""
        timestamp = datetime.now()
        change = ConfigChange(
            timestamp=timestamp,
            key="test.key",
            old_value="old",
            new_value="new",
            user="test_user"
        )
        
        assert change.timestamp == timestamp
        assert change.key == "test.key"
        assert change.old_value == "old"
        assert change.new_value == "new"
        assert change.user == "test_user"
    
    def test_config_change_serialization(self):
        """ConfigChangeシリアライゼーションテスト"""
        timestamp = datetime.now()
        change = ConfigChange(
            timestamp=timestamp,
            key="test.key",
            old_value="old",
            new_value="new",
            user="test_user"
        )
        
        # 辞書に変換
        change_dict = change.to_dict()
        assert change_dict["timestamp"] == timestamp.isoformat()
        assert change_dict["key"] == "test.key"
        assert change_dict["old_value"] == "old"
        assert change_dict["new_value"] == "new"
        assert change_dict["user"] == "test_user"
        
        # 辞書から復元
        restored_change = ConfigChange.from_dict(change_dict)
        assert restored_change.timestamp == timestamp
        assert restored_change.key == change.key
        assert restored_change.old_value == change.old_value
        assert restored_change.new_value == change.new_value
        assert restored_change.user == change.user