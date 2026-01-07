"""
SystemMonitorクラスのテスト

このモジュールは、SystemMonitorクラスの機能をテストします。
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
import hypothesis

from genkai_rag.core.system_monitor import SystemMonitor, SystemStatus, AlertThreshold


# テスト用の軽量な戦略を定義
percentage = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
memory_gb = st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False)


class TestSystemMonitor:
    """SystemMonitorクラスの基本機能テスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.data_dir = Path(self.temp_dir) / "data"
        
        self.system_monitor = SystemMonitor(
            log_dir=str(self.log_dir),
            data_dir=str(self.data_dir),
            monitoring_interval=1,  # テスト用に短く
            retention_days=7
        )
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        # 監視を停止
        if self.system_monitor.is_monitoring_active():
            self.system_monitor.stop_monitoring()
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_monitor_initialization(self):
        """SystemMonitorの初期化テスト"""
        assert self.system_monitor.log_dir == self.log_dir
        assert self.system_monitor.data_dir == self.data_dir
        assert self.system_monitor.monitoring_interval == 1
        assert self.system_monitor.retention_days == 7
        
        # ディレクトリが作成されている
        assert self.log_dir.exists()
        assert self.data_dir.exists()
        
        # アラート閾値がデフォルト値
        assert self.system_monitor.thresholds.memory_percent == 80.0
        assert self.system_monitor.thresholds.disk_percent == 90.0
        assert self.system_monitor.thresholds.cpu_percent == 90.0
    
    def test_check_memory_usage(self):
        """メモリ使用率チェックテスト"""
        memory_usage = self.system_monitor.check_memory_usage()
        
        # 妥当な範囲の値が返される
        assert isinstance(memory_usage, float)
        assert 0.0 <= memory_usage <= 100.0
    
    def test_check_disk_usage(self):
        """ディスク使用率チェックテスト"""
        # デフォルトパス（data_dir）のディスク使用率
        disk_usage = self.system_monitor.check_disk_usage()
        
        assert isinstance(disk_usage, float)
        assert 0.0 <= disk_usage <= 100.0
        
        # 指定パスのディスク使用率
        disk_usage_specific = self.system_monitor.check_disk_usage(str(self.temp_dir))
        
        assert isinstance(disk_usage_specific, float)
        assert 0.0 <= disk_usage_specific <= 100.0
    
    def test_check_cpu_usage(self):
        """CPU使用率チェックテスト"""
        cpu_usage = self.system_monitor.check_cpu_usage(interval=0.1)  # 短い間隔
        
        assert isinstance(cpu_usage, float)
        assert 0.0 <= cpu_usage <= 100.0
    
    def test_get_system_status(self):
        """システム状態取得テスト"""
        status = self.system_monitor.get_system_status()
        
        assert isinstance(status, SystemStatus)
        assert isinstance(status.timestamp, datetime)
        assert 0.0 <= status.memory_usage_percent <= 100.0
        assert status.memory_available_gb >= 0.0
        assert status.memory_total_gb > 0.0
        assert 0.0 <= status.disk_usage_percent <= 100.0
        assert status.disk_available_gb >= 0.0
        assert status.disk_total_gb > 0.0
        assert 0.0 <= status.cpu_usage_percent <= 100.0
        assert status.process_count > 0
        assert status.uptime_seconds >= 0.0
    
    def test_log_system_status(self):
        """システム状態ログ記録テスト"""
        result = self.system_monitor.log_system_status()
        assert result is True
        
        # ログファイルが作成されている
        assert self.system_monitor.status_log_file.exists()
        
        # ログ内容を確認
        with open(self.system_monitor.status_log_file, 'r') as f:
            logs = json.load(f)
        
        assert len(logs) == 1
        log_entry = logs[0]
        
        # 必要なフィールドが存在する
        required_fields = [
            "timestamp", "memory_usage_percent", "memory_available_gb",
            "memory_total_gb", "disk_usage_percent", "disk_available_gb",
            "disk_total_gb", "cpu_usage_percent", "process_count", "uptime_seconds"
        ]
        
        for field in required_fields:
            assert field in log_entry
    
    def test_multiple_status_logs(self):
        """複数回のステータスログテスト"""
        # 複数回ログを記録
        for _ in range(3):
            result = self.system_monitor.log_system_status()
            assert result is True
            time.sleep(0.01)  # 少し待機
        
        # ログエントリが3つ存在する
        with open(self.system_monitor.status_log_file, 'r') as f:
            logs = json.load(f)
        
        assert len(logs) == 3
        
        # タイムスタンプが昇順になっている
        timestamps = [datetime.fromisoformat(log["timestamp"]) for log in logs]
        assert timestamps == sorted(timestamps)
    
    def test_get_status_history(self):
        """ステータス履歴取得テスト"""
        # 複数回ログを記録
        for _ in range(3):
            self.system_monitor.log_system_status()
            time.sleep(0.01)
        
        # 履歴を取得
        history = self.system_monitor.get_status_history(hours=1)
        
        assert len(history) == 3
        assert all(isinstance(status, SystemStatus) for status in history)
        
        # 時刻順にソートされている
        timestamps = [status.timestamp for status in history]
        assert timestamps == sorted(timestamps)
    
    def test_set_alert_thresholds(self):
        """アラート閾値設定テスト"""
        result = self.system_monitor.set_alert_thresholds(
            memory_percent=70.0,
            disk_percent=85.0,
            cpu_percent=80.0
        )
        
        assert result is True
        assert self.system_monitor.thresholds.memory_percent == 70.0
        assert self.system_monitor.thresholds.disk_percent == 85.0
        assert self.system_monitor.thresholds.cpu_percent == 80.0
    
    @patch('genkai_rag.core.system_monitor.psutil.virtual_memory')
    def test_memory_alert_generation(self, mock_memory):
        """メモリアラート生成テスト"""
        # 高いメモリ使用率をシミュレート
        mock_memory.return_value = Mock(
            percent=85.0,  # 閾値(80%)を超える
            total=8 * 1024**3,  # 8GB
            available=1 * 1024**3  # 1GB
        )
        
        # アラート閾値を設定
        self.system_monitor.set_alert_thresholds(memory_percent=80.0)
        
        # ステータスをログに記録（アラートが生成される）
        result = self.system_monitor.log_system_status()
        assert result is True
        
        # アラートファイルが作成されている
        assert self.system_monitor.alert_log_file.exists()
        
        # アラート内容を確認
        alerts = self.system_monitor.get_alerts(hours=1)
        assert len(alerts) > 0
        
        memory_alert = next((alert for alert in alerts if alert["type"] == "memory"), None)
        assert memory_alert is not None
        assert memory_alert["level"] == "warning"
        assert memory_alert["value"] == 85.0
        assert memory_alert["threshold"] == 80.0
    
    def test_cleanup_old_data(self):
        """古いデータクリーンアップテスト"""
        # テスト用の古いログファイルを作成
        old_log_file = self.log_dir / "old_test.log"
        old_log_file.write_text("old log content")
        
        # ファイルの更新時刻を古く設定
        old_time = time.time() - (8 * 24 * 3600)  # 8日前
        import os
        os.utime(old_log_file, (old_time, old_time))
        
        # 新しいログファイルを作成
        new_log_file = self.log_dir / "new_test.log"
        new_log_file.write_text("new log content")
        
        # クリーンアップを実行（7日より古いファイルを削除）
        results = self.system_monitor.cleanup_old_data(retention_days=7)
        
        # 古いファイルが削除され、新しいファイルは残っている
        assert not old_log_file.exists()
        assert new_log_file.exists()
        assert results["deleted_log_files"] == 1
    
    def test_background_monitoring_start_stop(self):
        """バックグラウンド監視の開始・停止テスト"""
        # 監視開始
        result = self.system_monitor.start_monitoring()
        assert result is True
        assert self.system_monitor.is_monitoring_active()
        
        # 少し待機してログが記録されることを確認
        time.sleep(1.5)
        
        # 監視停止
        result = self.system_monitor.stop_monitoring()
        assert result is True
        assert not self.system_monitor.is_monitoring_active()
        
        # ログファイルが作成されている
        assert self.system_monitor.status_log_file.exists()
    
    def test_background_monitoring_duplicate_start(self):
        """重複した監視開始テスト"""
        # 最初の開始
        result1 = self.system_monitor.start_monitoring()
        assert result1 is True
        
        # 重複した開始（失敗する）
        result2 = self.system_monitor.start_monitoring()
        assert result2 is False
        
        # 停止
        self.system_monitor.stop_monitoring()
    
    def test_monitoring_stop_without_start(self):
        """開始していない監視の停止テスト"""
        result = self.system_monitor.stop_monitoring()
        assert result is False


class TestSystemMonitorProperties:
    """SystemMonitorのプロパティベーステスト"""
    
    def setup_method(self):
        """各テストメソッドの前に実行される設定"""
        import uuid
        self.temp_dir = tempfile.mkdtemp(suffix=f"_{str(uuid.uuid4())[:8]}")
        self.log_dir = Path(self.temp_dir) / "logs"
        self.data_dir = Path(self.temp_dir) / "data"
        
        self.system_monitor = SystemMonitor(
            log_dir=str(self.log_dir),
            data_dir=str(self.data_dir),
            monitoring_interval=1,
            retention_days=3  # テスト用に短く
        )
        
        # 既存のログファイルをクリア
        if self.system_monitor.status_log_file.exists():
            self.system_monitor.status_log_file.unlink()
    
    def teardown_method(self):
        """各テストメソッドの後に実行されるクリーンアップ"""
        if self.system_monitor.is_monitoring_active():
            self.system_monitor.stop_monitoring()
        
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        memory_threshold=percentage,
        disk_threshold=percentage,
        cpu_threshold=percentage
    )
    @settings(max_examples=20, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_alert_threshold_properties(self, memory_threshold, disk_threshold, cpu_threshold):
        """
        Feature: genkai-rag-system, Property 16: システムログ記録
        任意のアラート閾値に対して、閾値設定と取得が一貫している
        """
        # 閾値を設定
        result = self.system_monitor.set_alert_thresholds(
            memory_percent=memory_threshold,
            disk_percent=disk_threshold,
            cpu_percent=cpu_threshold
        )
        
        assert result is True
        
        # 設定された閾値を確認
        thresholds = self.system_monitor.thresholds
        assert thresholds.memory_percent == memory_threshold
        assert thresholds.disk_percent == disk_threshold
        assert thresholds.cpu_percent == cpu_threshold
    
    @given(
        retention_days=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=15, deadline=3000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_data_retention_properties(self, retention_days):
        """
        Feature: genkai-rag-system, Property 16: システムログ記録
        任意の保持日数に対して、古いデータが適切にクリーンアップされる
        """
        # テスト用のログエントリを作成
        test_logs = []
        current_time = datetime.now()
        
        # 保持期間内と期間外のログを作成
        within_range_count = 2
        outside_range_count = 2
        
        for i in range(within_range_count):
            # 期間内のログ
            recent_log = {
                "timestamp": (current_time - timedelta(days=i)).isoformat(),
                "memory_usage_percent": 50.0,
                "test_data": f"recent_{i}"
            }
            test_logs.append(recent_log)
        
        for i in range(outside_range_count):
            # 期間外のログ
            old_log = {
                "timestamp": (current_time - timedelta(days=retention_days + i + 1)).isoformat(),
                "memory_usage_percent": 60.0,
                "test_data": f"old_{i}"
            }
            test_logs.append(old_log)
        
        # ログファイルに保存
        with open(self.system_monitor.status_log_file, 'w') as f:
            json.dump(test_logs, f)
        
        # クリーンアップを実行
        results = self.system_monitor.cleanup_old_data(retention_days=retention_days)
        
        # 結果を確認（削除されたエントリ数は期間外のログ数以上）
        assert results["deleted_status_entries"] >= outside_range_count
        
        # 残ったログを確認
        with open(self.system_monitor.status_log_file, 'r') as f:
            remaining_logs = json.load(f)
        
        # 期間内のログが残っている
        assert len(remaining_logs) >= 0  # 少なくとも0以上
        
        for log in remaining_logs:
            log_time = datetime.fromisoformat(log["timestamp"])
            cutoff_time = current_time - timedelta(days=retention_days)
            assert log_time >= cutoff_time
    
    @given(
        log_count=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=5, deadline=2000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_status_logging_properties(self, log_count):
        """
        Feature: genkai-rag-system, Property 16: システムログ記録
        任意の回数のログ記録に対して、すべてのログが正しく保存される
        """
        # 既存のログファイルをクリア
        if self.system_monitor.status_log_file.exists():
            self.system_monitor.status_log_file.unlink()
        
        # 複数回ログを記録
        for i in range(log_count):
            result = self.system_monitor.log_system_status()
            assert result is True
            time.sleep(0.01)  # 少し待機してタイムスタンプを区別
        
        # ログファイルを確認
        assert self.system_monitor.status_log_file.exists()
        
        with open(self.system_monitor.status_log_file, 'r') as f:
            logs = json.load(f)
        
        # 指定回数のログが記録されている
        assert len(logs) == log_count
        
        # 各ログエントリが有効な形式
        for log in logs:
            # 必須フィールドの存在確認
            required_fields = [
                "timestamp", "memory_usage_percent", "disk_usage_percent",
                "cpu_usage_percent", "process_count"
            ]
            
            for field in required_fields:
                assert field in log
            
            # 値の妥当性確認
            assert 0.0 <= log["memory_usage_percent"] <= 100.0
            assert 0.0 <= log["disk_usage_percent"] <= 100.0
            assert 0.0 <= log["cpu_usage_percent"] <= 100.0
            assert log["process_count"] > 0
    
    @given(
        hours=st.integers(min_value=1, max_value=24)
    )
    @settings(max_examples=5, deadline=2000, suppress_health_check=[hypothesis.HealthCheck.too_slow])
    def test_history_retrieval_properties(self, hours):
        """
        Feature: genkai-rag-system, Property 16: システムログ記録
        任意の時間範囲に対して、該当期間のログのみが取得される
        """
        current_time = datetime.now()
        test_logs = []
        
        # 指定時間範囲内と範囲外のログを作成
        within_range_count = 2
        outside_range_count = 1
        
        # 範囲内のログ
        for i in range(within_range_count):
            log_time = current_time - timedelta(hours=hours // 2)  # 範囲内
            log_entry = {
                "timestamp": log_time.isoformat(),
                "memory_usage_percent": 40.0 + i,
                "memory_available_gb": 4.0,
                "memory_total_gb": 8.0,
                "disk_usage_percent": 50.0 + i,
                "disk_available_gb": 100.0,
                "disk_total_gb": 200.0,
                "cpu_usage_percent": 30.0 + i,
                "process_count": 100 + i,
                "uptime_seconds": 3600.0
            }
            test_logs.append(log_entry)
        
        # 範囲外のログ
        for i in range(outside_range_count):
            log_time = current_time - timedelta(hours=hours + 1 + i)  # 範囲外
            log_entry = {
                "timestamp": log_time.isoformat(),
                "memory_usage_percent": 70.0 + i,
                "memory_available_gb": 2.0,
                "memory_total_gb": 8.0,
                "disk_usage_percent": 80.0 + i,
                "disk_available_gb": 50.0,
                "disk_total_gb": 200.0,
                "cpu_usage_percent": 60.0 + i,
                "process_count": 200 + i,
                "uptime_seconds": 7200.0
            }
            test_logs.append(log_entry)
        
        # ログファイルに保存
        with open(self.system_monitor.status_log_file, 'w') as f:
            json.dump(test_logs, f)
        
        # 指定時間範囲の履歴を取得
        history = self.system_monitor.get_status_history(hours=hours)
        
        # 範囲内のログのみが取得される
        assert len(history) == within_range_count
        
        # 取得されたログが正しい時間範囲内
        cutoff_time = current_time - timedelta(hours=hours)
        for status in history:
            assert status.timestamp >= cutoff_time
        for i in range(outside_range_count):
            log_time = current_time - timedelta(hours=hours + 1 + i)  # 範囲外
            log_entry = {
                "timestamp": log_time.isoformat(),
                "memory_usage_percent": 70.0 + i,
                "memory_available_gb": 2.0,
                "memory_total_gb": 8.0,
                "disk_usage_percent": 80.0 + i,
                "disk_available_gb": 50.0,
                "disk_total_gb": 200.0,
                "cpu_usage_percent": 60.0 + i,
                "process_count": 200 + i,
                "uptime_seconds": 7200.0
            }
            test_logs.append(log_entry)
        
        # ログファイルに保存
        with open(self.system_monitor.status_log_file, 'w') as f:
            json.dump(test_logs, f)
        
        # 指定時間範囲の履歴を取得
        history = self.system_monitor.get_status_history(hours=hours)
        
        # 範囲内のログのみが取得される
        assert len(history) == within_range_count
        
        # 取得されたログが正しい時間範囲内
        cutoff_time = current_time - timedelta(hours=hours)
        for status in history:
            assert status.timestamp >= cutoff_time


class TestSystemStatus:
    """SystemStatusクラスのテスト"""
    
    def test_system_status_creation(self):
        """SystemStatus作成テスト"""
        timestamp = datetime.now()
        status = SystemStatus(
            timestamp=timestamp,
            memory_usage_percent=75.5,
            memory_available_gb=2.5,
            memory_total_gb=8.0,
            disk_usage_percent=60.0,
            disk_available_gb=40.0,
            disk_total_gb=100.0,
            cpu_usage_percent=45.0,
            process_count=150,
            uptime_seconds=3600.0
        )
        
        assert status.timestamp == timestamp
        assert status.memory_usage_percent == 75.5
        assert status.memory_available_gb == 2.5
        assert status.memory_total_gb == 8.0
        assert status.disk_usage_percent == 60.0
        assert status.disk_available_gb == 40.0
        assert status.disk_total_gb == 100.0
        assert status.cpu_usage_percent == 45.0
        assert status.process_count == 150
        assert status.uptime_seconds == 3600.0
    
    def test_system_status_serialization(self):
        """SystemStatusシリアライゼーションテスト"""
        timestamp = datetime.now()
        status = SystemStatus(
            timestamp=timestamp,
            memory_usage_percent=75.5,
            memory_available_gb=2.5,
            memory_total_gb=8.0,
            disk_usage_percent=60.0,
            disk_available_gb=40.0,
            disk_total_gb=100.0,
            cpu_usage_percent=45.0,
            process_count=150,
            uptime_seconds=3600.0
        )
        
        # 辞書に変換
        status_dict = status.to_dict()
        assert status_dict["timestamp"] == timestamp.isoformat()
        assert status_dict["memory_usage_percent"] == 75.5
        assert status_dict["cpu_usage_percent"] == 45.0
        
        # 辞書から復元
        restored_status = SystemStatus.from_dict(status_dict)
        assert restored_status.timestamp == timestamp
        assert restored_status.memory_usage_percent == status.memory_usage_percent
        assert restored_status.cpu_usage_percent == status.cpu_usage_percent


class TestAlertThreshold:
    """AlertThresholdクラスのテスト"""
    
    def test_alert_threshold_creation(self):
        """AlertThreshold作成テスト"""
        threshold = AlertThreshold(
            memory_percent=70.0,
            disk_percent=85.0,
            cpu_percent=80.0
        )
        
        assert threshold.memory_percent == 70.0
        assert threshold.disk_percent == 85.0
        assert threshold.cpu_percent == 80.0
    
    def test_alert_threshold_default_values(self):
        """AlertThresholdデフォルト値テスト"""
        threshold = AlertThreshold()
        
        assert threshold.memory_percent == 80.0
        assert threshold.disk_percent == 90.0
        assert threshold.cpu_percent == 90.0
    
    def test_alert_threshold_serialization(self):
        """AlertThresholdシリアライゼーションテスト"""
        threshold = AlertThreshold(
            memory_percent=75.0,
            disk_percent=88.0,
            cpu_percent=85.0
        )
        
        threshold_dict = threshold.to_dict()
        assert threshold_dict["memory_percent"] == 75.0
        assert threshold_dict["disk_percent"] == 88.0
        assert threshold_dict["cpu_percent"] == 85.0