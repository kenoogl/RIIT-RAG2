"""
SystemMonitorクラス

このモジュールは、システムリソースの監視とログ記録を行います。
メモリ使用量、ディスク使用量の監視、システム状態のログ記録、
古いデータのクリーンアップ機能を提供します。
"""

import psutil
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import threading
import time
from dataclasses import dataclass, asdict
import shutil
import asyncio
from functools import wraps
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ResponseTimeMetrics:
    """レスポンス時間メトリクスのデータクラス"""
    operation_type: str
    timestamp: datetime
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "operation_type": self.operation_type,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponseTimeMetrics':
        """辞書から復元"""
        return cls(
            operation_type=data["operation_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            response_time_ms=data["response_time_ms"],
            success=data["success"],
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )


@dataclass
class PerformanceStats:
    """パフォーマンス統計のデータクラス"""
    operation_type: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate_percent: float
    requests_per_minute: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class SystemStatus:
    """システム状態のデータクラス"""
    timestamp: datetime
    memory_usage_percent: float
    memory_available_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_available_gb: float
    disk_total_gb: float
    cpu_usage_percent: float
    process_count: int
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memory_usage_percent": self.memory_usage_percent,
            "memory_available_gb": self.memory_available_gb,
            "memory_total_gb": self.memory_total_gb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_available_gb": self.disk_available_gb,
            "disk_total_gb": self.disk_total_gb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "process_count": self.process_count,
            "uptime_seconds": self.uptime_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemStatus':
        """辞書から復元"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            memory_usage_percent=data["memory_usage_percent"],
            memory_available_gb=data["memory_available_gb"],
            memory_total_gb=data["memory_total_gb"],
            disk_usage_percent=data["disk_usage_percent"],
            disk_available_gb=data["disk_available_gb"],
            disk_total_gb=data["disk_total_gb"],
            cpu_usage_percent=data["cpu_usage_percent"],
            process_count=data["process_count"],
            uptime_seconds=data["uptime_seconds"]
        )


@dataclass
class AlertThreshold:
    """アラート閾値の設定"""
    memory_percent: float = 80.0
    disk_percent: float = 90.0
    cpu_percent: float = 90.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SystemMonitor:
    """
    システムリソース監視クラス
    
    メモリ、ディスク、CPU使用量の監視、ログ記録、アラート機能を提供します。
    バックグラウンドでの定期監視もサポートします。
    """
    
    def __init__(self, log_dir: str = "logs", data_dir: str = "data", 
                 monitoring_interval: int = 60, retention_days: int = 30):
        """
        SystemMonitorを初期化
        
        Args:
            log_dir: ログディレクトリ
            data_dir: データディレクトリ（監視対象）
            monitoring_interval: 監視間隔（秒）
            retention_days: ログ保持日数
        """
        self.log_dir = Path(log_dir)
        self.data_dir = Path(data_dir)
        self.monitoring_interval = monitoring_interval
        self.retention_days = retention_days
        
        # 監視ログファイル
        self.status_log_file = self.log_dir / "system_status.json"
        self.alert_log_file = self.log_dir / "system_alerts.json"
        self.performance_log_file = self.log_dir / "performance_metrics.json"
        
        # アラート閾値
        self.thresholds = AlertThreshold()
        
        # バックグラウンド監視用
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.RLock()
        
        # システム起動時刻
        self._boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        # レスポンス時間メトリクス用
        self._response_metrics: List[ResponseTimeMetrics] = []
        self._metrics_lock = threading.RLock()
        self._max_metrics_in_memory = 1000  # メモリ内に保持する最大メトリクス数
        
        # ディレクトリ作成
        self._ensure_directories()
        
        logger.info(f"SystemMonitor initialized with log_dir: {self.log_dir}")
    
    def _ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def check_memory_usage(self) -> float:
        """
        メモリ使用率を取得
        
        Returns:
            メモリ使用率（パーセント）
        """
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
            return 0.0
    
    def check_disk_usage(self, path: Optional[str] = None) -> float:
        """
        ディスク使用率を取得
        
        Args:
            path: チェック対象のパス（Noneの場合はdata_dir）
            
        Returns:
            ディスク使用率（パーセント）
        """
        try:
            target_path = Path(path) if path else self.data_dir
            
            # パスが存在しない場合は親ディレクトリをチェック
            while not target_path.exists() and target_path.parent != target_path:
                target_path = target_path.parent
            
            usage = shutil.disk_usage(target_path)
            total = usage.total
            used = usage.total - usage.free
            
            return (used / total) * 100 if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to check disk usage for {path}: {e}")
            return 0.0
    
    def check_cpu_usage(self, interval: float = 1.0) -> float:
        """
        CPU使用率を取得
        
        Args:
            interval: 測定間隔（秒）
            
        Returns:
            CPU使用率（パーセント）
        """
        try:
            return psutil.cpu_percent(interval=interval)
        except Exception as e:
            logger.error(f"Failed to check CPU usage: {e}")
            return 0.0
    
    def get_system_status(self) -> SystemStatus:
        """
        現在のシステム状態を取得
        
        Returns:
            SystemStatusオブジェクト
        """
        try:
            # メモリ情報
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            memory_usage_percent = memory.percent
            
            # ディスク情報
            disk_usage = shutil.disk_usage(self.data_dir)
            disk_total_gb = disk_usage.total / (1024**3)
            disk_available_gb = disk_usage.free / (1024**3)
            disk_usage_percent = ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100
            
            # CPU情報
            cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # プロセス情報
            process_count = len(psutil.pids())
            
            # アップタイム
            uptime_seconds = (datetime.now() - self._boot_time).total_seconds()
            
            return SystemStatus(
                timestamp=datetime.now(),
                memory_usage_percent=memory_usage_percent,
                memory_available_gb=memory_available_gb,
                memory_total_gb=memory_total_gb,
                disk_usage_percent=disk_usage_percent,
                disk_available_gb=disk_available_gb,
                disk_total_gb=disk_total_gb,
                cpu_usage_percent=cpu_usage_percent,
                process_count=process_count,
                uptime_seconds=uptime_seconds
            )
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            # エラー時はデフォルト値を返す
            return SystemStatus(
                timestamp=datetime.now(),
                memory_usage_percent=0.0,
                memory_available_gb=0.0,
                memory_total_gb=0.0,
                disk_usage_percent=0.0,
                disk_available_gb=0.0,
                disk_total_gb=0.0,
                cpu_usage_percent=0.0,
                process_count=0,
                uptime_seconds=0.0
            )
    
    def log_system_status(self) -> bool:
        """
        システム状態をログに記録
        
        Returns:
            ログ記録成功の場合True
        """
        try:
            status = self.get_system_status()
            
            # ログファイルに追記
            log_entry = status.to_dict()
            
            # 既存のログを読み込み
            logs = []
            if self.status_log_file.exists():
                with open(self.status_log_file, 'r', encoding='utf-8') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            
            # 新しいエントリを追加
            logs.append(log_entry)
            
            # ログサイズを制限（最新1000件まで）
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # ログを保存
            with open(self.status_log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            
            # アラートチェック
            self._check_alerts(status)
            
            logger.debug("System status logged successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log system status: {e}")
            return False
    
    def cleanup_old_data(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """
        古いデータをクリーンアップ
        
        Args:
            retention_days: 保持日数（Noneの場合はデフォルト値を使用）
            
        Returns:
            クリーンアップ結果の辞書
        """
        if retention_days is None:
            retention_days = self.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        results = {
            "deleted_log_files": 0,
            "deleted_status_entries": 0,
            "deleted_alert_entries": 0,
            "freed_space_mb": 0
        }
        
        try:
            # ログファイルのクリーンアップ
            results["deleted_log_files"] = self._cleanup_log_files(cutoff_date)
            
            # システム状態ログのクリーンアップ
            results["deleted_status_entries"] = self._cleanup_status_log(cutoff_date)
            
            # アラートログのクリーンアップ
            results["deleted_alert_entries"] = self._cleanup_alert_log(cutoff_date)
            
            logger.info(f"Cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return results
    
    def get_status_history(self, hours: int = 24) -> List[SystemStatus]:
        """
        指定時間内のシステム状態履歴を取得
        
        Args:
            hours: 取得する時間範囲（時間）
            
        Returns:
            SystemStatusのリスト
        """
        try:
            if not self.status_log_file.exists():
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with open(self.status_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # 指定時間内のログをフィルタ
            filtered_logs = []
            for log_entry in logs:
                try:
                    status = SystemStatus.from_dict(log_entry)
                    if status.timestamp >= cutoff_time:
                        filtered_logs.append(status)
                except (KeyError, ValueError):
                    # 無効なログエントリはスキップ
                    continue
            
            # 時刻順にソート
            filtered_logs.sort(key=lambda x: x.timestamp)
            return filtered_logs
            
        except Exception as e:
            logger.error(f"Failed to get status history: {e}")
            return []
    
    def get_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        指定時間内のアラート履歴を取得
        
        Args:
            hours: 取得する時間範囲（時間）
            
        Returns:
            アラート情報のリスト
        """
        try:
            if not self.alert_log_file.exists():
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with open(self.alert_log_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
            
            # 指定時間内のアラートをフィルタ
            filtered_alerts = []
            for alert in alerts:
                try:
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time >= cutoff_time:
                        filtered_alerts.append(alert)
                except (KeyError, ValueError):
                    continue
            
            # 時刻順にソート（新しい順）
            filtered_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
    
    def set_alert_thresholds(self, memory_percent: Optional[float] = None,
                           disk_percent: Optional[float] = None,
                           cpu_percent: Optional[float] = None) -> bool:
        """
        アラート閾値を設定
        
        Args:
            memory_percent: メモリ使用率の閾値
            disk_percent: ディスク使用率の閾値
            cpu_percent: CPU使用率の閾値
            
        Returns:
            設定成功の場合True
        """
        try:
            if memory_percent is not None:
                self.thresholds.memory_percent = memory_percent
            if disk_percent is not None:
                self.thresholds.disk_percent = disk_percent
            if cpu_percent is not None:
                self.thresholds.cpu_percent = cpu_percent
            
            logger.info(f"Alert thresholds updated: {self.thresholds.to_dict()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set alert thresholds: {e}")
            return False
    
    def start_monitoring(self) -> bool:
        """
        バックグラウンド監視を開始
        
        Returns:
            開始成功の場合True
        """
        with self._lock:
            if self._monitoring_active:
                logger.warning("Monitoring is already active")
                return False
            
            try:
                self._monitoring_active = True
                self._monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True
                )
                self._monitoring_thread.start()
                
                logger.info("Background monitoring started")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                self._monitoring_active = False
                return False
    
    def stop_monitoring(self) -> bool:
        """
        バックグラウンド監視を停止
        
        Returns:
            停止成功の場合True
        """
        with self._lock:
            if not self._monitoring_active:
                logger.warning("Monitoring is not active")
                return False
            
            try:
                self._monitoring_active = False
                
                if self._monitoring_thread and self._monitoring_thread.is_alive():
                    self._monitoring_thread.join(timeout=5.0)
                
                logger.info("Background monitoring stopped")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop monitoring: {e}")
                return False
    
    def is_monitoring_active(self) -> bool:
        """監視が有効かチェック"""
        return self._monitoring_active
    
    def record_response_time(self, operation_type: str, response_time_ms: float, 
                           success: bool = True, error_message: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        レスポンス時間を記録
        
        Args:
            operation_type: 操作タイプ（例: "query", "model_switch", "document_processing"）
            response_time_ms: レスポンス時間（ミリ秒）
            success: 操作が成功したかどうか
            error_message: エラーメッセージ（失敗時）
            metadata: 追加のメタデータ
        """
        try:
            with self._metrics_lock:
                # メトリクスを作成
                metric = ResponseTimeMetrics(
                    operation_type=operation_type,
                    timestamp=datetime.now(),
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=error_message,
                    metadata=metadata or {}
                )
                
                # メモリ内リストに追加
                self._response_metrics.append(metric)
                
                # メモリ内メトリクス数を制限
                if len(self._response_metrics) > self._max_metrics_in_memory:
                    # 古いメトリクスをファイルに保存してメモリから削除
                    self._flush_metrics_to_file()
                
                logger.debug(f"Recorded response time: {operation_type} = {response_time_ms:.2f}ms")
                
        except Exception as e:
            logger.error(f"Failed to record response time: {e}")
    
    def get_performance_stats(self, operation_type: Optional[str] = None, 
                            hours: int = 24) -> Dict[str, PerformanceStats]:
        """
        パフォーマンス統計を取得
        
        Args:
            operation_type: 特定の操作タイプ（Noneの場合は全タイプ）
            hours: 統計期間（時間）
            
        Returns:
            操作タイプ別のパフォーマンス統計
        """
        try:
            # 指定期間のメトリクスを取得
            metrics = self._get_metrics_in_timerange(hours)
            
            # 操作タイプでフィルタ
            if operation_type:
                metrics = [m for m in metrics if m.operation_type == operation_type]
            
            # 操作タイプ別に統計を計算
            stats_by_type = {}
            
            # 操作タイプごとにグループ化
            metrics_by_type = {}
            for metric in metrics:
                if metric.operation_type not in metrics_by_type:
                    metrics_by_type[metric.operation_type] = []
                metrics_by_type[metric.operation_type].append(metric)
            
            # 各操作タイプの統計を計算
            for op_type, type_metrics in metrics_by_type.items():
                stats_by_type[op_type] = self._calculate_performance_stats(op_type, type_metrics, hours)
            
            return stats_by_type
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {}
    
    def get_response_time_history(self, operation_type: Optional[str] = None,
                                hours: int = 24) -> List[ResponseTimeMetrics]:
        """
        レスポンス時間履歴を取得
        
        Args:
            operation_type: 特定の操作タイプ（Noneの場合は全タイプ）
            hours: 取得期間（時間）
            
        Returns:
            レスポンス時間メトリクスのリスト
        """
        try:
            # 指定期間のメトリクスを取得
            metrics = self._get_metrics_in_timerange(hours)
            
            # 操作タイプでフィルタ
            if operation_type:
                metrics = [m for m in metrics if m.operation_type == operation_type]
            
            # 時刻順にソート
            metrics.sort(key=lambda x: x.timestamp)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get response time history: {e}")
            return []
    
    def measure_response_time(self, operation_type: str, metadata: Optional[Dict[str, Any]] = None):
        """
        レスポンス時間測定用デコレータ
        
        Args:
            operation_type: 操作タイプ
            metadata: 追加のメタデータ
            
        Returns:
            デコレータ関数
        """
        def decorator(func: Callable):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    error_message = None
                    
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        error_message = str(e)
                        raise
                    finally:
                        response_time_ms = (time.time() - start_time) * 1000
                        self.record_response_time(
                            operation_type=operation_type,
                            response_time_ms=response_time_ms,
                            success=success,
                            error_message=error_message,
                            metadata=metadata
                        )
                
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    error_message = None
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        error_message = str(e)
                        raise
                    finally:
                        response_time_ms = (time.time() - start_time) * 1000
                        self.record_response_time(
                            operation_type=operation_type,
                            response_time_ms=response_time_ms,
                            success=success,
                            error_message=error_message,
                            metadata=metadata
                        )
                
                return sync_wrapper
        
        return decorator
    
    def clear_performance_metrics(self, operation_type: Optional[str] = None) -> int:
        """
        パフォーマンスメトリクスをクリア
        
        Args:
            operation_type: 特定の操作タイプ（Noneの場合は全タイプ）
            
        Returns:
            削除されたメトリクス数
        """
        try:
            with self._metrics_lock:
                if operation_type is None:
                    # 全メトリクスをクリア
                    cleared_count = len(self._response_metrics)
                    self._response_metrics.clear()
                else:
                    # 特定の操作タイプのみクリア
                    original_count = len(self._response_metrics)
                    self._response_metrics = [
                        m for m in self._response_metrics 
                        if m.operation_type != operation_type
                    ]
                    cleared_count = original_count - len(self._response_metrics)
                
                # ファイルからも削除（簡単のため全体を再書き込み）
                if operation_type is None and self.performance_log_file.exists():
                    self.performance_log_file.unlink()
                
                logger.info(f"Cleared {cleared_count} performance metrics")
                return cleared_count
                
        except Exception as e:
            logger.error(f"Failed to clear performance metrics: {e}")
            return 0
    
    def _monitoring_loop(self) -> None:
        """バックグラウンド監視のメインループ"""
        logger.info("Monitoring loop started")
        
        while self._monitoring_active:
            try:
                # システム状態をログに記録
                self.log_system_status()
                
                # 指定間隔で待機
                for _ in range(self.monitoring_interval):
                    if not self._monitoring_active:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # エラー時は短い間隔で再試行
        
        logger.info("Monitoring loop ended")
    
    def _check_alerts(self, status: SystemStatus) -> None:
        """アラート条件をチェックして必要に応じてアラートを記録"""
        alerts = []
        
        # メモリ使用率チェック
        if status.memory_usage_percent > self.thresholds.memory_percent:
            alerts.append({
                "type": "memory",
                "level": "warning",
                "message": f"High memory usage: {status.memory_usage_percent:.1f}%",
                "value": status.memory_usage_percent,
                "threshold": self.thresholds.memory_percent
            })
        
        # ディスク使用率チェック
        if status.disk_usage_percent > self.thresholds.disk_percent:
            alerts.append({
                "type": "disk",
                "level": "warning",
                "message": f"High disk usage: {status.disk_usage_percent:.1f}%",
                "value": status.disk_usage_percent,
                "threshold": self.thresholds.disk_percent
            })
        
        # CPU使用率チェック
        if status.cpu_usage_percent > self.thresholds.cpu_percent:
            alerts.append({
                "type": "cpu",
                "level": "warning",
                "message": f"High CPU usage: {status.cpu_usage_percent:.1f}%",
                "value": status.cpu_usage_percent,
                "threshold": self.thresholds.cpu_percent
            })
        
        # アラートを記録
        if alerts:
            self._record_alerts(alerts, status.timestamp)
    
    def _record_alerts(self, alerts: List[Dict[str, Any]], timestamp: datetime) -> None:
        """アラートをログに記録"""
        try:
            # 既存のアラートログを読み込み
            alert_logs = []
            if self.alert_log_file.exists():
                with open(self.alert_log_file, 'r', encoding='utf-8') as f:
                    try:
                        alert_logs = json.load(f)
                    except json.JSONDecodeError:
                        alert_logs = []
            
            # 新しいアラートを追加
            for alert in alerts:
                alert_entry = {
                    "timestamp": timestamp.isoformat(),
                    **alert
                }
                alert_logs.append(alert_entry)
                
                # ログにも出力
                logger.warning(f"ALERT: {alert['message']}")
            
            # ログサイズを制限（最新500件まで）
            if len(alert_logs) > 500:
                alert_logs = alert_logs[-500:]
            
            # アラートログを保存
            with open(self.alert_log_file, 'w', encoding='utf-8') as f:
                json.dump(alert_logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to record alerts: {e}")
    
    def _cleanup_log_files(self, cutoff_date: datetime) -> int:
        """古いログファイルを削除"""
        deleted_count = 0
        
        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.is_file():
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
                        
        except Exception as e:
            logger.error(f"Failed to cleanup log files: {e}")
        
        return deleted_count
    
    def _cleanup_status_log(self, cutoff_date: datetime) -> int:
        """古いシステム状態ログエントリを削除"""
        deleted_count = 0
        
        try:
            if not self.status_log_file.exists():
                return 0
            
            with open(self.status_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # 新しいログのみを保持
            filtered_logs = []
            for log_entry in logs:
                try:
                    log_time = datetime.fromisoformat(log_entry["timestamp"])
                    if log_time >= cutoff_date:
                        filtered_logs.append(log_entry)
                    else:
                        deleted_count += 1
                except (KeyError, ValueError):
                    deleted_count += 1  # 無効なエントリも削除
            
            # フィルタされたログを保存
            with open(self.status_log_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to cleanup status log: {e}")
        
        return deleted_count
    
    def _cleanup_alert_log(self, cutoff_date: datetime) -> int:
        """古いアラートログエントリを削除"""
        deleted_count = 0
        
        try:
            if not self.alert_log_file.exists():
                return 0
            
            with open(self.alert_log_file, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
            
            # 新しいアラートのみを保持
            filtered_alerts = []
            for alert in alerts:
                try:
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time >= cutoff_date:
                        filtered_alerts.append(alert)
                    else:
                        deleted_count += 1
                except (KeyError, ValueError):
                    deleted_count += 1  # 無効なエントリも削除
            
            # フィルタされたアラートを保存
            with open(self.alert_log_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_alerts, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to cleanup alert log: {e}")
        
        return deleted_count
    
    def _flush_metrics_to_file(self) -> None:
        """メモリ内のメトリクスをファイルに保存"""
        try:
            if not self._response_metrics:
                return
            
            # 既存のメトリクスを読み込み
            existing_metrics = []
            if self.performance_log_file.exists():
                with open(self.performance_log_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                        existing_metrics = existing_data if isinstance(existing_data, list) else []
                    except json.JSONDecodeError:
                        existing_metrics = []
            
            # 新しいメトリクスを追加
            new_metrics = [metric.to_dict() for metric in self._response_metrics]
            all_metrics = existing_metrics + new_metrics
            
            # ファイルサイズを制限（最新5000件まで）
            if len(all_metrics) > 5000:
                all_metrics = all_metrics[-5000:]
            
            # ファイルに保存
            with open(self.performance_log_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            
            # メモリ内メトリクスをクリア（半分だけ残す）
            keep_count = self._max_metrics_in_memory // 2
            self._response_metrics = self._response_metrics[-keep_count:]
            
        except Exception as e:
            logger.error(f"Failed to flush metrics to file: {e}")
    
    def _get_metrics_in_timerange(self, hours: int) -> List[ResponseTimeMetrics]:
        """指定時間範囲内のメトリクスを取得"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        all_metrics = []
        
        try:
            # メモリ内のメトリクスを追加
            with self._metrics_lock:
                for metric in self._response_metrics:
                    if metric.timestamp >= cutoff_time:
                        all_metrics.append(metric)
            
            # ファイルからメトリクスを読み込み
            if self.performance_log_file.exists():
                with open(self.performance_log_file, 'r', encoding='utf-8') as f:
                    try:
                        file_data = json.load(f)
                        for metric_dict in file_data:
                            try:
                                metric = ResponseTimeMetrics.from_dict(metric_dict)
                                if metric.timestamp >= cutoff_time:
                                    all_metrics.append(metric)
                            except (KeyError, ValueError):
                                continue  # 無効なメトリクスはスキップ
                    except json.JSONDecodeError:
                        pass  # ファイルが破損している場合はスキップ
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics in timerange: {e}")
            return all_metrics
    
    def _calculate_performance_stats(self, operation_type: str, 
                                   metrics: List[ResponseTimeMetrics], 
                                   hours: int) -> PerformanceStats:
        """パフォーマンス統計を計算"""
        if not metrics:
            return PerformanceStats(
                operation_type=operation_type,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                p50_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                error_rate_percent=0.0,
                requests_per_minute=0.0
            )
        
        # 基本統計
        total_requests = len(metrics)
        successful_requests = sum(1 for m in metrics if m.success)
        failed_requests = total_requests - successful_requests
        
        # レスポンス時間統計
        response_times = [m.response_time_ms for m in metrics]
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # パーセンタイル計算
        sorted_times = sorted(response_times)
        p50 = self._calculate_percentile(sorted_times, 50)
        p95 = self._calculate_percentile(sorted_times, 95)
        p99 = self._calculate_percentile(sorted_times, 99)
        
        # エラー率
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0.0
        
        # 分あたりのリクエスト数
        requests_per_minute = (total_requests / hours) * 60 if hours > 0 else 0.0
        
        return PerformanceStats(
            operation_type=operation_type,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            error_rate_percent=error_rate,
            requests_per_minute=requests_per_minute
        )
    
    def _calculate_percentile(self, sorted_values: List[float], percentile: int) -> float:
        """パーセンタイルを計算"""
        if not sorted_values:
            return 0.0
        
        if percentile <= 0:
            return sorted_values[0]
        if percentile >= 100:
            return sorted_values[-1]
        
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
        
        # 線形補間
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight