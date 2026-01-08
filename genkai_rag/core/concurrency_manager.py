"""
ConcurrencyManager: 同時アクセス処理管理

このモジュールは、同時アクセス処理の最適化、負荷分散、
キューイング機能を提供します。
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """リクエストメトリクス"""
    request_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None
    queue_time: Optional[float] = None
    status: str = "pending"  # pending, processing, completed, failed
    error: Optional[str] = None


@dataclass
class ConcurrencyConfig:
    """同時アクセス設定"""
    max_concurrent_requests: int = 10
    max_queue_size: int = 100
    request_timeout: float = 30.0
    rate_limit_per_minute: int = 60
    enable_request_queuing: bool = True
    enable_rate_limiting: bool = True
    connection_pool_size: int = 20
    connection_pool_timeout: float = 5.0


@dataclass
class QueuedRequest:
    """キューイングされたリクエスト"""
    request_id: str
    handler: Callable[..., Awaitable[Any]]
    args: tuple
    kwargs: dict
    future: asyncio.Future
    queued_at: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 0が最高優先度


class RateLimiter:
    """レート制限器"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        レート制限器を初期化
        
        Args:
            max_requests: 時間窓内での最大リクエスト数
            time_window: 時間窓（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = threading.RLock()
    
    def is_allowed(self, client_id: str = "default") -> bool:
        """
        リクエストが許可されるかチェック
        
        Args:
            client_id: クライアントID
            
        Returns:
            許可される場合True
        """
        with self._lock:
            now = datetime.now()
            
            # 古いリクエストを削除
            while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
                self.requests.popleft()
            
            # リクエスト数をチェック
            if len(self.requests) >= self.max_requests:
                return False
            
            # 新しいリクエストを記録
            self.requests.append(now)
            return True
    
    def get_remaining_requests(self) -> int:
        """残りリクエスト数を取得"""
        with self._lock:
            now = datetime.now()
            
            # 古いリクエストを削除
            while self.requests and self.requests[0] < now - timedelta(seconds=self.time_window):
                self.requests.popleft()
            
            return max(0, self.max_requests - len(self.requests))


class ConnectionPool:
    """コネクションプール"""
    
    def __init__(self, pool_size: int = 20, timeout: float = 5.0):
        """
        コネクションプールを初期化
        
        Args:
            pool_size: プールサイズ
            timeout: タイムアウト（秒）
        """
        self.pool_size = pool_size
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(pool_size)
        self.active_connections = 0
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def acquire_connection(self):
        """コネクションを取得"""
        async with self.semaphore:
            async with self._lock:
                self.active_connections += 1
            
            try:
                yield
            finally:
                async with self._lock:
                    self.active_connections -= 1
    
    def get_pool_status(self) -> Dict[str, Any]:
        """プール状態を取得"""
        return {
            "pool_size": self.pool_size,
            "active_connections": self.active_connections,
            "available_connections": self.pool_size - self.active_connections
        }


class ConcurrencyManager:
    """
    同時アクセス処理管理クラス
    
    機能:
    - 同時実行数制限
    - リクエストキューイング
    - レート制限
    - コネクションプール管理
    - メトリクス収集
    """
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """
        ConcurrencyManagerを初期化
        
        Args:
            config: 同時アクセス設定
        """
        self.config = config or ConcurrencyConfig()
        
        # 同時実行制御
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # リクエストキュー
        self.request_queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        
        # レート制限器
        self.rate_limiter = RateLimiter(
            max_requests=self.config.rate_limit_per_minute,
            time_window=60
        ) if self.config.enable_rate_limiting else None
        
        # コネクションプール
        self.connection_pool = ConnectionPool(
            pool_size=self.config.connection_pool_size,
            timeout=self.config.connection_pool_timeout
        )
        
        # メトリクス
        self.metrics: Dict[str, RequestMetrics] = {}
        self.metrics_lock = threading.RLock()
        
        # ワーカータスク
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info(f"ConcurrencyManager initialized with config: {self.config}")
    
    async def start(self) -> None:
        """同時アクセス管理を開始"""
        if self.is_running:
            logger.warning("ConcurrencyManager is already running")
            return
        
        self.is_running = True
        
        # ワーカータスクを開始
        if self.config.enable_request_queuing:
            for i in range(self.config.max_concurrent_requests):
                task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                self.worker_tasks.append(task)
        
        logger.info("ConcurrencyManager started")
    
    async def stop(self) -> None:
        """同時アクセス管理を停止"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # ワーカータスクを停止
        for task in self.worker_tasks:
            task.cancel()
        
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        
        logger.info("ConcurrencyManager stopped")
    
    async def execute_with_concurrency_control(
        self,
        handler: Callable[..., Awaitable[Any]],
        *args,
        request_id: Optional[str] = None,
        priority: int = 0,
        client_id: str = "default",
        **kwargs
    ) -> Any:
        """
        同時アクセス制御付きでハンドラーを実行
        
        Args:
            handler: 実行するハンドラー
            *args: ハンドラーの引数
            request_id: リクエストID
            priority: 優先度（0が最高）
            client_id: クライアントID
            **kwargs: ハンドラーのキーワード引数
            
        Returns:
            ハンドラーの実行結果
            
        Raises:
            asyncio.TimeoutError: タイムアウト時
            Exception: レート制限やキュー満杯時
        """
        if not request_id:
            request_id = f"req-{int(time.time() * 1000000)}"
        
        # レート制限チェック
        if self.rate_limiter and not self.rate_limiter.is_allowed(client_id):
            raise Exception(f"Rate limit exceeded for client {client_id}")
        
        # メトリクス記録開始
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=datetime.now()
        )
        
        with self.metrics_lock:
            self.metrics[request_id] = metrics
        
        try:
            if self.config.enable_request_queuing:
                # キューイング方式
                return await self._execute_with_queue(
                    handler, args, kwargs, request_id, priority, metrics
                )
            else:
                # 直接実行方式
                return await self._execute_direct(
                    handler, args, kwargs, request_id, metrics
                )
        
        except Exception as e:
            metrics.status = "failed"
            metrics.error = str(e)
            raise
        
        finally:
            metrics.end_time = datetime.now()
            if metrics.start_time and metrics.end_time:
                metrics.processing_time = (metrics.end_time - metrics.start_time).total_seconds()
    
    async def _execute_with_queue(
        self,
        handler: Callable[..., Awaitable[Any]],
        args: tuple,
        kwargs: dict,
        request_id: str,
        priority: int,
        metrics: RequestMetrics
    ) -> Any:
        """キューイング方式でハンドラーを実行"""
        future = asyncio.Future()
        
        queued_request = QueuedRequest(
            request_id=request_id,
            handler=handler,
            args=args,
            kwargs=kwargs,
            future=future,
            priority=priority
        )
        
        try:
            # キューに追加
            await asyncio.wait_for(
                self.request_queue.put(queued_request),
                timeout=1.0
            )
            
            # 結果を待機
            result = await asyncio.wait_for(
                future,
                timeout=self.config.request_timeout
            )
            
            metrics.status = "completed"
            return result
            
        except asyncio.TimeoutError as e:
            metrics.status = "failed"
            if future.done():
                # ハンドラー実行中のタイムアウト
                metrics.error = "timeout"
            else:
                # キュー追加のタイムアウト
                metrics.error = "queue_timeout"
            raise
        except Exception as e:
            if "queue is full" in str(e).lower():
                metrics.status = "failed"
                metrics.error = "queue_full"
            raise
    
    async def _execute_direct(
        self,
        handler: Callable[..., Awaitable[Any]],
        args: tuple,
        kwargs: dict,
        request_id: str,
        metrics: RequestMetrics
    ) -> Any:
        """直接実行方式でハンドラーを実行"""
        async with self.semaphore:
            async with self.connection_pool.acquire_connection():
                metrics.status = "processing"
                
                result = await asyncio.wait_for(
                    handler(*args, **kwargs),
                    timeout=self.config.request_timeout
                )
                
                metrics.status = "completed"
                return result
    
    async def _worker_loop(self, worker_id: str) -> None:
        """ワーカーループ"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # キューからリクエストを取得
                queued_request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=1.0
                )
                
                # キュー時間を計算
                queue_time = (datetime.now() - queued_request.queued_at).total_seconds()
                
                with self.metrics_lock:
                    if queued_request.request_id in self.metrics:
                        self.metrics[queued_request.request_id].queue_time = queue_time
                
                # セマフォとコネクションプールで制御
                async with self.semaphore:
                    async with self.connection_pool.acquire_connection():
                        try:
                            # メトリクス更新
                            with self.metrics_lock:
                                if queued_request.request_id in self.metrics:
                                    self.metrics[queued_request.request_id].status = "processing"
                            
                            # ハンドラーを実行
                            result = await asyncio.wait_for(
                                queued_request.handler(*queued_request.args, **queued_request.kwargs),
                                timeout=self.config.request_timeout
                            )
                            
                            # 結果を設定
                            if not queued_request.future.done():
                                queued_request.future.set_result(result)
                        
                        except asyncio.TimeoutError as e:
                            # タイムアウトエラーを設定
                            with self.metrics_lock:
                                if queued_request.request_id in self.metrics:
                                    self.metrics[queued_request.request_id].error = "timeout"
                            
                            if not queued_request.future.done():
                                queued_request.future.set_exception(e)
                        
                        except Exception as e:
                            # その他のエラーを設定
                            with self.metrics_lock:
                                if queued_request.request_id in self.metrics:
                                    self.metrics[queued_request.request_id].error = str(e)
                            
                            if not queued_request.future.done():
                                queued_request.future.set_exception(e)
                        
                        finally:
                            # キューのタスク完了を通知
                            self.request_queue.task_done()
            
            except asyncio.TimeoutError:
                # タイムアウトは正常（キューが空の場合）
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    def get_metrics(self, hours: int = 1) -> Dict[str, Any]:
        """メトリクスを取得"""
        with self.metrics_lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_metrics = [
                m for m in self.metrics.values()
                if m.start_time >= cutoff_time
            ]
            
            if not recent_metrics:
                return {
                    "total_requests": 0,
                    "completed_requests": 0,
                    "failed_requests": 0,
                    "average_processing_time": 0.0,
                    "average_queue_time": 0.0,
                    "success_rate": 0.0
                }
            
            completed = [m for m in recent_metrics if m.status == "completed"]
            failed = [m for m in recent_metrics if m.status == "failed"]
            
            # 完了したリクエストの処理時間のみを使用
            processing_times = [m.processing_time for m in completed if m.processing_time is not None]
            queue_times = [m.queue_time for m in recent_metrics if m.queue_time is not None]
            
            return {
                "total_requests": len(recent_metrics),
                "completed_requests": len(completed),
                "failed_requests": len(failed),
                "pending_requests": len([m for m in recent_metrics if m.status == "pending"]),
                "processing_requests": len([m for m in recent_metrics if m.status == "processing"]),
                "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
                "average_queue_time": sum(queue_times) / len(queue_times) if queue_times else 0.0,
                "success_rate": len(completed) / len(recent_metrics) if recent_metrics else 0.0,
                "queue_size": self.request_queue.qsize(),
                "active_connections": self.connection_pool.active_connections,
                "rate_limit_remaining": self.rate_limiter.get_remaining_requests() if self.rate_limiter else None
            }
    
    def cleanup_old_metrics(self, hours: int = 24) -> int:
        """古いメトリクスをクリーンアップ"""
        with self.metrics_lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            old_keys = [
                key for key, metrics in self.metrics.items()
                if metrics.start_time < cutoff_time
            ]
            
            for key in old_keys:
                del self.metrics[key]
            
            logger.info(f"Cleaned up {len(old_keys)} old metrics")
            return len(old_keys)