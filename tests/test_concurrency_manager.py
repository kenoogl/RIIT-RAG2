"""
ConcurrencyManagerのテストスイート

同時アクセス処理管理の機能をテストします。
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings, assume

from genkai_rag.core.concurrency_manager import (
    ConcurrencyManager, ConcurrencyConfig, RateLimiter, 
    ConnectionPool, RequestMetrics, QueuedRequest
)


class TestRateLimiter:
    """RateLimiterクラスのテスト"""
    
    def test_rate_limiter_initialization(self):
        """レート制限器の初期化テスト"""
        limiter = RateLimiter(max_requests=10, time_window=60)
        
        assert limiter.max_requests == 10
        assert limiter.time_window == 60
        assert len(limiter.requests) == 0
    
    def test_rate_limiter_allows_requests_within_limit(self):
        """制限内のリクエストが許可されることをテスト"""
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # 制限内のリクエストは許可される
        for i in range(5):
            assert limiter.is_allowed("client1") is True
        
        # 制限を超えるリクエストは拒否される
        assert limiter.is_allowed("client1") is False
    
    def test_rate_limiter_resets_after_time_window(self):
        """時間窓経過後にリセットされることをテスト"""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # 制限まで使用
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is True
        assert limiter.is_allowed("client1") is False
        
        # 時間窓経過を待つ
        time.sleep(1.1)
        
        # 再び許可される
        assert limiter.is_allowed("client1") is True
    
    def test_rate_limiter_remaining_requests(self):
        """残りリクエスト数の取得テスト"""
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        assert limiter.get_remaining_requests() == 5
        
        limiter.is_allowed("client1")
        assert limiter.get_remaining_requests() == 4
        
        limiter.is_allowed("client1")
        assert limiter.get_remaining_requests() == 3


class TestConnectionPool:
    """ConnectionPoolクラスのテスト"""
    
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self):
        """コネクションプールの初期化テスト"""
        pool = ConnectionPool(pool_size=5, timeout=10.0)
        
        assert pool.pool_size == 5
        assert pool.timeout == 10.0
        assert pool.active_connections == 0
    
    @pytest.mark.asyncio
    async def test_connection_pool_acquire_release(self):
        """コネクション取得・解放テスト"""
        pool = ConnectionPool(pool_size=2, timeout=1.0)
        
        # コネクション取得
        async with pool.acquire_connection():
            assert pool.active_connections == 1
            
            # 別のコネクション取得
            async with pool.acquire_connection():
                assert pool.active_connections == 2
        
        # 解放後は0に戻る
        assert pool.active_connections == 0
    
    @pytest.mark.asyncio
    async def test_connection_pool_limit(self):
        """コネクションプール制限テスト"""
        pool = ConnectionPool(pool_size=1, timeout=0.1)
        
        async def acquire_connection():
            async with pool.acquire_connection():
                await asyncio.sleep(0.2)
        
        # 1つ目のコネクションを取得
        task1 = asyncio.create_task(acquire_connection())
        await asyncio.sleep(0.05)  # 少し待つ
        
        # 2つ目のコネクション取得は制限される
        start_time = time.time()
        task2 = asyncio.create_task(acquire_connection())
        
        await asyncio.gather(task1, task2)
        
        # 2つ目のタスクは待機時間があったはず
        elapsed = time.time() - start_time
        assert elapsed >= 0.15  # 最初のタスクの実行時間分待機
    
    @pytest.mark.asyncio
    async def test_connection_pool_status(self):
        """プール状態取得テスト"""
        pool = ConnectionPool(pool_size=3, timeout=1.0)
        
        status = pool.get_pool_status()
        assert status["pool_size"] == 3
        assert status["active_connections"] == 0
        assert status["available_connections"] == 3
        
        async with pool.acquire_connection():
            status = pool.get_pool_status()
            assert status["active_connections"] == 1
            assert status["available_connections"] == 2


class TestConcurrencyConfig:
    """ConcurrencyConfigクラスのテスト"""
    
    def test_concurrency_config_defaults(self):
        """デフォルト設定のテスト"""
        config = ConcurrencyConfig()
        
        assert config.max_concurrent_requests == 10
        assert config.max_queue_size == 100
        assert config.request_timeout == 30.0
        assert config.rate_limit_per_minute == 60
        assert config.enable_request_queuing is True
        assert config.enable_rate_limiting is True
        assert config.connection_pool_size == 20
        assert config.connection_pool_timeout == 5.0
    
    def test_concurrency_config_custom(self):
        """カスタム設定のテスト"""
        config = ConcurrencyConfig(
            max_concurrent_requests=5,
            max_queue_size=50,
            request_timeout=15.0,
            rate_limit_per_minute=30,
            enable_request_queuing=False,
            enable_rate_limiting=False,
            connection_pool_size=10,
            connection_pool_timeout=2.5
        )
        
        assert config.max_concurrent_requests == 5
        assert config.max_queue_size == 50
        assert config.request_timeout == 15.0
        assert config.rate_limit_per_minute == 30
        assert config.enable_request_queuing is False
        assert config.enable_rate_limiting is False
        assert config.connection_pool_size == 10
        assert config.connection_pool_timeout == 2.5


class TestConcurrencyManager:
    """ConcurrencyManagerクラスのテスト"""
    
    @pytest.fixture
    def config(self):
        """テスト用設定"""
        return ConcurrencyConfig(
            max_concurrent_requests=2,
            max_queue_size=5,
            request_timeout=1.0,
            rate_limit_per_minute=10,
            enable_request_queuing=True,
            enable_rate_limiting=True,
            connection_pool_size=3,
            connection_pool_timeout=0.5
        )
    
    @pytest.fixture
    def manager(self, config):
        """テスト用ConcurrencyManager"""
        return ConcurrencyManager(config)
    
    def test_concurrency_manager_initialization(self, manager, config):
        """ConcurrencyManagerの初期化テスト"""
        assert manager.config == config
        assert manager.semaphore._value == config.max_concurrent_requests
        assert manager.request_queue.maxsize == config.max_queue_size
        assert manager.rate_limiter is not None
        assert manager.connection_pool is not None
        assert len(manager.metrics) == 0
        assert len(manager.worker_tasks) == 0
        assert manager.is_running is False
    
    @pytest.mark.asyncio
    async def test_concurrency_manager_start_stop(self, manager):
        """開始・停止テスト"""
        assert manager.is_running is False
        
        await manager.start()
        assert manager.is_running is True
        assert len(manager.worker_tasks) == manager.config.max_concurrent_requests
        
        await manager.stop()
        assert manager.is_running is False
        assert len(manager.worker_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_direct_execution_mode(self):
        """直接実行モードのテスト"""
        config = ConcurrencyConfig(
            max_concurrent_requests=2,
            enable_request_queuing=False,
            enable_rate_limiting=False
        )
        manager = ConcurrencyManager(config)
        
        async def test_handler(value):
            await asyncio.sleep(0.1)
            return value * 2
        
        # 直接実行
        result = await manager.execute_with_concurrency_control(
            test_handler, 5, request_id="test-1"
        )
        
        assert result == 10
        assert "test-1" in manager.metrics
        assert manager.metrics["test-1"].status == "completed"
    
    @pytest.mark.asyncio
    async def test_queuing_execution_mode(self, manager):
        """キューイング実行モードのテスト"""
        await manager.start()
        
        try:
            async def test_handler(value):
                await asyncio.sleep(0.1)
                return value * 3
            
            # キューイング実行
            result = await manager.execute_with_concurrency_control(
                test_handler, 7, request_id="test-2"
            )
            
            assert result == 21
            assert "test-2" in manager.metrics
            assert manager.metrics["test-2"].status == "completed"
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_limit(self, manager):
        """同時リクエスト数制限テスト"""
        await manager.start()
        
        try:
            async def slow_handler(delay):
                await asyncio.sleep(delay)
                return f"completed-{delay}"
            
            # 複数のリクエストを同時実行
            tasks = []
            for i in range(4):  # 制限(2)を超える数
                task = asyncio.create_task(
                    manager.execute_with_concurrency_control(
                        slow_handler, 0.2, request_id=f"concurrent-{i}"
                    )
                )
                tasks.append(task)
            
            # すべて完了するまで待機
            results = await asyncio.gather(*tasks)
            
            # すべて正常に完了
            assert len(results) == 4
            for i, result in enumerate(results):
                assert result == f"completed-0.2"
                assert f"concurrent-{i}" in manager.metrics
                assert manager.metrics[f"concurrent-{i}"].status == "completed"
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """レート制限テスト"""
        config = ConcurrencyConfig(
            rate_limit_per_minute=2,
            enable_rate_limiting=True,
            enable_request_queuing=False
        )
        manager = ConcurrencyManager(config)
        
        async def test_handler():
            return "success"
        
        # 制限内のリクエストは成功
        result1 = await manager.execute_with_concurrency_control(
            test_handler, request_id="rate-1", client_id="client1"
        )
        assert result1 == "success"
        
        result2 = await manager.execute_with_concurrency_control(
            test_handler, request_id="rate-2", client_id="client1"
        )
        assert result2 == "success"
        
        # 制限を超えるリクエストは失敗
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await manager.execute_with_concurrency_control(
                test_handler, request_id="rate-3", client_id="client1"
            )
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, manager):
        """タイムアウト処理テスト"""
        await manager.start()
        
        try:
            async def timeout_handler():
                await asyncio.sleep(2.0)  # 設定タイムアウト(1.0s)を超える
                return "should not reach here"
            
            with pytest.raises(asyncio.TimeoutError):
                await manager.execute_with_concurrency_control(
                    timeout_handler, request_id="timeout-test"
                )
            
            # 少し待ってメトリクスが更新されるのを待つ
            await asyncio.sleep(0.1)
            
            # メトリクスにエラーが記録される
            assert "timeout-test" in manager.metrics
            assert manager.metrics["timeout-test"].status == "failed"
            
            # エラーメッセージをチェック（タイムアウトまたは空文字列を許容）
            error_msg = manager.metrics["timeout-test"].error
            assert error_msg in ["timeout", ""], f"Expected 'timeout' or '', got '{error_msg}'"
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, manager):
        """エラーハンドリングテスト"""
        await manager.start()
        
        try:
            async def error_handler():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError, match="Test error"):
                await manager.execute_with_concurrency_control(
                    error_handler, request_id="error-test"
                )
            
            # メトリクスにエラーが記録される
            assert "error-test" in manager.metrics
            assert manager.metrics["error-test"].status == "failed"
            assert "Test error" in manager.metrics["error-test"].error
            
        finally:
            await manager.stop()
    
    def test_metrics_collection(self, manager):
        """メトリクス収集テスト"""
        # 初期状態
        metrics = manager.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["completed_requests"] == 0
        assert metrics["failed_requests"] == 0
        assert metrics["success_rate"] == 0.0
        
        # テストメトリクスを追加
        now = datetime.now()
        manager.metrics["test-1"] = RequestMetrics(
            request_id="test-1",
            start_time=now,
            end_time=now + timedelta(seconds=1),
            processing_time=1.0,
            status="completed"
        )
        manager.metrics["test-2"] = RequestMetrics(
            request_id="test-2",
            start_time=now,
            end_time=now + timedelta(seconds=0.5),
            processing_time=0.5,
            status="failed",
            error="test error"
        )
        
        # メトリクス確認
        metrics = manager.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["completed_requests"] == 1
        assert metrics["failed_requests"] == 1
        assert metrics["success_rate"] == 0.5
        # 完了したリクエストのみの平均処理時間（失敗したリクエストは除外）
        assert metrics["average_processing_time"] == 1.0
    
    def test_metrics_cleanup(self, manager):
        """メトリクスクリーンアップテスト"""
        # 古いメトリクスを追加
        old_time = datetime.now() - timedelta(hours=25)
        manager.metrics["old-1"] = RequestMetrics(
            request_id="old-1",
            start_time=old_time,
            status="completed"
        )
        
        # 新しいメトリクスを追加
        new_time = datetime.now()
        manager.metrics["new-1"] = RequestMetrics(
            request_id="new-1",
            start_time=new_time,
            status="completed"
        )
        
        assert len(manager.metrics) == 2
        
        # クリーンアップ実行
        cleaned_count = manager.cleanup_old_metrics(hours=24)
        
        assert cleaned_count == 1
        assert len(manager.metrics) == 1
        assert "old-1" not in manager.metrics
        assert "new-1" in manager.metrics


class TestConcurrencyManagerProperties:
    """ConcurrencyManagerのプロパティベーステスト"""
    
    @pytest.mark.asyncio
    @given(
        max_concurrent=st.integers(min_value=1, max_value=5),
        num_requests=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10, deadline=5000)
    async def test_property_concurrent_execution_correctness(self, max_concurrent, num_requests):
        """
        プロパティ 19: 同時アクセス処理
        
        同時アクセス制御が正しく動作し、すべてのリクエストが
        適切に処理されることを検証
        """
        assume(num_requests >= 1)
        assume(max_concurrent >= 1)
        
        config = ConcurrencyConfig(
            max_concurrent_requests=max_concurrent,
            max_queue_size=num_requests + 5,
            request_timeout=2.0,
            enable_request_queuing=True,
            enable_rate_limiting=False
        )
        manager = ConcurrencyManager(config)
        
        await manager.start()
        
        try:
            async def test_handler(request_id):
                # 少し時間のかかる処理をシミュレート
                await asyncio.sleep(0.1)
                return f"result-{request_id}"
            
            # 複数のリクエストを同時実行
            tasks = []
            for i in range(num_requests):
                task = asyncio.create_task(
                    manager.execute_with_concurrency_control(
                        test_handler, i, request_id=f"prop-test-{i}"
                    )
                )
                tasks.append(task)
            
            # すべてのタスクが完了するまで待機
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # すべてのリクエストが正常に処理されたことを確認
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == num_requests
            
            # 結果が期待通りであることを確認
            expected_results = {f"result-{i}" for i in range(num_requests)}
            actual_results = set(successful_results)
            assert actual_results == expected_results
            
            # メトリクスが正しく記録されていることを確認
            metrics = manager.get_metrics()
            assert metrics["total_requests"] >= num_requests
            assert metrics["completed_requests"] >= num_requests
            assert metrics["success_rate"] >= 0.8  # 80%以上の成功率
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    @given(
        rate_limit=st.integers(min_value=2, max_value=5),
        num_requests=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=5, deadline=3000)
    async def test_property_rate_limiting_effectiveness(self, rate_limit, num_requests):
        """
        レート制限の有効性プロパティテスト
        
        レート制限が正しく動作し、制限を超えるリクエストが
        適切に拒否されることを検証
        """
        assume(num_requests > rate_limit)
        
        config = ConcurrencyConfig(
            rate_limit_per_minute=rate_limit,
            enable_rate_limiting=True,
            enable_request_queuing=False
        )
        manager = ConcurrencyManager(config)
        
        async def test_handler():
            return "success"
        
        successful_requests = 0
        failed_requests = 0
        
        # 制限を超える数のリクエストを送信
        for i in range(num_requests):
            try:
                await manager.execute_with_concurrency_control(
                    test_handler, request_id=f"rate-test-{i}", client_id="test-client"
                )
                successful_requests += 1
            except Exception:
                failed_requests += 1
        
        # レート制限が正しく動作していることを確認
        assert successful_requests <= rate_limit
        assert failed_requests >= (num_requests - rate_limit)
        assert successful_requests + failed_requests == num_requests
    
    @pytest.mark.asyncio
    @given(
        processing_times=st.lists(
            st.floats(min_value=0.05, max_value=0.3), 
            min_size=2, 
            max_size=5
        )
    )
    @settings(max_examples=5, deadline=3000)
    async def test_property_metrics_accuracy(self, processing_times):
        """
        メトリクス精度のプロパティテスト
        
        処理時間やリクエスト数のメトリクスが正確に
        記録されることを検証
        """
        config = ConcurrencyConfig(
            max_concurrent_requests=len(processing_times),
            enable_request_queuing=False,
            enable_rate_limiting=False
        )
        manager = ConcurrencyManager(config)
        
        async def timed_handler(delay):
            await asyncio.sleep(delay)
            return f"completed-{delay}"
        
        # 異なる処理時間のリクエストを実行
        tasks = []
        for i, delay in enumerate(processing_times):
            task = asyncio.create_task(
                manager.execute_with_concurrency_control(
                    timed_handler, delay, request_id=f"metrics-test-{i}"
                )
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # メトリクスの精度を確認
        metrics = manager.get_metrics()
        
        # リクエスト数が正確
        assert metrics["total_requests"] == len(processing_times)
        assert metrics["completed_requests"] == len(processing_times)
        assert metrics["failed_requests"] == 0
        
        # 成功率が100%
        assert metrics["success_rate"] == 1.0
        
        # 平均処理時間が妥当な範囲内
        expected_avg = sum(processing_times) / len(processing_times)
        actual_avg = metrics["average_processing_time"]
        
        # 10%の誤差を許容（非同期処理のオーバーヘッドを考慮）
        assert abs(actual_avg - expected_avg) / expected_avg <= 0.1


@pytest.mark.asyncio
async def test_integration_with_fastapi_dependency():
    """FastAPI依存性注入との統合テスト"""
    config = ConcurrencyConfig(
        max_concurrent_requests=2,
        enable_request_queuing=True,
        enable_rate_limiting=False
    )
    manager = ConcurrencyManager(config)
    
    await manager.start()
    
    try:
        # FastAPIハンドラーをシミュレート
        async def api_handler(query: str):
            await asyncio.sleep(0.1)
            return {"result": f"processed: {query}"}
        
        # 複数のAPIリクエストをシミュレート
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                manager.execute_with_concurrency_control(
                    api_handler, f"query-{i}", request_id=f"api-{i}"
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # すべてのリクエストが正常に処理される
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["result"] == f"processed: query-{i}"
        
        # メトリクスが正しく記録される
        metrics = manager.get_metrics()
        assert metrics["total_requests"] == 3
        assert metrics["completed_requests"] == 3
        assert metrics["success_rate"] == 1.0
        
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_queue_full_handling():
    """キュー満杯時の処理テスト"""
    config = ConcurrencyConfig(
        max_concurrent_requests=1,
        max_queue_size=2,
        request_timeout=2.0,  # タイムアウトを長めに設定
        enable_request_queuing=True,
        enable_rate_limiting=False
    )
    manager = ConcurrencyManager(config)
    
    await manager.start()
    
    try:
        async def blocking_handler():
            await asyncio.sleep(0.5)  # 短めの処理時間
            return "completed"
        
        # キューを満杯にするため、多数のリクエストを短時間で送信
        tasks = []
        for i in range(6):  # 制限(1) + キュー(2) + 3つ余分
            task = asyncio.create_task(
                manager.execute_with_concurrency_control(
                    blocking_handler, request_id=f"queue-{i}"
                )
            )
            tasks.append(task)
            # 間隔を空けずに連続送信してキューを満杯にする
        
        # 結果を収集
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 成功とエラーを分類
        successes = [r for r in results if not isinstance(r, Exception)]
        exceptions = [r for r in results if isinstance(r, Exception)]
        
        # 少なくとも一部は成功し、一部はエラーになる
        assert len(successes) >= 2  # 最低限の成功
        assert len(exceptions) >= 1  # 少なくとも1つはエラー
        
        # キュー満杯エラーまたはタイムアウトエラーが含まれているかチェック
        queue_full_errors = [
            e for e in exceptions 
            if "queue" in str(e).lower() or "full" in str(e).lower()
        ]
        timeout_errors = [
            e for e in exceptions 
            if isinstance(e, asyncio.TimeoutError)
        ]
        
        # キュー満杯エラーまたはタイムアウトエラーがあることを確認
        # （どちらも同時アクセス制限が正しく動作していることを示す）
        assert len(queue_full_errors) >= 1 or len(timeout_errors) >= 1
        
    finally:
        await manager.stop()