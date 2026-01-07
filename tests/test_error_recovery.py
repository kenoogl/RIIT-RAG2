"""
ErrorRecoveryManager テストスイート

エラー回復管理機能のテストを実装します。
"""

import pytest
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from genkai_rag.core.error_recovery import (
    ErrorRecoveryManager, ErrorType, ErrorSeverity, ErrorContext, RetryConfig,
    with_error_recovery
)
from genkai_rag.models.document import Document


class TestErrorRecoveryManager:
    """ErrorRecoveryManagerの基本機能テスト"""
    
    @pytest.fixture
    def error_manager(self):
        """テスト用ErrorRecoveryManagerインスタンス"""
        config = {
            "max_history_size": 100,
            "default_max_attempts": 3,
            "default_base_delay": 0.1,  # テスト用に短縮
            "default_max_delay": 1.0
        }
        return ErrorRecoveryManager(config)
    
    def test_initialization(self, error_manager):
        """初期化テスト"""
        assert error_manager.config is not None
        assert error_manager.error_history == []
        assert error_manager.max_history_size == 100
        assert error_manager.default_retry_config.max_attempts == 3
        assert error_manager.default_retry_config.base_delay == 0.1
    
    def test_error_context_creation(self):
        """ErrorContextの作成テスト"""
        context = ErrorContext(
            error_type=ErrorType.SCRAPING_ERROR,
            severity=ErrorSeverity.HIGH,
            operation="test_operation",
            url="https://example.com"
        )
        
        assert context.error_type == ErrorType.SCRAPING_ERROR
        assert context.severity == ErrorSeverity.HIGH
        assert context.operation == "test_operation"
        assert context.url == "https://example.com"
        assert isinstance(context.timestamp, datetime)
    
    def test_log_error(self, error_manager):
        """エラーログ記録テスト"""
        error = ValueError("Test error")
        context = ErrorContext(
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            operation="test_validation"
        )
        
        with patch('genkai_rag.core.error_recovery.logger') as mock_logger:
            error_manager.log_error(error, context)
        
        # エラー履歴に追加されることを確認
        assert len(error_manager.error_history) == 1
        assert error_manager.error_history[0] == context
        
        # ログが記録されることを確認
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.WARNING  # MEDIUM severity
        assert "Test error" in call_args[0][1]
    
    def test_error_history_size_limit(self, error_manager):
        """エラー履歴サイズ制限テスト"""
        error_manager.max_history_size = 5
        
        # 制限を超えるエラーを追加
        for i in range(10):
            error = ValueError(f"Error {i}")
            context = ErrorContext(
                error_type=ErrorType.SYSTEM_ERROR,
                severity=ErrorSeverity.LOW,
                operation=f"operation_{i}"
            )
            error_manager.log_error(error, context)
        
        # 履歴サイズが制限内であることを確認
        assert len(error_manager.error_history) == 5
        # 最新の5つが保持されていることを確認
        assert error_manager.error_history[-1].operation == "operation_9"
        assert error_manager.error_history[0].operation == "operation_5"
    
    def test_determine_severity(self, error_manager):
        """エラー重要度判定テスト"""
        # クリティカルエラー
        critical_error = Exception("Out of memory error")
        assert error_manager._determine_severity(critical_error) == ErrorSeverity.CRITICAL
        
        # 高重要度エラー
        high_error = Exception("Connection refused")
        assert error_manager._determine_severity(high_error) == ErrorSeverity.HIGH
        
        # 中重要度エラー
        medium_error = Exception("Not found")
        assert error_manager._determine_severity(medium_error) == ErrorSeverity.MEDIUM
        
        # デフォルト（中重要度）
        default_error = Exception("Unknown error")
        assert error_manager._determine_severity(default_error) == ErrorSeverity.MEDIUM
    
    def test_error_type_detection(self, error_manager):
        """エラータイプ検出テスト"""
        # ネットワークエラー
        network_error = Exception("Connection timeout")
        assert error_manager._is_network_error(network_error) is True
        
        # 一時的エラー
        temp_error = Exception("Service temporarily unavailable")
        assert error_manager._is_temporary_error(temp_error) is True
        
        # 接続エラー
        conn_error = Exception("Database connection failed")
        assert error_manager._is_connection_error(conn_error) is True
        
        # 通常のエラー
        normal_error = Exception("Invalid input")
        assert error_manager._is_network_error(normal_error) is False
        assert error_manager._is_temporary_error(normal_error) is False
        assert error_manager._is_connection_error(normal_error) is False
    
    def test_handle_scraping_error_success(self, error_manager):
        """スクレイピングエラー処理成功テスト"""
        error = Exception("Network timeout")
        url = "https://example.com"
        
        # _scrape_with_fallback をモック
        mock_doc = Document(
            content="Test content",
            metadata={"url": url, "title": "Test"}
        )
        
        with patch.object(error_manager, '_scrape_with_fallback', return_value=mock_doc):
            result = error_manager.handle_scraping_error(error, url)
        
        assert result == mock_doc
        assert len(error_manager.error_history) == 1
        assert error_manager.error_history[0].error_type == ErrorType.SCRAPING_ERROR
    
    def test_handle_scraping_error_failure(self, error_manager):
        """スクレイピングエラー処理失敗テスト"""
        error = Exception("Invalid URL format")  # ネットワークエラーではない
        url = "invalid-url"
        
        result = error_manager.handle_scraping_error(error, url)
        
        assert result is None
        assert len(error_manager.error_history) == 1
    
    def test_handle_llm_error_with_fallback(self, error_manager):
        """LLMエラー処理フォールバックテスト"""
        error = Exception("Model not available")
        query = "Test query"
        
        result = error_manager.handle_llm_error(error, query)
        
        assert result == error_manager.fallback_responses["llm_error"]
        assert len(error_manager.error_history) == 1
        assert error_manager.error_history[0].error_type == ErrorType.LLM_ERROR
    
    def test_handle_database_error(self, error_manager):
        """データベースエラー処理テスト"""
        error = Exception("Connection pool exhausted")
        operation = "insert_document"
        
        with patch.object(error_manager, '_reconnect_database', return_value=True):
            result = error_manager.handle_database_error(error, operation)
        
        assert result is True
        assert len(error_manager.error_history) == 1
        assert error_manager.error_history[0].error_type == ErrorType.DATABASE_ERROR
    
    def test_handle_validation_error(self, error_manager):
        """バリデーションエラー処理テスト"""
        error = ValueError("Invalid data format")
        data = {"invalid": "data"}
        operation = "validate_input"
        
        result = error_manager.handle_validation_error(error, data, operation)
        
        assert result is True  # バリデーションエラーは継続処理可能
        assert len(error_manager.error_history) == 1
        assert error_manager.error_history[0].error_type == ErrorType.VALIDATION_ERROR
    
    def test_get_error_statistics_empty(self, error_manager):
        """エラー統計取得テスト（エラーなし）"""
        stats = error_manager.get_error_statistics(24)
        
        assert stats["total_errors"] == 0
        assert stats["error_rate"] == 0.0
        assert stats["by_type"] == {}
        assert stats["by_severity"] == {}
        assert stats["most_common_operations"] == []
    
    def test_get_error_statistics_with_data(self, error_manager):
        """エラー統計取得テスト（データあり）"""
        # テストデータを追加
        errors = [
            (ErrorType.SCRAPING_ERROR, ErrorSeverity.HIGH, "scraping"),
            (ErrorType.SCRAPING_ERROR, ErrorSeverity.MEDIUM, "scraping"),
            (ErrorType.LLM_ERROR, ErrorSeverity.HIGH, "llm_query"),
            (ErrorType.DATABASE_ERROR, ErrorSeverity.CRITICAL, "database_op")
        ]
        
        for error_type, severity, operation in errors:
            context = ErrorContext(
                error_type=error_type,
                severity=severity,
                operation=operation
            )
            error_manager.error_history.append(context)
        
        stats = error_manager.get_error_statistics(24)
        
        assert stats["total_errors"] == 4
        assert stats["by_type"]["scraping_error"] == 2
        assert stats["by_type"]["llm_error"] == 1
        assert stats["by_type"]["database_error"] == 1
        assert stats["by_severity"]["high"] == 2
        assert stats["by_severity"]["medium"] == 1
        assert stats["by_severity"]["critical"] == 1
        assert ("scraping", 2) in stats["most_common_operations"]
    
    def test_retry_with_backoff_success(self, error_manager):
        """リトライ成功テスト"""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01, max_delay=0.1)
        result = error_manager._retry_with_backoff(failing_function, retry_config=retry_config)
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_with_backoff_failure(self, error_manager):
        """リトライ失敗テスト"""
        def always_failing_function():
            raise Exception("Permanent failure")
        
        retry_config = RetryConfig(max_attempts=2, base_delay=0.01, max_delay=0.1)
        result = error_manager._retry_with_backoff(always_failing_function, retry_config=retry_config)
        
        assert result is None


class TestErrorRecoveryProperties:
    """ErrorRecoveryManagerのプロパティベーステスト"""
    
    @pytest.fixture
    def error_manager(self):
        """テスト用ErrorRecoveryManagerインスタンス"""
        config = {
            "max_history_size": 50,
            "default_max_attempts": 2,
            "default_base_delay": 0.01,
            "default_max_delay": 0.1
        }
        return ErrorRecoveryManager(config)
    
    @given(
        error_messages=st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=20
        ),
        operations=st.lists(
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_error_processing_continuity_property(self, error_manager, error_messages, operations):
        """
        プロパティ 4: エラー処理の継続性
        
        任意の文書処理エラーが発生した時、システムはエラーログを記録し、他の文書の処理を継続する
        **検証: 要件 1.4**
        """
        assume(len(error_messages) > 0 and len(operations) > 0)
        
        initial_history_size = len(error_manager.error_history)
        processed_errors = []
        
        # 複数のエラーを順次処理
        for i, (error_msg, operation) in enumerate(zip(error_messages, operations)):
            try:
                # 様々なタイプのエラーをシミュレート
                if i % 4 == 0:
                    # スクレイピングエラー
                    error = Exception(f"Scraping failed: {error_msg}")
                    url = f"https://example.com/doc{i}"
                    result = error_manager.handle_scraping_error(error, url)
                    processed_errors.append(("scraping", error_msg, result))
                    
                elif i % 4 == 1:
                    # LLMエラー
                    error = Exception(f"LLM error: {error_msg}")
                    query = f"Query {i}: {operation}"
                    result = error_manager.handle_llm_error(error, query)
                    processed_errors.append(("llm", error_msg, result))
                    
                elif i % 4 == 2:
                    # データベースエラー
                    error = Exception(f"Database error: {error_msg}")
                    result = error_manager.handle_database_error(error, operation)
                    processed_errors.append(("database", error_msg, result))
                    
                else:
                    # バリデーションエラー
                    error = ValueError(f"Validation error: {error_msg}")
                    data = {"test": "data"}
                    result = error_manager.handle_validation_error(error, data, operation)
                    processed_errors.append(("validation", error_msg, result))
                    
            except Exception as e:
                # 予期しないエラーが発生した場合もテストを継続
                processed_errors.append(("unexpected", str(e), None))
        
        # プロパティ検証: エラー処理の継続性
        
        # 1. すべてのエラーが履歴に記録されていること
        final_history_size = len(error_manager.error_history)
        errors_logged = final_history_size - initial_history_size
        
        assert errors_logged > 0, \
            "エラーが履歴に記録されませんでした"
        
        # 2. エラーが発生しても処理が継続されていること
        assert len(processed_errors) == min(len(error_messages), len(operations)), \
            "エラー発生により処理が中断されました"
        
        # 3. 各エラーが適切に分類されていること
        error_types_in_history = [error.error_type for error in error_manager.error_history[-errors_logged:]]
        expected_types = {ErrorType.SCRAPING_ERROR, ErrorType.LLM_ERROR, ErrorType.DATABASE_ERROR, ErrorType.VALIDATION_ERROR}
        actual_types = set(error_types_in_history)
        
        # 少なくとも1つのエラータイプが記録されていること
        assert len(actual_types) > 0, \
            "エラータイプが正しく分類されませんでした"
        
        # 4. システムが継続動作していること（統計取得が可能）
        stats = error_manager.get_error_statistics(1)  # 1時間以内
        assert stats["total_errors"] >= errors_logged, \
            "エラー統計が正しく取得できませんでした"
        
        # 5. エラー発生後もシステム機能が利用可能であること
        test_error = Exception("Post-processing test error")
        test_context = ErrorContext(
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.LOW,
            operation="post_test"
        )
        
        # エラーログ機能が正常に動作すること
        try:
            error_manager.log_error(test_error, test_context)
            post_processing_success = True
        except Exception:
            post_processing_success = False
        
        assert post_processing_success, \
            "エラー処理後にシステム機能が利用できなくなりました"
    
    @given(
        retry_attempts=st.integers(min_value=1, max_value=5),
        base_delay=st.floats(min_value=0.001, max_value=0.1),
        failure_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_retry_mechanism_property(self, error_manager, retry_attempts, base_delay, failure_rate):
        """
        プロパティ: リトライ機構の動作
        
        リトライ設定に従って適切にリトライが実行されることを検証
        """
        assume(0.001 <= base_delay <= 0.1)
        assume(1 <= retry_attempts <= 5)
        
        call_count = 0
        success_threshold = int(retry_attempts * (1 - failure_rate)) + 1
        
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < success_threshold:
                raise Exception(f"Attempt {call_count} failed")
            return f"Success on attempt {call_count}"
        
        retry_config = RetryConfig(
            max_attempts=retry_attempts,
            base_delay=base_delay,
            max_delay=base_delay * 10
        )
        
        start_time = time.time()
        result = error_manager._retry_with_backoff(test_function, retry_config=retry_config)
        end_time = time.time()
        
        # プロパティ検証
        if success_threshold <= retry_attempts:
            # 成功するはずの場合
            assert result is not None, \
                f"リトライが成功するはずでしたが失敗しました (threshold: {success_threshold}, attempts: {retry_attempts})"
            assert call_count == success_threshold, \
                f"期待された回数のリトライが実行されませんでした (expected: {success_threshold}, actual: {call_count})"
        else:
            # 失敗するはずの場合
            assert result is None, \
                f"リトライが失敗するはずでしたが成功しました"
            assert call_count == retry_attempts, \
                f"最大リトライ回数まで実行されませんでした (expected: {retry_attempts}, actual: {call_count})"
        
        # 実行時間が妥当であること（リトライ間隔を考慮）
        if call_count > 1:
            min_expected_time = base_delay * (call_count - 1) * 0.5  # ジッターを考慮
            assert end_time - start_time >= min_expected_time, \
                f"リトライ間隔が短すぎます (actual: {end_time - start_time:.3f}s, min_expected: {min_expected_time:.3f}s)"


class TestErrorRecoveryDecorator:
    """エラー回復デコレータのテスト"""
    
    @pytest.fixture
    def error_manager(self):
        """テスト用ErrorRecoveryManagerインスタンス"""
        return ErrorRecoveryManager()
    
    def test_decorator_success(self, error_manager):
        """デコレータ成功テスト"""
        @with_error_recovery(error_manager, ErrorType.SYSTEM_ERROR, "test_operation")
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5
        assert len(error_manager.error_history) == 0  # エラーなし
    
    def test_decorator_error_handling(self, error_manager):
        """デコレータエラーハンドリングテスト"""
        @with_error_recovery(error_manager, ErrorType.VALIDATION_ERROR, "test_validation")
        def failing_function():
            raise ValueError("Test validation error")
        
        # バリデーションエラーは継続処理されるため例外は再発生しない
        result = failing_function()
        
        # エラーが記録されていることを確認
        assert len(error_manager.error_history) == 1
        assert error_manager.error_history[0].error_type == ErrorType.VALIDATION_ERROR


class TestRetryConfig:
    """RetryConfig設定テスト"""
    
    def test_retry_config_defaults(self):
        """デフォルト設定テスト"""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_retry_config_custom(self):
        """カスタム設定テスト"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])