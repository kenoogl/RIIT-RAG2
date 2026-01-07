"""
エラー回復管理モジュール

このモジュールは、システム全体のエラーハンドリングと回復機能を提供します。
各種エラーの分類、回復戦略、フォールバック機能、リトライ機構を実装します。
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable, List, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from functools import wraps

from ..models.document import Document


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """エラータイプの分類"""
    SCRAPING_ERROR = "scraping_error"
    LLM_ERROR = "llm_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """エラーの重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """エラーコンテキスト情報"""
    error_type: ErrorType
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = ""
    url: Optional[str] = None
    query: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """リトライ設定"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ErrorRecoveryManager:
    """
    エラー回復管理クラス
    
    システム全体のエラーハンドリングと回復機能を提供します。
    各種エラーの分類、ログ記録、回復戦略、フォールバック機能を実装します。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        ErrorRecoveryManagerを初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config or {}
        self.error_history: List[ErrorContext] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        self.default_retry_config = RetryConfig(
            max_attempts=self.config.get("default_max_attempts", 3),
            base_delay=self.config.get("default_base_delay", 1.0),
            max_delay=self.config.get("default_max_delay", 60.0)
        )
        
        # フォールバック応答
        self.fallback_responses = {
            "llm_error": "申し訳ございませんが、現在システムに問題が発生しています。しばらく時間をおいてから再度お試しください。",
            "scraping_error": "文書の取得に失敗しました。ネットワーク接続を確認してください。",
            "database_error": "データベースへのアクセスに問題が発生しました。システム管理者にお問い合わせください。"
        }
        
        logger.info("ErrorRecoveryManager initialized")
    
    def handle_scraping_error(
        self, 
        error: Exception, 
        url: str,
        retry_config: Optional[RetryConfig] = None
    ) -> Optional[Document]:
        """
        Webスクレイピングエラーを処理
        
        Args:
            error: 発生したエラー
            url: スクレイピング対象URL
            retry_config: リトライ設定
            
        Returns:
            回復できた場合はDocument、できなかった場合はNone
        """
        context = ErrorContext(
            error_type=ErrorType.SCRAPING_ERROR,
            severity=self._determine_severity(error),
            operation="web_scraping",
            url=url
        )
        
        self.log_error(error, context)
        
        # ネットワークエラーの場合はリトライ
        if self._is_network_error(error):
            return self._retry_with_backoff(
                self._scrape_with_fallback,
                args=(url,),
                retry_config=retry_config or self.default_retry_config,
                context=context
            )
        
        # その他のエラーの場合は継続処理のためNoneを返す
        logger.warning(f"Scraping failed for {url}, continuing with other documents")
        return None
    
    def handle_llm_error(
        self, 
        error: Exception, 
        query: str,
        retry_config: Optional[RetryConfig] = None
    ) -> str:
        """
        LLMエラーを処理
        
        Args:
            error: 発生したエラー
            query: 処理中のクエリ
            retry_config: リトライ設定
            
        Returns:
            フォールバック応答またはリトライ結果
        """
        context = ErrorContext(
            error_type=ErrorType.LLM_ERROR,
            severity=self._determine_severity(error),
            operation="llm_query",
            query=query
        )
        
        self.log_error(error, context)
        
        # 一時的なエラーの場合はリトライ
        if self._is_temporary_error(error):
            result = self._retry_with_backoff(
                self._query_llm_with_fallback,
                args=(query,),
                retry_config=retry_config or self.default_retry_config,
                context=context
            )
            if result:
                return result
        
        # フォールバック応答を返す
        return self.fallback_responses.get("llm_error", "エラーが発生しました。")
    
    def handle_database_error(
        self, 
        error: Exception, 
        operation: str,
        retry_config: Optional[RetryConfig] = None
    ) -> bool:
        """
        データベースエラーを処理
        
        Args:
            error: 発生したエラー
            operation: 実行中の操作
            retry_config: リトライ設定
            
        Returns:
            回復に成功した場合True、失敗した場合False
        """
        context = ErrorContext(
            error_type=ErrorType.DATABASE_ERROR,
            severity=self._determine_severity(error),
            operation=operation
        )
        
        self.log_error(error, context)
        
        # 接続エラーの場合はリトライ
        if self._is_connection_error(error):
            return self._retry_with_backoff(
                self._reconnect_database,
                retry_config=retry_config or self.default_retry_config,
                context=context
            ) is not None
        
        # その他のエラーは継続処理
        logger.error(f"Database operation '{operation}' failed, continuing")
        return False
    
    def handle_validation_error(
        self, 
        error: Exception, 
        data: Any,
        operation: str
    ) -> bool:
        """
        バリデーションエラーを処理
        
        Args:
            error: 発生したエラー
            data: バリデーション対象データ
            operation: 実行中の操作
            
        Returns:
            処理を継続できる場合True、できない場合False
        """
        context = ErrorContext(
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            operation=operation,
            additional_data={"data_type": type(data).__name__}
        )
        
        self.log_error(error, context)
        
        # バリデーションエラーは通常継続処理可能
        logger.warning(f"Validation failed for {operation}, skipping invalid data")
        return True
    
    def log_error(self, error: Exception, context: ErrorContext) -> None:
        """
        エラーをログに記録し、履歴に保存
        
        Args:
            error: 発生したエラー
            context: エラーコンテキスト
        """
        # エラー履歴に追加
        self.error_history.append(context)
        
        # 履歴サイズ制限
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
        
        # ログレベルを重要度に応じて決定
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(context.severity, logging.ERROR)
        
        # 詳細なエラー情報をログに記録
        error_details = {
            "error_type": context.error_type.value,
            "severity": context.severity.value,
            "operation": context.operation,
            "timestamp": context.timestamp.isoformat(),
            "error_message": str(error),
            "error_class": error.__class__.__name__,
            "traceback": traceback.format_exc()
        }
        
        if context.url:
            error_details["url"] = context.url
        if context.query:
            error_details["query"] = context.query[:100]  # クエリは100文字まで
        if context.session_id:
            error_details["session_id"] = context.session_id
        
        error_details.update(context.additional_data)
        
        logger.log(log_level, f"Error in {context.operation}: {str(error)}", extra=error_details)
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        指定時間内のエラー統計を取得
        
        Args:
            hours: 統計対象時間（時間）
            
        Returns:
            エラー統計情報
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= cutoff_time
        ]
        
        if not recent_errors:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "by_type": {},
                "by_severity": {},
                "most_common_operations": []
            }
        
        # エラータイプ別統計
        by_type = {}
        for error in recent_errors:
            error_type = error.error_type.value
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        # 重要度別統計
        by_severity = {}
        for error in recent_errors:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # 操作別統計
        operations = {}
        for error in recent_errors:
            op = error.operation
            operations[op] = operations.get(op, 0) + 1
        
        most_common_operations = sorted(
            operations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "total_errors": len(recent_errors),
            "error_rate": len(recent_errors) / hours,
            "by_type": by_type,
            "by_severity": by_severity,
            "most_common_operations": most_common_operations,
            "time_range_hours": hours
        }
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """エラーの重要度を判定"""
        error_class = error.__class__.__name__
        error_message = str(error).lower()
        
        # クリティカルエラー
        if any(keyword in error_message for keyword in [
            "out of memory", "disk full", "permission denied", "access denied"
        ]):
            return ErrorSeverity.CRITICAL
        
        # 高重要度エラー
        if any(keyword in error_message for keyword in [
            "connection refused", "timeout", "authentication failed"
        ]):
            return ErrorSeverity.HIGH
        
        # 中重要度エラー
        if any(keyword in error_message for keyword in [
            "not found", "invalid", "bad request"
        ]):
            return ErrorSeverity.MEDIUM
        
        # デフォルトは中重要度
        return ErrorSeverity.MEDIUM
    
    def _is_network_error(self, error: Exception) -> bool:
        """ネットワークエラーかどうか判定"""
        error_message = str(error).lower()
        return any(keyword in error_message for keyword in [
            "connection", "timeout", "network", "dns", "unreachable"
        ])
    
    def _is_temporary_error(self, error: Exception) -> bool:
        """一時的なエラーかどうか判定"""
        error_message = str(error).lower()
        return any(keyword in error_message for keyword in [
            "timeout", "temporary", "temporarily", "busy", "overloaded", "rate limit"
        ])
    
    def _is_connection_error(self, error: Exception) -> bool:
        """接続エラーかどうか判定"""
        error_message = str(error).lower()
        return any(keyword in error_message for keyword in [
            "connection", "connect", "database", "pool"
        ])
    
    def _retry_with_backoff(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[ErrorContext] = None
    ) -> Any:
        """
        指数バックオフでリトライ実行
        
        Args:
            func: 実行する関数
            args: 関数の引数
            kwargs: 関数のキーワード引数
            retry_config: リトライ設定
            context: エラーコンテキスト
            
        Returns:
            関数の実行結果、失敗した場合はNone
        """
        if kwargs is None:
            kwargs = {}
        
        config = retry_config or self.default_retry_config
        
        for attempt in range(config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == config.max_attempts - 1:
                    # 最後の試行でも失敗
                    if context:
                        logger.error(
                            f"All retry attempts failed for {context.operation}: {str(e)}"
                        )
                    return None
                
                # 待機時間を計算
                delay = min(
                    config.base_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                
                if config.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)
                
                if context:
                    logger.info(
                        f"Retry attempt {attempt + 1}/{config.max_attempts} "
                        f"for {context.operation} in {delay:.2f}s"
                    )
                
                time.sleep(delay)
        
        return None
    
    def _scrape_with_fallback(self, url: str) -> Optional[Document]:
        """フォールバック付きスクレイピング（実装は他のコンポーネントに依存）"""
        # 実際の実装では WebScraper を使用
        # ここではプレースホルダー
        raise NotImplementedError("Actual scraping implementation needed")
    
    def _query_llm_with_fallback(self, query: str) -> Optional[str]:
        """フォールバック付きLLMクエリ（実装は他のコンポーネントに依存）"""
        # 実際の実装では LLMManager を使用
        # ここではプレースホルダー
        raise NotImplementedError("Actual LLM query implementation needed")
    
    def _reconnect_database(self) -> bool:
        """データベース再接続（実装は他のコンポーネントに依存）"""
        # 実際の実装では データベース接続管理を使用
        # ここではプレースホルダー
        raise NotImplementedError("Actual database reconnection implementation needed")


def with_error_recovery(
    error_manager: ErrorRecoveryManager,
    error_type: ErrorType,
    operation: str,
    retry_config: Optional[RetryConfig] = None
):
    """
    エラー回復機能付きデコレータ
    
    Args:
        error_manager: ErrorRecoveryManagerインスタンス
        error_type: エラータイプ
        operation: 操作名
        retry_config: リトライ設定
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error_type=error_type,
                    severity=error_manager._determine_severity(e),
                    operation=operation
                )
                
                error_manager.log_error(e, context)
                
                # エラータイプに応じた処理（ログは既に記録済み）
                if error_type == ErrorType.SCRAPING_ERROR:
                    # スクレイピングエラーは継続処理のためNoneを返す
                    return None
                elif error_type == ErrorType.LLM_ERROR:
                    # LLMエラーはフォールバック応答を返す
                    return error_manager.fallback_responses.get("llm_error", "エラーが発生しました。")
                elif error_type == ErrorType.DATABASE_ERROR:
                    # データベースエラーは継続処理のためFalseを返す
                    return False
                elif error_type == ErrorType.VALIDATION_ERROR:
                    # バリデーションエラーは継続処理可能なのでTrueを返す
                    return True
                else:
                    # その他のエラーは再発生
                    raise
        
        return wrapper
    return decorator