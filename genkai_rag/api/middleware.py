"""
FastAPIミドルウェア実装

このモジュールは、FastAPIアプリケーション用のカスタムミドルウェアを提供します。
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    リクエスト/レスポンスのログ記録ミドルウェア
    
    すべてのHTTPリクエストとレスポンスをログに記録し、
    処理時間とリクエストIDを追跡します。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        リクエストを処理し、ログを記録
        
        Args:
            request: HTTPリクエスト
            call_next: 次のミドルウェア/ハンドラー
            
        Returns:
            HTTPレスポンス
        """
        # リクエストIDを生成
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 開始時刻を記録
        start_time = time.time()
        
        # クライアント情報を取得
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # リクエストログ
        logger.info(
            f"Request started - ID: {request_id}, "
            f"Method: {request.method}, "
            f"URL: {request.url}, "
            f"Client: {client_ip}, "
            f"User-Agent: {user_agent}"
        )
        
        try:
            # リクエストを処理
            response = await call_next(request)
            
            # 処理時間を計算
            process_time = time.time() - start_time
            
            # レスポンスヘッダーにリクエストIDと処理時間を追加
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # レスポンスログ
            logger.info(
                f"Request completed - ID: {request_id}, "
                f"Status: {response.status_code}, "
                f"Process Time: {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # エラー時の処理時間を計算
            process_time = time.time() - start_time
            
            # エラーログ
            logger.error(
                f"Request failed - ID: {request_id}, "
                f"Error: {str(e)}, "
                f"Process Time: {process_time:.3f}s",
                exc_info=True
            )
            
            # エラーレスポンスを返す
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Process-Time": str(process_time)
                }
            )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    エラーハンドリングミドルウェア
    
    アプリケーション全体のエラーを統一的に処理し、
    適切なエラーレスポンスを返します。
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        リクエストを処理し、エラーをハンドリング
        
        Args:
            request: HTTPリクエスト
            call_next: 次のミドルウェア/ハンドラー
            
        Returns:
            HTTPレスポンス
        """
        try:
            response = await call_next(request)
            return response
            
        except ValueError as e:
            # バリデーションエラー
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation Error",
                    "message": str(e),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except FileNotFoundError as e:
            # ファイル/リソースが見つからない
            logger.warning(f"Resource not found: {str(e)}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Not Found",
                    "message": "The requested resource was not found",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except PermissionError as e:
            # 権限エラー
            logger.warning(f"Permission denied: {str(e)}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "message": "Access to the requested resource is forbidden",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except TimeoutError as e:
            # タイムアウトエラー
            logger.error(f"Timeout error: {str(e)}")
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Gateway Timeout",
                    "message": "The request timed out",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except ConnectionError as e:
            # 接続エラー
            logger.error(f"Connection error: {str(e)}")
            return JSONResponse(
                status_code=502,
                content={
                    "error": "Bad Gateway",
                    "message": "Unable to connect to external service",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except Exception as e:
            # その他の予期しないエラー
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    セキュリティミドルウェア
    
    セキュリティヘッダーの追加とリクエストの検証を行います。
    """
    
    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        セキュリティチェックを実行
        
        Args:
            request: HTTPリクエスト
            call_next: 次のミドルウェア/ハンドラー
            
        Returns:
            HTTPレスポンス
        """
        # リクエストサイズのチェック
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(f"Request too large: {content_length} bytes")
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Payload Too Large",
                    "message": f"Request size exceeds maximum allowed size of {self.max_request_size} bytes",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        # リクエストを処理
        response = await call_next(request)
        
        # セキュリティヘッダーを追加
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    レート制限ミドルウェア
    
    IPアドレス単位でのリクエスト制限を実装します。
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.last_reset = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        レート制限をチェック
        
        Args:
            request: HTTPリクエスト
            call_next: 次のミドルウェア/ハンドラー
            
        Returns:
            HTTPレスポンス
        """
        # 1分ごとにカウンターをリセット
        current_time = time.time()
        if current_time - self.last_reset > 60:
            self.request_counts.clear()
            self.last_reset = current_time
        
        # クライアントIPを取得
        client_ip = request.client.host if request.client else "unknown"
        
        # リクエスト数をカウント
        self.request_counts[client_ip] = self.request_counts.get(client_ip, 0) + 1
        
        # レート制限をチェック
        if self.request_counts[client_ip] > self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute allowed",
                    "request_id": getattr(request.state, "request_id", None)
                },
                headers={
                    "Retry-After": "60"
                }
            )
        
        # リクエストを処理
        response = await call_next(request)
        
        # レート制限情報をヘッダーに追加
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - self.request_counts[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(self.last_reset + 60))
        
        return response