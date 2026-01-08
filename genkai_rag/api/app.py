"""
FastAPIアプリケーションのメインファイル

このモジュールは、FastAPIアプリケーションの作成と設定を行います。
依存性注入によるコンポーネント管理をサポートします。
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..core.config_manager import ConfigManager
from ..core.rag_engine import RAGEngine
from ..core.llm_manager import LLMManager
from ..core.chat_manager import ChatManager
from ..core.system_monitor import SystemMonitor
from ..core.processor import DocumentProcessor
from ..core.error_recovery import ErrorRecoveryManager
from ..core.scraper import WebScraper
from .middleware import LoggingMiddleware, ErrorHandlingMiddleware

logger = logging.getLogger(__name__)


class AppState:
    """アプリケーション状態管理クラス"""
    
    def __init__(self, dependencies: Optional[Dict[str, Any]] = None):
        """
        アプリケーション状態を初期化
        
        Args:
            dependencies: 依存性注入用のコンポーネント辞書
        """
        self.dependencies = dependencies or {}
        
        # 依存性注入されたコンポーネントを設定
        self.config_manager: ConfigManager = self.dependencies.get("config_manager")
        self.rag_engine: RAGEngine = self.dependencies.get("rag_engine")
        self.llm_manager: LLMManager = self.dependencies.get("llm_manager")
        self.chat_manager: ChatManager = self.dependencies.get("chat_manager")
        self.system_monitor: SystemMonitor = self.dependencies.get("system_monitor")
        self.document_processor: DocumentProcessor = self.dependencies.get("document_processor")
        self.error_recovery_manager: ErrorRecoveryManager = self.dependencies.get("error_recovery_manager")
        self.web_scraper: WebScraper = self.dependencies.get("web_scraper")
        
        # テンプレートエンジン
        self.templates: Jinja2Templates = None


# グローバル状態インスタンス
app_state: Optional[AppState] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """アプリケーションのライフサイクル管理"""
    # 起動時の初期化
    logger.info("Starting Genkai RAG System...")
    
    try:
        # 依存性注入されたコンポーネントがない場合は従来の初期化を実行
        if not app_state or not app_state.dependencies:
            await _initialize_legacy_components()
        
        # テンプレートエンジンの初期化
        app_state.templates = Jinja2Templates(directory="genkai_rag/templates")
        
        logger.info("Genkai RAG System started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        if app_state and app_state.error_recovery_manager:
            app_state.error_recovery_manager.handle_validation_error(
                e, {}, "application_startup"
            )
        raise
    
    finally:
        # 終了時のクリーンアップ
        logger.info("Shutting down Genkai RAG System...")
        
        if app_state and app_state.system_monitor:
            app_state.system_monitor.stop_monitoring()
        
        logger.info("Genkai RAG System shutdown complete")


async def _initialize_legacy_components():
    """従来の方式でコンポーネントを初期化（後方互換性のため）"""
    logger.info("Initializing components in legacy mode...")
    
    # 設定管理の初期化
    if not app_state.config_manager:
        app_state.config_manager = ConfigManager()
    
    config = app_state.config_manager.load_config()
    
    # システム監視の初期化
    if not app_state.system_monitor:
        app_state.system_monitor = SystemMonitor()
        app_state.system_monitor.start_monitoring()
    
    # LLMマネージャーの初期化
    if not app_state.llm_manager:
        llm_config = config.get("llm", {})
        app_state.llm_manager = LLMManager(
            ollama_url=llm_config.get("ollama_url", "http://localhost:11434")
        )
    
    # 文書プロセッサーの初期化
    if not app_state.document_processor:
        app_state.document_processor = DocumentProcessor()
    
    # RAGエンジンの初期化
    if not app_state.rag_engine:
        app_state.rag_engine = RAGEngine(
            llm_manager=app_state.llm_manager,
            document_processor=app_state.document_processor
        )
    
    # チャットマネージャーの初期化
    if not app_state.chat_manager:
        chat_config = config.get("chat", {})
        app_state.chat_manager = ChatManager(
            max_history_size=chat_config.get("max_history_size", 50)
        )


def create_app(dependencies: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """
    FastAPIアプリケーションを作成
    
    Args:
        dependencies: 依存性注入用のコンポーネント辞書
        config: Web設定
        
    Returns:
        設定済みのFastAPIアプリケーション
    """
    global app_state
    
    # アプリケーション状態を初期化
    app_state = AppState(dependencies)
    
    # 設定の取得
    web_config = config or {}
    if not web_config and app_state.config_manager:
        try:
            full_config = app_state.config_manager.load_config()
            web_config = full_config.get("web", {})
        except Exception as e:
            logger.warning(f"Failed to load web config: {e}")
            web_config = {}
    
    # FastAPIアプリケーションの作成
    app = FastAPI(
        title="Genkai RAG System",
        description="九州大学スーパーコンピュータ玄界システム用RAGシステム",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
        debug=web_config.get("debug", False)
    )
    
    # 依存性注入の設定
    app.state.dependencies = dependencies or {}
    app.state.app_state = app_state
    
    # CORS設定
    cors_origins = web_config.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # 信頼できるホストの設定
    allowed_hosts = web_config.get("allowed_hosts", ["*"])
    if allowed_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )
    
    # カスタムミドルウェアの追加
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # APIルーターの登録
    from .routes import query_router, model_router, chat_router, system_router, health_router
    app.include_router(query_router, prefix="/api", tags=["query"])
    app.include_router(model_router, prefix="/api", tags=["models"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(system_router, prefix="/api", tags=["system"])
    app.include_router(health_router, prefix="/api", tags=["health"])
    
    # 静的ファイルの設定
    try:
        app.mount("/static", StaticFiles(directory="genkai_rag/static"), name="static")
    except Exception as e:
        logger.warning(f"Failed to mount static files: {e}")
    
    # ルートエンドポイント（Webインターフェイス）
    @app.get("/")
    async def root(request: Request):
        """メインのWebインターフェイス"""
        if not app_state.templates:
            # テンプレートが初期化されていない場合は簡単なHTMLを返す
            html_content = """
            <!DOCTYPE html>
            <html lang="ja">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>玄界RAGシステム</title>
            </head>
            <body>
                <main role="main">
                    <h1>玄界RAGシステム</h1>
                    <div id="messagesContainer" aria-live="polite"></div>
                </main>
            </body>
            </html>
            """
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=html_content)
        
        return app_state.templates.TemplateResponse(
            "index.html", 
            {"request": request, "title": "玄界RAGシステム"}
        )
    
    # ヘルスチェックエンドポイント
    @app.get("/health")
    async def health_check():
        """システムヘルスチェック"""
        try:
            # 基本的なヘルスチェック
            status = {
                "status": "healthy",
                "components": {
                    "config_manager": app_state.config_manager is not None,
                    "rag_engine": app_state.rag_engine is not None,
                    "llm_manager": app_state.llm_manager is not None,
                    "chat_manager": app_state.chat_manager is not None,
                    "system_monitor": app_state.system_monitor is not None,
                    "error_recovery_manager": app_state.error_recovery_manager is not None,
                    "web_scraper": app_state.web_scraper is not None,
                    "document_processor": app_state.document_processor is not None
                }
            }
            
            # システム監視情報を追加
            if app_state.system_monitor:
                system_status = app_state.system_monitor.get_system_status()
                status["timestamp"] = system_status.timestamp.isoformat() if hasattr(system_status.timestamp, 'isoformat') else str(system_status.timestamp)
                status["system_metrics"] = {
                    "memory_usage": getattr(system_status, 'memory_usage', system_status.memory_usage_mb),
                    "disk_usage": getattr(system_status, 'disk_usage', system_status.disk_usage_mb),
                    "cpu_usage": getattr(system_status, 'cpu_usage', 0.0)
                }
            
            # LLMの健全性チェック
            if app_state.llm_manager:
                try:
                    llm_health = await app_state.llm_manager.check_model_health()
                    status["components"]["llm_health"] = llm_health
                except Exception as e:
                    logger.warning(f"LLM health check failed: {e}")
                    status["components"]["llm_health"] = False
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            if app_state.error_recovery_manager:
                app_state.error_recovery_manager.handle_validation_error(
                    e, {}, "health_check"
                )
            raise HTTPException(status_code=503, detail="Service unavailable")
    
    # グローバル例外ハンドラー
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """グローバル例外ハンドラー"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        # エラー回復管理でログ記録
        if app_state.error_recovery_manager:
            try:
                app_state.error_recovery_manager.handle_validation_error(
                    exc, {"path": str(request.url.path)}, "global_exception"
                )
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    return app


def get_app_state() -> AppState:
    """アプリケーション状態を取得"""
    return app_state