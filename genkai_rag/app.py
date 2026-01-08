"""
玄界RAGシステム - メインアプリケーション

全コンポーネントを統合し、依存性注入とコンポーネント間の配線を行います。
設定ファイルからの初期化処理を提供します。
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

from .utils.config import ConfigManager as UtilsConfigManager
from .utils.logging import setup_logging
from .core import (
    WebScraper, DocumentProcessor, LLMManager, RAGEngine, 
    ChatManager, ConfigManager, SystemMonitor, ErrorRecoveryManager
)
from .api.app import create_app


logger = logging.getLogger(__name__)


class GenkaiRAGSystem:
    """
    玄界RAGシステムのメインクラス
    
    全コンポーネントを統合し、システム全体の初期化と管理を行います。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        システムを初期化
        
        Args:
            config_path: 設定ファイルのパス（デフォルト: config/config.yaml）
        """
        self.config_path = config_path or "config/config.yaml"
        self.config: Optional[Dict[str, Any]] = None
        
        # コンポーネント
        self.utils_config_manager: Optional[UtilsConfigManager] = None
        self.config_manager: Optional[ConfigManager] = None
        self.error_recovery_manager: Optional[ErrorRecoveryManager] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.web_scraper: Optional[WebScraper] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.llm_manager: Optional[LLMManager] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.chat_manager: Optional[ChatManager] = None
        
        # FastAPIアプリケーション
        self.app = None
        
        # 初期化フラグ
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        システム全体を初期化
        
        依存性注入とコンポーネント間の配線を行います。
        """
        if self._initialized:
            logger.warning("System already initialized")
            return
        
        try:
            logger.info("Initializing Genkai RAG System...")
            
            # 1. 設定管理の初期化
            await self._initialize_config()
            
            # 2. ログ設定の初期化
            self._initialize_logging()
            
            # 3. エラー回復管理の初期化
            self._initialize_error_recovery()
            
            # 4. システム監視の初期化
            await self._initialize_system_monitor()
            
            # 5. コアコンポーネントの初期化
            await self._initialize_core_components()
            
            # 6. FastAPIアプリケーションの初期化
            await self._initialize_web_app()
            
            self._initialized = True
            logger.info("Genkai RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {str(e)}")
            if self.error_recovery_manager:
                self.error_recovery_manager.handle_validation_error(
                    e, {"config_path": self.config_path}, "system_initialization"
                )
            raise
    
    async def _initialize_config(self) -> None:
        """設定管理を初期化"""
        logger.info("Initializing configuration management...")
        
        # ユーティリティ設定マネージャー（既存）
        self.utils_config_manager = UtilsConfigManager()
        self.config = self.utils_config_manager.get_all()
        
        # コア設定マネージャー（新規）
        config_dir = Path(self.config_path).parent
        self.config_manager = ConfigManager(str(config_dir))
        
        # 設定ファイルが存在しない場合はデフォルト設定を作成
        if not Path(self.config_path).exists():
            logger.info("Creating default configuration...")
            await self._create_default_config()
    
    def _initialize_logging(self) -> None:
        """ログ設定を初期化"""
        logger.info("Initializing logging configuration...")
        
        log_config = self.config.get("logging", {})
        setup_logging(
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file", "logs/genkai_rag.log"),
            max_log_size_mb=log_config.get("max_bytes", 10485760) // (1024 * 1024),  # バイトをMBに変換
            backup_count=log_config.get("backup_count", 5)
        )
    
    def _initialize_error_recovery(self) -> None:
        """エラー回復管理を初期化"""
        logger.info("Initializing error recovery management...")
        
        error_config = self.config.get("error_recovery", {})
        self.error_recovery_manager = ErrorRecoveryManager(error_config)
    
    async def _initialize_system_monitor(self) -> None:
        """システム監視を初期化"""
        logger.info("Initializing system monitoring...")
        
        monitor_config = self.config.get("system_monitor", {})
        self.system_monitor = SystemMonitor(
            log_dir=monitor_config.get("log_dir", "logs"),
            data_dir=monitor_config.get("data_dir", "data"),
            monitoring_interval=monitor_config.get("monitoring_interval", 60),
            retention_days=monitor_config.get("retention_days", 30)
        )
        
        # バックグラウンド監視を開始
        if monitor_config.get("enable_background_monitoring", True):
            self.system_monitor.start_monitoring()
    
    async def _initialize_core_components(self) -> None:
        """コアコンポーネントを初期化"""
        logger.info("Initializing core components...")
        
        # Webスクレイパー
        scraper_config = self.config.get("scraper", {})
        self.web_scraper = WebScraper(
            timeout=scraper_config.get("timeout", 30),
            max_retries=scraper_config.get("max_retries", 3),
            request_delay=scraper_config.get("delay", 1.0)
        )
        
        # 文書プロセッサー
        processor_config = self.config.get("document_processor", {})
        self.document_processor = DocumentProcessor(
            chunk_size=processor_config.get("chunk_size", 1000),
            chunk_overlap=processor_config.get("chunk_overlap", 200),
            embedding_model=processor_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        # LLMマネージャー
        llm_config = self.config.get("llm", {})
        self.llm_manager = LLMManager(
            ollama_base_url=llm_config.get("base_url", "http://localhost:11434"),
            system_monitor=self.system_monitor
        )
        
        # RAGエンジン
        rag_config = self.config.get("rag", {})
        self.rag_engine = RAGEngine(
            llm_manager=self.llm_manager,
            document_processor=self.document_processor,
            max_retrieved_docs=rag_config.get("similarity_top_k", 5),
            max_context_docs=rag_config.get("rerank_top_n", 3),
            system_monitor=self.system_monitor
        )
        
        # チャットマネージャー
        chat_config = self.config.get("chat", {})
        self.chat_manager = ChatManager(
            max_history_size=chat_config.get("max_history_size", 100),
            max_session_age_days=chat_config.get("session_timeout_hours", 24) // 24  # 時間を日に変換
        )
    
    async def _initialize_web_app(self) -> None:
        """FastAPIアプリケーションを初期化"""
        logger.info("Initializing web application...")
        
        # 依存性注入用の設定
        dependencies = {
            "rag_engine": self.rag_engine,
            "chat_manager": self.chat_manager,
            "llm_manager": self.llm_manager,
            "system_monitor": self.system_monitor,
            "config_manager": self.config_manager,
            "error_recovery_manager": self.error_recovery_manager,
            "web_scraper": self.web_scraper,
            "document_processor": self.document_processor
        }
        
        web_config = self.config.get("web", {})
        self.app = create_app(
            dependencies=dependencies,
            config=web_config
        )
    
    async def _create_default_config(self) -> None:
        """デフォルト設定ファイルを作成"""
        default_config = {
            "logging": {
                "level": "INFO",
                "file": "logs/genkai_rag.log",
                "max_bytes": 10485760,
                "backup_count": 5
            },
            "error_recovery": {
                "max_history_size": 1000,
                "default_max_attempts": 3,
                "default_base_delay": 1.0,
                "default_max_delay": 60.0
            },
            "system_monitor": {
                "enable_background_monitoring": True,
                "monitoring_interval": 60,
                "memory_threshold": 80.0,
                "disk_threshold": 90.0,
                "cpu_threshold": 80.0
            },
            "scraper": {
                "timeout": 30,
                "max_retries": 3,
                "delay": 1.0,
                "user_agent": "Genkai RAG System/1.0"
            },
            "document_processor": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            },
            "llm": {
                "base_url": "http://localhost:11434",
                "default_model": "llama3.2:3b",
                "timeout": 60,
                "japanese_model": "elyza:jp-7b",
                "large_model": "llama3.1:70b",
                "small_model": "llama3.2:1b",
                "code_model": "codellama:7b"
            },
            "rag": {
                "similarity_top_k": 5,
                "rerank_top_n": 3,
                "response_mode": "compact"
            },
            "chat": {
                "max_history_size": 100,
                "session_timeout_hours": 24,
                "enable_context_compression": True
            },
            "web": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "cors_origins": ["*"],
                "rate_limit": {
                    "enabled": True,
                    "requests_per_minute": 60
                }
            }
        }
        
        # 設定ファイルを保存
        config_path = Path(self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        # 設定を再読み込み
        self.utils_config_manager = UtilsConfigManager()
        self.config = self.utils_config_manager.get_all()
    
    async def start_server(self, host: str = None, port: int = None) -> None:
        """
        Webサーバーを開始
        
        Args:
            host: ホストアドレス
            port: ポート番号
        """
        if not self._initialized:
            await self.initialize()
        
        web_config = self.config.get("web", {})
        host = host or web_config.get("host", "0.0.0.0")
        port = port or web_config.get("port", 8000)
        
        logger.info(f"Starting web server on {host}:{port}")
        
        import uvicorn
        await uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
    
    async def shutdown(self) -> None:
        """システムをシャットダウン"""
        logger.info("Shutting down Genkai RAG System...")
        
        try:
            # システム監視を停止
            if self.system_monitor:
                self.system_monitor.stop_monitoring()
            
            # チャットマネージャーのクリーンアップ
            if self.chat_manager:
                self.chat_manager.cleanup_old_sessions()
            
            # 設定の保存
            if self.config_manager:
                # 必要に応じて設定変更を保存
                pass
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            if self.error_recovery_manager:
                self.error_recovery_manager.handle_validation_error(
                    e, {}, "system_shutdown"
                )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        システム状態を取得
        
        Returns:
            システム状態情報
        """
        if not self._initialized:
            return {"status": "not_initialized"}
        
        status = {
            "status": "running",
            "initialized": self._initialized,
            "components": {
                "config_manager": self.config_manager is not None,
                "error_recovery_manager": self.error_recovery_manager is not None,
                "system_monitor": self.system_monitor is not None,
                "web_scraper": self.web_scraper is not None,
                "document_processor": self.document_processor is not None,
                "llm_manager": self.llm_manager is not None,
                "rag_engine": self.rag_engine is not None,
                "chat_manager": self.chat_manager is not None,
                "web_app": self.app is not None
            }
        }
        
        # システム監視情報を追加
        if self.system_monitor:
            try:
                system_status = self.system_monitor.get_system_status()
                status["system_metrics"] = {
                    "memory_usage": system_status.memory_usage,
                    "disk_usage": system_status.disk_usage,
                    "cpu_usage": system_status.cpu_usage
                }
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {str(e)}")
        
        # エラー統計を追加
        if self.error_recovery_manager:
            try:
                error_stats = self.error_recovery_manager.get_error_statistics(24)
                status["error_statistics"] = error_stats
            except Exception as e:
                logger.warning(f"Failed to get error statistics: {str(e)}")
        
        return status


# グローバルシステムインスタンス
_system_instance: Optional[GenkaiRAGSystem] = None


def get_system() -> GenkaiRAGSystem:
    """
    グローバルシステムインスタンスを取得
    
    Returns:
        GenkaiRAGSystemインスタンス
    """
    global _system_instance
    if _system_instance is None:
        _system_instance = GenkaiRAGSystem()
    return _system_instance


async def initialize_system(config_path: Optional[str] = None) -> GenkaiRAGSystem:
    """
    システムを初期化
    
    Args:
        config_path: 設定ファイルのパス
        
    Returns:
        初期化されたGenkaiRAGSystemインスタンス
    """
    global _system_instance
    _system_instance = GenkaiRAGSystem(config_path)
    await _system_instance.initialize()
    return _system_instance


async def shutdown_system() -> None:
    """システムをシャットダウン"""
    global _system_instance
    if _system_instance:
        await _system_instance.shutdown()
        _system_instance = None