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
from ..core.concurrency_manager import ConcurrencyManager, ConcurrencyConfig
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
        self.concurrency_manager: ConcurrencyManager = self.dependencies.get("concurrency_manager")
        
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
        
        # ConcurrencyManagerの開始
        if app_state.concurrency_manager:
            await app_state.concurrency_manager.start()
        
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
        
        if app_state and app_state.concurrency_manager:
            await app_state.concurrency_manager.stop()
        
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
            ollama_base_url=llm_config.get("base_url", "http://localhost:11434")
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
    
    # 同時アクセス管理の初期化
    if not app_state.concurrency_manager:
        concurrency_config = config.get("concurrency", {})
        app_state.concurrency_manager = ConcurrencyManager(
            ConcurrencyConfig(
                max_concurrent_requests=concurrency_config.get("max_concurrent_requests", 10),
                max_queue_size=concurrency_config.get("max_queue_size", 100),
                request_timeout=concurrency_config.get("request_timeout", 30.0),
                rate_limit_per_minute=concurrency_config.get("rate_limit_per_minute", 60),
                enable_request_queuing=concurrency_config.get("enable_request_queuing", True),
                enable_rate_limiting=concurrency_config.get("enable_rate_limiting", True),
                connection_pool_size=concurrency_config.get("connection_pool_size", 20),
                connection_pool_timeout=concurrency_config.get("connection_pool_timeout", 5.0)
            )
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
    
    # 静的ファイルの設定（MIMEタイプを明示的に設定）
    try:
        from fastapi.staticfiles import StaticFiles
        import mimetypes
        
        # JavaScriptファイルのMIMEタイプを確実に設定
        mimetypes.add_type('application/javascript', '.js')
        mimetypes.add_type('text/css', '.css')
        
        app.mount("/static", StaticFiles(directory="genkai_rag/static"), name="static")
        logger.info("Static files mounted successfully")
    except Exception as e:
        logger.warning(f"Failed to mount static files: {e}")
    
    # ルートエンドポイント（Webインターフェイス）
    @app.get("/")
    async def root(request: Request):
        """メインのWebインターフェイス"""
        # 一時的にインラインJavaScriptバージョンを使用
        html_content = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta name="description" content="九州大学スーパーコンピュータ玄界システム用RAG質問応答システム">
            <title>玄界RAGシステム</title>
            <link rel="stylesheet" href="/static/css/style.css">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&display=swap" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <header class="header" role="banner">
                    <h1 class="title">玄界RAGシステム</h1>
                    <p class="subtitle">九州大学スーパーコンピュータ玄界システム用質問応答システム</p>
                </header>

                <main class="main" role="main">
                    <!-- システム状態表示 -->
                    <section class="status-panel" id="statusPanel" aria-label="システム状態">
                        <div class="status-item">
                            <span class="status-label">システム状態:</span>
                            <span class="status-value" id="systemStatus" aria-live="polite">確認中...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">使用モデル:</span>
                            <span class="status-value" id="currentModel" aria-live="polite">確認中...</span>
                        </div>
                        <div class="status-item">
                            <span class="status-label">アクティブセッション:</span>
                            <span class="status-value" id="activeSessions" aria-live="polite">0</span>
                        </div>
                    </section>

                    <!-- モデル選択 -->
                    <section class="model-selector" aria-label="モデル選択">
                        <label for="modelSelect" class="model-label">使用モデル:</label>
                        <select id="modelSelect" class="model-select">
                            <option value="">読み込み中...</option>
                        </select>
                        <button id="switchModelBtn" class="btn btn-secondary">モデル切り替え</button>
                    </section>

                    <!-- 質問入力フォーム -->
                    <section class="query-form" aria-label="質問入力">
                        <form id="queryForm" novalidate>
                            <div class="input-group">
                                <label for="questionInput" class="input-label">質問を入力してください:</label>
                                <textarea 
                                    id="questionInput" 
                                    class="question-input" 
                                    placeholder="玄界システムについて質問してください..."
                                    rows="4"
                                    required
                                ></textarea>
                            </div>
                            
                            <div class="form-options">
                                <div class="checkbox-group">
                                    <input type="checkbox" id="includeHistory" checked>
                                    <label for="includeHistory">会話履歴を含める</label>
                                </div>
                                
                                <div class="input-group-inline">
                                    <label for="maxSources">最大出典数:</label>
                                    <input type="number" id="maxSources" value="5" min="1" max="20" class="number-input">
                                </div>
                            </div>
                            
                            <div class="button-group">
                                <button type="submit" id="submitBtn" class="btn btn-primary">質問する</button>
                                <button type="button" id="clearBtn" class="btn btn-secondary">履歴クリア</button>
                            </div>
                        </form>
                    </section>

                    <!-- 処理中表示 -->
                    <div class="loading" id="loadingIndicator" style="display: none;" role="status" aria-live="assertive">
                        <div class="loading-spinner" aria-hidden="true"></div>
                        <p class="loading-text">回答を生成中...</p>
                    </div>

                    <!-- 会話履歴表示 -->
                    <section class="conversation" id="conversationArea" aria-label="会話履歴">
                        <h2 class="conversation-title">会話履歴</h2>
                        <div class="messages" id="messagesContainer" role="log" aria-live="polite" aria-label="メッセージ一覧">
                            <!-- メッセージがここに動的に追加される -->
                        </div>
                    </section>
                </main>

                <footer class="footer" role="contentinfo">
                    <p>&copy; 2024 九州大学情報基盤研究開発センター</p>
                </footer>
            </div>

            <!-- エラーモーダル -->
            <div class="modal" id="errorModal" style="display: none;">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>エラー</h3>
                        <button class="modal-close" id="closeErrorModal">&times;</button>
                    </div>
                    <div class="modal-body">
                        <p id="errorMessage"></p>
                    </div>
                    <div class="modal-footer">
                        <button class="btn btn-primary" id="errorOkBtn">OK</button>
                    </div>
                </div>
            </div>

            <script>
                console.log('玄界RAGシステム - インラインJavaScript開始');
                
                class GenkaiRAGApp {
                    constructor() {
                        console.log('GenkaiRAGApp constructor called');
                        this.sessionId = this.generateSessionId();
                        this.currentModel = null;
                        this.isProcessing = false;
                        
                        this.initializeElements();
                        this.bindEvents();
                        this.loadInitialData();
                        
                        // 定期的にシステム状態を更新
                        setInterval(() => this.updateSystemStatus(), 30000);
                        console.log('GenkaiRAGApp initialization complete');
                    }
                    
                    initializeElements() {
                        console.log('Initializing DOM elements...');
                        
                        // フォーム要素
                        this.questionInput = document.getElementById('questionInput');
                        this.submitBtn = document.getElementById('submitBtn');
                        this.clearBtn = document.getElementById('clearBtn');
                        this.includeHistoryCheckbox = document.getElementById('includeHistory');
                        this.maxSourcesInput = document.getElementById('maxSources');
                        
                        // モデル選択
                        this.modelSelect = document.getElementById('modelSelect');
                        this.switchModelBtn = document.getElementById('switchModelBtn');
                        
                        // ステータス表示
                        this.systemStatus = document.getElementById('systemStatus');
                        this.currentModelDisplay = document.getElementById('currentModel');
                        this.activeSessions = document.getElementById('activeSessions');
                        
                        // 会話表示
                        this.messagesContainer = document.getElementById('messagesContainer');
                        this.loadingIndicator = document.getElementById('loadingIndicator');
                        
                        // モーダル
                        this.errorModal = document.getElementById('errorModal');
                        this.errorMessage = document.getElementById('errorMessage');
                        this.closeErrorModal = document.getElementById('closeErrorModal');
                        this.errorOkBtn = document.getElementById('errorOkBtn');
                        
                        // 重要な要素の存在確認
                        console.log('DOM elements check:');
                        console.log('questionInput:', !!this.questionInput);
                        console.log('submitBtn:', !!this.submitBtn);
                        console.log('messagesContainer:', !!this.messagesContainer);
                        
                        if (!this.messagesContainer) {
                            console.error('CRITICAL: messagesContainer not found!');
                        }
                    }
                    
                    bindEvents() {
                        console.log('Binding events...');
                        
                        // フォーム送信
                        const queryForm = document.getElementById('queryForm');
                        if (queryForm) {
                            queryForm.addEventListener('submit', (e) => {
                                e.preventDefault();
                                this.submitQuery();
                            });
                        }
                        
                        if (this.submitBtn) {
                            this.submitBtn.addEventListener('click', (e) => {
                                e.preventDefault();
                                this.submitQuery();
                            });
                        }
                        
                        if (this.questionInput) {
                            this.questionInput.addEventListener('keydown', (e) => {
                                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                    e.preventDefault();
                                    this.submitQuery();
                                }
                            });
                            
                            this.questionInput.addEventListener('input', () => {
                                this.validateInput();
                            });
                        }
                        
                        // 履歴クリア
                        if (this.clearBtn) {
                            this.clearBtn.addEventListener('click', () => this.clearHistory());
                        }
                        
                        // モデル切り替え
                        if (this.switchModelBtn) {
                            this.switchModelBtn.addEventListener('click', () => this.switchModel());
                        }
                        
                        // モーダル閉じる
                        if (this.closeErrorModal) {
                            this.closeErrorModal.addEventListener('click', () => this.hideErrorModal());
                        }
                        if (this.errorOkBtn) {
                            this.errorOkBtn.addEventListener('click', () => this.hideErrorModal());
                        }
                        
                        // モーダル外クリックで閉じる
                        if (this.errorModal) {
                            this.errorModal.addEventListener('click', (e) => {
                                if (e.target === this.errorModal) {
                                    this.hideErrorModal();
                                }
                            });
                        }
                        
                        console.log('Event binding complete');
                    }
                    
                    async loadInitialData() {
                        console.log('Loading initial data...');
                        try {
                            await Promise.all([
                                this.loadModels(),
                                this.updateSystemStatus(),
                                this.loadChatHistory()
                            ]);
                            console.log('Initial data loaded successfully');
                        } catch (error) {
                            console.error('Failed to load initial data:', error);
                            this.showError('初期データの読み込みに失敗しました');
                        }
                    }
                    
                    validateInput() {
                        const question = this.questionInput.value.trim();
                        const isValid = question.length >= 1;
                        
                        this.submitBtn.disabled = !isValid || this.isProcessing;
                    }
                    
                    async loadModels() {
                        try {
                            const response = await fetch('/api/models');
                            if (!response.ok) {
                                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            }
                            
                            const data = await response.json();
                            this.populateModelSelect(data.models);
                            this.currentModel = data.current_model;
                            if (this.currentModelDisplay) {
                                this.currentModelDisplay.textContent = this.currentModel;
                            }
                            
                        } catch (error) {
                            console.error('Failed to load models:', error);
                            if (this.modelSelect) {
                                this.modelSelect.innerHTML = '<option value="">エラー: モデル読み込み失敗</option>';
                            }
                        }
                    }
                    
                    populateModelSelect(models) {
                        if (!this.modelSelect) return;
                        
                        this.modelSelect.innerHTML = '';
                        
                        models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.name;
                            option.textContent = model.display_name || model.name;
                            option.selected = model.is_default;
                            
                            if (!model.available) {
                                option.disabled = true;
                                option.textContent += ' (利用不可)';
                            }
                            
                            this.modelSelect.appendChild(option);
                        });
                    }
                    
                    async updateSystemStatus() {
                        try {
                            const response = await fetch('/api/system/status');
                            if (!response.ok) {
                                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            }
                            
                            const status = await response.json();
                            
                            // ステータス表示を更新
                            if (this.systemStatus) {
                                this.systemStatus.textContent = this.getStatusText(status.status);
                                this.systemStatus.className = `status-value ${status.status}`;
                            }
                            
                            if (this.currentModelDisplay) {
                                this.currentModelDisplay.textContent = status.performance_metrics?.current_model || 'unknown';
                            }
                            if (this.activeSessions) {
                                this.activeSessions.textContent = status.system_metrics?.active_sessions || 0;
                            }
                            
                        } catch (error) {
                            console.error('Failed to update system status:', error);
                            if (this.systemStatus) {
                                this.systemStatus.textContent = 'エラー';
                                this.systemStatus.className = 'status-value unhealthy';
                            }
                        }
                    }
                    
                    getStatusText(status) {
                        const statusMap = {
                            'healthy': '正常',
                            'degraded': '低下',
                            'unhealthy': '異常'
                        };
                        return statusMap[status] || status;
                    }
                    
                    async loadChatHistory() {
                        try {
                            const response = await fetch(`/api/chat/history?session_id=${this.sessionId}&limit=20`);
                            if (!response.ok) {
                                if (response.status === 404) {
                                    return; // 履歴がない場合は正常
                                }
                                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            }
                            
                            const data = await response.json();
                            this.displayMessages(data.messages);
                            
                        } catch (error) {
                            console.error('Failed to load chat history:', error);
                        }
                    }
                    
                    async submitQuery() {
                        console.log('submitQuery called');
                        const question = this.questionInput.value.trim();
                        console.log('Question:', question);
                        
                        if (!question) {
                            this.showError('質問を入力してください');
                            this.questionInput.focus();
                            return;
                        }
                        
                        if (this.isProcessing) {
                            console.log('Already processing, ignoring request');
                            return;
                        }
                        
                        console.log('Starting query processing...');
                        this.setProcessingState(true);
                        
                        try {
                            // ユーザーメッセージを即座に表示
                            console.log('Adding user message to UI');
                            this.addMessage('user', question);
                            
                            // APIリクエストを送信
                            const requestData = {
                                question: question,
                                session_id: this.sessionId,
                                model_name: this.modelSelect.value || null,
                                max_sources: parseInt(this.maxSourcesInput.value) || 5,
                                include_history: this.includeHistoryCheckbox.checked
                            };
                            
                            console.log('Sending API request:', requestData);
                            
                            const response = await fetch('/api/query', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify(requestData)
                            });
                            
                            console.log('API response status:', response.status);
                            
                            if (!response.ok) {
                                const errorData = await response.json().catch(() => ({}));
                                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
                            }
                            
                            const data = await response.json();
                            console.log('API response data:', data);
                            console.log('Response text length:', data.response ? data.response.length : 'undefined');
                            console.log('Source documents count:', data.source_documents ? data.source_documents.length : 'undefined');
                            
                            // アシスタントの回答を表示
                            console.log('Adding assistant message to UI');
                            this.addMessage('assistant', data.response, data.source_documents, {
                                processing_time: data.processing_time,
                                model_used: data.model_used
                            });
                            
                            // 入力フィールドをクリア
                            this.questionInput.value = '';
                            
                        } catch (error) {
                            console.error('Failed to submit query:', error);
                            this.showError(`質問の処理に失敗しました: ${error.message}`);
                            
                            // エラーメッセージを表示
                            this.addMessage('assistant', 'エラーが発生しました。もう一度お試しください。', [], {
                                error: true
                            });
                            
                        } finally {
                            console.log('Query processing complete, resetting state');
                            this.setProcessingState(false);
                        }
                    }
                    
                    generateSessionId() {
                        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                    }
                    
                    setProcessingState(processing) {
                        this.isProcessing = processing;
                        
                        if (processing) {
                            this.submitBtn.disabled = true;
                            this.submitBtn.textContent = '処理中...';
                            if (this.loadingIndicator) {
                                this.loadingIndicator.style.display = 'block';
                            }
                        } else {
                            this.submitBtn.disabled = false;
                            this.submitBtn.textContent = '質問する';
                            if (this.loadingIndicator) {
                                this.loadingIndicator.style.display = 'none';
                            }
                            
                            // 入力検証を再実行
                            this.validateInput();
                        }
                    }
                    
                    addMessage(role, content, sources = [], metadata = {}) {
                        console.log('addMessage called:', { 
                            role, 
                            contentLength: content ? content.length : 0,
                            contentPreview: content ? content.substring(0, 100) : 'undefined', 
                            sourcesCount: sources ? sources.length : 0,
                            metadata 
                        });
                        
                        if (!this.messagesContainer) {
                            console.error('messagesContainer not found!');
                            return;
                        }
                        
                        console.log('messagesContainer found:', this.messagesContainer);
                        
                        const messageDiv = document.createElement('div');
                        messageDiv.className = `message ${role}`;
                        console.log('Created messageDiv with class:', messageDiv.className);
                        
                        const headerDiv = document.createElement('div');
                        headerDiv.className = 'message-header';
                        
                        const roleSpan = document.createElement('span');
                        roleSpan.className = 'message-role';
                        roleSpan.textContent = role === 'user' ? 'ユーザー' : 'アシスタント';
                        
                        const timeSpan = document.createElement('span');
                        timeSpan.className = 'message-time';
                        timeSpan.textContent = new Date().toLocaleString('ja-JP');
                        
                        headerDiv.appendChild(roleSpan);
                        headerDiv.appendChild(timeSpan);
                        
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'message-content';
                        contentDiv.textContent = content;
                        console.log('Content set to contentDiv:', contentDiv.textContent.substring(0, 100));
                        
                        messageDiv.appendChild(headerDiv);
                        messageDiv.appendChild(contentDiv);
                        
                        // 出典情報を追加
                        if (sources && sources.length > 0) {
                            console.log('Adding sources:', sources);
                            const sourcesDiv = document.createElement('div');
                            sourcesDiv.className = 'message-sources';
                            
                            const sourcesTitle = document.createElement('div');
                            sourcesTitle.className = 'sources-title';
                            sourcesTitle.textContent = '出典:';
                            sourcesDiv.appendChild(sourcesTitle);
                            
                            sources.forEach(source => {
                                const sourceItem = document.createElement('div');
                                sourceItem.className = 'source-item';
                                
                                const sourceLink = document.createElement('a');
                                sourceLink.className = 'source-url';
                                sourceLink.href = source.url || '#';
                                sourceLink.target = '_blank';
                                sourceLink.textContent = source.url || 'ドキュメント';
                                
                                const sourceTitle = document.createElement('span');
                                sourceTitle.className = 'source-title';
                                sourceTitle.textContent = source.title || '';
                                
                                sourceItem.appendChild(sourceLink);
                                if (source.title) {
                                    sourceItem.appendChild(sourceTitle);
                                }
                                
                                sourcesDiv.appendChild(sourceItem);
                            });
                            
                            messageDiv.appendChild(sourcesDiv);
                        }
                        
                        // メタデータ情報を追加（デバッグ用）
                        if (metadata.processing_time) {
                            const metaDiv = document.createElement('div');
                            metaDiv.style.fontSize = '0.8rem';
                            metaDiv.style.color = '#666';
                            metaDiv.style.marginTop = '10px';
                            metaDiv.textContent = `処理時間: ${metadata.processing_time.toFixed(2)}秒`;
                            if (metadata.model_used) {
                                metaDiv.textContent += ` | モデル: ${metadata.model_used}`;
                            }
                            messageDiv.appendChild(metaDiv);
                        }
                        
                        console.log('About to append messageDiv to container');
                        console.log('Container children count before:', this.messagesContainer.children.length);
                        
                        this.messagesContainer.appendChild(messageDiv);
                        
                        console.log('Container children count after:', this.messagesContainer.children.length);
                        console.log('MessageDiv appended successfully');
                        
                        // スクロールを最下部に移動
                        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
                        console.log('Scrolled to bottom');
                    }
                    
                    displayMessages(messages) {
                        if (!this.messagesContainer) return;
                        
                        this.messagesContainer.innerHTML = '';
                        
                        messages.forEach(msg => {
                            const role = msg.role === 'user' ? 'user' : 'assistant';
                            const sources = msg.metadata?.sources || [];
                            this.addMessage(role, msg.content, sources);
                        });
                    }
                    
                    async clearHistory() {
                        if (!confirm('会話履歴をクリアしますか？')) {
                            return;
                        }
                        
                        try {
                            const response = await fetch(`/api/chat/history/${this.sessionId}`, {
                                method: 'DELETE'
                            });
                            
                            if (!response.ok) {
                                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                            }
                            
                            // 表示をクリア
                            if (this.messagesContainer) {
                                this.messagesContainer.innerHTML = '';
                            }
                            
                            // 新しいセッションIDを生成
                            this.sessionId = this.generateSessionId();
                            
                        } catch (error) {
                            console.error('Failed to clear history:', error);
                            this.showError('履歴のクリアに失敗しました');
                        }
                    }
                    
                    async switchModel() {
                        const selectedModel = this.modelSelect.value;
                        if (!selectedModel) {
                            this.showError('モデルを選択してください');
                            return;
                        }
                        
                        if (selectedModel === this.currentModel) {
                            this.showError('既に選択されているモデルです');
                            return;
                        }
                        
                        try {
                            this.switchModelBtn.disabled = true;
                            this.switchModelBtn.textContent = '切り替え中...';
                            
                            const response = await fetch('/api/models/switch', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json'
                                },
                                body: JSON.stringify({
                                    model_name: selectedModel,
                                    force: false
                                })
                            });
                            
                            if (!response.ok) {
                                const errorData = await response.json().catch(() => ({}));
                                throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
                            }
                            
                            const data = await response.json();
                            
                            if (data.success) {
                                this.currentModel = selectedModel;
                                if (this.currentModelDisplay) {
                                    this.currentModelDisplay.textContent = selectedModel;
                                }
                                this.addMessage('assistant', `モデルを ${selectedModel} に切り替えました。`, [], {
                                    system: true
                                });
                            } else {
                                throw new Error(data.message || 'モデルの切り替えに失敗しました');
                            }
                            
                        } catch (error) {
                            console.error('Failed to switch model:', error);
                            this.showError(`モデルの切り替えに失敗しました: ${error.message}`);
                            
                        } finally {
                            this.switchModelBtn.disabled = false;
                            this.switchModelBtn.textContent = 'モデル切り替え';
                        }
                    }
                    
                    showError(message) {
                        console.error('Error:', message);
                        if (this.errorMessage && this.errorModal) {
                            this.errorMessage.textContent = message;
                            this.errorModal.style.display = 'flex';
                        } else {
                            alert(message); // フォールバック
                        }
                    }
                    
                    hideErrorModal() {
                        if (this.errorModal) {
                            this.errorModal.style.display = 'none';
                        }
                        if (this.questionInput) {
                            this.questionInput.focus();
                        }
                    }
                }

                // アプリケーションを初期化
                document.addEventListener('DOMContentLoaded', () => {
                    console.log('DOM loaded, initializing GenkaiRAGApp...');
                    console.log('Document ready state:', document.readyState);
                    
                    // DOM要素の存在確認
                    const messagesContainer = document.getElementById('messagesContainer');
                    console.log('messagesContainer found during init:', !!messagesContainer);
                    
                    try {
                        window.genkaiApp = new GenkaiRAGApp();
                        console.log('GenkaiRAGApp initialized successfully');
                        console.log('App instance:', window.genkaiApp);
                    } catch (error) {
                        console.error('Failed to initialize GenkaiRAGApp:', error);
                        console.error('Error stack:', error.stack);
                    }
                });
                
                console.log('玄界RAGシステム - インラインJavaScript完了');
            </script>
        </body>
        </html>
        """
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    
    # テスト用エンドポイント
    @app.get("/test")
    async def test_page(request: Request):
        """JavaScript テストページ"""
        # インラインJavaScriptのみのテストページ
        html_content = """
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JavaScript テスト</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                #messagesContainer { 
                    border: 2px solid #333; 
                    min-height: 200px; 
                    padding: 15px; 
                    margin: 20px 0;
                    background-color: #f9f9f9;
                }
                button { 
                    padding: 10px 20px; 
                    font-size: 16px; 
                    background-color: #007bff;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                button:hover { background-color: #0056b3; }
                .message { 
                    padding: 8px; 
                    margin: 5px 0; 
                    background-color: #e9ecef;
                    border-left: 4px solid #007bff;
                }
            </style>
        </head>
        <body>
            <h1>JavaScript 動作テスト</h1>
            
            <div>
                <button id="testBtn">テストボタン</button>
                <button id="apiTestBtn">API テスト</button>
            </div>
            
            <div id="messagesContainer">
                <p><strong>メッセージ表示エリア</strong></p>
                <p>ボタンをクリックしてJavaScriptの動作を確認してください。</p>
            </div>
            
            <div id="console-log" style="background: #000; color: #0f0; padding: 10px; font-family: monospace; margin-top: 20px;">
                <strong>コンソールログ:</strong><br>
            </div>
            
            <script>
                // カスタムログ関数
                function log(message) {
                    console.log(message);
                    const logDiv = document.getElementById('console-log');
                    logDiv.innerHTML += '<br>' + new Date().toLocaleTimeString() + ': ' + message;
                }
                
                log('インラインスクリプト読み込み完了');
                
                document.addEventListener('DOMContentLoaded', function() {
                    log('DOM読み込み完了');
                    
                    const testBtn = document.getElementById('testBtn');
                    const apiTestBtn = document.getElementById('apiTestBtn');
                    const messagesContainer = document.getElementById('messagesContainer');
                    
                    log('DOM要素取得: testBtn=' + !!testBtn + ', messagesContainer=' + !!messagesContainer);
                    
                    if (testBtn) {
                        testBtn.addEventListener('click', function() {
                            log('テストボタンクリック');
                            
                            if (messagesContainer) {
                                const messageDiv = document.createElement('div');
                                messageDiv.className = 'message';
                                messageDiv.innerHTML = '<strong>テストメッセージ:</strong> ' + new Date().toLocaleString();
                                messagesContainer.appendChild(messageDiv);
                                log('メッセージ追加成功');
                            } else {
                                log('エラー: messagesContainer が見つかりません');
                            }
                        });
                    }
                    
                    if (apiTestBtn) {
                        apiTestBtn.addEventListener('click', async function() {
                            log('API テストボタンクリック');
                            
                            try {
                                const response = await fetch('/api/system/status');
                                log('API レスポンス状態: ' + response.status);
                                
                                if (response.ok) {
                                    const data = await response.json();
                                    log('API データ取得成功: ' + JSON.stringify(data).substring(0, 100) + '...');
                                    
                                    if (messagesContainer) {
                                        const messageDiv = document.createElement('div');
                                        messageDiv.className = 'message';
                                        messageDiv.innerHTML = '<strong>API テスト成功:</strong> システム状態 = ' + data.status;
                                        messagesContainer.appendChild(messageDiv);
                                    }
                                } else {
                                    log('API エラー: ' + response.status);
                                }
                            } catch (error) {
                                log('API 例外: ' + error.message);
                            }
                        });
                    }
                    
                    log('イベントリスナー設定完了');
                });
                
                log('スクリプト実行完了');
            </script>
        </body>
        </html>
        """
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    
    # 元のテンプレートベースのページ
    @app.get("/original")
    async def original_page(request: Request):
        """元のテンプレートベースのページ"""
        if not app_state.templates:
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content="<h1>Templates not initialized</h1>")
        
        return app_state.templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "title": "玄界RAGシステム",
                "timestamp": int(__import__('time').time())
            }
        )
    
    # JavaScriptファイル専用エンドポイント
    @app.get("/static/js/{filename}")
    async def serve_js_file(filename: str):
        """JavaScriptファイルを正しいMIMEタイプで配信"""
        from fastapi.responses import FileResponse
        import os
        
        file_path = f"genkai_rag/static/js/{filename}"
        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                media_type="application/javascript",
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="File not found")
    
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
                    "document_processor": app_state.document_processor is not None,
                    "concurrency_manager": app_state.concurrency_manager is not None
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
                    llm_health = app_state.llm_manager.check_model_health()
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


# デフォルトアプリケーションインスタンス（uvicorn用）
app = create_app()