"""
FastAPI ルーター実装

このモジュールは、FastAPIアプリケーション用のAPIルーターを提供します。
"""

import logging
import time
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models.api import (
    StatusResponse,
    ModelInfo, ModelListResponse, ModelSwitchRequest,
    ChatHistoryRequest, ChatHistoryResponse,
    SystemStatusResponse, ErrorResponse,
    create_success_response, create_error_response, create_api_error_response
)
from ..models.chat import QueryRequest, QueryResponse
from ..models.chat import Message, create_user_message, create_assistant_message
from ..core.rag_engine import RAGEngine
from ..core.llm_manager import LLMManager
from ..core.chat_manager import ChatManager
from ..core.system_monitor import SystemMonitor
from ..core.concurrency_manager import ConcurrencyManager

logger = logging.getLogger(__name__)

# ルーターの作成
query_router = APIRouter()
model_router = APIRouter()
chat_router = APIRouter()
system_router = APIRouter()
health_router = APIRouter()


# 依存性注入用の関数（後でapp.pyから注入される）
def get_rag_engine() -> RAGEngine:
    """RAGエンジンを取得"""
    from .app import get_app_state
    return get_app_state().rag_engine


def get_llm_manager() -> LLMManager:
    """LLMマネージャーを取得"""
    from .app import get_app_state
    return get_app_state().llm_manager


def get_chat_manager() -> ChatManager:
    """チャットマネージャーを取得"""
    from .app import get_app_state
    return get_app_state().chat_manager


def get_system_monitor() -> SystemMonitor:
    """システムモニターを取得"""
    from .app import get_app_state
    return get_app_state().system_monitor


def get_concurrency_manager() -> ConcurrencyManager:
    """同時アクセス管理を取得"""
    from .app import get_app_state
    return get_app_state().concurrency_manager


# 質問応答API
@query_router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    chat_manager: ChatManager = Depends(get_chat_manager),
    concurrency_manager: ConcurrencyManager = Depends(get_concurrency_manager)
) -> QueryResponse:
    """
    文書に対する質問応答を実行
    
    Args:
        request: 質問リクエスト
        background_tasks: バックグラウンドタスク
        rag_engine: RAGエンジン
        chat_manager: チャットマネージャー
        concurrency_manager: 同時アクセス管理
        
    Returns:
        質問応答レスポンス
        
    Raises:
        HTTPException: 処理エラー時
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query for session {request.session_id}: {request.question[:100]}...")
        
        # 同時アクセス制御付きでクエリを実行
        if concurrency_manager:
            result = await concurrency_manager.execute_with_concurrency_control(
                _process_query_internal,
                request,
                rag_engine,
                chat_manager,
                request_id=f"query-{request.session_id}-{int(time.time() * 1000)}",
                client_id=request.session_id
            )
        else:
            # フォールバック: 直接実行
            result = await _process_query_internal(request, rag_engine, chat_manager)
        
        # 処理時間を計算
        processing_time = time.time() - start_time
        
        # 使用されたモデル名を取得
        model_used = getattr(result, "model_used", request.model_name or "default")
        
        # レスポンスを作成
        response = QueryResponse(
            response=result.answer,
            source_documents=getattr(result, "sources", []),
            processing_time=processing_time,
            model_used=model_used,
            session_id=request.session_id,
            metadata=getattr(result, "metadata", {})
        )
        
        # バックグラウンドで会話履歴を保存
        background_tasks.add_task(
            save_conversation_history,
            chat_manager,
            request.session_id,
            request.question,
            response.response,
            response.source_documents
        )
        
        logger.info(f"Query completed for session {request.session_id} in {processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in query: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def _process_query_internal(
    request: QueryRequest,
    rag_engine: RAGEngine,
    chat_manager: ChatManager
) -> Any:
    """
    内部クエリ処理関数
    
    Args:
        request: 質問リクエスト
        rag_engine: RAGエンジン
        chat_manager: チャットマネージャー
        
    Returns:
        RAGエンジンの実行結果
    """
    # 会話履歴を取得（必要に応じて）
    context_messages = []
    if request.include_history:
        history = chat_manager.get_chat_history(request.session_id, limit=5)
        context_messages = history  # Messageオブジェクトのリストとして渡す
    
    # RAGエンジンで質問応答を実行
    return rag_engine.query(
        question=request.question,
        chat_history=context_messages,
        model_name=request.model_name or "llama3.2:1b"  # デフォルトモデルを指定
    )


async def save_conversation_history(
    chat_manager: ChatManager,
    session_id: str,
    question: str,
    answer: str,
    sources: List[Any]
) -> None:
    """
    会話履歴をバックグラウンドで保存
    
    Args:
        chat_manager: チャットマネージャー
        session_id: セッションID
        question: 質問
        answer: 回答
        sources: 出典情報
    """
    try:
        # ユーザーメッセージを保存
        user_message = create_user_message(session_id, question)
        chat_manager.save_message(session_id, user_message)
        
        # アシスタントメッセージを保存
        assistant_message = create_assistant_message(
            session_id, 
            answer, 
            [source.url if hasattr(source, 'url') else str(source) for source in sources]
        )
        assistant_message.metadata = {"sources": [source.to_dict() if hasattr(source, 'to_dict') else source for source in sources]}
        chat_manager.save_message(session_id, assistant_message)
        
    except Exception as e:
        logger.error(f"Failed to save conversation history: {str(e)}")


# モデル管理API
@model_router.get("/models", response_model=ModelListResponse)
async def list_models(
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> ModelListResponse:
    """
    利用可能なモデル一覧を取得
    
    Args:
        llm_manager: LLMマネージャー
        
    Returns:
        モデル一覧レスポンス
    """
    try:
        # 利用可能なモデルを取得
        available_models = llm_manager.get_available_models()
        current_model = llm_manager.get_current_model()
        
        # ModelInfoオブジェクトのリストを作成
        models = []
        for model_info in available_models:
            models.append(ModelInfo(
                name=model_info.name,
                display_name=model_info.name,
                description=f"Size: {model_info.size}, Modified: {model_info.modified_at}",
                available=True,
                parameters=model_info.details
            ))
        
        return ModelListResponse(
            models=models,
            current_model=current_model or "not_set"
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")


@model_router.post("/models/switch", response_model=StatusResponse)
async def switch_model(
    request: ModelSwitchRequest,
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> StatusResponse:
    """
    使用するモデルを切り替え
    
    Args:
        request: モデル切り替えリクエスト
        llm_manager: LLMマネージャー
        
    Returns:
        ステータスレスポンス
    """
    try:
        # モデルの切り替えを実行
        success = llm_manager.switch_model(request.model_name)
        
        if success:
            return create_success_response(
                f"Model switched to {request.model_name}",
                {"model_name": request.model_name}
            )
        else:
            return create_error_response(
                f"Failed to switch to model {request.model_name}",
                {"model_name": request.model_name}
            )
            
    except ValueError as e:
        logger.warning(f"Invalid model switch request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error switching model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to switch model")


@model_router.get("/models/current", response_model=Dict[str, str])
async def get_current_model(
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> Dict[str, str]:
    """
    現在使用中のモデルを取得
    
    Args:
        llm_manager: LLMマネージャー
        
    Returns:
        現在のモデル情報
    """
    try:
        current_model = llm_manager.get_current_model()
        return {"current_model": current_model}
        
    except Exception as e:
        logger.error(f"Error getting current model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get current model")


# チャット履歴API
@chat_router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 10,
    include_sources: bool = True,
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> ChatHistoryResponse:
    """
    チャット履歴を取得
    
    Args:
        session_id: セッションID
        limit: 取得するメッセージ数の上限
        include_sources: 出典情報を含めるかどうか
        chat_manager: チャットマネージャー
        
    Returns:
        チャット履歴レスポンス
    """
    try:
        # 履歴を取得
        messages = chat_manager.get_chat_history(session_id, limit=limit)
        
        # セッション情報を取得してメッセージ数を確認
        session_info = chat_manager.get_session_info(session_id)
        total_count = session_info.message_count if session_info else 0
        
        # メッセージを辞書形式に変換
        message_dicts = []
        for msg in messages:
            msg_dict = msg.to_dict()
            if not include_sources and "sources" in msg_dict.get("metadata", {}):
                del msg_dict["metadata"]["sources"]
            message_dicts.append(msg_dict)
        
        return ChatHistoryResponse(
            session_id=session_id,
            messages=message_dicts,
            total_count=total_count,
            has_more=(total_count > len(messages))
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")


@chat_router.delete("/chat/history/{session_id}", response_model=StatusResponse)
async def clear_chat_history(
    session_id: str,
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> StatusResponse:
    """
    チャット履歴をクリア
    
    Args:
        session_id: セッションID
        chat_manager: チャットマネージャー
        
    Returns:
        ステータスレスポンス
    """
    try:
        chat_manager.clear_history(session_id)
        return create_success_response(
            f"Chat history cleared for session {session_id}",
            {"session_id": session_id}
        )
        
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear chat history")


@chat_router.get("/chat/sessions", response_model=Dict[str, Any])
async def list_chat_sessions(
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> Dict[str, Any]:
    """
    アクティブなチャットセッション一覧を取得
    
    Args:
        chat_manager: チャットマネージャー
        
    Returns:
        セッション一覧
    """
    try:
        sessions = chat_manager.list_sessions()
        return {
            "sessions": sessions,
            "total_count": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"Error listing chat sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list chat sessions")


# システム管理API
@system_router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(
    system_monitor: SystemMonitor = Depends(get_system_monitor),
    llm_manager: LLMManager = Depends(get_llm_manager),
    chat_manager: ChatManager = Depends(get_chat_manager),
    concurrency_manager: ConcurrencyManager = Depends(get_concurrency_manager)
) -> SystemStatusResponse:
    """
    システムステータスを取得
    
    Args:
        system_monitor: システムモニター
        llm_manager: LLMマネージャー
        chat_manager: チャットマネージャー
        concurrency_manager: 同時アクセス管理
        
    Returns:
        システムステータスレスポンス
    """
    try:
        # システム状態を取得
        status = system_monitor.get_system_status()
        
        # アクティブセッション数を取得
        try:
            active_sessions = len(chat_manager.list_sessions())
        except Exception as e:
            logger.warning(f"Failed to get active sessions: {e}")
            active_sessions = 0
        
        # 現在のモデルを取得
        try:
            current_model = llm_manager.get_current_model() or "not_set"
        except Exception as e:
            logger.warning(f"Failed to get current model: {e}")
            current_model = "unknown"
        
        # 同時アクセスメトリクスを取得
        concurrency_metrics = {}
        try:
            if concurrency_manager:
                concurrency_metrics = concurrency_manager.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get concurrency metrics: {e}")
        
        # パフォーマンス統計を取得
        performance_stats = {}
        try:
            performance_stats = system_monitor.get_performance_stats(hours=24)
        except Exception as e:
            logger.warning(f"Failed to get performance stats: {e}")
        
        # システム状態の属性を安全に取得
        try:
            uptime_seconds = getattr(status, 'uptime_seconds', 0.0)
            memory_usage_mb = getattr(status, 'memory_usage_mb', 0.0)
            disk_usage_mb = getattr(status, 'disk_usage_mb', 0.0)
            
            # 属性名が異なる場合の対応
            if memory_usage_mb == 0.0:
                memory_total_gb = getattr(status, 'memory_total_gb', 0.0)
                memory_usage_percent = getattr(status, 'memory_usage_percent', 0.0)
                if memory_total_gb > 0:
                    memory_usage_mb = memory_total_gb * 1024 * (memory_usage_percent / 100)
            
            if disk_usage_mb == 0.0:
                disk_total_gb = getattr(status, 'disk_total_gb', 0.0)
                disk_usage_percent = getattr(status, 'disk_usage_percent', 0.0)
                if disk_total_gb > 0:
                    disk_usage_mb = disk_total_gb * 1024 * (disk_usage_percent / 100)
                    
        except Exception as e:
            logger.warning(f"Failed to parse system status attributes: {e}")
            uptime_seconds = 0.0
            memory_usage_mb = 0.0
            disk_usage_mb = 0.0
        
        return SystemStatusResponse(
            status="healthy",
            version="1.0.0",
            uptime=uptime_seconds,
            components={
                "llm_manager": True,
                "document_processor": True,
                "rag_engine": True,
                "chat_manager": True,
                "concurrency_manager": True
            },
            system_metrics={
                "memory_usage_mb": memory_usage_mb,
                "disk_usage_mb": disk_usage_mb,
                "active_sessions": active_sessions
            },
            performance_metrics={
                "total_queries": concurrency_metrics.get("total_requests", 0),
                "current_model": current_model,
                "concurrency_metrics": concurrency_metrics,
                "performance_stats": performance_stats
            },
            error_statistics={}
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get system status")


@system_router.post("/system/health-check", response_model=Dict[str, Any])
async def health_check(
    llm_manager: LLMManager = Depends(get_llm_manager)
) -> Dict[str, Any]:
    """
    詳細なヘルスチェックを実行
    
    Args:
        llm_manager: LLMマネージャー
        
    Returns:
        ヘルスチェック結果
    """
    try:
        # LLMの健全性をチェック
        llm_health = llm_manager.check_model_health()
        
        # 各コンポーネントの状態をチェック
        health_status = {
            "overall": "healthy",
            "components": {
                "llm_manager": "healthy" if llm_health else "unhealthy",
                "database": "healthy",  # TODO: 実際のDBヘルスチェック
                "storage": "healthy"    # TODO: 実際のストレージチェック
            },
            "details": {
                "llm_health": llm_health
            }
        }
        
        # 全体的な健全性を判定
        if not all(status == "healthy" for status in health_status["components"].values()):
            health_status["overall"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}", exc_info=True)
        return {
            "overall": "unhealthy",
            "error": str(e)
        }


# ヘルスチェックAPI
@health_router.get("/health")
async def health_check_simple() -> Dict[str, Any]:
    """
    シンプルなヘルスチェック（依存関係なし）
    
    Returns:
        基本的なヘルスチェック結果
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "genkai-rag-system",
        "version": "1.0.0"
    }


@health_router.get("/health/detailed")
async def health_check_detailed(
    system_monitor: SystemMonitor = Depends(get_system_monitor),
    llm_manager: LLMManager = Depends(get_llm_manager),
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> Dict[str, Any]:
    """
    詳細なヘルスチェック
    
    Args:
        system_monitor: システムモニター
        llm_manager: LLMマネージャー
        chat_manager: チャットマネージャー
        
    Returns:
        詳細なヘルスチェック結果
    """
    try:
        # システムリソースをチェック
        system_status = system_monitor.get_system_status()
        
        # LLMの健全性をチェック
        llm_health = llm_manager.check_model_health()
        
        # チャットマネージャーの状態をチェック
        active_sessions = len(chat_manager.list_sessions())
        
        # 各コンポーネントの状態
        components = {
            "system_monitor": "healthy",
            "llm_manager": "healthy" if llm_health else "unhealthy",
            "chat_manager": "healthy",
            "database": "healthy"  # TODO: 実際のDBヘルスチェック
        }
        
        # リソース使用率をチェック
        memory_usage_percent = 0
        disk_usage_percent = 0
        
        try:
            if hasattr(system_status, 'memory_usage_percent') and hasattr(system_status, 'memory_total_gb'):
                if system_status.memory_total_gb and system_status.memory_total_gb > 0:
                    memory_usage_percent = system_status.memory_usage_percent
            
            if hasattr(system_status, 'disk_usage_percent') and hasattr(system_status, 'disk_total_gb'):
                if system_status.disk_total_gb and system_status.disk_total_gb > 0:
                    disk_usage_percent = system_status.disk_usage_percent
        except (TypeError, AttributeError):
            # モック環境での計算エラーを回避
            memory_usage_percent = 0
            disk_usage_percent = 0
        
        # 警告レベルをチェック
        warnings = []
        if memory_usage_percent > 80:
            warnings.append(f"High memory usage: {memory_usage_percent:.1f}%")
            components["system_monitor"] = "warning"
        
        if disk_usage_percent > 90:
            warnings.append(f"High disk usage: {disk_usage_percent:.1f}%")
            components["system_monitor"] = "warning"
        
        # 全体的な健全性を判定
        overall_status = "healthy"
        if any(status == "unhealthy" for status in components.values()):
            overall_status = "unhealthy"
        elif any(status == "warning" for status in components.values()):
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "service": "genkai-rag-system",
            "version": "1.0.0",
            "components": components,
            "metrics": {
                "memory_usage_percent": memory_usage_percent,
                "disk_usage_percent": disk_usage_percent,
                "active_sessions": active_sessions,
                "uptime_seconds": system_status.uptime_seconds
            },
            "warnings": warnings
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "service": "genkai-rag-system",
            "version": "1.0.0",
            "error": str(e)
        }


@system_router.get("/system/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    operation_type: Optional[str] = None,
    hours: int = 24,
    system_monitor: SystemMonitor = Depends(get_system_monitor)
) -> Dict[str, Any]:
    """
    パフォーマンスメトリクスを取得
    
    Args:
        operation_type: 特定の操作タイプ（Noneの場合は全タイプ）
        hours: 統計期間（時間）
        system_monitor: システムモニター
        
    Returns:
        パフォーマンスメトリクス
    """
    try:
        # パフォーマンス統計を取得
        stats = system_monitor.get_performance_stats(operation_type=operation_type, hours=hours)
        
        # レスポンス時間履歴を取得
        history = system_monitor.get_response_time_history(operation_type=operation_type, hours=hours)
        
        return {
            "performance_stats": {op_type: stat.to_dict() for op_type, stat in stats.items()},
            "response_time_history": [metric.to_dict() for metric in history[-100:]],  # 最新100件
            "total_metrics": len(history),
            "time_range_hours": hours,
            "operation_type_filter": operation_type
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@system_router.delete("/system/performance", response_model=StatusResponse)
async def clear_performance_metrics(
    operation_type: Optional[str] = None,
    system_monitor: SystemMonitor = Depends(get_system_monitor)
) -> StatusResponse:
    """
    パフォーマンスメトリクスをクリア
    
    Args:
        operation_type: 特定の操作タイプ（Noneの場合は全タイプ）
        system_monitor: システムモニター
        
    Returns:
        ステータスレスポンス
    """
    try:
        cleared_count = system_monitor.clear_performance_metrics(operation_type=operation_type)
        
        return create_success_response(
            f"Cleared {cleared_count} performance metrics",
            {"cleared_count": cleared_count, "operation_type": operation_type}
        )
        
    except Exception as e:
        logger.error(f"Error clearing performance metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear performance metrics")