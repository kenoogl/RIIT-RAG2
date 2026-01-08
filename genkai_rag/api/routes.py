"""
FastAPI ルーター実装

このモジュールは、FastAPIアプリケーション用のAPIルーターを提供します。
"""

import logging
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models.api import (
    QueryRequest, QueryResponse, StatusResponse,
    ModelInfo, ModelListResponse, ModelSwitchRequest,
    ChatHistoryRequest, ChatHistoryResponse,
    SystemStatusResponse, ErrorResponse,
    create_success_response, create_error_response, create_api_error_response
)
from ..models.chat import Message, create_user_message, create_assistant_message
from ..core.rag_engine import RAGEngine
from ..core.llm_manager import LLMManager
from ..core.chat_manager import ChatManager
from ..core.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

# ルーターの作成
query_router = APIRouter()
model_router = APIRouter()
chat_router = APIRouter()
system_router = APIRouter()


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


# 質問応答API
@query_router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    rag_engine: RAGEngine = Depends(get_rag_engine),
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> QueryResponse:
    """
    文書に対する質問応答を実行
    
    Args:
        request: 質問リクエスト
        background_tasks: バックグラウンドタスク
        rag_engine: RAGエンジン
        chat_manager: チャットマネージャー
        
    Returns:
        質問応答レスポンス
        
    Raises:
        HTTPException: 処理エラー時
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query for session {request.session_id}: {request.question[:100]}...")
        
        # 会話履歴を取得（必要に応じて）
        context_messages = []
        if request.include_history:
            history = chat_manager.get_chat_history(request.session_id, limit=5)
            context_messages = [msg.to_dict() for msg in history]
        
        # RAGエンジンで質問応答を実行
        result = await rag_engine.query(
            question=request.question,
            model_name=request.model_name,
            max_sources=request.max_sources,
            context_messages=context_messages
        )
        
        # 処理時間を計算
        processing_time = time.time() - start_time
        
        # 使用されたモデル名を取得
        model_used = result.get("model_used", request.model_name or "default")
        
        # レスポンスを作成
        response = QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            processing_time=processing_time,
            model_used=model_used,
            session_id=request.session_id,
            metadata=result.get("metadata", {})
        )
        
        # バックグラウンドで会話履歴を保存
        background_tasks.add_task(
            save_conversation_history,
            chat_manager,
            request.session_id,
            request.question,
            response.answer,
            response.sources
        )
        
        logger.info(f"Query completed for session {request.session_id} in {processing_time:.3f}s")
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error in query: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


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
        chat_manager.save_message(request.session_id, user_message)
        
        # アシスタントメッセージを保存
        assistant_message = create_assistant_message(
            session_id, 
            answer, 
            [source.url for source in sources]
        )
        assistant_message.metadata = {"sources": [source.to_dict() for source in sources]}
        chat_manager.save_message(request.session_id, assistant_message)
        
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
        available_models = await llm_manager.list_available_models()
        current_model = llm_manager.get_current_model()
        
        # ModelInfoオブジェクトのリストを作成
        models = []
        for model_name, model_info in available_models.items():
            models.append(ModelInfo(
                name=model_name,
                display_name=model_info.get("display_name", model_name),
                description=model_info.get("description"),
                is_available=model_info.get("is_available", True),
                is_default=(model_name == current_model),
                parameters=model_info.get("parameters", {})
            ))
        
        return ModelListResponse(
            models=models,
            current_model=current_model
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
        success = await llm_manager.switch_model(request.model_name, force=request.force)
        
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
    chat_manager: ChatManager = Depends(get_chat_manager)
) -> SystemStatusResponse:
    """
    システムステータスを取得
    
    Args:
        system_monitor: システムモニター
        llm_manager: LLMマネージャー
        chat_manager: チャットマネージャー
        
    Returns:
        システムステータスレスポンス
    """
    try:
        # システム状態を取得
        status = system_monitor.get_system_status()
        
        # アクティブセッション数を取得
        active_sessions = len(chat_manager.list_sessions())
        
        # 現在のモデルを取得
        current_model = llm_manager.get_current_model()
        
        return SystemStatusResponse(
            status="healthy",  # TODO: 実際のヘルスチェックロジックを実装
            version="1.0.0",
            uptime_seconds=status.uptime_seconds,
            memory_usage_mb=status.memory_usage_mb,
            disk_usage_mb=status.disk_usage_mb,
            active_sessions=active_sessions,
            total_queries=0,  # TODO: クエリカウンターを実装
            current_model=current_model
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
        llm_health = await llm_manager.check_model_health()
        
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