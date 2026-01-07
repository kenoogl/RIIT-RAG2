"""API データモデルのテスト"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from genkai_rag.models.api import (
    QueryRequest, QueryResponse, StatusResponse,
    ModelInfo, ModelListResponse, ModelSwitchRequest,
    ChatHistoryRequest, ChatHistoryResponse,
    SystemStatusResponse, ErrorResponse,
    create_success_response, create_error_response, create_api_error_response
)
from genkai_rag.models.document import DocumentSource


class TestQueryRequest:
    """QueryRequestモデルのテスト"""
    
    def test_valid_query_request(self):
        """有効な質問リクエストのテスト"""
        request = QueryRequest(
            question="玄界システムの使い方を教えてください",
            session_id="test-session-123"
        )
        
        assert request.question == "玄界システムの使い方を教えてください"
        assert request.session_id == "test-session-123"
        assert request.model_name is None
        assert request.max_sources == 5
        assert request.include_history is True
    
    def test_query_request_with_optional_fields(self):
        """オプションフィールド付きの質問リクエストのテスト"""
        request = QueryRequest(
            question="バッチジョブの投入方法は？",
            session_id="session-456",
            model_name="llama3.2:3b",
            max_sources=10,
            include_history=False
        )
        
        assert request.model_name == "llama3.2:3b"
        assert request.max_sources == 10
        assert request.include_history is False
    
    def test_empty_question_validation(self):
        """空の質問のバリデーションテスト"""
        with pytest.raises(ValidationError):
            QueryRequest(
                question="",
                session_id="test-session"
            )
        
        with pytest.raises(ValidationError):
            QueryRequest(
                question="   ",  # 空白のみ
                session_id="test-session"
            )
    
    def test_empty_session_id_validation(self):
        """空のセッションIDのバリデーションテスト"""
        with pytest.raises(ValidationError):
            QueryRequest(
                question="テスト質問",
                session_id=""
            )
    
    def test_question_length_validation(self):
        """質問の長さのバリデーションテスト"""
        # 長すぎる質問
        long_question = "あ" * 2001
        with pytest.raises(ValidationError):
            QueryRequest(
                question=long_question,
                session_id="test-session"
            )
    
    def test_max_sources_validation(self):
        """max_sourcesのバリデーションテスト"""
        # 範囲外の値
        with pytest.raises(ValidationError):
            QueryRequest(
                question="テスト質問",
                session_id="test-session",
                max_sources=0  # 最小値未満
            )
        
        with pytest.raises(ValidationError):
            QueryRequest(
                question="テスト質問",
                session_id="test-session",
                max_sources=21  # 最大値超過
            )


class TestQueryResponse:
    """QueryResponseモデルのテスト"""
    
    def test_valid_query_response(self):
        """有効な質問回答レスポンスのテスト"""
        sources = [
            DocumentSource(
                title="玄界システム利用ガイド",
                url="https://example.com/guide",
                section="第1章",
                relevance_score=0.95
            )
        ]
        
        response = QueryResponse(
            answer="玄界システムは九州大学のスーパーコンピュータです。",
            sources=sources,
            processing_time=1.5,
            model_used="llama3.2:3b",
            session_id="test-session"
        )
        
        assert response.answer == "玄界システムは九州大学のスーパーコンピュータです。"
        assert len(response.sources) == 1
        assert response.processing_time == 1.5
        assert response.model_used == "llama3.2:3b"
        assert response.session_id == "test-session"
        assert isinstance(response.timestamp, datetime)
    
    def test_negative_processing_time_validation(self):
        """負の処理時間のバリデーションテスト"""
        with pytest.raises(ValidationError):
            QueryResponse(
                answer="テスト回答",
                processing_time=-1.0,  # 負の値
                model_used="test-model",
                session_id="test-session"
            )


class TestStatusResponse:
    """StatusResponseモデルのテスト"""
    
    def test_success_status_response(self):
        """成功ステータスレスポンスのテスト"""
        response = StatusResponse(
            success=True,
            message="操作が正常に完了しました"
        )
        
        assert response.success is True
        assert response.message == "操作が正常に完了しました"
        assert response.data is None
        assert isinstance(response.timestamp, datetime)
    
    def test_error_status_response(self):
        """エラーステータスレスポンスのテスト"""
        response = StatusResponse(
            success=False,
            message="エラーが発生しました",
            data={"error_code": "E001"}
        )
        
        assert response.success is False
        assert response.message == "エラーが発生しました"
        assert response.data == {"error_code": "E001"}


class TestModelInfo:
    """ModelInfoモデルのテスト"""
    
    def test_model_info_creation(self):
        """モデル情報の作成テスト"""
        model = ModelInfo(
            name="llama3.2:3b",
            display_name="LLaMA 3.2 3B",
            description="軽量な日本語対応モデル",
            is_available=True,
            is_default=True,
            parameters={"temperature": 0.7, "max_tokens": 2048}
        )
        
        assert model.name == "llama3.2:3b"
        assert model.display_name == "LLaMA 3.2 3B"
        assert model.is_available is True
        assert model.is_default is True
        assert model.parameters["temperature"] == 0.7


class TestModelSwitchRequest:
    """ModelSwitchRequestモデルのテスト"""
    
    def test_valid_model_switch_request(self):
        """有効なモデル切り替えリクエストのテスト"""
        request = ModelSwitchRequest(
            model_name="gemma2:2b",
            force=True
        )
        
        assert request.model_name == "gemma2:2b"
        assert request.force is True
    
    def test_empty_model_name_validation(self):
        """空のモデル名のバリデーションテスト"""
        with pytest.raises(ValidationError):
            ModelSwitchRequest(model_name="")
        
        with pytest.raises(ValidationError):
            ModelSwitchRequest(model_name="   ")


class TestChatHistoryRequest:
    """ChatHistoryRequestモデルのテスト"""
    
    def test_valid_chat_history_request(self):
        """有効なチャット履歴リクエストのテスト"""
        request = ChatHistoryRequest(
            session_id="test-session",
            limit=20,
            include_sources=False
        )
        
        assert request.session_id == "test-session"
        assert request.limit == 20
        assert request.include_sources is False
    
    def test_limit_validation(self):
        """limitのバリデーションテスト"""
        # 範囲外の値
        with pytest.raises(ValidationError):
            ChatHistoryRequest(
                session_id="test-session",
                limit=0  # 最小値未満
            )
        
        with pytest.raises(ValidationError):
            ChatHistoryRequest(
                session_id="test-session",
                limit=101  # 最大値超過
            )


class TestSystemStatusResponse:
    """SystemStatusResponseモデルのテスト"""
    
    def test_system_status_response(self):
        """システムステータスレスポンスのテスト"""
        response = SystemStatusResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=3600.0,
            memory_usage_mb=512.0,
            disk_usage_mb=1024.0,
            active_sessions=5,
            total_queries=100,
            current_model="llama3.2:3b"
        )
        
        assert response.status == "healthy"
        assert response.version == "0.1.0"
        assert response.uptime_seconds == 3600.0
        assert response.active_sessions == 5
        assert response.total_queries == 100
    
    def test_negative_values_validation(self):
        """負の値のバリデーションテスト"""
        with pytest.raises(ValidationError):
            SystemStatusResponse(
                status="healthy",
                version="0.1.0",
                uptime_seconds=-1.0,  # 負の値
                memory_usage_mb=512.0,
                disk_usage_mb=1024.0,
                active_sessions=5,
                total_queries=100,
                current_model="test-model"
            )


class TestErrorResponse:
    """ErrorResponseモデルのテスト"""
    
    def test_error_response_creation(self):
        """エラーレスポンスの作成テスト"""
        response = ErrorResponse(
            error="ValidationError",
            message="入力データが無効です",
            details={"field": "question", "issue": "empty"},
            request_id="req-123"
        )
        
        assert response.error == "ValidationError"
        assert response.message == "入力データが無効です"
        assert response.details == {"field": "question", "issue": "empty"}
        assert response.request_id == "req-123"
        assert isinstance(response.timestamp, datetime)


class TestHelperFunctions:
    """ヘルパー関数のテスト"""
    
    def test_create_success_response(self):
        """成功レスポンス作成のテスト"""
        response = create_success_response(
            "操作が完了しました",
            {"result": "success"}
        )
        
        assert response.success is True
        assert response.message == "操作が完了しました"
        assert response.data == {"result": "success"}
    
    def test_create_error_response(self):
        """エラーレスポンス作成のテスト"""
        response = create_error_response(
            "エラーが発生しました",
            {"error_code": "E001"}
        )
        
        assert response.success is False
        assert response.message == "エラーが発生しました"
        assert response.data == {"error_code": "E001"}
    
    def test_create_api_error_response(self):
        """APIエラーレスポンス作成のテスト"""
        response = create_api_error_response(
            "ValidationError",
            "入力データが無効です",
            {"field": "question"},
            "req-456"
        )
        
        assert response.error == "ValidationError"
        assert response.message == "入力データが無効です"
        assert response.details == {"field": "question"}
        assert response.request_id == "req-456"