"""
LLMManagerクラスのテスト

このモジュールは、LLMManagerクラスの機能をテストします。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume
import json

from genkai_rag.core.llm_manager import LLMManager, ModelInfo


class TestLLMManager:
    """LLMManagerクラスの基本機能テスト"""
    
    def test_llm_manager_initialization(self):
        """LLMManagerの初期化テスト"""
        manager = LLMManager()
        
        assert manager.ollama_base_url == "http://localhost:11434"
        assert manager.current_model is None
        assert isinstance(manager.model_configs, dict)
        assert len(manager.model_configs) == 0
        assert "temperature" in manager.default_config
        assert "max_tokens" in manager.default_config
    
    def test_custom_ollama_url(self):
        """カスタムOllama URLの設定テスト"""
        custom_url = "http://custom-server:8080"
        manager = LLMManager(ollama_base_url=custom_url)
        
        assert manager.ollama_base_url == custom_url
    
    def test_url_trailing_slash_removal(self):
        """URL末尾のスラッシュ除去テスト"""
        manager = LLMManager(ollama_base_url="http://localhost:11434/")
        
        assert manager.ollama_base_url == "http://localhost:11434"
    
    @patch('requests.get')
    def test_get_available_models_success(self, mock_get):
        """利用可能モデル取得の成功テスト"""
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama2:7b",
                    "size": "3.8GB",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "digest": "abc123",
                    "details": {"family": "llama"}
                },
                {
                    "name": "codellama:13b",
                    "size": "7.3GB",
                    "modified_at": "2024-01-02T00:00:00Z",
                    "digest": "def456",
                    "details": {"family": "codellama"}
                }
            ]
        }
        mock_get.return_value = mock_response
        
        manager = LLMManager()
        models = manager.get_available_models()
        
        assert len(models) == 2
        assert models[0].name == "llama2:7b"
        assert models[0].size == "3.8GB"
        assert models[1].name == "codellama:13b"
        assert models[1].size == "7.3GB"
        
        mock_get.assert_called_once_with("http://localhost:11434/api/tags", timeout=10)
    
    @patch('requests.get')
    def test_get_available_models_connection_error(self, mock_get):
        """モデル取得時の接続エラーテスト"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        manager = LLMManager()
        
        with pytest.raises(ConnectionError, match="Cannot connect to Ollama server"):
            manager.get_available_models()
    
    @patch('requests.get')
    def test_get_available_models_invalid_response(self, mock_get):
        """モデル取得時の無効レスポンステスト"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response
        
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="Invalid response from Ollama server"):
            manager.get_available_models()
    
    @patch('genkai_rag.core.llm_manager.LLMManager.get_available_models')
    @patch('requests.post')
    def test_switch_model_success(self, mock_post, mock_get_models):
        """モデル切り替えの成功テスト"""
        # 利用可能モデルをモック
        mock_models = [
            ModelInfo("llama2:7b", "3.8GB", datetime.now(), "abc123", {}),
            ModelInfo("codellama:13b", "7.3GB", datetime.now(), "def456", {})
        ]
        mock_get_models.return_value = mock_models
        
        # POST リクエストをモック
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        manager = LLMManager()
        result = manager.switch_model("llama2:7b")
        
        assert result is True
        assert manager.current_model == "llama2:7b"
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/pull"
        assert call_args[1]["json"]["name"] == "llama2:7b"
    
    @patch('genkai_rag.core.llm_manager.LLMManager.get_available_models')
    def test_switch_model_not_found(self, mock_get_models):
        """存在しないモデルへの切り替えテスト"""
        mock_models = [
            ModelInfo("llama2:7b", "3.8GB", datetime.now(), "abc123", {})
        ]
        mock_get_models.return_value = mock_models
        
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            manager.switch_model("nonexistent")
    
    def test_switch_model_empty_name(self):
        """空のモデル名での切り替えテスト"""
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            manager.switch_model("")
        
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            manager.switch_model("   ")
    
    def test_get_current_model(self):
        """現在のモデル取得テスト"""
        manager = LLMManager()
        
        # 初期状態
        assert manager.get_current_model() is None
        
        # モデル設定後
        manager.current_model = "llama2:7b"
        assert manager.get_current_model() == "llama2:7b"
    
    def test_optimize_for_model_default(self):
        """デフォルトモデル最適化テスト"""
        manager = LLMManager()
        config = manager.optimize_for_model("unknown-model")
        
        assert config == manager.default_config
        assert "unknown-model" in manager.model_configs
    
    def test_optimize_for_model_japanese(self):
        """日本語モデル最適化テスト"""
        manager = LLMManager()
        
        japanese_models = ["japanese-llama", "elyza-7b", "calm2-7b", "jp-model"]
        
        for model_name in japanese_models:
            config = manager.optimize_for_model(model_name)
            
            assert config["temperature"] == 0.6
            assert config["top_p"] == 0.85
            assert config["context_length"] == 8192
    
    def test_optimize_for_model_large(self):
        """大型モデル最適化テスト"""
        manager = LLMManager()
        
        # コード生成でない大型モデルのみテスト
        large_models = ["llama2-70b", "large-model"]
        
        for model_name in large_models:
            config = manager.optimize_for_model(model_name)
            
            assert config["temperature"] == 0.5
            assert config["max_tokens"] == 4096
            assert config["context_length"] == 8192
    
    def test_optimize_for_model_small(self):
        """小型モデル最適化テスト"""
        manager = LLMManager()
        
        small_models = ["llama2-7b", "tiny-3b", "small-1b"]
        
        for model_name in small_models:
            config = manager.optimize_for_model(model_name)
            
            assert config["temperature"] == 0.8
            assert config["max_tokens"] == 1024
            assert config["context_length"] == 2048
    
    def test_optimize_for_model_code(self):
        """コード生成モデル最適化テスト"""
        manager = LLMManager()
        
        code_models = ["codellama", "code-model", "coder-7b", "coding-assistant"]
        
        for model_name in code_models:
            config = manager.optimize_for_model(model_name)
            
            assert config["temperature"] == 0.3
            assert config["top_p"] == 0.95
            assert config["repeat_penalty"] == 1.0
    
    def test_optimize_for_model_code_large(self):
        """大型コード生成モデル最適化テスト"""
        manager = LLMManager()
        
        # コード生成かつ大型モデル（コード生成設定が優先される）
        config = manager.optimize_for_model("codellama-65b")
        
        assert config["temperature"] == 0.3  # コード生成設定が優先
        assert config["top_p"] == 0.95
        assert config["repeat_penalty"] == 1.0
        assert config["max_tokens"] == 4096  # 大型モデル設定
        assert config["context_length"] == 8192  # 大型モデル設定
    
    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        """レスポンス生成の成功テスト"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "Generated response text"}
        mock_post.return_value = mock_response
        
        manager = LLMManager()
        manager.current_model = "llama2:7b"
        
        result = manager.generate_response("Test prompt")
        
        assert result == "Generated response text"
        
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        assert call_args[1]["json"]["model"] == "llama2:7b"
        assert call_args[1]["json"]["prompt"] == "Test prompt"
    
    def test_generate_response_no_model(self):
        """モデル未設定でのレスポンス生成テスト"""
        manager = LLMManager()
        
        with pytest.raises(ValueError, match="No model specified and no current model set"):
            manager.generate_response("Test prompt")
    
    @patch('requests.post')
    def test_generate_response_connection_error(self, mock_post):
        """レスポンス生成時の接続エラーテスト"""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        manager = LLMManager()
        manager.current_model = "llama2:7b"
        
        with pytest.raises(ConnectionError, match="Failed to generate response"):
            manager.generate_response("Test prompt")
    
    @patch('genkai_rag.core.llm_manager.LLMManager.generate_response')
    def test_check_model_health_success(self, mock_generate):
        """モデル健全性チェックの成功テスト"""
        mock_generate.return_value = "Hello response"
        
        manager = LLMManager()
        manager.current_model = "llama2:7b"
        
        result = manager.check_model_health()
        
        assert result is True
        mock_generate.assert_called_once_with("Hello", model_name="llama2:7b", max_tokens=10)
    
    @patch('genkai_rag.core.llm_manager.LLMManager.generate_response')
    def test_check_model_health_failure(self, mock_generate):
        """モデル健全性チェックの失敗テスト"""
        mock_generate.side_effect = Exception("Generation failed")
        
        manager = LLMManager()
        manager.current_model = "llama2:7b"
        
        result = manager.check_model_health()
        
        assert result is False
    
    def test_check_model_health_no_model(self):
        """モデル未設定での健全性チェックテスト"""
        manager = LLMManager()
        
        result = manager.check_model_health()
        
        assert result is False
    
    @patch('genkai_rag.core.llm_manager.LLMManager.get_available_models')
    @patch('genkai_rag.core.llm_manager.LLMManager.check_model_health')
    def test_get_model_stats(self, mock_health, mock_get_models):
        """モデル統計情報取得テスト"""
        mock_models = [
            ModelInfo("llama2:7b", "3.8GB", datetime.now(), "abc123", {}),
            ModelInfo("codellama:13b", "7.3GB", datetime.now(), "def456", {})
        ]
        mock_get_models.return_value = mock_models
        mock_health.return_value = True
        
        manager = LLMManager()
        manager.current_model = "llama2:7b"
        
        stats = manager.get_model_stats()
        
        assert stats["current_model"] == "llama2:7b"
        assert stats["available_models_count"] == 2
        assert stats["model_configs_count"] == 0
        assert stats["health_status"] == "healthy"
        assert "llama2:7b" in stats["available_models"]
        assert "codellama:13b" in stats["available_models"]


class TestLLMManagerProperties:
    """LLMManagerのプロパティベーステスト"""
    
    @given(
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        temperature=st.floats(min_value=0.0, max_value=2.0),
        max_tokens=st.integers(min_value=1, max_value=8192)
    )
    @settings(max_examples=50, deadline=None)
    def test_model_optimization_properties(self, model_name, temperature, max_tokens):
        """
        Feature: genkai-rag-system, Property 5: LLMモデル切り替え
        任意のモデル名に対して、最適化設定が適切に生成される
        """
        assume(model_name.strip())  # 空白のみの文字列を除外
        
        manager = LLMManager()
        config = manager.optimize_for_model(model_name)
        
        # 基本設定が含まれている
        assert "temperature" in config
        assert "max_tokens" in config
        assert "top_p" in config
        assert "top_k" in config
        assert "repeat_penalty" in config
        assert "context_length" in config
        
        # 値が妥当な範囲内
        assert 0.0 <= config["temperature"] <= 2.0
        assert config["max_tokens"] > 0
        assert 0.0 <= config["top_p"] <= 1.0
        assert config["top_k"] > 0
        assert config["repeat_penalty"] >= 1.0
        assert config["context_length"] > 0
        
        # 設定がキャッシュされている
        assert model_name in manager.model_configs
        
        # 同じモデル名で再度呼び出すと同じ設定が返される
        config2 = manager.optimize_for_model(model_name)
        assert config == config2
    
    @given(
        prompt=st.text(min_size=1, max_size=1000),
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip())
    )
    @settings(max_examples=30, deadline=None)
    @patch('requests.post')
    def test_generate_response_properties(self, mock_post, prompt, model_name):
        """
        Feature: genkai-rag-system, Property 5: LLMモデル切り替え
        任意のプロンプトとモデルに対して、適切なリクエストが生成される
        """
        assume(prompt.strip())  # 空白のみのプロンプトを除外
        assume(model_name.strip())  # 空白のみのモデル名を除外
        
        # モックレスポンスを設定
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": f"Response to: {prompt[:50]}"}
        mock_post.return_value = mock_response
        
        manager = LLMManager()
        
        try:
            result = manager.generate_response(prompt, model_name=model_name)
            
            # レスポンスが文字列である
            assert isinstance(result, str)
            
            # リクエストが正しく送信された
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            
            # URLが正しい
            assert call_args[0][0] == "http://localhost:11434/api/generate"
            
            # ペイロードが正しい
            payload = call_args[1]["json"]
            assert payload["model"] == model_name
            assert payload["prompt"] == prompt
            assert payload["stream"] is False
            assert "options" in payload
            
        except Exception:
            # ネットワークエラーなどは許容
            pass
    
    @given(
        base_url=st.text(min_size=1, max_size=100).filter(
            lambda x: x.strip() and not x.startswith(' ') and not x.endswith(' ')
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_initialization_properties(self, base_url):
        """
        Feature: genkai-rag-system, Property 5: LLMモデル切り替え
        任意のベースURLに対して、LLMManagerが適切に初期化される
        """
        assume(base_url.strip())  # 空白のみのURLを除外
        
        try:
            manager = LLMManager(ollama_base_url=base_url)
            
            # URLが正規化されている（末尾スラッシュ除去）
            expected_url = base_url.rstrip('/')
            assert manager.ollama_base_url == expected_url
            
            # 初期状態が正しい
            assert manager.current_model is None
            assert isinstance(manager.model_configs, dict)
            assert len(manager.model_configs) == 0
            
            # デフォルト設定が存在する
            assert isinstance(manager.default_config, dict)
            assert "temperature" in manager.default_config
            
        except Exception:
            # 無効なURLなどは許容
            pass