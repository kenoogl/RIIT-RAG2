"""
LLMManager: Ollama統合とモデル管理機能

このモジュールは、Ollamaとの統合、モデル管理、最適化設定を提供します。
"""

import logging
from typing import Dict, List, Optional, Any
import requests
import json
from dataclasses import dataclass
from datetime import datetime

from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """LLMモデル情報"""
    name: str
    size: str
    modified_at: datetime
    digest: str
    details: Dict[str, Any]


class LLMManager:
    """
    Ollama LLMモデル管理クラス
    
    機能:
    - Ollamaサーバーとの通信
    - モデルの切り替えとロード
    - モデル固有の最適化設定
    - モデル情報の取得と管理
    """
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        """
        LLMManagerを初期化
        
        Args:
            ollama_base_url: OllamaサーバーのベースURL
        """
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.config_manager = ConfigManager()
        self.current_model: Optional[str] = None
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
        # デフォルトモデル設定
        self.default_config = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "context_length": 4096
        }
        
        logger.info(f"LLMManager initialized with Ollama URL: {self.ollama_base_url}")
    
    def get_available_models(self) -> List[ModelInfo]:
        """
        利用可能なモデル一覧を取得
        
        Returns:
            利用可能なモデル情報のリスト
            
        Raises:
            ConnectionError: Ollamaサーバーに接続できない場合
            ValueError: レスポンスの解析に失敗した場合
        """
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("models", []):
                model_info = ModelInfo(
                    name=model_data["name"],
                    size=model_data.get("size", "unknown"),
                    modified_at=datetime.fromisoformat(
                        model_data.get("modified_at", "1970-01-01T00:00:00Z").replace("Z", "+00:00")
                    ),
                    digest=model_data.get("digest", ""),
                    details=model_data.get("details", {})
                )
                models.append(model_info)
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            raise ConnectionError(f"Cannot connect to Ollama server at {self.ollama_base_url}: {e}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            raise ValueError(f"Invalid response from Ollama server: {e}")
    
    def switch_model(self, model_name: str) -> bool:
        """
        指定されたモデルに切り替え
        
        Args:
            model_name: 切り替え先のモデル名
            
        Returns:
            切り替えが成功した場合True
            
        Raises:
            ValueError: 指定されたモデルが存在しない場合
            ConnectionError: モデルのロードに失敗した場合
        """
        if not model_name or not model_name.strip():
            raise ValueError("Model name cannot be empty")
        
        # モデルが利用可能かチェック
        available_models = self.get_available_models()
        model_names = [model.name for model in available_models]
        
        if model_name not in model_names:
            raise ValueError(f"Model '{model_name}' not found. Available models: {model_names}")
        
        try:
            # モデルをロード（プリロード）
            load_payload = {
                "name": model_name
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/pull",
                json=load_payload,
                timeout=60
            )
            response.raise_for_status()
            
            # 現在のモデルを更新
            old_model = self.current_model
            self.current_model = model_name
            
            # モデル固有の設定を適用
            self._apply_model_optimization(model_name)
            
            logger.info(f"Successfully switched from '{old_model}' to '{model_name}'")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise ConnectionError(f"Failed to load model '{model_name}': {e}")
    
    def get_current_model(self) -> Optional[str]:
        """
        現在アクティブなモデル名を取得
        
        Returns:
            現在のモデル名、設定されていない場合はNone
        """
        return self.current_model
    
    def optimize_for_model(self, model_name: str) -> Dict[str, Any]:
        """
        指定されたモデルに最適化された設定を取得
        
        Args:
            model_name: モデル名
            
        Returns:
            モデル固有の最適化設定
        """
        # モデル固有の設定があるかチェック
        if model_name in self.model_configs:
            return self.model_configs[model_name]
        
        # モデル名に基づいた最適化設定
        config = self.default_config.copy()
        
        # 優先順位に基づいて設定を適用（後の条件が優先される）
        
        # 1. サイズベースの最適化
        if any(size_indicator in model_name.lower() for size_indicator in ["70b", "65b", "large"]):
            config.update({
                "temperature": 0.5,  # 大型モデルは低い温度で安定
                "max_tokens": 4096,
                "context_length": 8192
            })
        elif any(size_indicator in model_name.lower() for size_indicator in ["7b", "3b", "1b", "small"]):
            config.update({
                "temperature": 0.8,  # 小型モデルは高めの温度で創造性を
                "max_tokens": 1024,
                "context_length": 2048
            })
        
        # 2. 特殊用途の最適化（サイズ設定を上書き）
        if any(code_indicator in model_name.lower() for code_indicator in ["code", "coder", "coding"]):
            config.update({
                "temperature": 0.3,  # コード生成は低い温度
                "top_p": 0.95,
                "repeat_penalty": 1.0
            })
        elif any(jp_indicator in model_name.lower() for jp_indicator in ["japanese", "ja", "jp", "elyza", "calm"]):
            config.update({
                "temperature": 0.6,  # 日本語では少し低めの温度
                "top_p": 0.85,
                "repeat_penalty": 1.05,
                "context_length": 8192  # 日本語は文脈が重要
            })
        
        self.model_configs[model_name] = config
        logger.info(f"Generated optimization config for model '{model_name}': {config}")
        
        return config
    
    def _apply_model_optimization(self, model_name: str) -> None:
        """
        モデル固有の最適化設定を適用
        
        Args:
            model_name: 最適化を適用するモデル名
        """
        config = self.optimize_for_model(model_name)
        
        # 設定をConfigManagerに保存
        try:
            current_config = self.config_manager.load_config()
            if "llm" not in current_config:
                current_config["llm"] = {}
            
            current_config["llm"]["current_model"] = model_name
            current_config["llm"]["model_configs"] = current_config["llm"].get("model_configs", {})
            current_config["llm"]["model_configs"][model_name] = config
            
            self.config_manager.save_config(current_config)
            logger.info(f"Applied optimization settings for model '{model_name}'")
            
        except Exception as e:
            logger.warning(f"Failed to save model configuration: {e}")
    
    def generate_response(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """
        指定されたプロンプトに対してレスポンスを生成
        
        Args:
            prompt: 入力プロンプト
            model_name: 使用するモデル名（Noneの場合は現在のモデル）
            **kwargs: 追加の生成パラメータ
            
        Returns:
            生成されたレスポンス
            
        Raises:
            ValueError: モデルが設定されていない場合
            ConnectionError: 生成に失敗した場合
        """
        target_model = model_name or self.current_model
        if not target_model:
            raise ValueError("No model specified and no current model set")
        
        # モデル固有の設定を取得
        model_config = self.optimize_for_model(target_model)
        
        # kwargsで設定を上書き
        generation_config = model_config.copy()
        generation_config.update(kwargs)
        
        try:
            payload = {
                "model": target_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": generation_config.get("temperature", 0.7),
                    "num_predict": generation_config.get("max_tokens", 2048),
                    "top_p": generation_config.get("top_p", 0.9),
                    "top_k": generation_config.get("top_k", 40),
                    "repeat_penalty": generation_config.get("repeat_penalty", 1.1)
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.info(f"Generated response using model '{target_model}' (length: {len(generated_text)})")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate response with model '{target_model}': {e}")
            raise ConnectionError(f"Failed to generate response: {e}")
    
    def check_model_health(self, model_name: Optional[str] = None) -> bool:
        """
        指定されたモデルの健全性をチェック
        
        Args:
            model_name: チェックするモデル名（Noneの場合は現在のモデル）
            
        Returns:
            モデルが正常に動作する場合True
        """
        target_model = model_name or self.current_model
        if not target_model:
            return False
        
        try:
            # 簡単なテストプロンプトで動作確認
            test_response = self.generate_response(
                "Hello", 
                model_name=target_model,
                max_tokens=10
            )
            return len(test_response.strip()) > 0
            
        except Exception as e:
            logger.warning(f"Model health check failed for '{target_model}': {e}")
            return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """
        モデル統計情報を取得
        
        Returns:
            モデル統計情報
        """
        stats = {
            "current_model": self.current_model,
            "available_models_count": 0,
            "model_configs_count": len(self.model_configs),
            "ollama_url": self.ollama_base_url,
            "health_status": "unknown"
        }
        
        try:
            available_models = self.get_available_models()
            stats["available_models_count"] = len(available_models)
            stats["available_models"] = [model.name for model in available_models]
            
            if self.current_model:
                stats["health_status"] = "healthy" if self.check_model_health() else "unhealthy"
            
        except Exception as e:
            logger.warning(f"Failed to get model stats: {e}")
            stats["health_status"] = "error"
        
        return stats