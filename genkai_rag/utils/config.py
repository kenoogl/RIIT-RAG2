"""設定管理ユーティリティ"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """設定ファイルの読み込みと管理を行うクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        設定マネージャーを初期化
        
        Args:
            config_path: 設定ファイルのパス（デフォルト: config/default.yaml）
        """
        if config_path is None:
            config_path = "config/default.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """設定ファイルを読み込む"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"警告: 設定ファイル {self.config_path} が見つかりません。デフォルト設定を使用します。")
            self._config = self._get_default_config()
        except yaml.YAMLError as e:
            print(f"エラー: 設定ファイルの解析に失敗しました: {e}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "llm": {
                "default_model": "llama3.2:3b",
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "timeout": 60
                }
            },
            "vector_store": {
                "type": "chroma",
                "persist_directory": "./data/chroma_db",
                "collection_name": "genkai_documents"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得
        
        Args:
            key: 設定キー（ドット記法対応、例: "llm.default_model"）
            default: デフォルト値
            
        Returns:
            設定値
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        設定値を更新
        
        Args:
            key: 設定キー（ドット記法対応）
            value: 設定値
        """
        keys = key.split('.')
        config = self._config
        
        # 最後のキー以外まで辿る
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 最後のキーに値を設定
        config[keys[-1]] = value
    
    def save(self) -> bool:
        """
        設定をファイルに保存
        
        Returns:
            保存成功時True
        """
        try:
            # ディレクトリが存在しない場合は作成
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"エラー: 設定ファイルの保存に失敗しました: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """
        全設定を取得
        
        Returns:
            設定辞書のコピー
        """
        return self._config.copy()


# グローバル設定インスタンス
config = ConfigManager()