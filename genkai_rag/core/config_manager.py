"""
ConfigManagerクラス

このモジュールは、システム設定の管理を行います。
設定ファイルの読み書き、検証、履歴管理、ロールバック機能を提供します。
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil
import threading
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfigChange:
    """設定変更履歴のデータクラス"""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    user: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "user": self.user
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigChange':
        """辞書から復元"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            key=data["key"],
            old_value=data["old_value"],
            new_value=data["new_value"],
            user=data.get("user", "system")
        )


class ConfigManager:
    """
    システム設定管理クラス
    
    設定ファイルの読み書き、検証、履歴管理、ロールバック機能を提供します。
    スレッドセーフな操作をサポートします。
    """
    
    def __init__(self, config_dir: str = "config", backup_count: int = 10):
        """
        ConfigManagerを初期化
        
        Args:
            config_dir: 設定ファイルディレクトリ
            backup_count: 保持するバックアップ数
        """
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / "config.yaml"
        self.history_file = self.config_dir / "config_history.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_count = backup_count
        
        # スレッドセーフティのためのロック
        self._lock = threading.RLock()
        
        # 現在の設定をメモリにキャッシュ
        self._config_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        
        # ディレクトリ作成
        self._ensure_directories()
        
        # デフォルト設定の初期化
        self._initialize_default_config()
        
        logger.info(f"ConfigManager initialized with config_dir: {self.config_dir}")
    
    def _ensure_directories(self) -> None:
        """必要なディレクトリを作成"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_default_config(self) -> None:
        """デフォルト設定を初期化"""
        if not self.config_file.exists():
            default_config = {
                "llm": {
                    "default_model": "llama2:7b",
                    "ollama_url": "http://localhost:11434",
                    "models": {
                        "llama2:7b": {
                            "temperature": 0.7,
                            "max_tokens": 2048,
                            "top_p": 0.9,
                            "context_window": 4096
                        },
                        "codellama:7b": {
                            "temperature": 0.1,
                            "max_tokens": 4096,
                            "top_p": 0.95,
                            "context_window": 16384
                        }
                    }
                },
                "rag": {
                    "similarity_threshold": 0.7,
                    "max_retrieved_docs": 10,
                    "max_context_docs": 5,
                    "chunk_size": 1000,
                    "chunk_overlap": 200
                },
                "chat": {
                    "max_history_size": 50,
                    "max_session_age_days": 30,
                    "cleanup_interval_hours": 24
                },
                "system": {
                    "log_level": "INFO",
                    "max_log_size_mb": 100,
                    "log_retention_days": 30,
                    "memory_threshold_percent": 80,
                    "disk_threshold_percent": 90
                },
                "web": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "cors_origins": ["*"],
                    "request_timeout": 300
                }
            }
            
            self.save_config(default_config)
            logger.info("Default configuration created")
    
    def load_config(self) -> Dict[str, Any]:
        """
        設定ファイルを読み込み
        
        Returns:
            設定辞書
            
        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合
            yaml.YAMLError: YAML解析エラーの場合
        """
        with self._lock:
            try:
                # キャッシュの有効性をチェック
                if self._is_cache_valid():
                    return self._config_cache.copy()
                
                if not self.config_file.exists():
                    raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
                
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # キャッシュを更新
                self._config_cache = config
                self._cache_timestamp = datetime.now()
                
                logger.debug("Configuration loaded successfully")
                return config.copy()
                
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse configuration file: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise
    
    def save_config(self, config: Dict[str, Any], user: str = "system") -> bool:
        """
        設定を保存
        
        Args:
            config: 保存する設定辞書
            user: 変更を行ったユーザー
            
        Returns:
            保存成功の場合True
        """
        with self._lock:
            try:
                # 設定の検証
                if not self._validate_config(config):
                    logger.error("Configuration validation failed")
                    return False
                
                # 現在の設定を取得（履歴記録用）
                old_config = {}
                if self.config_file.exists():
                    old_config = self.load_config()
                
                # バックアップを作成
                self._create_backup()
                
                # 設定を保存
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
                
                # 変更履歴を記録
                self._record_changes(old_config, config, user)
                
                # キャッシュを更新
                self._config_cache = config
                self._cache_timestamp = datetime.now()
                
                logger.info("Configuration saved successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                return False
    
    def get_llm_config(self, model_name: str) -> Dict[str, Any]:
        """
        指定されたLLMモデルの設定を取得
        
        Args:
            model_name: モデル名
            
        Returns:
            モデル設定辞書
        """
        config = self.load_config()
        llm_config = config.get("llm", {})
        models = llm_config.get("models", {})
        
        if model_name not in models:
            # デフォルト設定を返す
            return {
                "temperature": 0.7,
                "max_tokens": 2048,
                "top_p": 0.9,
                "context_window": 4096
            }
        
        return models[model_name].copy()
    
    def update_llm_config(self, model_name: str, model_config: Dict[str, Any], user: str = "system") -> bool:
        """
        LLMモデル設定を更新
        
        Args:
            model_name: モデル名
            model_config: モデル設定
            user: 変更を行ったユーザー
            
        Returns:
            更新成功の場合True
        """
        try:
            config = self.load_config()
            
            # LLM設定セクションを確保
            if "llm" not in config:
                config["llm"] = {}
            if "models" not in config["llm"]:
                config["llm"]["models"] = {}
            
            # モデル設定を更新
            config["llm"]["models"][model_name] = model_config
            
            return self.save_config(config, user)
            
        except Exception as e:
            logger.error(f"Failed to update LLM config for {model_name}: {e}")
            return False
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        ドット記法でネストした設定値を取得
        
        Args:
            key_path: "llm.default_model" のようなキーパス
            default: デフォルト値
            
        Returns:
            設定値
        """
        try:
            config = self.load_config()
            keys = key_path.split('.')
            
            value = config
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to get config value for {key_path}: {e}")
            return default
    
    def set_config_value(self, key_path: str, value: Any, user: str = "system") -> bool:
        """
        ドット記法でネストした設定値を設定
        
        Args:
            key_path: "llm.default_model" のようなキーパス
            value: 設定値
            user: 変更を行ったユーザー
            
        Returns:
            設定成功の場合True
        """
        try:
            config = self.load_config()
            keys = key_path.split('.')
            
            # ネストした辞書を作成
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # 最終キーに値を設定
            current[keys[-1]] = value
            
            return self.save_config(config, user)
            
        except Exception as e:
            logger.error(f"Failed to set config value for {key_path}: {e}")
            return False
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """
        設定変更履歴を取得
        
        Args:
            limit: 取得する履歴数の上限
            
        Returns:
            変更履歴のリスト
        """
        try:
            if not self.history_file.exists():
                return []
            
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            changes = [ConfigChange.from_dict(item) for item in history_data]
            
            # 新しい順にソートして制限数まで返す
            changes.sort(key=lambda x: x.timestamp, reverse=True)
            return changes[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get change history: {e}")
            return []
    
    def rollback_to_backup(self, backup_timestamp: str, user: str = "system") -> bool:
        """
        指定されたバックアップにロールバック
        
        Args:
            backup_timestamp: バックアップのタイムスタンプ
            user: ロールバックを実行するユーザー
            
        Returns:
            ロールバック成功の場合True
        """
        try:
            backup_file = self.backup_dir / f"config_{backup_timestamp}.yaml"
            
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                return False
            
            # バックアップから設定を読み込み
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_config = yaml.safe_load(f)
            
            # 現在の設定をバックアップ
            self._create_backup()
            
            # バックアップ設定を復元
            success = self.save_config(backup_config, user)
            
            if success:
                logger.info(f"Successfully rolled back to backup: {backup_timestamp}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback to backup {backup_timestamp}: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        利用可能なバックアップのリストを取得
        
        Returns:
            バックアップ情報のリスト
        """
        try:
            backups = []
            
            for backup_file in self.backup_dir.glob("config_*.yaml"):
                # ファイル名からタイムスタンプを抽出
                timestamp_str = backup_file.stem.replace("config_", "")
                
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("_", ":"))
                    file_size = backup_file.stat().st_size
                    
                    backups.append({
                        "timestamp": timestamp.isoformat(),
                        "filename": backup_file.name,
                        "size_bytes": file_size,
                        "created_at": timestamp
                    })
                except ValueError:
                    # 無効なタイムスタンプ形式のファイルはスキップ
                    continue
            
            # 新しい順にソート
            backups.sort(key=lambda x: x["created_at"], reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def cleanup_old_backups(self) -> int:
        """
        古いバックアップファイルをクリーンアップ
        
        Returns:
            削除されたファイル数
        """
        try:
            backups = self.list_backups()
            
            if len(backups) <= self.backup_count:
                return 0
            
            # 保持数を超えた古いバックアップを削除
            old_backups = backups[self.backup_count:]
            deleted_count = 0
            
            for backup in old_backups:
                backup_file = self.backup_dir / backup["filename"]
                if backup_file.exists():
                    backup_file.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backup files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    def _is_cache_valid(self) -> bool:
        """キャッシュが有効かチェック"""
        if self._config_cache is None or self._cache_timestamp is None:
            return False
        
        # ファイルの更新時刻をチェック
        if not self.config_file.exists():
            return False
        
        file_mtime = datetime.fromtimestamp(self.config_file.stat().st_mtime)
        return file_mtime <= self._cache_timestamp
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """設定の妥当性を検証"""
        try:
            # 必須セクションの存在確認
            required_sections = ["llm", "rag", "chat", "system", "web"]
            for section in required_sections:
                if section not in config:
                    logger.error(f"Missing required section: {section}")
                    return False
            
            # LLM設定の検証
            llm_config = config.get("llm", {})
            if "default_model" not in llm_config:
                logger.error("Missing llm.default_model")
                return False
            
            # 数値範囲の検証
            rag_config = config.get("rag", {})
            if "similarity_threshold" in rag_config:
                threshold = rag_config["similarity_threshold"]
                if not (0.0 <= threshold <= 1.0):
                    logger.error(f"Invalid similarity_threshold: {threshold}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _create_backup(self) -> bool:
        """現在の設定のバックアップを作成"""
        try:
            if not self.config_file.exists():
                return True
            
            timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
            backup_file = self.backup_dir / f"config_{timestamp}.yaml"
            
            shutil.copy2(self.config_file, backup_file)
            
            # 古いバックアップをクリーンアップ
            self.cleanup_old_backups()
            
            logger.debug(f"Configuration backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def _record_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], user: str) -> None:
        """設定変更を履歴に記録"""
        try:
            changes = self._find_config_changes(old_config, new_config, user)
            
            if not changes:
                return
            
            # 既存の履歴を読み込み
            history = []
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 新しい変更を追加
            for change in changes:
                history.append(change.to_dict())
            
            # 履歴サイズを制限（最新1000件まで）
            if len(history) > 1000:
                history = history[-1000:]
            
            # 履歴を保存
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Recorded {len(changes)} configuration changes")
            
        except Exception as e:
            logger.error(f"Failed to record configuration changes: {e}")
    
    def _find_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any], 
                           user: str, prefix: str = "") -> List[ConfigChange]:
        """設定の変更点を再帰的に検出"""
        changes = []
        timestamp = datetime.now()
        
        # 新しい設定のキーをチェック
        for key, new_value in new_config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            old_value = old_config.get(key)
            
            if isinstance(new_value, dict) and isinstance(old_value, dict):
                # 再帰的にネストした辞書をチェック
                changes.extend(self._find_config_changes(old_value, new_value, user, full_key))
            elif old_value != new_value:
                # 値が変更された
                changes.append(ConfigChange(
                    timestamp=timestamp,
                    key=full_key,
                    old_value=old_value,
                    new_value=new_value,
                    user=user
                ))
        
        # 削除されたキーをチェック
        for key, old_value in old_config.items():
            if key not in new_config:
                full_key = f"{prefix}.{key}" if prefix else key
                changes.append(ConfigChange(
                    timestamp=timestamp,
                    key=full_key,
                    old_value=old_value,
                    new_value=None,
                    user=user
                ))
        
        return changes