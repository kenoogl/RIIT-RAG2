"""ログ設定ユーティリティ"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

from .config import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_log_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None
) -> logging.Logger:
    """
    ログ設定を初期化
    
    Args:
        log_level: ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: ログファイルパス
        max_log_size_mb: ログファイルの最大サイズ（MB）
        backup_count: バックアップファイル数
        
    Returns:
        設定されたロガー
    """
    # 設定から値を取得（引数が指定されていない場合）
    log_level = log_level or config.get("monitoring.log_level", "INFO")
    log_file = log_file or config.get("monitoring.log_file", "./logs/genkai_rag.log")
    max_log_size_mb = max_log_size_mb or config.get("monitoring.max_log_size_mb", 100)
    backup_count = backup_count or config.get("monitoring.backup_count", 5)
    
    # ログレベルを設定
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # ルートロガーを取得
    logger = logging.getLogger("genkai_rag")
    logger.setLevel(numeric_level)
    
    # 既存のハンドラーをクリア
    logger.handlers.clear()
    
    # フォーマッターを作成
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラーを追加
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラーを追加
    if log_file:
        # ログディレクトリを作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ローテーティングファイルハンドラーを作成
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_log_size_mb * 1024 * 1024,  # MBをバイトに変換
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    指定された名前のロガーを取得
    
    Args:
        name: ロガー名
        
    Returns:
        ロガーインスタンス
    """
    return logging.getLogger(f"genkai_rag.{name}")


# デフォルトロガーを設定
default_logger = setup_logging()