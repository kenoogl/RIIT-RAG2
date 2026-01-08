"""
玄界RAGシステム - メインエントリーポイント

このファイルは、玄界RAGシステムの起動とコマンドライン操作を提供します。
統合されたシステムアーキテクチャを使用します。
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from genkai_rag.app import GenkaiRAGSystem, initialize_system, shutdown_system
from genkai_rag.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def start_server_sync(config_path: str = None, host: str = None, port: int = None):
    """
    Webサーバーを同期的に開始
    
    Args:
        config_path: 設定ファイルのパス
        host: ホストアドレス
        port: ポート番号
    """
    try:
        # ログ設定
        setup_logging(log_level="INFO")
        
        # 設定の読み込み
        from genkai_rag.utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_all()
        
        web_config = config.get("web", {})
        host = host or web_config.get("host", "0.0.0.0")
        port = port or web_config.get("port", 8000)
        
        logger.info(f"Starting web server on {host}:{port}")
        
        # uvicornで直接起動
        import uvicorn
        uvicorn.run(
            "genkai_rag.api.app:app",
            host=host,
            port=port,
            log_level="info",
            reload=False
        )
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise


async def run_query(query: str, url: str = None, config_path: str = None):
    """
    単発のクエリを実行
    
    Args:
        query: 質問文
        url: 文書URL（オプション）
        config_path: 設定ファイルのパス
    """
    try:
        # システムを初期化
        system = await initialize_system(config_path)
        
        # 文書を処理（URLが指定された場合）
        if url:
            logger.info(f"Processing document from URL: {url}")
            document = await system.web_scraper.scrape_url(url)
            if document:
                await system.document_processor.add_document(document)
                logger.info("Document processed successfully")
            else:
                logger.warning("Failed to process document")
        
        # クエリを実行
        logger.info(f"Processing query: {query}")
        response = await system.rag_engine.query(query)
        
        # 結果を表示
        print("\n" + "="*50)
        print("質問:", query)
        if url:
            print("文書URL:", url)
        print("="*50)
        print("回答:")
        print(response.response)
        
        if response.source_documents:
            print("\n参考文書:")
            for i, doc in enumerate(response.source_documents, 1):
                print(f"{i}. {doc.metadata.get('title', 'Unknown')}")
                if doc.metadata.get('url'):
                    print(f"   URL: {doc.metadata['url']}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise
    finally:
        # システムをシャットダウン
        await shutdown_system()


async def check_system_status(config_path: str = None):
    """
    システム状態をチェック
    
    Args:
        config_path: 設定ファイルのパス
    """
    try:
        # システムを初期化
        system = await initialize_system(config_path)
        
        # システム状態を取得
        status = system.get_system_status()
        
        # 状態を表示
        print("\n" + "="*50)
        print("玄界RAGシステム状態")
        print("="*50)
        print(f"ステータス: {status['status']}")
        print(f"初期化済み: {status['initialized']}")
        
        print("\nコンポーネント:")
        for component, available in status['components'].items():
            status_str = "✓" if available else "✗"
            print(f"  {status_str} {component}")
        
        if 'system_metrics' in status:
            metrics = status['system_metrics']
            print(f"\nシステムメトリクス:")
            print(f"  メモリ使用率: {metrics['memory_usage']:.1f}%")
            print(f"  ディスク使用率: {metrics['disk_usage']:.1f}%")
            print(f"  CPU使用率: {metrics['cpu_usage']:.1f}%")
        
        if 'error_statistics' in status:
            error_stats = status['error_statistics']
            print(f"\nエラー統計（24時間）:")
            print(f"  総エラー数: {error_stats['total_errors']}")
            print(f"  エラー率: {error_stats['error_rate']:.2f}/時間")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise
    finally:
        # システムをシャットダウン
        await shutdown_system()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="玄界RAGシステム - 九州大学スーパーコンピュータ玄界システム用RAGシステム"
    )
    
    # サブコマンドの設定
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")
    
    # サーバー起動コマンド
    server_parser = subparsers.add_parser("server", help="Webサーバーを起動")
    server_parser.add_argument("--config", "-c", help="設定ファイルのパス")
    server_parser.add_argument("--host", default="0.0.0.0", help="ホストアドレス")
    server_parser.add_argument("--port", type=int, default=8000, help="ポート番号")
    
    # クエリ実行コマンド
    query_parser = subparsers.add_parser("query", help="単発のクエリを実行")
    query_parser.add_argument("query", help="質問文")
    query_parser.add_argument("--url", "-u", help="文書URL")
    query_parser.add_argument("--config", "-c", help="設定ファイルのパス")
    
    # ステータスチェックコマンド
    status_parser = subparsers.add_parser("status", help="システム状態をチェック")
    status_parser.add_argument("--config", "-c", help="設定ファイルのパス")
    
    # 共通オプション
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="ログレベル")
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(log_level=args.log_level)
    
    # コマンドが指定されていない場合はサーバーを起動
    if not args.command:
        args.command = "server"
        args.config = None
        args.host = "0.0.0.0"
        args.port = 8000
    
    try:
        # コマンドに応じて処理を実行
        if args.command == "server":
            start_server_sync(
                config_path=args.config,
                host=args.host,
                port=args.port
            )
        elif args.command == "query":
            asyncio.run(run_query(
                query=args.query,
                url=args.url,
                config_path=args.config
            ))
        elif args.command == "status":
            asyncio.run(check_system_status(
                config_path=args.config
            ))
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("操作が中断されました")
    except Exception as e:
        logger.error(f"実行エラー: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()