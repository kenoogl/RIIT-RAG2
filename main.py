"""
玄界RAGシステム メインエントリーポイント

使用方法:
    python main.py
"""

import uvicorn
from genkai_rag.utils.config import config
from genkai_rag.utils.logging import setup_logging, get_logger


def main():
    """メイン関数"""
    # ログ設定を初期化
    setup_logging()
    logger = get_logger("main")
    
    logger.info("玄界RAGシステムを開始しています...")
    
    # 設定を読み込み
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    
    logger.info(f"サーバーを起動します: http://{host}:{port}")
    
    # FastAPIアプリケーションを起動
    # 注意: この時点ではAPIアプリケーションはまだ実装されていません
    # 後のタスクで実装予定
    try:
        # uvicorn.run(
        #     "genkai_rag.api.app:app",
        #     host=host,
        #     port=port,
        #     reload=True
        # )
        logger.info("APIアプリケーションはまだ実装されていません。後のタスクで実装予定です。")
        print("プロジェクト構造とコア依存関係の設定が完了しました。")
        print("次のタスクでデータモデルを実装します。")
        
    except Exception as e:
        logger.error(f"サーバー起動エラー: {e}")
        raise


if __name__ == "__main__":
    main()