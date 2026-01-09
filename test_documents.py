#!/usr/bin/env python3
"""
テスト用ドキュメントを追加してRAGシステムを初期化するスクリプト
"""

import sys
import os
import asyncio
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from genkai_rag.core.config_manager import ConfigManager
from genkai_rag.core.processor import DocumentProcessor
from genkai_rag.models.document import Document, DocumentMetadata
from datetime import datetime


async def main():
    """メイン処理"""
    print("玄界RAGシステム - テストドキュメント追加")
    
    try:
        # 設定管理を初期化
        config_manager = ConfigManager()
        
        # DocumentProcessorを初期化
        processor = DocumentProcessor(
            index_dir="./data/index",
            chunk_size=1024,
            chunk_overlap=200
        )
        
        # テスト用ドキュメントを作成
        test_documents = [
            Document(
                content="""
                玄界システムは、九州大学情報基盤研究開発センターが開発・運用している
                高性能計算システムです。研究・教育活動を支援するため、大規模な
                計算リソースを提供しています。
                
                主な特徴：
                - 高性能計算ノード
                - 大容量ストレージ
                - 高速ネットワーク
                - 24時間365日運用
                
                利用者は、研究や教育目的でシステムを利用することができます。
                """,
                metadata=DocumentMetadata(
                    title="玄界システムについて",
                    url="https://www.cc.kyushu-u.ac.jp/scp/system/genkai/",
                    source="web",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00")
                )
            ),
            Document(
                content="""
                玄界システムを利用するには、以下の手順が必要です：
                
                1. アカウント申請
                   - 九州大学の教職員または学生であること
                   - 指導教員の承認が必要
                
                2. ログイン方法
                   - SSHクライアントを使用
                   - 多要素認証が必要
                
                3. ジョブ投入
                   - SLURMワークロードマネージャーを使用
                   - キューシステムによる資源管理
                
                4. データ管理
                   - ホームディレクトリ
                   - 共有ストレージ
                   - バックアップ機能
                """,
                metadata=DocumentMetadata(
                    title="玄界システムの利用方法",
                    url="https://www.cc.kyushu-u.ac.jp/scp/system/genkai/usage/",
                    source="web",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00")
                )
            ),
            Document(
                content="""
                玄界システムの技術仕様：
                
                計算ノード：
                - CPU: Intel Xeon プロセッサー
                - メモリ: 最大512GB
                - ノード数: 200台以上
                
                ストレージ：
                - 総容量: 10PB以上
                - ファイルシステム: Lustre
                - 高速アクセス対応
                
                ネットワーク：
                - InfiniBand接続
                - 高帯域・低遅延
                - 100Gbps対応
                
                ソフトウェア：
                - OS: Linux (CentOS/RHEL)
                - コンパイラ: GCC, Intel, PGI
                - MPI: OpenMPI, Intel MPI
                - 各種科学計算ライブラリ
                """,
                metadata=DocumentMetadata(
                    title="玄界システムの技術仕様",
                    url="https://www.cc.kyushu-u.ac.jp/scp/system/genkai/specs/",
                    source="web",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00")
                )
            )
        ]
        
        print(f"テストドキュメント {len(test_documents)} 件を処理中...")
        
        # ドキュメントを処理してインデックスに追加
        success_count = 0
        for doc in test_documents:
            try:
                result = processor.process_single_document(doc)
                if result:
                    success_count += 1
                    print(f"✓ ドキュメント '{doc.metadata.title}' を追加しました")
                else:
                    print(f"✗ ドキュメント '{doc.metadata.title}' の追加に失敗しました")
            except Exception as e:
                print(f"✗ ドキュメント '{doc.metadata.title}' の処理中にエラー: {e}")
        
        print(f"\n処理完了: {success_count}/{len(test_documents)} 件のドキュメントを追加")
        
        # インデックス統計を表示
        stats = processor.get_index_statistics()
        print(f"インデックス統計: {stats}")
        
        # インデックスを保存
        processor._save_index()
        print("インデックスを保存しました")
        
        return success_count > 0
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)