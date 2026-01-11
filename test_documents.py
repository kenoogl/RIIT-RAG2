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
                - 高性能計算ノード: Intel Xeon プロセッサー搭載
                - 大容量ストレージ: 10PB以上の容量
                - 高速ネットワーク: InfiniBand接続
                - 24時間365日運用: 安定したサービス提供
                
                利用者は、研究や教育目的でシステムを利用することができます。
                システムは主に科学技術計算、データ解析、機械学習などの用途に使用されています。
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
                   - 申請フォームをWebサイトから提出
                   - 審査期間は通常1-2週間
                
                2. ログイン方法
                   - SSHクライアントを使用してアクセス
                   - ホスト名: genkai.cc.kyushu-u.ac.jp
                   - ポート: 22 (SSH)
                   - 多要素認証が必要（ワンタイムパスワード）
                
                3. ジョブ投入
                   - SLURMワークロードマネージャーを使用
                   - sbatchコマンドでジョブスクリプトを投入
                   - squeueコマンドでジョブ状況を確認
                   - scancelコマンドでジョブをキャンセル
                
                4. データ管理
                   - ホームディレクトリ: /home/username (容量制限あり)
                   - 共有ストレージ: /work/username (大容量データ用)
                   - バックアップ機能: 定期的な自動バックアップ
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
                - CPU: Intel Xeon Gold 6248R (3.0GHz, 24コア)
                - メモリ: 192GB DDR4-2933
                - ノード数: 256台
                - 総コア数: 6,144コア
                
                ストレージ：
                - 総容量: 12PB
                - ファイルシステム: Lustre 2.12
                - 転送速度: 最大100GB/s
                - RAID構成による冗長化
                
                ネットワーク：
                - InfiniBand HDR (200Gbps)
                - Fat-treeトポロジー
                - 低遅延通信対応
                - Ethernet 10GbE管理ネットワーク
                
                ソフトウェア環境：
                - OS: CentOS 8 Stream
                - コンパイラ: GCC 8.5, Intel oneAPI 2023
                - MPI: OpenMPI 4.1, Intel MPI 2021
                - 数値計算ライブラリ: Intel MKL, FFTW, BLAS/LAPACK
                - 開発環境: Python 3.9, R 4.2, MATLAB R2023a
                """,
                metadata=DocumentMetadata(
                    title="玄界システムの技術仕様",
                    url="https://www.cc.kyushu-u.ac.jp/scp/system/genkai/specs/",
                    source="web",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00")
                )
            ),
            Document(
                content="""
                玄界システムでのジョブ投入方法：
                
                基本的なジョブスクリプトの例：
                #!/bin/bash
                #SBATCH --job-name=my_job
                #SBATCH --nodes=1
                #SBATCH --ntasks-per-node=24
                #SBATCH --time=01:00:00
                #SBATCH --partition=cpu
                #SBATCH --output=job_%j.out
                #SBATCH --error=job_%j.err
                
                module load gcc/8.5.0
                module load openmpi/4.1.0
                
                mpirun ./my_program
                
                ジョブ投入コマンド：
                sbatch job_script.sh
                
                ジョブ状況確認：
                squeue -u $USER
                
                ジョブキャンセル：
                scancel <job_id>
                
                利用可能なパーティション：
                - cpu: 一般的なCPU計算用
                - gpu: GPU計算用（Tesla V100搭載）
                - bigmem: 大容量メモリ用（最大1TB）
                - debug: デバッグ用（短時間実行）
                """,
                metadata=DocumentMetadata(
                    title="ジョブ投入方法",
                    url="https://www.cc.kyushu-u.ac.jp/scp/system/genkai/job/",
                    source="web",
                    created_at=datetime.fromisoformat("2024-01-01T00:00:00"),
                    updated_at=datetime.fromisoformat("2024-01-01T00:00:00")
                )
            ),
            Document(
                content="""
                玄界システムでよくあるトラブルと解決方法：
                
                1. ログインできない場合
                   - ネットワーク接続を確認
                   - ユーザー名とパスワードを再確認
                   - 多要素認証の設定を確認
                   - VPN接続が必要な場合があります
                
                2. ジョブが実行されない場合
                   - squeue -u $USER でジョブ状況を確認
                   - リソース要求が適切か確認（ノード数、時間制限）
                   - パーティションの指定が正しいか確認
                   - ジョブスクリプトの構文エラーをチェック
                
                3. ファイルアクセスエラー
                   - ディスク容量制限を確認（quota -u $USER）
                   - ファイルパーミッションを確認
                   - ファイルシステムの状態を確認
                
                4. 計算が遅い場合
                   - 並列化の設定を確認
                   - メモリ使用量を最適化
                   - I/O処理を効率化
                   - 適切なコンパイラオプションを使用
                
                サポート連絡先：
                - メール: genkai-support@cc.kyushu-u.ac.jp
                - 電話: 092-802-2683
                - 受付時間: 平日 9:00-17:00
                """,
                metadata=DocumentMetadata(
                    title="トラブルシューティング",
                    url="https://www.cc.kyushu-u.ac.jp/scp/system/genkai/trouble/",
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