# 玄界RAGシステム

九州大学情報基盤研究開発センターのスーパーコンピュータ玄界システム用RAG（Retrieval-Augmented Generation）質問応答システムです。

## 概要

このシステムは、玄界システムの公式文書を基に、利用者からの質問に自動的に回答を提供します。LlamaIndexを使用したRAGアーキテクチャにより、正確で関連性の高い情報を提供します。

## 主な機能

- **文書自動取得**: 玄界システム公式サイトからの文書スクレイピング
- **インテリジェント検索**: ベクトル検索とrerankingによる高精度な情報検索
- **日本語対応**: 日本語文書の処理と自然な日本語回答生成
- **オンプレミス運用**: ローカルLLMモデル（Ollama）による完全オンプレミス運用
- **Webインターフェイス**: 使いやすいWeb UI
- **会話履歴管理**: セッション単位での会話履歴保持

## 技術スタック

- **RAGフレームワーク**: LlamaIndex
- **Webフレームワーク**: FastAPI
- **ローカルLLM**: Ollama
- **ベクトルDB**: Chroma
- **Webスクレイピング**: BeautifulSoup
- **フロントエンド**: HTML/CSS/JavaScript

## インストール

### 前提条件

- Python 3.8以上
- Ollama（ローカルLLMモデル用）

### セットアップ

1. リポジトリをクローン:
```bash
git clone <repository-url>
cd genkai-rag-system
```

2. 依存関係をインストール:
```bash
pip install -r requirements.txt
```

3. Ollamaをインストールし、モデルをダウンロード:
```bash
# Ollamaのインストール（https://ollama.ai/）
ollama pull llama3.2:3b
ollama pull gemma2:2b
```

4. 設定ファイルを確認・編集:
```bash
cp config/default.yaml config/local.yaml
# config/local.yamlを必要に応じて編集
```

## 使用方法

### 開発モード

```bash
python main.py
```

### プロダクションモード

```bash
uvicorn genkai_rag.api.app:app --host 0.0.0.0 --port 8000
```

## プロジェクト構造

```
genkai-rag-system/
├── genkai_rag/           # メインパッケージ
│   ├── models/           # データモデル
│   ├── core/             # コアコンポーネント
│   ├── api/              # FastAPI アプリケーション
│   └── utils/            # ユーティリティ
├── tests/                # テストファイル
├── config/               # 設定ファイル
├── data/                 # データストレージ
├── logs/                 # ログファイル
└── static/               # 静的ファイル（HTML/CSS/JS）
```

## 設定

主要な設定項目は `config/default.yaml` で管理されています：

- **LLMモデル設定**: 使用するモデルとパラメータ
- **ベクトルDB設定**: Chromaデータベースの設定
- **スクレイピング設定**: 文書取得の設定
- **API設定**: Webサーバーの設定

## 開発

### テスト実行

```bash
pytest tests/
```

### コード品質チェック

```bash
black genkai_rag/
isort genkai_rag/
mypy genkai_rag/
```

## ライセンス

このプロジェクトは九州大学情報基盤研究開発センター向けに開発されています。

## 貢献

バグ報告や機能要望は、GitHubのIssueでお知らせください。

## サポート

技術的な質問やサポートが必要な場合は、開発チームまでお問い合わせください。