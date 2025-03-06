# LangChain RAGシステム

このプロジェクトは、LangChainを使用したシンプルなRAG（Retrieval-Augmented Generation）システムを実装しています。HTMLファイルからテキストを抽出し、ベクトルデータベースに保存して、ユーザーの質問に対して関連情報を検索し回答を生成します。

## 機能

- HTMLファイルからのテキスト抽出
- テキストのチャンク分割と埋め込み（OpenAIまたはCohereの埋め込みモデルをサポート）
- Chromaベクトルデータベースへの保存
- 質問応答システム（コマンドライン、インタラクティブモード、APIサーバー）
- トークン使用量の追跡と統計情報の記録

## ファイル構成

- `config.py`: システム設定（埋め込みプロバイダー、ディレクトリパスなど）
- `rag_utils.py`: 共通ユーティリティ関数（HTMLローダー、ドキュメント分割、ベクトルストア操作など）
- `rag_generator.py`: ベクトルストア生成スクリプト
- `rag_query.py`: 質問応答スクリプト（コマンドラインとインタラクティブモード）
- `api.py`: FastAPIを使用したAPIサーバー
- `requirements.txt`: 依存パッケージリスト

## セットアップ

1. 依存パッケージのインストール:

```bash
pip install -r requirements.txt
```

2. 環境変数の設定:

`.env`ファイルを作成し、以下の内容を記述します:

```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
EMBEDDING_PROVIDER=cohere  # 'openai'または'cohere'を指定
```

3. HTMLデータの準備:

`rag_base_data/html/`ディレクトリにHTMLファイルを配置します。

## 使用方法

### ベクトルストアの生成

```bash
python rag_generator.py
```

環境変数`EMBEDDING_PROVIDER`で指定された埋め込みモデル（OpenAIまたはCohere）を使用してベクトルストアが生成されます。

### 質問応答（コマンドライン）

```bash
python rag_query.py --query "あなたの質問"
```

### 質問応答（インタラクティブモード）

```bash
python rag_query.py
```

### APIサーバーの起動

```bash
python api.py
```

APIサーバーは`http://localhost:8000`で起動し、以下のエンドポイントを提供します:

- `GET /health`: ヘルスチェック（埋め込みプロバイダー情報も返します）
- `POST /query`: 質問応答API
- `GET /docs`: Swagger UIによるAPIドキュメント

## APIの使用例

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "あなたの質問"}'
```

## 埋め込みモデルの切り替え

システムは2つの埋め込みプロバイダーをサポートしています：

1. **OpenAI埋め込み**：デフォルトでは`text-embedding-ada-002`モデルを使用
2. **Cohere埋め込み**：`embed-multilingual-v3.0`モデルを使用

埋め込みプロバイダーを切り替えるには、`.env`ファイルの`EMBEDDING_PROVIDER`変数を変更します：

```
EMBEDDING_PROVIDER=openai  # OpenAI埋め込みを使用
```

または

```
EMBEDDING_PROVIDER=cohere  # Cohere埋め込みを使用
```

**重要**: 埋め込みプロバイダーを変更した場合は、以下の手順が必要です：

1. 既存の`chroma_db`ディレクトリを削除する必要があります：
   ```bash
   # Windowsの場合
   Remove-Item -Recurse -Force chroma_db
   
   # Linuxの場合
   rm -rf chroma_db
   ```

2. 新しいベクトルストアを生成するために`rag_generator.py`を再実行します：
   ```bash
   python rag_generator.py
   ```

これは、異なる埋め込みプロバイダーが生成するベクトルの次元数が異なるため、互換性がないためです。埋め込みプロバイダーを変更せずに新しいドキュメントを追加する場合は、ベクトルストアを削除する必要はありません。

## ディレクトリ構造

- `chroma_db/`: ベクトルストアデータ
- `rag_base_data/html/`: HTMLソースファイル
- `stats/`: トークン使用量統計情報

## 注意事項

- OpenAI APIキーとCohere APIキーが必要です
- HTMLファイルは`rag_base_data/html/`ディレクトリに配置する必要があります
- ベクトルストアは`chroma_db/`ディレクトリに保存されます

## プロジェクト構造

- `config.py`: システム設定ファイル
- `rag_generator.py`: HTMLファイルの読み込みとベクトルストア生成用スクリプト
- `rag_query.py`: ベクトルストアへの問い合わせ用スクリプト
- `requirements.txt`: 必要なPythonライブラリ
- `.env`: 環境変数の設定ファイル（.gitignoreに含まれており、リポジトリには保存されません）
- `chroma_db/`: ベクトルストアの保存ディレクトリ（自動生成、.gitignoreに含まれています）
- `rag_base_data/html/`: HTMLファイルが格納されているディレクトリ

## 将来の拡張

- 異なる埋め込みモデル（Local等）へのサポート追加
- 複数言語のテキスト処理の改善
- WebアプリケーションUIの追加

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。 