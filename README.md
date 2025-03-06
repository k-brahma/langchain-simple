# HTML RAGシステム

このプロジェクトは、HTMLファイルをベースにしたRAG（Retrieval Augmented Generation）システムを構築します。  
`rag_base_data/html`ディレクトリ内のHTMLファイルを読み込み、ベクトルストアを作成し、質問応答システムを提供します。

## 機能

- HTMLファイルの読み込みと処理
- テキストのチャンク分割
- OpenAI埋め込みによるベクトル化とChromaベクトルストアへの保存
- OpenAI GPTモデルを使用した質問応答
- コマンドライン対話インターフェース

## セットアップ

1. 必要なライブラリをインストールします：

```bash
pip install -r requirements.txt
```

2. 環境変数を設定します：

このリポジトリには`.env_sample`ファイルが含まれています。以下の手順で環境変数を設定してください：

**Windowsの場合:**
```bash
copy .env_sample .env
```

**macOS/Linuxの場合:**
```bash
cp .env_sample .env
```

3. `.env`ファイルをテキストエディタで開き、OpenAI APIキーを設定します：

```
# .envファイルの中身
OPENAI_API_KEY=sk-あなたのAPIキーをここに入力してください
```

> **OpenAI APIキーの取得方法:**
> 1. [OpenAIのウェブサイト](https://platform.openai.com/)にアクセスしてアカウントを作成またはログインします
> 2. 右上のプロフィールアイコンをクリックし、「API keys」を選択します
> 3. 「Create new secret key」ボタンをクリックして新しいAPIキーを作成します
> 4. 作成されたキーをコピーして`.env`ファイルに貼り付けます（このキーは一度しか表示されないので注意してください）

## 使用方法

このシステムは2つの独立したスクリプトで構成されています：

1. `rag_generator.py` - HTMLファイルからベクトルストアを生成するスクリプト
2. `rag_query.py` - 生成されたベクトルストアに対して問い合わせを行うスクリプト

### 1. ベクトルストアの生成

まず、HTMLファイルをロードしてベクトルストアを作成します：

```bash
python rag_generator.py
```

このスクリプトは、HTMLファイルを読み込み、テキストを抽出し、チャンクに分割してベクトル化し、`chroma_db`ディレクトリに保存します。

### 2. RAGシステムへの問い合わせ

ベクトルストアが作成された後、以下のコマンドで問い合わせを行うことができます：

```bash
# 対話モード
python rag_query.py

# コマンドラインで直接質問
python rag_query.py -q "LangChainとは何ですか？"
```

対話モードでは、複数の質問を連続して入力できます。終了するには「exit」または「quit」と入力します。

## よくあるエラーと解決方法

### APIキー関連のエラー

```
Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.
```

**解決方法:** `.env`ファイルが正しく作成され、有効なAPIキーが設定されていることを確認してください。

### ベクトルストアが見つからないエラー

```
Error: Vector store not found. Please run rag_generator.py first.
```

**解決方法:** 先に`python rag_generator.py`を実行して、ベクトルストアを生成してください。

## プロジェクト構造

- `rag_generator.py`: HTMLファイルの読み込みとベクトルストア生成用スクリプト
- `rag_query.py`: ベクトルストアへの問い合わせ用スクリプト
- `requirements.txt`: 必要なPythonライブラリ
- `.env`: 環境変数の設定ファイル（.gitignoreに含まれており、リポジトリには保存されません）
- `.env_sample`: 環境変数設定のサンプルファイル
- `chroma_db/`: ベクトルストアの保存ディレクトリ（自動生成、.gitignoreに含まれています）
- `rag_base_data/html/`: HTMLファイルが格納されているディレクトリ

## 将来の拡張

- 異なる埋め込みモデル（Cohere、Local等）へのサポート追加
- 複数言語のテキスト処理の改善
- WebアプリケーションUIの追加

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。 