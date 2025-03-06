"""
rag_utils.py

このモジュールは、RAG（Retrieval-Augmented Generation）システムの共通ユーティリティを提供します。
HTMLファイルの読み込み、ドキュメントの分割、ベクトルストアの操作など、
複数のモジュールで使用される共通機能が含まれています。

主な機能:
- HTMLファイルの読み込みと解析
- ドキュメントのチャンクへの分割
- ベクトルストアの作成と読み込み
- RAGチェーンの構築
"""

import glob
import json
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 設定を読み込む
from config import (
    CHROMA_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HTML_DATA_DIRECTORY,
    get_embedding_provider,
    get_llm_provider,
    is_cohere_trial,
)

# 環境変数を読み込む
load_dotenv()

# 埋め込みプロバイダーの設定
DEFAULT_EMBEDDING_PROVIDER = "openai"  # "openai" または "cohere"


class CustomHTMLLoader:
    """
    カスタムHTMLローダー。

    指定されたHTMLファイルを読み込み、テキストとメタデータを抽出して
    Documentオブジェクトとして返します。

    属性:
        file_path (str): 読み込むHTMLファイルのパス。

    メソッド:
        load(): HTMLファイルを読み込み、Documentオブジェクトのリストを返します。
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """
        HTMLファイルを読み込み、Documentオブジェクトのリストを返します。

        戻り値:
            list: Documentオブジェクトのリスト。
        """
        try:
            # UTF-8でファイルを読み込む
            print(f"ファイル読み込み開始: {self.file_path}")
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"ファイルサイズ: {len(content)} バイト")

            # BeautifulSoupを使用してHTMLを解析
            print(f"HTMLの解析開始: {self.file_path}")
            soup = BeautifulSoup(content, "html.parser")

            # 不要なタグを削除
            for script in soup(["script", "style"]):
                script.extract()

            # テキストを取得
            text = soup.get_text(separator="\n")

            # 余分な空白行を削除
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            print(f"抽出されたテキストサイズ: {len(text)} 文字")

            # メタデータを作成
            metadata = {
                "source": os.path.basename(self.file_path),
                "title": soup.title.string if soup.title else os.path.basename(self.file_path),
            }
            print(f"メタデータ: {metadata}")

            # Documentオブジェクトを作成
            document = Document(page_content=text, metadata=metadata)
            print(f"ドキュメント作成完了: {self.file_path}")

            return [document]
        except Exception as e:
            print(f"ファイル読み込みエラー: {self.file_path} - {str(e)}")
            traceback.print_exc()
            return []


def load_html_files(html_files=None):
    """
    HTMLファイルを読み込み、ドキュメントのリストを返します。

    Args:
        html_files (list, optional): 読み込むHTMLファイルのリスト。
            Noneの場合は、デフォルトのディレクトリからすべてのHTMLファイルを読み込みます。

    Returns:
        list: ドキュメントのリスト
    """
    if html_files is None:
        # デフォルトのディレクトリからHTMLファイルを検索
        html_files = glob.glob(os.path.join(HTML_DATA_DIRECTORY, "**/*.html"), recursive=True)

    print(f"検出されたHTMLファイル数: {len(html_files)}")
    print(f"検出されたファイルリスト: {', '.join(html_files)}")

    documents = []
    embedding_provider = get_embedding_provider()

    for file_path in html_files:
        print(f"処理開始: {file_path}")
        loader = CustomHTMLLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
        print(
            f"読み込み完了: {os.path.basename(file_path)} から {len(docs)} 件のドキュメントを読み込みました"
        )

        # Cohereの場合はレート制限対策として遅延を入れる
        if embedding_provider == "cohere" and len(html_files) > 1:
            delay_time = 1  # 一律1秒の遅延時間
            print(f"Cohereのレート制限対策として{delay_time}秒間待機します...")
            time.sleep(delay_time)

    return documents


def split_documents(documents):
    """
    ドキュメントをチャンクに分割する関数。

    指定されたドキュメントをRecursiveCharacterTextSplitterを使用して
    チャンクに分割し、分割されたチャンクのリストを返します。

    パラメータ:
        documents (list): 分割するDocumentオブジェクトのリスト。

    戻り値:
        list: 分割されたチャンクのリスト。
    """
    print(f"ドキュメント分割開始: {len(documents)} 件のドキュメント")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"分割完了: {len(chunks)} チャンクに分割されました")

    # サンプルチャンクのログ
    if chunks:
        sample_chunk = chunks[0]
        print(f"サンプルチャンク内容: {sample_chunk.page_content[:100]}...")
        print(f"サンプルチャンクのメタデータ: {sample_chunk.metadata}")

    return chunks


def create_embeddings(embedding_provider=None):
    """
    埋め込みモデルを作成する関数。

    指定された埋め込みプロバイダーに基づいて、適切な埋め込みモデルを作成します。

    パラメータ:
        embedding_provider (str, optional): 使用する埋め込みプロバイダー。
            "openai"または"cohere"を指定できます。
            Noneの場合は、get_embedding_provider()の値を使用します。

    戻り値:
        Embeddings: 作成された埋め込みモデル。
    """
    if embedding_provider is None:
        embedding_provider = get_embedding_provider()

    print(f"{embedding_provider}埋め込みを初期化中...")

    if embedding_provider == "openai":
        # OpenAI埋め込みを使用
        embeddings = OpenAIEmbeddings()
    elif embedding_provider == "cohere":
        # Cohere埋め込みを使用
        try:
            # Cohereの埋め込みを初期化
            embeddings = CohereEmbeddings(
                model="embed-multilingual-v3.0",  # モデルを指定
                client=None,
                async_client=None,
            )
        except Exception as e:
            print(f"Cohere埋め込み初期化エラー: {e}")
            # フォールバック方法
            embeddings = CohereEmbeddings(
                model="embed-multilingual-v3.0",  # モデルを指定
                client=None,
                async_client=None,
            )
    else:
        raise ValueError(f"サポートされていない埋め込みプロバイダー: {embedding_provider}")

    return embeddings


def create_vector_store(chunks, embedding_provider=None):
    """
    チャンクからベクトルストアを作成します。

    Args:
        chunks (list): ドキュメントチャンクのリスト
        embedding_provider (str, optional): 使用する埋め込みプロバイダー

    Returns:
        Chroma: 作成されたベクトルストア
    """
    print("ベクトルストアの作成を開始します")

    # 埋め込みを作成
    embeddings = create_embeddings(embedding_provider)

    # Cohereのレート制限対策として、チャンクを小さなバッチに分割して処理
    if (
        embedding_provider == "cohere"
        or (embedding_provider is None and get_embedding_provider() == "cohere")
    ) and is_cohere_trial():
        print("Cohereの無料トライアルキーを使用しているため、レート制限対策を適用します")
        batch_size = 5  # 一度に処理するチャンク数
        all_chunks = chunks.copy()
        processed_chunks = []
        total_tokens = 0

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            print(
                f"バッチ処理中: {i+1}〜{min(i+batch_size, len(all_chunks))}/{len(all_chunks)}チャンク"
            )

            # バッチのトークン数を概算（1文字あたり約0.3トークンと仮定）
            batch_text = " ".join([chunk.page_content for chunk in batch])
            estimated_tokens = len(batch_text) * 0.3
            print(f"バッチの推定トークン数: 約{int(estimated_tokens)}トークン")
            total_tokens += estimated_tokens

            # 処理開始時間を記録
            batch_start_time = time.time()

            # 小さなバッチでベクトルストアを作成
            if i == 0:
                # 最初のバッチでベクトルストアを初期化
                vector_store = Chroma.from_documents(
                    documents=batch, embedding=embeddings, persist_directory=CHROMA_DIRECTORY
                )
                processed_chunks.extend(batch)
            else:
                # 既存のベクトルストアに追加
                vector_store = Chroma(
                    persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings
                )
                vector_store.add_documents(documents=batch)
                processed_chunks.extend(batch)

            # 処理時間を計算
            batch_time = time.time() - batch_start_time
            print(f"バッチ処理時間: {batch_time:.2f}秒")
            print(f"1トークンあたりの処理時間: {(batch_time / estimated_tokens) * 1000:.2f}ミリ秒")

            # バッチ間に遅延を追加
            if i + batch_size < len(all_chunks):
                delay_time = 1  # 一律1秒の遅延時間

                print(f"次のバッチ処理前に{delay_time}秒間待機します...")
                time.sleep(delay_time)

        print(f"全{len(processed_chunks)}チャンクの処理が完了しました")
        print(f"総推定トークン数: 約{int(total_tokens)}トークン")
    else:
        # 通常の処理（OpenAIまたはCohere有料プラン）
        if embedding_provider == "cohere" or (
            embedding_provider is None and get_embedding_provider() == "cohere"
        ):
            print("Cohereの有料プランを使用しているため、レート制限対策を適用しません")

        print(f"Chromaベクトルストアを作成中... ({len(chunks)} チャンク)")
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIRECTORY
        )

    # ベクトルストアを保存
    print("ベクトルストアを保存中...")
    # 最新バージョンではpersist()メソッドが不要な場合があります
    # vector_store.persist()
    print(f"ベクトルストア作成完了: {CHROMA_DIRECTORY}")

    return vector_store


def load_vector_store(embedding_provider=None):
    """
    保存されたベクトルストアを読み込みます。

    Args:
        embedding_provider (str, optional): 使用する埋め込みプロバイダー

    Returns:
        Chroma: 読み込まれたベクトルストア
    """
    print(f"ベクトルストアディレクトリ: {CHROMA_DIRECTORY}")

    # ディレクトリの内容を確認
    if os.path.exists(CHROMA_DIRECTORY):
        print("ベクトルストアディレクトリの内容を確認:")
        for item in os.listdir(CHROMA_DIRECTORY):
            print(f"  - {item}")

    # 埋め込みを作成
    embeddings = create_embeddings(embedding_provider)

    # ベクトルストアを読み込む
    print("Chromaベクトルストアを読み込み中...")
    vector_store = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)
    print("ベクトルストアの読み込み完了")

    return vector_store


def create_llm(llm_provider=None):
    """
    指定されたプロバイダーに基づいてLLMを作成します。

    Args:
        llm_provider (str, optional): 使用するLLMプロバイダー

    Returns:
        BaseChatModel: 初期化されたLLMインスタンス
    """
    if llm_provider is None:
        llm_provider = get_llm_provider()

    print(f"LLMプロバイダー: {llm_provider}")

    try:
        if llm_provider.lower() == "openai":
            # OpenAI APIキーを確認
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("OpenAI APIキーが設定されていません")
                return None

            # OpenAI LLMを作成
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
            )
        else:
            print(f"未サポートのLLMプロバイダー: {llm_provider}")
            return None
    except Exception as e:
        print(f"LLMの作成中にエラーが発生しました: {str(e)}")
        return None


class ConversationMemory:
    """
    会話履歴を管理するクラス。
    ユーザーの質問とシステムの回答を保存し、会話のコンテキストを維持します。
    """

    def __init__(self, max_history: int = 5):
        """
        ConversationMemoryを初期化します。

        Args:
            max_history (int): 保持する会話履歴の最大数
        """
        self.history = []
        self.max_history = max_history

    def add_interaction(self, user_query: str, system_response: Dict[str, Any]):
        """
        ユーザーの質問とシステムの回答を履歴に追加します。

        Args:
            user_query (str): ユーザーからの質問
            system_response (Dict[str, Any]): システムの回答（RAGチェーンの出力）
        """
        # system_responseから必要な情報を抽出
        answer = system_response.get("answer", "")

        # 履歴に追加する情報を整理
        self.history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "system_response": {
                    "answer": answer,
                    # 他の必要なフィールドがあれば追加
                },
            }
        )

        # 履歴が最大数を超えた場合、古いものから削除
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_formatted_history(self) -> str:
        """
        会話履歴をフォーマットして文字列として返します。

        Returns:
            str: フォーマットされた会話履歴
        """
        if not self.history:
            return ""

        formatted = "以下は過去の会話履歴です：\n\n"
        for i, interaction in enumerate(self.history):
            formatted += f"質問 {i+1}: {interaction['user_query']}\n"

            # system_responseが辞書型かどうかを確認
            if isinstance(interaction["system_response"], dict):
                answer = interaction["system_response"].get("answer", "")
            else:
                # 古い形式の場合の対応
                answer = str(interaction["system_response"])

            formatted += f"回答 {i+1}: {answer}\n\n"

        return formatted

    def save_to_file(self, filename: str = "conversation_history.json"):
        """
        会話履歴をJSONファイルに保存します。

        Args:
            filename (str): 保存するファイル名
        """
        # 履歴がない場合は保存しない
        if not self.history:
            print("保存する会話履歴がありません")
            return False

        # JSONシリアライズ可能な形式に変換
        serializable_history = []
        for interaction in self.history:
            # system_responseの処理
            if isinstance(interaction["system_response"], dict):
                system_response = {"answer": interaction["system_response"].get("answer", "")}
            else:
                # 古い形式の場合の対応
                system_response = {"answer": str(interaction["system_response"])}

            serializable_interaction = {
                "timestamp": interaction["timestamp"],
                "user_query": interaction["user_query"],
                "system_response": system_response,
            }
            serializable_history.append(serializable_interaction)

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
            print(f"会話履歴を {filename} に保存しました（{len(serializable_history)}件）")
            return True
        except Exception as e:
            print(f"会話履歴の保存中にエラーが発生しました: {str(e)}")
            return False

    def load_from_file(self, filename: str = "conversation_history.json") -> bool:
        """
        JSONファイルから会話履歴を読み込みます。

        Args:
            filename (str): 読み込むファイル名

        Returns:
            bool: 読み込みに成功した場合はTrue、失敗した場合はFalse
        """
        try:
            if not os.path.exists(filename):
                print(f"ファイル {filename} が見つかりません")
                return False

            with open(filename, "r", encoding="utf-8") as f:
                loaded_history = json.load(f)

            # 読み込んだデータの検証
            if not isinstance(loaded_history, list):
                print(f"無効な会話履歴形式: リストではありません")
                return False

            for item in loaded_history:
                if not isinstance(item, dict):
                    print(f"無効な会話履歴形式: 項目が辞書ではありません")
                    return False
                if "user_query" not in item or "system_response" not in item:
                    print(f"無効な会話履歴形式: 必須フィールドがありません")
                    return False

            self.history = loaded_history
            print(f"{filename} から会話履歴を読み込みました（{len(self.history)}件）")
            return True
        except json.JSONDecodeError as e:
            print(f"JSONデコードエラー: {str(e)}")
            return False
        except Exception as e:
            print(f"会話履歴の読み込み中にエラーが発生しました: {str(e)}")
            return False


def create_rag_chain(embedding_provider=None, llm_provider=None, memory=None):
    """
    RAGチェーンを作成します。

    パラメータ:
        embedding_provider (str, optional): 使用する埋め込みプロバイダー。
            Noneの場合は、get_embedding_provider()の値を使用します。
        llm_provider (str, optional): 使用するLLMプロバイダー。
            Noneの場合は、get_llm_provider()の値を使用します。
        memory (ConversationMemory, optional): 会話履歴を管理するメモリオブジェクト

    戻り値:
        Chain: 作成されたRAGチェーン
    """
    print("RAGチェーンの作成を開始します")

    # 埋め込みプロバイダーとLLMプロバイダーを取得
    if embedding_provider is None:
        embedding_provider = get_embedding_provider()
    if llm_provider is None:
        llm_provider = get_llm_provider()

    # ベクトルストアを読み込む
    vector_store = load_vector_store(embedding_provider)
    if vector_store is None:
        print("ベクトルストアの読み込みに失敗しました。")
        return None

    # リトリーバーを作成
    print("リトリーバーを作成中...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("リトリーバーの作成完了")

    # LLMを初期化
    print(f"LLMを初期化中... (プロバイダー: {llm_provider})")
    llm = create_llm(llm_provider)
    if llm is None:
        print("LLMの初期化に失敗しました。")
        return None
    print("LLMの初期化完了")

    # プロンプトテンプレートを作成
    print("プロンプトテンプレートを作成中...")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """あなたは親切で丁寧な日本語アシスタントです。
以下の情報源を使用して、ユーザーの質問に答えてください。
情報源に基づいて回答し、情報源にない内容は含めないでください。

{context}

会話の文脈を考慮して回答してください。ユーザーが前の会話を参照している場合は、
その文脈を理解した上で回答してください。例えば「それはどういう意味ですか？」
「もっと詳しく教えてください」などの質問には、前の回答の内容を踏まえて詳細に説明してください。

会話履歴:
{conversation_history}
""",
            ),
            ("human", "{input}"),
        ]
    )
    print("プロンプトテンプレートの作成完了")

    # ドキュメント結合チェーンを作成
    print("ドキュメント結合チェーンを作成中...")
    document_chain = create_stuff_documents_chain(llm, prompt)
    print("ドキュメント結合チェーンの作成完了")

    # RAGチェーンを作成
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("RAGチェーンの作成完了")

    return rag_chain


def query_rag(rag_chain, query, memory=None):
    """
    RAGチェーンを使用してクエリを実行します。

    パラメータ:
        rag_chain (Chain): 使用するRAGチェーン
        query (str): 実行するクエリ
        memory (ConversationMemory, optional): 会話履歴を管理するメモリオブジェクト

    戻り値:
        dict: RAGチェーンからの応答
    """
    print(f"クエリを実行中: {query}")
    try:
        # 会話履歴がある場合は、それを含める
        input_data = {"input": query}

        if memory and memory.history:
            formatted_history = memory.get_formatted_history()
            input_data["conversation_history"] = formatted_history
        else:
            # 会話履歴がない場合は空の文字列を設定
            input_data["conversation_history"] = ""

        print(f"入力データ: {input_data}")
        response = rag_chain.invoke(input_data)

        # 会話履歴に追加
        if memory:
            memory.add_interaction(query, response)

        return response
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"answer": f"エラーが発生しました: {str(e)}"}


def print_document_sources(response):
    """
    参照ドキュメントのソースを表示する関数。

    RAGチェーンの応答から参照ドキュメントのソースを抽出し、表示します。

    パラメータ:
        response (dict): RAGチェーンからの応答。
    """
    try:
        # 最新のLangChainでのレスポンス形式に対応
        source_documents = response.get("context", [])
        if not source_documents and "documents" in response:
            # 新しい形式の場合
            source_documents = response.get("documents", [])

        if source_documents:
            print("\n参照ドキュメント:")
            sources = set()
            for doc in source_documents:
                source = doc.metadata.get("source", "不明")
                sources.add(source)

            for source in sources:
                print(f"  - {source}")
        else:
            print("\n参照ドキュメントはありません")
    except Exception as e:
        print(f"ソース表示中にエラー: {str(e)}")
        traceback.print_exc()
