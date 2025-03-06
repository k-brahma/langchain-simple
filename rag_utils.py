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


def create_rag_chain(embedding_provider=None, llm_provider=None):
    """
    RAGチェーンを作成します。

    パラメータ:
        embedding_provider (str, optional): 使用する埋め込みプロバイダー。
            Noneの場合は、get_embedding_provider()の値を使用します。
        llm_provider (str, optional): 使用するLLMプロバイダー。
            Noneの場合は、get_llm_provider()の値を使用します。

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
        raise ValueError("ベクトルストアの読み込みに失敗しました")

    # リトリーバーを作成
    print("リトリーバーを作成します")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    print("リトリーバーの作成完了")

    # LLMを初期化
    print(f"{llm_provider}モデルを初期化します")
    llm = ChatOpenAI()
    print(f"{llm_provider}モデルの初期化完了")

    # プロンプトテンプレートを作成
    print("プロンプトテンプレートを作成します")
    prompt = ChatPromptTemplate.from_template(
        """
        あなたは親切なアシスタントです。以下の情報を元に質問に答えてください。
        
        コンテキスト情報:
        {context}
        
        質問: {input}
        
        回答:
        """
    )
    print("プロンプトテンプレートの作成完了")

    # ドキュメント結合チェーンを作成
    print("ドキュメント結合チェーンを作成します")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    print("ドキュメント結合チェーンの作成完了")

    # 最終的なRAGチェーンを作成
    print("最終的なRAGチェーンを作成します")
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("RAGチェーンの作成完了")

    return rag_chain


def query_rag(rag_chain, query):
    """
    RAGチェーンで問い合わせを実行する関数。

    指定されたRAGチェーンを使用して、ユーザーからの問い合わせに応答します。

    パラメータ:
        rag_chain (RetrievalChain): 使用するRAGチェーン。
        query (str): ユーザーからの問い合わせ。

    戻り値:
        dict: RAGチェーンからの応答。
    """
    print(f"RAGチェーンに問い合わせを実行します: {query}")
    try:
        response = rag_chain.invoke({"input": query})
        print("生のレスポンス:", response)
        return response
    except Exception as e:
        print(f"query_rag内でエラーが発生: {str(e)}")
        traceback.print_exc()
        raise  # 元の例外を再度スローして、呼び出し元で処理できるようにする


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
