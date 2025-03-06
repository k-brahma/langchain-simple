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
import traceback
from datetime import datetime

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 環境変数を読み込む
load_dotenv()

# ディレクトリ設定
CHROMA_DIRECTORY = "chroma_db"
HTML_DATA_DIRECTORY = "rag_base_data/html"


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


def load_html_files():
    """
    HTMLファイルを読み込む関数。

    指定されたディレクトリ内のすべてのHTMLファイルを検出し、
    各ファイルをCustomHTMLLoaderを使用して読み込みます。
    読み込まれたドキュメントはリストとして返されます。

    戻り値:
        list: 読み込まれたDocumentオブジェクトのリスト。
    """
    html_files = glob.glob(f"{HTML_DATA_DIRECTORY}/*.html")
    print(f"検出されたHTMLファイル数: {len(html_files)}")
    print(f"検出されたファイルリスト: {', '.join(html_files)}")
    all_documents = []

    if not html_files:
        print("HTMLファイルが見つかりませんでした。ディレクトリパスを確認してください。")
        print(f"現在の作業ディレクトリ: {os.getcwd()}")
        print(
            f"ディレクトリ内容: {os.listdir(HTML_DATA_DIRECTORY) if os.path.exists(HTML_DATA_DIRECTORY) else '存在しません'}"
        )
        return all_documents

    for file_path in html_files:
        try:
            print(f"処理開始: {file_path}")
            # カスタムHTMLローダーを使用
            loader = CustomHTMLLoader(file_path)
            documents = loader.load()

            if not documents:
                print(f"ファイルからドキュメントを抽出できませんでした: {file_path}")
                continue

            # ファイル名をメタデータに追加
            file_name = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source"] = file_name
                # ドキュメントの内容の一部をログに出力
                content_preview = (
                    doc.page_content[:100] + "..."
                    if len(doc.page_content) > 100
                    else doc.page_content
                )
                print(f"抽出されたコンテンツ (先頭100文字): {content_preview}")

            all_documents.extend(documents)
            print(
                f"読み込み完了: {file_name} から {len(documents)} 件のドキュメントを読み込みました"
            )
        except Exception as e:
            print(f"ファイル読み込みエラー: {file_path} - {str(e)}")
            traceback.print_exc()

    print(f"合計 {len(all_documents)} 件のドキュメントを読み込みました")
    return all_documents


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


def create_vector_store(chunks):
    """
    ベクトルストアを作成する関数。

    指定されたチャンクを使用して、Chromaベクトルストアを作成し、
    指定されたディレクトリに保存します。

    パラメータ:
        chunks (list): ベクトルストアに追加するチャンクのリスト。

    戻り値:
        Chroma: 作成されたベクトルストアオブジェクト。
    """
    print("ベクトルストアの作成を開始します")

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(CHROMA_DIRECTORY):
        os.makedirs(CHROMA_DIRECTORY)
        print(f"ディレクトリを作成しました: {CHROMA_DIRECTORY}")

    # OpenAIの埋め込みを使用
    print("OpenAI埋め込みを初期化中...")
    embeddings = OpenAIEmbeddings()
    print("埋め込みの初期化完了")

    # Chromaベクトルストアを作成
    print(f"Chromaベクトルストアを作成中... ({len(chunks)} チャンク)")
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIRECTORY
    )

    # ベクトルストアを保存 - 最新のlangchain-chromaでは不要な場合があります
    # 最新バージョンではfrom_documentsで自動的に保存されるため
    print("ベクトルストアを保存中...")
    # vector_store.persist()  # 最新バージョンでは不要な場合があります
    print(f"ベクトルストア作成完了: {CHROMA_DIRECTORY}")

    return vector_store


def load_vector_store():
    """
    保存されたベクトルストアを読み込む関数。

    ベクトルストアが存在しない場合はエラーメッセージを表示し、Noneを返します。
    ベクトルストアが存在する場合は、Chromaオブジェクトを返します。

    戻り値:
        Chroma: ベクトルストアオブジェクト、またはNone。
    """
    try:
        # 環境変数の読み込み
        load_dotenv()
        print("環境変数を読み込みました")

        # ベクトルストアの保存先ディレクトリ
        print(f"ベクトルストアディレクトリ: {CHROMA_DIRECTORY}")

        # ディレクトリの存在確認
        if not os.path.exists(CHROMA_DIRECTORY):
            print(f"エラー: ベクトルストア '{CHROMA_DIRECTORY}' が見つかりません。")
            print("rag_generator.pyを先に実行して、ベクトルストアを生成してください。")
            return None

        print(f"ベクトルストアディレクトリの内容を確認:")
        for item in os.listdir(CHROMA_DIRECTORY):
            print(f"  - {item}")

        # OpenAIの埋め込みを初期化
        print("OpenAI埋め込みを初期化中...")
        embeddings = OpenAIEmbeddings()
        print("埋め込みの初期化完了")

        print(f"Chromaベクトルストアを読み込み中...")
        vector_store = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)
        print("ベクトルストアの読み込み完了")
        return vector_store
    except Exception as e:
        print(f"ベクトルストア読み込み中にエラーが発生しました: {e}")
        print("詳細なエラー情報:")
        traceback.print_exc()
        return None


def create_rag_chain(vector_store):
    """
    RAGチェーンを作成する関数。

    指定されたベクトルストアを使用して、RAGチェーンを構築します。
    チェーンはリトリーバー、LLM、プロンプトテンプレートを含みます。

    パラメータ:
        vector_store (Chroma): 使用するベクトルストア。

    戻り値:
        RetrievalChain: 構築されたRAGチェーン。
    """
    try:
        print("RAGチェーンの作成を開始します")
        # リトリーバーの作成
        print("リトリーバーを作成します")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("リトリーバーの作成完了")

        # LLMを初期化 (temperature=0で決定論的な回答に)
        print("ChatOpenAIモデルを初期化します")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        print("ChatOpenAIモデルの初期化完了")

        # プロンプトテンプレートを作成
        print("プロンプトテンプレートを作成します")
        prompt = ChatPromptTemplate.from_template(
            """以下の情報をもとに質問に日本語で回答してください。
            与えられた情報に基づいた回答のみを行い、情報がない場合は「提供された情報からは回答できません」と答えてください。

            コンテキスト:
            {context}

            質問: {input}
            """
        )
        print("プロンプトテンプレートの作成完了")

        # ドキュメント結合チェーンの作成
        print("ドキュメント結合チェーンを作成します")
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        print("ドキュメント結合チェーンの作成完了")

        # 最終的なRAGチェーンの作成
        print("最終的なRAGチェーンを作成します")
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("RAGチェーンの作成完了")

        return rag_chain
    except Exception as e:
        print(f"RAGチェーン作成中にエラーが発生: {str(e)}")
        print("詳細なエラー情報:")
        traceback.print_exc()
        raise


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
