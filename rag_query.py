import argparse
import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# 埋め込みモジュールをインポート
# from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# from langchain_openai import OpenAIEmbeddings

# 環境変数を読み込む
load_dotenv()

# ベクトルストアの保存先
PERSIST_DIRECTORY = "vectorstore"

# 埋め込みプロバイダーと設定
EMBEDDING_PROVIDER = "cohere"
COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"


def load_vector_store():
    """保存されたベクトルストアを読み込む"""
    try:
        # 環境変数の読み込み
        load_dotenv()

        # ベクトルストアの保存先ディレクトリ
        persist_directory = "chroma_db"

        # ディレクトリの存在確認
        if not os.path.exists(persist_directory):
            print(f"エラー: ベクトルストア '{persist_directory}' が見つかりません。")
            print("rag_generator.pyを先に実行して、ベクトルストアを生成してください。")
            return None

        # OpenAIの埋め込みを初期化
        embeddings = OpenAIEmbeddings()

        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vector_store
    except Exception as e:
        print(f"ベクトルストア読み込み中にエラーが発生しました: {e}")
        return None


def create_rag_chain(vector_store):
    """RAGチェーンを作成する関数"""
    # リトリーバーの作成
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # LLMを初期化 (temperature=0で決定論的な回答に)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_template(
        """以下の情報をもとに質問に日本語で回答してください。
        与えられた情報に基づいた回答のみを行い、情報がない場合は「提供された情報からは回答できません」と答えてください。

        コンテキスト:
        {context}

        質問: {question}
        """
    )

    # ドキュメント結合チェーンの作成
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 最終的なRAGチェーンの作成
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def query_rag(rag_chain, query):
    """RAGチェーンで問い合わせを実行する関数"""
    response = rag_chain.invoke({"question": query})
    return response


def print_document_sources(response):
    """参照ドキュメントのソースを表示する関数"""
    source_documents = response.get("context", [])
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


def interactive_mode():
    """インタラクティブモード（対話型）の関数"""
    # ベクトルストアを読み込む
    vector_store = load_vector_store()
    if not vector_store:
        return

    # RAGチェーンを作成
    rag_chain = create_rag_chain(vector_store)

    print("\nRAGシステムの対話モードを開始します（終了するには 'exit' または 'quit' と入力）")
    print("質問を入力してください:")

    while True:
        user_input = input("\n> ")
        user_input = user_input.strip()

        if user_input.lower() in ["exit", "quit", "終了"]:
            print("対話モードを終了します。")
            break

        if not user_input:
            continue

        try:
            print("問い合わせ中...")
            response = query_rag(rag_chain, user_input)
            print(f"\n回答: {response['answer']}")
            print_document_sources(response)
        except Exception as e:
            print(f"エラーが発生しました: {e}")


def single_query(query):
    """単一の問い合わせを実行する関数"""
    try:
        # ベクトルストアを読み込む
        vector_store = load_vector_store()
        if not vector_store:
            return

        # RAGチェーンを作成
        rag_chain = create_rag_chain(vector_store)

        # 問い合わせを実行
        print(f"問い合わせ: {query}")
        response = query_rag(rag_chain, query)

        # 結果を表示
        print(f"\n回答: {response['answer']}")
        print_document_sources(response)
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="RAGシステムへの問い合わせ")
    parser.add_argument("-q", "--query", help="単一の問い合わせを実行する")
    args = parser.parse_args()

    if args.query:
        single_query(args.query)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
