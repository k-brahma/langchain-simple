"""
rag_query.py

このスクリプトは、RAG（Retrieval-Augmented Generation）システムを使用して、
ユーザーの質問に回答するためのインターフェースを提供します。
コマンドラインから単一の質問に回答する機能と、対話モードでの質問応答機能を提供します。
"""

import argparse
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv

from config import (
    get_embedding_provider,
    get_llm_provider,
    set_embedding_provider,
    set_llm_provider,
)
from rag_utils import create_rag_chain

# 環境変数を読み込む
load_dotenv()


def print_document_sources(response):
    """
    レスポンスからドキュメントソースを抽出して表示します。

    Args:
        response (dict): RAGチェーンからのレスポンス
    """
    sources = []
    if "context" in response and response["context"]:
        for doc in response["context"]:
            if "source" in doc.metadata:
                sources.append(doc.metadata["source"])

    if sources:
        print("\n参照ドキュメント:")
        for source in set(sources):
            print(f"  - {source}")


def interactive_mode():
    """
    対話モードで質問応答を行います。
    ユーザーからの入力を受け取り、RAGシステムを使用して回答を生成します。
    'exit'または'quit'と入力されるまで続けます。
    """
    print("RAGシステム対話モードを開始します")
    print("質問を入力してください。終了するには 'exit' または 'quit' と入力してください。")

    try:
        # 埋め込みプロバイダーとLLMプロバイダーを取得
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider()
        print(f"使用する埋め込みプロバイダー: {embedding_provider}")
        print(f"使用するLLMプロバイダー: {llm_provider}")

        # RAGチェーンを作成
        rag_chain = create_rag_chain(embedding_provider, llm_provider)
        if not rag_chain:
            print("RAGチェーンの作成に失敗しました。終了します。")
            return

        while True:
            # ユーザー入力を取得
            user_input = input("\n質問を入力してください: ")

            # 終了コマンドをチェック
            if user_input.lower() in ["exit", "quit", "終了"]:
                print("対話モードを終了します。")
                break

            # 空の入力をスキップ
            if not user_input.strip():
                continue

            try:
                # RAGチェーンに問い合わせ
                result = rag_chain.invoke({"input": user_input})
                print(f"生のレスポンス: {result}")

                # 回答を表示
                if "answer" in result:
                    print("\n回答:", result["answer"])
                else:
                    print("\n回答を生成できませんでした。")

                # ドキュメントソースを表示
                print_document_sources(result)

            except Exception as e:
                print(f"エラーが発生しました: {e}")
                import traceback

                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n対話モードを終了します。")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def single_query(query):
    """
    単一の質問に回答します。

    Args:
        query (str): 質問文字列
    """
    print(f"質問: {query}")

    try:
        # 埋め込みプロバイダーとLLMプロバイダーを取得
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider()
        print(f"使用する埋め込みプロバイダー: {embedding_provider}")
        print(f"使用するLLMプロバイダー: {llm_provider}")

        # RAGチェーンを作成
        rag_chain = create_rag_chain(embedding_provider, llm_provider)
        if not rag_chain:
            print("RAGチェーンの作成に失敗しました。終了します。")
            return

        # RAGチェーンに問い合わせ
        result = rag_chain.invoke({"input": query})
        print(f"生のレスポンス: {result}")

        # 回答を表示
        if "answer" in result:
            print("\n回答:", result["answer"])
        else:
            print("\n回答を生成できませんでした。")

        # ドキュメントソースを表示
        print_document_sources(result)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="RAGシステムを使用した質問応答")
    parser.add_argument(
        "--query", type=str, help="回答する質問。指定しない場合は対話モードで実行します。"
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default=get_embedding_provider(),
        choices=["openai", "cohere"],
        help="埋め込みプロバイダー (openai または cohere)",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default=get_llm_provider(),
        choices=["openai"],
        help="LLMプロバイダー (現在は openai のみサポート)",
    )
    args = parser.parse_args()

    # 埋め込みプロバイダーとLLMプロバイダーを設定
    set_embedding_provider(args.embedding_provider)
    set_llm_provider(args.llm_provider)

    if args.query:
        # 単一の質問に回答
        single_query(args.query)
    else:
        # 対話モードで実行
        interactive_mode()


if __name__ == "__main__":
    main()
