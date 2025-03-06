"""
rag_query.py

このスクリプトは、RAG（Retrieval-Augmented Generation）システムを使用して、
ユーザーの質問に回答するためのインターフェースを提供します。
コマンドラインから単一の質問に回答する機能と、対話モードでの質問応答機能を提供します。
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv

from config import (
    get_embedding_provider,
    get_llm_provider,
    set_embedding_provider,
    set_llm_provider,
)
from rag_utils import ConversationMemory, create_rag_chain, query_rag

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


def interactive_mode(history_file=None):
    """
    対話モードで質問応答を行います。
    ユーザーからの入力を受け取り、RAGシステムを使用して回答を生成します。
    'exit'または'quit'と入力されるまで続けます。

    Args:
        history_file (str, optional): 会話履歴を保存・読み込むファイルパス
    """
    print("RAGシステム対話モードを開始します")
    print("質問を入力してください。終了するには 'exit' または 'quit' と入力してください。")
    print("特殊コマンド:")
    print("  !save [filename] - 会話履歴を保存")
    print("  !load [filename] - 会話履歴を読み込み")
    print("  !clear - 会話履歴をクリア")
    print("  !history - 会話履歴を表示")

    try:
        # 埋め込みプロバイダーとLLMプロバイダーを取得
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider()
        print(f"使用する埋め込みプロバイダー: {embedding_provider}")
        print(f"使用するLLMプロバイダー: {llm_provider}")

        # 会話履歴を初期化
        memory = ConversationMemory()

        # 履歴ファイルが指定されていない場合はデフォルトを使用
        if history_file is None:
            history_file = "conversation_history.json"

        # 履歴ファイルが存在する場合は読み込む
        if os.path.exists(history_file):
            success = memory.load_from_file(history_file)
            if success:
                print(f"会話履歴を {history_file} から読み込みました")
            else:
                print(f"会話履歴の読み込みに失敗しました: {history_file}")

        # RAGチェーンを作成
        rag_chain = create_rag_chain(embedding_provider, llm_provider, memory)
        if not rag_chain:
            print("RAGチェーンの作成に失敗しました。終了します。")
            return

        while True:
            # ユーザー入力を取得
            user_input = input("\n質問を入力してください: ")

            # 終了コマンドをチェック
            if user_input.lower() in ["exit", "quit", "終了"]:
                # 終了時に会話履歴を保存
                if memory.history:
                    memory.save_to_file(history_file)
                    print(f"会話履歴を {history_file} に保存しました")
                print("対話モードを終了します。")
                break

            # 特殊コマンドの処理
            if user_input.startswith("!"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()

                if command == "!save":
                    # 会話履歴を保存
                    save_file = parts[1] if len(parts) > 1 else history_file
                    memory.save_to_file(save_file)
                    continue

                elif command == "!load":
                    # 会話履歴を読み込み
                    load_file = parts[1] if len(parts) > 1 else history_file
                    if memory.load_from_file(load_file):
                        print(f"会話履歴を {load_file} から読み込みました")
                        # 履歴を読み込んだ後、RAGチェーンを再作成
                        rag_chain = create_rag_chain(embedding_provider, llm_provider, memory)
                    else:
                        print(f"会話履歴の読み込みに失敗しました: {load_file}")
                    continue

                elif command == "!clear":
                    # 会話履歴をクリア
                    memory = ConversationMemory()
                    print("会話履歴をクリアしました")
                    # 履歴をクリアした後、RAGチェーンを再作成
                    rag_chain = create_rag_chain(embedding_provider, llm_provider, memory)
                    continue

                elif command == "!history":
                    # 会話履歴を表示
                    if memory.history:
                        print("\n=== 会話履歴 ===")
                        print(memory.get_formatted_history())
                        print("================")
                    else:
                        print("会話履歴はありません")
                    continue

            # 空の入力をスキップ
            if not user_input.strip():
                continue

            try:
                # RAGチェーンに問い合わせ
                result = query_rag(rag_chain, user_input, memory)

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
        # Ctrl+Cで終了時も会話履歴を保存
        if history_file and memory.history:
            memory.save_to_file(history_file)
            print(f"\n会話履歴を {history_file} に保存しました")
        print("\n対話モードを終了します。")
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def single_query(query, history_file=None):
    """
    単一の質問に回答します。

    Args:
        query (str): 質問文字列
        history_file (str, optional): 会話履歴を保存・読み込むファイルパス
    """
    print(f"質問: {query}")

    try:
        # 埋め込みプロバイダーとLLMプロバイダーを取得
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider()
        print(f"使用する埋め込みプロバイダー: {embedding_provider}")
        print(f"使用するLLMプロバイダー: {llm_provider}")

        # 会話履歴を初期化
        memory = ConversationMemory()

        # 履歴ファイルが指定されていて、存在する場合は読み込む
        if history_file and os.path.exists(history_file):
            memory.load_from_file(history_file)
            print(f"会話履歴を {history_file} から読み込みました")

        # RAGチェーンを作成
        rag_chain = create_rag_chain(embedding_provider, llm_provider, memory)
        if not rag_chain:
            print("RAGチェーンの作成に失敗しました。終了します。")
            return

        # RAGチェーンに問い合わせ
        result = query_rag(rag_chain, query, memory)

        # 回答を表示
        if "answer" in result:
            print("\n回答:", result["answer"])
        else:
            print("\n回答を生成できませんでした。")

        # ドキュメントソースを表示
        print_document_sources(result)

        # 会話履歴を保存
        if history_file:
            memory.save_to_file(history_file)
            print(f"会話履歴を {history_file} に保存しました")

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
    parser.add_argument(
        "--history_file",
        type=str,
        default="conversation_history.json",
        help="会話履歴を保存・読み込むファイルパス",
    )
    args = parser.parse_args()

    # 埋め込みプロバイダーとLLMプロバイダーを設定
    set_embedding_provider(args.embedding_provider)
    set_llm_provider(args.llm_provider)

    if args.query:
        # 単一の質問に回答
        single_query(args.query, args.history_file)
    else:
        # 対話モードで実行
        interactive_mode(args.history_file)


if __name__ == "__main__":
    main()
