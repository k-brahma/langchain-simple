"""
rag_query.py

このスクリプトは、RAG（Retrieval-Augmented Generation）システムを使用して
ユーザーからの質問に回答します。

主な機能:
- ベクトルストアからの情報検索
- 質問応答チェーンの実行
- インタラクティブモードでのユーザー対話
- コマンドライン引数による単一質問への回答
"""

import argparse
import os
import sys
import traceback

from dotenv import load_dotenv

# 共通ユーティリティをインポート
from rag_utils import create_rag_chain, load_vector_store, print_document_sources, query_rag

# 環境変数を読み込む
load_dotenv()


def interactive_mode():
    """
    インタラクティブモードでRAGシステムを実行する関数。

    ユーザーからの質問を受け付け、RAGシステムを使用して回答します。
    'exit'または'quit'と入力されるまで繰り返し質問を受け付けます。
    """
    print("\nRAGシステム インタラクティブモード")
    print("質問を入力してください。終了するには 'exit' または 'quit' と入力してください。")

    # ベクトルストアを読み込む
    vector_store = load_vector_store()
    if not vector_store:
        print("ベクトルストアの読み込みに失敗しました。終了します。")
        return

    # RAGチェーンを作成
    try:
        rag_chain = create_rag_chain(vector_store)
    except Exception as e:
        print(f"RAGチェーンの作成に失敗しました: {str(e)}")
        traceback.print_exc()
        return

    # インタラクティブループ
    while True:
        try:
            # ユーザーからの入力を受け付ける
            query = input("\n質問> ")

            # 終了コマンドのチェック
            if query.lower() in ["exit", "quit"]:
                print("インタラクティブモードを終了します。")
                break

            # 空の入力をスキップ
            if not query.strip():
                print("質問を入力してください。")
                continue

            # RAGチェーンに問い合わせ
            try:
                response = query_rag(rag_chain, query)

                # レスポンスの構造を確認（デバッグ用）
                print("\nレスポンスキー:", response.keys())

                # 回答を表示
                if "answer" in response:
                    print("\n回答:", response["answer"])
                elif "result" in response:
                    print("\n回答:", response["result"])
                else:
                    print("\n回答: レスポンス形式が認識できません。")

                # 参照ドキュメントを表示
                print_document_sources(response)

            except Exception as e:
                print(f"エラーが発生しました: {str(e)}")
                traceback.print_exc()

        except KeyboardInterrupt:
            print("\nインタラクティブモードを終了します。")
            break
        except Exception as e:
            print(f"予期しないエラーが発生しました: {str(e)}")
            traceback.print_exc()


def single_query(query):
    """
    単一の質問に対してRAGシステムを実行する関数。

    パラメータ:
        query (str): 処理する質問。

    戻り値:
        bool: 処理が成功した場合はTrue、失敗した場合はFalse。
    """
    print(f"質問: {query}")

    # ベクトルストアを読み込む
    vector_store = load_vector_store()
    if not vector_store:
        print("ベクトルストアの読み込みに失敗しました。終了します。")
        return False

    # RAGチェーンを作成
    try:
        rag_chain = create_rag_chain(vector_store)
    except Exception as e:
        print(f"RAGチェーンの作成に失敗しました: {str(e)}")
        traceback.print_exc()
        return False

    # RAGチェーンに問い合わせ
    try:
        response = query_rag(rag_chain, query)

        # レスポンスの構造を確認（デバッグ用）
        print("\nレスポンスキー:", response.keys())

        # 回答を表示
        if "answer" in response:
            print("\n回答:", response["answer"])
        elif "result" in response:
            print("\n回答:", response["result"])
        else:
            print("\n回答: レスポンス形式が認識できません。")

        # 参照ドキュメントを表示
        print_document_sources(response)

        return True
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """
    メイン関数。

    コマンドライン引数を解析し、適切なモードでRAGシステムを実行します。
    引数が指定されていない場合はインタラクティブモードで実行します。
    """
    parser = argparse.ArgumentParser(description="RAGシステムを使用して質問に回答します。")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="回答する質問。指定しない場合はインタラクティブモードで実行します。",
    )
    args = parser.parse_args()

    try:
        if args.query:
            # 単一の質問に回答
            success = single_query(args.query)
            sys.exit(0 if success else 1)
        else:
            # インタラクティブモード
            interactive_mode()
            sys.exit(0)
    except Exception as e:
        print(f"予期しないエラーが発生しました: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
