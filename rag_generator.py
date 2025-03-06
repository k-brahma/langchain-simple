"""
rag_generator.py

このスクリプトは、RAG（Retrieval-Augmented Generation）システム用のベクトルストアを生成します。
HTMLファイルを読み込み、ドキュメントに変換し、チャンクに分割して、ベクトルストアを作成します。
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv

from config import (
    CHROMA_DIRECTORY,
    HTML_DATA_DIRECTORY,
    STATS_DIRECTORY,
    get_embedding_provider,
    set_embedding_provider,
)
from rag_utils import create_vector_store, load_html_files, split_documents

# 環境変数を読み込む
load_dotenv()


class TokenUsageTracker:
    """
    トークン使用量を追跡するクラス。

    埋め込みトークン、完了トークン、合計トークン、処理時間などの統計情報を記録します。
    """

    def __init__(self):
        """トークン使用量トラッカーを初期化します。"""
        self.stats = {
            "embedding_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "chunks_count": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": 0,
        }

    def update_tokens(self, embedding_tokens=0, completion_tokens=0):
        """
        トークン使用量を更新します。

        Args:
            embedding_tokens (int): 埋め込みトークン数
            completion_tokens (int): 完了トークン数
        """
        self.stats["embedding_tokens"] += embedding_tokens
        self.stats["completion_tokens"] += completion_tokens
        self.stats["total_tokens"] = (
            self.stats["embedding_tokens"] + self.stats["completion_tokens"]
        )

    def set_chunks_count(self, count):
        """
        チャンク数を設定します。

        Args:
            count (int): チャンク数
        """
        self.stats["chunks_count"] = count

    def finish(self):
        """処理の終了を記録します。"""
        self.stats["end_time"] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.stats["start_time"])
        end_time = datetime.fromisoformat(self.stats["end_time"])
        self.stats["duration_seconds"] = (end_time - start_time).total_seconds()

    def save_stats(self):
        """統計情報をJSONファイルに保存します。"""
        # 統計ディレクトリが存在しない場合は作成
        if not os.path.exists(STATS_DIRECTORY):
            os.makedirs(STATS_DIRECTORY)

        # ファイル名に現在の日時を含める
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STATS_DIRECTORY}/token_usage_{timestamp}.json"

        # 統計情報をJSONファイルに書き込む
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        print(f"統計情報を保存しました: {filename}")
        print(f"合計トークン使用量: {self.stats['total_tokens']}")
        print(f"処理時間: {self.stats['duration_seconds']} 秒")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="RAGシステム用ベクトルストア生成")
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default=get_embedding_provider(),
        choices=["openai", "cohere"],
        help="埋め込みプロバイダー (openai または cohere)",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.html",
        help="処理するHTMLファイルのパターン (デフォルト: *.html)",
    )
    args = parser.parse_args()

    embedding_provider = args.embedding_provider
    file_pattern = args.file_pattern

    # 環境変数に埋め込みプロバイダーを設定
    set_embedding_provider(embedding_provider)

    print("RAGシステム用ベクトルストア生成を開始します")

    # トークン使用量トラッカーの初期化
    token_tracker = TokenUsageTracker()
    start_time = time.time()

    try:
        # HTMLファイルの読み込み
        html_files = glob.glob(os.path.join(HTML_DATA_DIRECTORY, file_pattern))
        print(f"検出されたHTMLファイル数: {len(html_files)}")
        print(f"検出されたファイルリスト: {', '.join(html_files)}")

        # HTMLファイルからドキュメントを読み込む
        documents = load_html_files(html_files)
        if not documents:
            print("処理するドキュメントがありません。終了します。")
            return

        print(f"合計 {len(documents)} 件のドキュメントを読み込みました")

        # ドキュメントをチャンクに分割
        chunks = split_documents(documents)
        if not chunks:
            print("チャンクの生成に失敗しました。終了します。")
            return

        # チャンク数を記録
        token_tracker.set_chunks_count(len(chunks))

        # ベクトルストアを作成
        vector_store = create_vector_store(chunks, embedding_provider=embedding_provider)
        if not vector_store:
            print("ベクトルストアの作成に失敗しました。終了します。")
            return

        # 統計情報を更新して保存
        token_tracker.finish()
        token_tracker.save_stats()

        # 成功メッセージ
        print(f"ベクトルストアの生成が完了しました: {CHROMA_DIRECTORY}")
        print(f"チャンク数: {len(chunks)}")
        print(f"処理時間: {token_tracker.stats['duration_seconds']} 秒")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
