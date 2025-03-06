"""
rag_generator.py

このスクリプトは、HTMLファイルからRAG（Retrieval-Augmented Generation）システム用の
ベクトルストアを生成します。

主な機能:
- HTMLファイルの読み込みと解析
- ドキュメントのチャンクへの分割
- ベクトルストアの作成と保存
- トークン使用量の追跡と統計の記録
"""

import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv

# 共通ユーティリティをインポート
from rag_utils import CHROMA_DIRECTORY, create_vector_store, load_html_files, split_documents

# 環境変数を読み込む
load_dotenv()

# ディレクトリ設定
STATS_DIRECTORY = "stats"


class TokenUsageTracker:
    """
    トークン使用量を追跡するクラス。

    OpenAIのAPIを使用する際のトークン使用量を追跡し、
    統計情報を記録します。

    属性:
        stats (dict): トークン使用量の統計情報。
        start_time (float): 処理開始時間。
    """

    def __init__(self):
        self.stats = {
            "embedding_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "start_time": time.time(),
            "end_time": None,
            "duration_seconds": None,
            "timestamp": datetime.now().isoformat(),
        }
        self.start_time = time.time()

    def update_embedding_tokens(self, tokens):
        """
        埋め込みトークン数を更新します。

        パラメータ:
            tokens (int): 追加する埋め込みトークン数。
        """
        self.stats["embedding_tokens"] += tokens
        self.stats["total_tokens"] += tokens

    def update_completion_tokens(self, tokens):
        """
        補完トークン数を更新します。

        パラメータ:
            tokens (int): 追加する補完トークン数。
        """
        self.stats["completion_tokens"] += tokens
        self.stats["total_tokens"] += tokens

    def finalize(self):
        """
        統計情報を確定します。

        処理時間を計算し、統計情報を更新します。
        """
        end_time = time.time()
        self.stats["end_time"] = end_time
        self.stats["duration_seconds"] = end_time - self.start_time

    def save_stats(self):
        """
        統計情報をJSONファイルに保存します。

        統計情報を指定されたディレクトリに保存します。
        ディレクトリが存在しない場合は作成します。
        """
        if not os.path.exists(STATS_DIRECTORY):
            os.makedirs(STATS_DIRECTORY)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STATS_DIRECTORY}/token_usage_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

        print(f"統計情報を保存しました: {filename}")
        print(f"合計トークン使用量: {self.stats['total_tokens']}")
        print(f"処理時間: {self.stats['duration_seconds']:.2f} 秒")


def main():
    """
    メイン関数。

    HTMLファイルを読み込み、ドキュメントを分割し、
    ベクトルストアを作成して保存します。
    トークン使用量を追跡し、統計情報を記録します。
    """
    print("RAGシステム用ベクトルストア生成を開始します")

    # トークン使用量追跡を初期化
    token_tracker = TokenUsageTracker()

    try:
        # HTMLファイルを読み込む
        documents = load_html_files()
        if not documents:
            print("ドキュメントが読み込めませんでした。処理を終了します。")
            return

        # ドキュメントをチャンクに分割
        chunks = split_documents(documents)
        if not chunks:
            print("ドキュメントの分割に失敗しました。処理を終了します。")
            return

        # ベクトルストアを作成
        vector_store = create_vector_store(chunks)

        # 統計情報を確定して保存
        token_tracker.finalize()
        token_tracker.save_stats()

        print(f"ベクトルストアの生成が完了しました: {CHROMA_DIRECTORY}")
        print(f"チャンク数: {len(chunks)}")
        print(f"処理時間: {token_tracker.stats['duration_seconds']:.2f} 秒")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
