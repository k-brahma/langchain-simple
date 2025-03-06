import glob
import json
import logging
import os
import time
from datetime import datetime

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# from langchain_openai import OpenAIEmbeddings

# ロギングの設定
LOG_DIRECTORY = "logs"
LOG_FILE = os.path.join(
    LOG_DIRECTORY, f"rag_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

if not os.path.exists(LOG_DIRECTORY):
    os.makedirs(LOG_DIRECTORY)

# ロガーの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),  # 標準出力にも表示
    ],
)
logger = logging.getLogger(__name__)

# 環境変数を読み込む
load_dotenv()

# ベクトルストアの保存先
PERSIST_DIRECTORY = "vectorstore"
# 使用統計情報の保存先
STATS_DIRECTORY = "stats"
STATS_FILE = os.path.join(STATS_DIRECTORY, "embedding_stats.json")

# 埋め込みプロバイダーと設定
# 使用する埋め込みプロバイダー（"cohere" または "openai"）
EMBEDDING_PROVIDER = "cohere"
# モデル名
COHERE_EMBEDDING_MODEL = "embed-multilingual-v3.0"
# OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"


class TokenUsageTracker:
    """トークン使用量を追跡するクラス"""

    def __init__(self):
        self.total_tokens = 0
        self.total_chunks = 0
        self.start_time = None
        self.end_time = None
        self.provider = EMBEDDING_PROVIDER
        self.model_name = COHERE_EMBEDDING_MODEL

    def start(self):
        """処理開始時刻を記録"""
        self.start_time = time.time()

    def stop(self):
        """処理終了時刻を記録"""
        self.end_time = time.time()

    def add_tokens(self, token_count):
        """トークン数を追加"""
        self.total_tokens += token_count

    def add_chunks(self, chunk_count):
        """チャンク数を追加"""
        self.total_chunks += chunk_count

    def get_duration(self):
        """処理時間を取得（秒）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def get_stats(self):
        """統計情報を取得"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "provider": self.provider,
            "model": self.model_name,
            "total_tokens": self.total_tokens,
            "total_chunks": self.total_chunks,
            "duration_seconds": self.get_duration(),
            "tokens_per_chunk": (
                self.total_tokens / self.total_chunks if self.total_chunks > 0 else 0
            ),
        }

    def save_stats(self):
        """統計情報をJSONファイルに保存"""
        if not os.path.exists(STATS_DIRECTORY):
            os.makedirs(STATS_DIRECTORY)

        stats = self.get_stats()

        # 既存の統計情報があれば読み込む
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r", encoding="utf-8") as f:
                    existing_stats = json.load(f)
                    if not isinstance(existing_stats, list):
                        existing_stats = [existing_stats]
            except:
                existing_stats = []
        else:
            existing_stats = []

        # 新しい統計情報を追加
        existing_stats.append(stats)

        # 保存
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_stats, f, ensure_ascii=False, indent=2)

        logger.info(f"統計情報を保存しました: {STATS_FILE}")
        return stats


# トークン使用量トラッカーのインスタンスを作成
token_tracker = TokenUsageTracker()


class CustomHTMLLoader:
    """カスタムHTMLローダー"""

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """HTMLファイルを読み込み、Documentを返す"""
        try:
            # UTF-8でファイルを読み込む
            logger.info(f"ファイル読み込み開始: {self.file_path}")
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"ファイルサイズ: {len(content)} バイト")

            # BeautifulSoupを使用してHTMLを解析
            logger.debug(f"HTMLの解析開始: {self.file_path}")
            soup = BeautifulSoup(content, "html.parser")

            # 不要なタグを削除
            for script in soup(["script", "style"]):
                script.extract()

            # テキストを取得
            text = soup.get_text(separator="\n")

            # 余分な空白行を削除
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            logger.debug(f"抽出されたテキストサイズ: {len(text)} 文字")

            # メタデータを作成
            metadata = {
                "source": os.path.basename(self.file_path),
                "title": soup.title.string if soup.title else os.path.basename(self.file_path),
            }
            logger.debug(f"メタデータ: {metadata}")

            # Documentオブジェクトを作成
            document = Document(page_content=text, metadata=metadata)
            logger.info(f"ドキュメント作成完了: {self.file_path}")

            return [document]
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {self.file_path} - {str(e)}", exc_info=True)
            return []


def load_html_files():
    """HTMLファイルを読み込む関数"""
    html_files = glob.glob("rag_base_data/html/*.html")
    logger.info(f"検出されたHTMLファイル数: {len(html_files)}")
    logger.info(f"検出されたファイルリスト: {', '.join(html_files)}")
    all_documents = []

    if not html_files:
        logger.warning("HTMLファイルが見つかりませんでした。ディレクトリパスを確認してください。")
        logger.info(f"現在の作業ディレクトリ: {os.getcwd()}")
        logger.info(
            f"ディレクトリ内容: {os.listdir('rag_base_data') if os.path.exists('rag_base_data') else '存在しません'}"
        )
        return all_documents

    for file_path in html_files:
        try:
            logger.info(f"処理開始: {file_path}")
            # カスタムHTMLローダーを使用
            loader = CustomHTMLLoader(file_path)
            documents = loader.load()

            if not documents:
                logger.warning(f"ファイルからドキュメントを抽出できませんでした: {file_path}")
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
                logger.debug(f"抽出されたコンテンツ (先頭100文字): {content_preview}")

            all_documents.extend(documents)
            logger.info(
                f"読み込み完了: {file_name} から {len(documents)} 件のドキュメントを読み込みました"
            )
        except Exception as e:
            logger.error(f"ファイル読み込みエラー: {file_path} - {str(e)}", exc_info=True)

    logger.info(f"合計 {len(all_documents)} 件のドキュメントを読み込みました")
    return all_documents


def split_documents(documents):
    """ドキュメントをチャンクに分割する関数"""
    logger.info(f"ドキュメント分割開始: {len(documents)} 件のドキュメント")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"分割完了: {len(chunks)} チャンクに分割されました")

    # チャンク数を記録
    token_tracker.add_chunks(len(chunks))

    # サンプルチャンクのログ
    if chunks:
        sample_chunk = chunks[0]
        logger.debug(f"サンプルチャンク内容: {sample_chunk.page_content[:100]}...")
        logger.debug(f"サンプルチャンクのメタデータ: {sample_chunk.metadata}")

    return chunks


def estimate_tokens(text):
    """テキストのトークン数を推定する関数（簡易版）"""
    # OpenAIのトークナイザーに近い簡易的な推定
    # 英語は平均的に1単語あたり約1.3トークン
    # 日本語は1文字あたり約1.3トークン（大まかな推定）
    return int(len(text) * 1.3)


def create_vector_store(chunks):
    """テキストチャンクからベクトルストアを作成する"""
    logger.info("ベクトルストアの作成を開始します")

    try:
        # 環境変数からAPIキーを取得
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEYが設定されていません")

        # OpenAIの埋め込みを初期化
        embeddings = OpenAIEmbeddings()

        # ベクトルストアの保存先ディレクトリ
        persist_directory = "chroma_db"

        # ベクトルストアが既に存在するか確認
        if os.path.exists(persist_directory):
            logger.info(f"既存のベクトルストア '{persist_directory}' を見つけました")
            overwrite = True  # オーバーライドするかどうか

            if not overwrite:
                logger.info("既存のベクトルストアを使用します")
                return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            else:
                logger.info("既存のベクトルストアを上書きします")

        # ベクトルストアを作成
        logger.info(f"新しいベクトルストアを作成します: '{persist_directory}'")
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=persist_directory
        )

        # ベクトルストアを保存
        vector_store.persist()
        logger.info(f"ベクトルストアを '{persist_directory}' に保存しました")

        return vector_store
    except Exception as e:
        logger.error(f"ベクトルストア作成中にエラーが発生しました: {e}")
        raise


def main():
    """メイン関数"""
    logger.info("===== RAG生成処理開始 =====")
    logger.info(f"使用する埋め込みプロバイダー: {EMBEDDING_PROVIDER}")
    # トークン追跡開始
    token_tracker.start()

    if os.path.exists(PERSIST_DIRECTORY):
        logger.warning(f"警告: {PERSIST_DIRECTORY} が既に存在します。")
        choice = input("既存のベクトルストアを上書きしますか？ (y/n): ")
        if choice.lower() != "y":
            logger.info("処理を中止します。")
            return

    logger.info("新しいベクトルストアを作成します...")

    # HTMLファイルの読み込み
    documents = load_html_files()

    if not documents:
        logger.error("エラー: ドキュメントが読み込まれませんでした。")
        return

    # ドキュメントの分割
    chunks = split_documents(documents)

    if not chunks:
        logger.error("エラー: チャンクが作成されませんでした。")
        return

    # ベクトルストアの作成
    vector_store = create_vector_store(chunks)

    if not vector_store:
        logger.error("エラー: ベクトルストアの作成に失敗しました。")
        return

    # トークン追跡終了
    token_tracker.stop()
    stats = token_tracker.save_stats()

    logger.info("===== RAG生成処理完了 =====")
    logger.info(f"{len(chunks)}個のチャンクがインデックス化されました。")
    logger.info(f"保存先: {os.path.abspath(PERSIST_DIRECTORY)}")
    logger.info(f"使用モデル: {stats['model']}")
    logger.info(f"推定トークン数: {stats['total_tokens']}")
    logger.info(f"処理時間: {stats['duration_seconds']:.2f}秒")


if __name__ == "__main__":
    main()
