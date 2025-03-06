"""
config.py

このモジュールは、RAGシステムの設定を管理します。
環境変数や設定ファイルから設定を読み込み、アプリケーション全体で使用できるようにします。
"""

import os

from dotenv import load_dotenv

# 環境変数を読み込む
load_dotenv()

# ディレクトリ設定
CHROMA_DIRECTORY = "chroma_db"
HTML_DATA_DIRECTORY = "rag_base_data/html"
STATS_DIRECTORY = "stats"

# 埋め込み設定
# デフォルトは環境変数から読み込み、設定されていない場合は "openai" を使用
DEFAULT_EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")

# LLMの設定
DEFAULT_LLM_PROVIDER = "openai"  # 現在はOpenAIのみサポート
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"  # デフォルトモデル

# APIサーバー設定
API_HOST = "0.0.0.0"
API_PORT = 8000

# チャンク分割設定
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embedding_provider():
    """
    現在の埋め込みプロバイダーを取得します。
    環境変数 EMBEDDING_PROVIDER が設定されている場合はその値を、
    そうでない場合はデフォルト値を返します。
    """
    return os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)


def set_embedding_provider(provider):
    """
    埋め込みプロバイダーを設定します。
    この関数は環境変数を直接変更します。

    Args:
        provider (str): 使用する埋め込みプロバイダー ("openai" または "cohere")
    """
    if provider not in ["openai", "cohere"]:
        raise ValueError("埋め込みプロバイダーは 'openai' または 'cohere' である必要があります")

    os.environ["EMBEDDING_PROVIDER"] = provider


def get_llm_provider():
    """
    現在のLLMプロバイダーを取得します。
    環境変数 LLM_PROVIDER が設定されている場合はその値を、
    そうでない場合はデフォルト値を返します。
    """
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)


def set_llm_provider(provider):
    """
    LLMプロバイダーを設定します。
    この関数は環境変数を直接変更します。

    Args:
        provider (str): 使用するLLMプロバイダー (現在は "openai" のみサポート)
    """
    if provider != "openai":
        raise ValueError("現在サポートされているLLMプロバイダーは 'openai' のみです")

    os.environ["LLM_PROVIDER"] = provider


def is_cohere_trial():
    """
    Cohereの無料トライアルキーを使用しているかどうかを確認します。
    環境変数 IS_COHERE_TRIAL が設定されている場合はその値を、
    そうでない場合はデフォルト値（True）を返します。

    Returns:
        bool: Cohereの無料トライアルキーを使用している場合はTrue、そうでない場合はFalse
    """
    trial_setting = os.getenv("IS_COHERE_TRIAL", "true").lower()
    return trial_setting == "true"
