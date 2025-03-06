"""
api.py

このモジュールは、RAG（Retrieval-Augmented Generation）システムのAPIサーバーを提供します。
FastAPIを使用して、クエリに対する回答を生成するエンドポイントを公開します。
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config import API_HOST, API_PORT, get_embedding_provider, get_llm_provider
from rag_utils import create_rag_chain

# 環境変数を読み込む
load_dotenv()

# FastAPIアプリケーションの作成
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation APIサーバー",
    version="1.0.0",
)

# RAGチェーン
rag_chain = None


class QueryRequest(BaseModel):
    """
    クエリリクエストモデル。

    属性:
        query (str): ユーザーからのクエリ文字列。
    """

    query: str


class QueryResponse(BaseModel):
    """
    クエリレスポンスモデル。

    属性:
        answer (str): 生成された回答。
        sources (List[str]): 回答の生成に使用されたソースドキュメントのリスト。
    """

    answer: str
    sources: List[str]


@app.on_event("startup")
async def startup_event():
    """
    アプリケーション起動時に実行される関数。
    RAGチェーンを初期化します。
    """
    global rag_chain

    try:
        print("RAGチェーンを初期化中...")
        # 環境変数から埋め込みプロバイダーとLLMプロバイダーを取得
        embedding_provider = get_embedding_provider()
        llm_provider = get_llm_provider()
        print(f"使用する埋め込みプロバイダー: {embedding_provider}")
        print(f"使用するLLMプロバイダー: {llm_provider}")

        # 埋め込みプロバイダーとLLMプロバイダーを明示的に指定してRAGチェーンを作成
        rag_chain = create_rag_chain(embedding_provider, llm_provider)
        print("RAGチェーンの初期化完了")
    except Exception as e:
        print(f"RAGチェーンの初期化中にエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


@app.get("/health")
async def health_check():
    """
    ヘルスチェックエンドポイント。
    アプリケーションの状態を確認します。

    Returns:
        dict: アプリケーションの状態を示す辞書。
    """
    return {
        "status": "healthy",
        "embedding_provider": get_embedding_provider(),
        "llm_provider": get_llm_provider(),
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    クエリエンドポイント。
    ユーザーからのクエリに対して回答を生成します。

    Args:
        request (QueryRequest): クエリリクエスト。

    Returns:
        QueryResponse: 生成された回答とソースドキュメント。

    Raises:
        HTTPException: RAGチェーンが初期化されていない場合や、クエリの処理中にエラーが発生した場合。
    """
    global rag_chain

    if rag_chain is None:
        raise HTTPException(status_code=500, detail="RAGチェーンが初期化されていません")

    try:
        print(f"RAGチェーンに問い合わせを実行します: {request.query}")
        result = rag_chain.invoke({"input": request.query})
        print(f"生のレスポンス: {result}")

        # レスポンスからソースを抽出
        sources = []
        if "context" in result and result["context"]:
            for doc in result["context"]:
                if "source" in doc.metadata:
                    sources.append(doc.metadata["source"])

        # 回答を抽出
        answer = result.get("answer", "回答を生成できませんでした")

        return QueryResponse(answer=answer, sources=list(set(sources)))

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"クエリの処理中にエラーが発生しました: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # UvicornでAPIサーバーを起動
    uvicorn.run(app, host=API_HOST, port=API_PORT)
