"""
api.py

このスクリプトは、RAG（Retrieval-Augmented Generation）システムのAPIサーバーを提供します。
FastAPIを使用して、RESTful APIエンドポイントを公開します。

主な機能:
- RAGシステムへの問い合わせAPIエンドポイント
- ヘルスチェックエンドポイント
- Swagger UIによるAPIドキュメント
"""

import os
import traceback
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 共通ユーティリティをインポート
from rag_utils import create_rag_chain, load_vector_store, query_rag

# 環境変数を読み込む
load_dotenv()

# FastAPIアプリケーションを作成
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation APIサーバー",
    version="1.0.0",
)

# グローバル変数
rag_chain = None


class QueryRequest(BaseModel):
    """
    問い合わせリクエストモデル。

    属性:
        query (str): ユーザーからの問い合わせ。
    """

    query: str


class QueryResponse(BaseModel):
    """
    問い合わせレスポンスモデル。

    属性:
        answer (str): 問い合わせに対する回答。
        sources (List[str]): 回答の参照元ドキュメントのリスト。
    """

    answer: str
    sources: List[str]


class HealthResponse(BaseModel):
    """
    ヘルスチェックレスポンスモデル。

    属性:
        status (str): サーバーのステータス。
        version (str): APIのバージョン。
    """

    status: str
    version: str


@app.on_event("startup")
async def startup_event():
    """
    アプリケーション起動時に実行される関数。

    RAGチェーンを初期化します。
    """
    global rag_chain
    try:
        print("RAGチェーンを初期化中...")
        vector_store = load_vector_store()
        if not vector_store:
            print("警告: ベクトルストアの読み込みに失敗しました。APIは正常に動作しません。")
            return

        rag_chain = create_rag_chain(vector_store)
        print("RAGチェーンの初期化完了")
    except Exception as e:
        print(f"RAGチェーン初期化中にエラーが発生: {str(e)}")
        traceback.print_exc()


@app.get("/health", response_model=HealthResponse, tags=["システム"])
async def health_check():
    """
    ヘルスチェックエンドポイント。

    サーバーの状態を確認するためのエンドポイントです。

    戻り値:
        HealthResponse: サーバーのステータスとバージョン情報。
    """
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    RAGシステムへの問い合わせエンドポイント。

    ユーザーからの問い合わせを受け付け、RAGシステムを使用して回答を生成します。

    パラメータ:
        request (QueryRequest): 問い合わせリクエスト。

    戻り値:
        QueryResponse: 問い合わせに対する回答と参照元。

    例外:
        HTTPException: RAGチェーンが初期化されていない場合や、
                      問い合わせ処理中にエラーが発生した場合。
    """
    global rag_chain
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="RAGチェーンが初期化されていません。サーバーを再起動してください。",
        )

    try:
        print(f"問い合わせを処理中: {request.query}")
        response = query_rag(rag_chain, request.query)

        # 回答を取得
        answer = ""
        if "answer" in response:
            answer = response["answer"]
        elif "result" in response:
            answer = response["result"]
        else:
            answer = "回答を生成できませんでした。"

        # ソースを取得
        sources = []
        try:
            source_documents = response.get("context", [])
            if not source_documents and "documents" in response:
                source_documents = response.get("documents", [])

            if source_documents:
                for doc in source_documents:
                    source = doc.metadata.get("source", "不明")
                    if source not in sources:
                        sources.append(source)
        except Exception as e:
            print(f"ソース抽出中にエラー: {str(e)}")
            traceback.print_exc()

        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        print(f"問い合わせ処理中にエラーが発生: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # UvicornでAPIサーバーを起動
    uvicorn.run(app, host="0.0.0.0", port=8000)
