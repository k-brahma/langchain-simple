import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from html_rag_loader import create_rag_chain, load_vector_store

# 環境変数を読み込む
load_dotenv()

# FastAPIアプリケーションの初期化
app = FastAPI(
    title="HTML RAG API", description="HTMLファイルを基にしたRAGシステムのAPI", version="1.0.0"
)


# リクエストモデル
class QueryRequest(BaseModel):
    query: str


# レスポンスモデル
class QueryResponse(BaseModel):
    answer: str
    source_documents: list = []


# 起動時にベクトルストアを読み込む
@app.on_event("startup")
async def startup_event():
    global rag_chain, vector_store

    # ベクトルストアを読み込む
    vector_store = load_vector_store()

    if vector_store is None:
        raise HTTPException(
            status_code=500,
            detail="ベクトルストアが見つかりません。先にhtml_rag_loader.pyを実行してください。",
        )

    # RAGチェーンを作成
    rag_chain = create_rag_chain(vector_store)


# クエリエンドポイント
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not hasattr(app.state, "rag_chain"):
        # RAGチェーンがない場合は初期化
        vector_store = load_vector_store()
        if vector_store is None:
            raise HTTPException(
                status_code=500,
                detail="ベクトルストアが見つかりません。先にhtml_rag_loader.pyを実行してください。",
            )
        app.state.rag_chain = create_rag_chain(vector_store)

    try:
        # RAGチェーンに質問
        response = app.state.rag_chain.invoke({"input": request.query})

        # ソースドキュメントの情報を取得
        source_documents = []
        if "context" in response:
            for doc in response["context"]:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source_documents.append(
                        {
                            "source": doc.metadata["source"],
                            "page_content": (
                                doc.page_content[:100] + "..."
                                if len(doc.page_content) > 100
                                else doc.page_content
                            ),
                        }
                    )

        return {"answer": response["answer"], "source_documents": source_documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"エラーが発生しました: {str(e)}")


# ヘルスチェックエンドポイント
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# メインエンドポイント
@app.get("/")
async def root():
    return {
        "message": "HTML RAG API",
        "usage": "POST /query エンドポイントに query パラメータを送信してください",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
