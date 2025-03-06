import glob
import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 環境変数を読み込む
load_dotenv()

# ベクトルストアの保存先
PERSIST_DIRECTORY = "vectorstore"


class CustomHTMLLoader:
    """カスタムHTMLローダー"""

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """HTMLファイルを読み込み、Documentを返す"""
        try:
            # UTF-8でファイルを読み込む
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # BeautifulSoupを使用してHTMLを解析
            soup = BeautifulSoup(content, "html.parser")

            # 不要なタグを削除
            for script in soup(["script", "style"]):
                script.extract()

            # テキストを取得
            text = soup.get_text(separator="\n")

            # 余分な空白行を削除
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)

            # メタデータを作成
            metadata = {
                "source": os.path.basename(self.file_path),
                "title": soup.title.string if soup.title else os.path.basename(self.file_path),
            }

            # Documentオブジェクトを作成
            document = Document(page_content=text, metadata=metadata)

            return [document]
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            return []


def load_html_files():
    """HTMLファイルを読み込む関数"""
    html_files = glob.glob("rag_base_data/html/*.html")
    all_documents = []

    print(f"Loading {len(html_files)} HTML files...")

    for file_path in html_files:
        try:
            print(f"Processing: {file_path}")
            # カスタムHTMLローダーを使用
            loader = CustomHTMLLoader(file_path)
            documents = loader.load()

            # ファイル名をメタデータに追加
            file_name = os.path.basename(file_path)
            for doc in documents:
                doc.metadata["source"] = file_name

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} document(s) from {file_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return all_documents


def split_documents(documents):
    """ドキュメントをチャンクに分割する関数"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks):
    """ベクトルストアを作成する関数"""
    # OpenAIのembeddingsを使用
    embeddings = OpenAIEmbeddings()

    # Chromaベクトルストアを作成
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
    )

    # ベクトルストアを保存
    vector_store.persist()
    print(f"Vector store created and saved to {PERSIST_DIRECTORY}")

    return vector_store


def load_vector_store():
    """保存されたベクトルストアを読み込む関数"""
    embeddings = OpenAIEmbeddings()

    if os.path.exists(PERSIST_DIRECTORY):
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        return vector_store
    else:
        print("No existing vector store found.")
        return None


def create_rag_chain(vector_store):
    """RAGチェーンを作成する関数"""
    # リトリーバーを作成
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # LLMを初期化
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # プロンプトテンプレートを作成
    prompt = ChatPromptTemplate.from_template(
        """
    次の質問に対して、提供されたコンテキスト情報を使用して答えてください。
    コンテキストに情報がない場合は、「提供された情報からは回答できません」と正直に答えてください。
    
    コンテキスト情報:
    {context}
    
    質問: {input}
    
    回答:
    """
    )

    # ドキュメント結合チェーンを作成
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # 検索チェーンを作成
    rag_chain = create_retrieval_chain(
        retriever=retriever, combine_docs_chain=question_answer_chain
    )

    return rag_chain


def query_rag(rag_chain, query):
    """RAGチェーンに質問する関数"""
    response = rag_chain.invoke({"input": query})
    return response


def main():
    # ベクトルストアが既に存在するか確認
    existing_vector_store = load_vector_store()

    if existing_vector_store:
        print("Using existing vector store.")
        vector_store = existing_vector_store
    else:
        print("Creating new vector store...")
        # HTMLファイルの読み込み
        documents = load_html_files()

        # ドキュメントの分割
        if documents:
            chunks = split_documents(documents)

            # ベクトルストアの作成
            if chunks:
                vector_store = create_vector_store(chunks)
            else:
                print("エラー: チャンクが作成されませんでした。")
                return
        else:
            print("エラー: ドキュメントが読み込まれませんでした。")
            return

    # RAGチェーンの作成
    rag_chain = create_rag_chain(vector_store)

    # 対話ループ
    while True:
        query = input("質問を入力してください (終了するには 'exit' を入力): ")

        if query.lower() == "exit":
            break

        response = query_rag(rag_chain, query)
        print(f"\n回答: {response['answer']}\n")


if __name__ == "__main__":
    main()
