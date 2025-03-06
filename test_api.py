import json

import requests


def test_api():
    """APIにリクエストを送信するテスト関数"""
    url = "http://localhost:8000/query"
    headers = {"Content-Type": "application/json"}
    data = {"query": "LangChainとは何ですか？"}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # エラーがあれば例外を発生させる

        # レスポンスを表示
        print("ステータスコード:", response.status_code)
        print("レスポンスヘッダー:", response.headers)
        print("レスポンス本文:")
        result = response.json()
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 回答と参照ドキュメントを表示
        if "answer" in result:
            print("\n回答:", result["answer"])

        if "sources" in result:
            print("\n参照ドキュメント:")
            for source in result["sources"]:
                print(f"  - {source}")

    except requests.exceptions.RequestException as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    test_api()
