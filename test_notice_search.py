import os
import requests

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")


question = (
    "When is the registration deadline "
    "for the AI innovation competition?"
)


# 生成问题 embedding
endpoint = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-embedding-001:embedContent"
    f"?key={GEMINI_API_KEY}"
)

response = requests.post(
    endpoint,
    json={
        "content": {
            "parts": [
                {
                    "text": question
                }
            ]
        }
    },
    timeout=30
)

response.raise_for_status()

query_embedding = (
    response.json()["embedding"]["values"]
)


# MongoDB Vector Search
client = MongoClient(MONGO_URI)

collection = (
    client["campus_ai"]["notice_vector"]
)

pipeline = [
    {
        "$vectorSearch": {
            "index": "default",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 100,
            "limit": 3
        }
    },
    {
        "$project": {
            "_id": 0,
            "post_id": 1,
            "content": 1,
            "metadata": 1,
            "score": {
                "$meta": "vectorSearchScore"
            }
        }
    }
]

results = list(
    collection.aggregate(pipeline)
)


print("Question:")
print(question)

print("\nResults:")

for result in results:
    print("-" * 60)
    print("Post ID:", result["post_id"])
    print("Score:", result["score"])
    print("Title:", result["metadata"]["title"])
    print("Content:", result["content"])