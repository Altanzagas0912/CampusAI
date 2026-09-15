import os
import hashlib
import requests

from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not found in .env")


# 1. 一条虚构的测试通知
# 注意：这里只用于验证技术，不是真实学校通知
content = (
    "Campus AI Innovation Competition registration closes on "
    "30 September 2026. The final presentation will be held on "
    "15 October 2026 in the university innovation center."
)

title = "Campus AI Innovation Competition Notice"

post_id = "test_notice_ai_competition"


# 2. 调用和原 CampusAI 相同的 Gemini Embedding
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
                    "text": content
                }
            ]
        }
    },
    timeout=30
)

if response.status_code != 200:
    raise RuntimeError(
        f"Gemini API Error: {response.status_code}\n"
        f"{response.text}"
    )

embedding = response.json()["embedding"]["values"]

print("Gemini embedding generated.")
print("Vector dimensions:", len(embedding))


# 3. 存进 MongoDB
client = MongoClient(MONGO_URI)

db = client["campus_ai"]
collection = db["notice_vector"]

content_hash = hashlib.sha256(
    (title + content).encode("utf-8")
).hexdigest()

document = {
    "post_id": post_id,
    "content": content,
    "embedding": embedding,
    "metadata": {
        "title": title,
        "date": "2026-09-15",
        "notice_url": "https://example.com/notices/ai-competition",
        "pdf_url": None,
        "department": "CSE",
        "type": "notice",
        "content_hash": content_hash,
        "created_at": datetime.now(timezone.utc)
    }
}

collection.update_one(
    {
        "post_id": post_id
    },
    {
        "$set": document
    },
    upsert=True
)

print("Notice inserted successfully.")
print("notice_vector count:", collection.count_documents({}))
