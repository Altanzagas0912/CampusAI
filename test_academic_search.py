import os

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["campus_ai"]
collection = db["academic_vector"]

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

query = "How much attendance does a student need for exams?"

query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

pipeline = [
    {
        "$vectorSearch": {
            "index": "academic_index",
            "path": "values",
            "queryVector": query_vector,
            "numCandidates": 100,
            "limit": 3
        }
    },
    {
        "$project": {
            "_id": 0,
            "id": 1,
            "metadata": 1,
            "score": {
                "$meta": "vectorSearchScore"
            }
        }
    }
]

results = list(collection.aggregate(pipeline))

print("Question:")
print(query)

print("\nSearch results:")

for result in results:
    print("-" * 50)
    print("ID:", result["id"])
    print("Score:", result["score"])
    print("Text:", result["metadata"]["text"])