import os

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["campus_ai"]
collection = db["academic_vector"]

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    {
        "id": "attendance_policy",
        "text": (
            "Students must maintain at least 75 percent attendance "
            "to be eligible to appear in the end semester examination."
        ),
    },
    {
        "id": "exam_rule",
        "text": (
            "Students must carry their university identity card "
            "when appearing for an examination."
        ),
    },
    {
        "id": "library_rule",
        "text": (
            "The university library is open from 9 AM to 8 PM "
            "on working days."
        ),
    },
]

for doc in documents:
    vector = model.encode(
        doc["text"],
        normalize_embeddings=True
    ).tolist()

    collection.update_one(
        {"id": doc["id"]},
        {
            "$set": {
                "id": doc["id"],
                "metadata": {
                    "text": doc["text"]
                },
                "values": vector,
            }
        },
        upsert=True,
    )

    print(
        f"Inserted/updated: {doc['id']} | "
        f"vector dimensions: {len(vector)}"
    )

print("Academic test data inserted successfully.")