# CampusAI

CampusAI is a university-focused **Agentic RAG (Retrieval-Augmented Generation)** assistant designed to answer campus-specific questions (academic policies, syllabus, notices, placements, etc.) and also help students find **Previous Year Question Papers (PYQs)**.

The project contains:
- A **FastAPI backend** that exposes a `/query` endpoint.
- An **agentic RAG pipeline** built with **LangGraph** + **LangChain**.
- **MongoDB** as the knowledge store (including vector search collections).
- Data ingestion pipelines for:
  - **Notices** (scrape → chunk → embed → store in MongoDB vector collection)
  - **PYQs** (scan a GitHub repo folder tree for PDFs → parse metadata → upsert into MongoDB)
- A simple web crawler in `Web_Scrapper/` used to scrape site content into `.txt` files.

> Language composition (GitHub): Jupyter Notebook (~86%) and Python (~14%).

---

## High-level architecture

At runtime, CampusAI:
1. Receives a user question.
2. Uses an LLM decision node to determine whether retrieval is needed (`RETRIEVE` vs `GENERATE`).
3. Routes the query into one of the intent categories:
   - `academicQuery`
   - `noticeQuery`
   - `pyqQuery`
4. Retrieves the relevant context from MongoDB.
5. Generates a final answer using an LLM.

You can find diagrams in the repository root:
- `rag_architecture.png`
- `RAG_Pipeline.png`
- `CampusAI-RAG.png`

---

## Repository structure

```text
CampusAI/
├── backend/
│   ├── main.py                 # FastAPI app (health + query endpoints)
│   ├── agentic_rag.py           # LangGraph workflow (decide → route → retrieve → generate)
│   └── schema.py                # Pydantic models for API + agent state
├── Data_Ingestion/
│   ├── notices_ingestion/
│   │   ├── main.py              # Local entrypoint to run notice ingestion
│   │   ├── lambda_function.py   # AWS Lambda handler to schedule/trigger ingestion
│   │   ├── scraper_ingest.py    # Scrapes notices and ingests into MongoDB vector store
│   │   └── ...                  # chunker/embedder/db/retriever helpers
│   └── pyq_ingestion/
│       ├── main.py              # AWS Lambda handler: scans GitHub repo tree for PDFs, upserts metadata
│       └── query.py             # Local query helper for searching PYQs in MongoDB
├── Web_Scrapper/
│   ├── scrapper.py              # Deep crawler for https://gehu.ac.in/
│   ├── gehu-homepage.txt        # Scraped output (large)
│   └── gehu-top-placement.txt   # Scraped output
├── requirements.txt
└── *.ipynb                      # Notebooks for experimentation/prototyping
```

---

## Backend API

### Health check

- **GET** `/health`

Returns:
```json
{ "status": "API is running live." }
```

### Ask a question

- **POST** `/query`

Request body:
```json
{ "question": "What is the attendance policy for CSE?" }
```

Response:
```json
{ "answer": "..." }
```

---

## Agentic RAG workflow (backend/agentic_rag.py)

The workflow is built with **LangGraph** and uses these steps:

- `decide_retrieval`: LLM decides if the system should retrieve from the knowledge base or answer directly.
- `route_query_type_node`: LLM routes the query to one of: `pyqQuery`, `noticeQuery`, `academicQuery`.
- Retrieval nodes:
  - `retrieve_notice_context`: Uses **Gemini Embeddings API** (`models/gemini-embedding-001`) + MongoDB `$vectorSearch` on the `notice_vector` collection.
  - `retrieve_academic_context`: Uses **SentenceTransformers** embeddings and MongoDB `$vectorSearch` on the `academic_vector` collection.
  - `retrieve_pyq_context`: Parses user intent (semester, exam type, subject code/name) and queries the `pyq_papers` collection.
- `generate_ans`: Generates the final response using **Groq (Llama 3.1)**.

---

## Data ingestion

### 1) Notices ingestion

Location: `Data_Ingestion/notices_ingestion/`

What it does:
- Scrapes notices from `http://btechcsegehu.in/notices-2/`.
- Stops once it reaches a previously ingested post.
- Chunks notice text, generates embeddings, and stores them in MongoDB (`notice_vector`) with metadata like title/date/urls.

Run locally:
```bash
python Data_Ingestion/notices_ingestion/main.py
```

Run on AWS Lambda:
- Use `Data_Ingestion/notices_ingestion/lambda_function.py` as the handler.

### 2) PYQ ingestion

Location: `Data_Ingestion/pyq_ingestion/`

What it does:
- Uses the GitHub API to fetch a repository tree (recursive).
- Finds PDFs under `btech/cse/`.
- Parses metadata (semester, subject code, year, exam type, optional set) from the path/filename.
- Upserts records into MongoDB (`pyq_papers`).

This is designed to run as an AWS Lambda job.

---

## Setup & running locally

### 1) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Environment variables

Create a `.env` file (or set environment variables) with at least:

```bash
# Mongo
MONGO_URI="mongodb+srv://..."

# LLM keys
GROQ_API_KEY="..."
GROQ_API_KEY2="..."  # used by generate_ans
GEMINI_API_KEY="..." # used for notice embeddings

# Optional: NVIDIA/DeepSeek script
DEEPSEEK_API_KEY="..."
```

> Note: The backend connects to MongoDB database `campus_ai`.

### 4) Run the backend

```bash
uvicorn backend.main:app --reload
```

Then test:
- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/query`

---

## Requirements

Key libraries used (see `requirements.txt`):
- FastAPI + Uvicorn
- LangChain + LangGraph
- sentence-transformers
- pymongo
- google-genai (Gemini embeddings)
- langchain-groq

---

## Notes / TODOs

- Add a `.env.example` to make setup easier.
- Add Docker support (optional).
- Consider adding a frontend client.
- Ensure MongoDB Atlas has the required vector search indexes (`notice_vector`, `academic_vector`).

---

## License

Add a license if you plan to open-source this project.
