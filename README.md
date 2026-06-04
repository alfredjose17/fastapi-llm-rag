# RAG App — PDF Injection + LLaMA 3 Generation

A FastAPI-based Retrieval Augmented Generation (RAG) system that lets you inject PDF documents and ask questions about them using LLaMA 3 running locally via Ollama.

## How It Works

```
INJECTION:
PDF upload → extract text (PyPDFLoader)
           → split into chunks (RecursiveCharacterTextSplitter)
           → embed each chunk (nomic-embed-text via Ollama)
           → store in ChromaDB (persisted to ./chroma_db)

GENERATION:
Question → embed (nomic-embed-text)
         → similarity search in ChromaDB → top 5 relevant chunks
         → build prompt with chunks as context
         → generate answer (LLaMA 3 via Ollama)
```

## Project Structure

```
rag-app/
├── main.py           # FastAPI app + endpoints
├── rag/
│   ├── __init__.py   # makes rag a package
│   ├── config.py     # vectorstore, embeddings, llm setup
│   ├── loader.py     # load_pdf, split_docs
│   └── engine.py     # retrieve, prompt, generate
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd rag-app

# 2. Create and activate virtual environment
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull required Ollama models
ollama pull llama3
ollama pull nomic-embed-text

# 5. Start Ollama (keep this running in a separate terminal)
ollama serve

# 6. Start the API
uvicorn main:app --reload
```

API is now running at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/inject` | Upload a PDF → chunk → embed → store in ChromaDB |
| POST | `/generate` | Ask a question → retrieve chunks → LLaMA 3 answer |
| GET | `/health` | Health check |

## Usage

### Inject a PDF
```bash
curl -X POST http://localhost:8000/inject \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "filename": "your_document.pdf",
  "chunks_stored": 15
}
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the candidate most recent job?", "top_k": 5}'
```

Response:
```json
{
  "answer": "The candidate's most recent role was Cloud Engineer (Internship) at OracleLens..."
}
```

## Configuration

All AI models and settings are in `rag/config.py`:

```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3")
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 500 | Smaller = more precise retrieval |
| `chunk_overlap` | 50 | Higher = less context lost at boundaries |
| `top_k` | 5 | More chunks = more context for the LLM |

## Key Concepts

- **Embeddings** — numerical vector representations of text. Similar meaning = similar vectors
- **ChromaDB** — vector database that stores chunks and their embeddings, persisted to `./chroma_db`
- **Similarity search** — finds chunks whose vectors are closest to the question vector (cosine similarity)
- **RAG** — grounds the LLM's answer in your documents, reducing hallucination
