from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag.config import vectorstore
from rag.loader import load_pdf, split_docs
from rag.engine import generate

app = FastAPI(title="RAG App")


@app.post("/inject")
async def inject_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs supported")

    file_bytes = await file.read()
    docs = load_pdf(file_bytes)
    chunks = split_docs(docs)

    vectorstore.add_documents(chunks)

    return {"filename": file.filename, "chunks_stored": len(chunks)}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/generate")
def generate_answer(request: QueryRequest):
    answer = generate(request.question, request.top_k)
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}