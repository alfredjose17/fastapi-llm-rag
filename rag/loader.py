from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


def load_pdf(file_bytes: bytes) -> list:
    # Write bytes to temp file because PyPDFLoader needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()
    os.unlink(tmp_path)
    return docs


def split_docs(docs) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(docs)