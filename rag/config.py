from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Embedding model - converts text to vectors
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Vector store - persists to ./chroma_db on disk
vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# LLM - generates answers
llm = OllamaLLM(model="llama3", temperature=0.1)