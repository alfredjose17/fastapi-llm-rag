from rag.config import vectorstore, llm


def retrieve_chunks(question: str, top_k: int) -> list:
    # Embed the question and find similar chunks in ChromaDB
    return vectorstore.similarity_search(question, k=top_k)


def build_prompt(question: str, chunks: list) -> str:
    # Join all retrieved chunks into one context block
    context = "\n\n".join(doc.page_content for doc in chunks)

    return f"""You are an expert resume analyzer.
Answer the question below using ONLY the context provided.
Be specific and cite exact details like company names, dates, and technologies.
Do NOT make up information. If the answer is not in the context, say "I don't know based on the provided resume."

Context:
{context}

Question: {question}

Answer:"""


def generate(question: str, top_k: int) -> str:
    chunks = retrieve_chunks(question, top_k)
    prompt = build_prompt(question, chunks)
    return llm.invoke(prompt)