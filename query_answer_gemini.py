import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

MODEL_NAME = "all-MiniLM-L6-v2"

def init_pc():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "pdf-chunks-index")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing. Put it in .env")
    pc = Pinecone(api_key=api_key)
    return pc, index_name

def build_source_filter():
    """Filter only chunks from the current uploaded PDF (optional)."""
    source_url = os.getenv("PDF_URL")
    source_path = os.getenv("PDF_PATH")

    if source_path and not source_url:
        source_url = f"local://{os.path.basename(source_path)}"

    if source_url:
        return {"source": {"$eq": source_url}}
    return None

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing. Put it in .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def rerank_with_gemini(model, query, matches):
    """Ask Gemini to select the most relevant chunks before answering."""
    chunks = [m.get("metadata", {}).get("text", "") for m in matches]
    joined = "\n\n".join(chunks)

    rerank_prompt = f"""
You are given text chunks retrieved from a PDF. 
Some chunks may be irrelevant, overlapping, or noisy. 
Carefully read them and pick only the most relevant ones to answer the question.

Question: {query}

Chunks:
{joined}

Return the most relevant information only (concise, cleaned text).
"""
    resp = model.generate_content(rerank_prompt)
    return resp.text.strip()

def generate_answer(model, query, filtered_context):
    prompt = f"""
You are an expert assistant. You are given cleaned, relevant context extracted from a PDF. 
Answer the question in a clear, well-structured, human-readable explanation. 
Do not copy chunks directly. Write naturally as if explaining to a colleague.

Context:
{filtered_context}

Question:
{query}

Final Answer:
"""
    resp = model.generate_content(prompt)
    return resp.text

def query_pinecone_and_answer(query: str):
    pc, index_name = init_pc()
    index = pc.Index(index_name)
    embed_model = SentenceTransformer(MODEL_NAME)

    metadata_filter = build_source_filter()

    q_vec = embed_model.encode([query], convert_to_numpy=True)[0].tolist()
    query_args = {
        "vector": q_vec,
        "top_k": int(os.getenv("TOP_K", "5")),
        "include_metadata": True
    }
    if metadata_filter:
        query_args["filter"] = metadata_filter

    res = index.query(**query_args)
    matches = res.get("matches", [])

    if not matches:
        return "⚠️ No matches found in Pinecone.", []

    # Gemini rerank + answer
    gemini_model = init_gemini()
    filtered_context = rerank_with_gemini(gemini_model, query, matches)
    final_answer = generate_answer(gemini_model, query, filtered_context)

    retrieved_chunks = [m.get("metadata", {}).get("text", "") for m in matches]
    return final_answer, retrieved_chunks

def main():
    user_q = input("Enter your question: ").strip()
    answer, chunks = query_pinecone_and_answer(user_q)

    print("\nTop retrieved chunks (debug):")
    for i, ch in enumerate(chunks, 1):
        snippet = ch.replace("\n", " ")
        print(f"\n[{i}] {snippet[:200]}{'…' if len(snippet)>200 else ''}")

    print("\n=== Gemini Final Answer ===\n")
    print(answer)

if __name__ == "__main__":
    os.environ["USE_TF"] = "0"
    main()
