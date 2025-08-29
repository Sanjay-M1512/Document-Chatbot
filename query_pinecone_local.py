# query_pinecone_local.py (Pinecone v3) — supports PDF_URL or PDF_PATH filtering
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

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
    """
    Build a Pinecone metadata filter based on either PDF_URL or PDF_PATH.
    For local files we used source_id = 'local://<filename>' when upserting.
    """
    source_url = os.getenv("PDF_URL")
    source_path = os.getenv("PDF_PATH")

    if source_path and not source_url:
        # match the upsert convention in pdf_to_pinecone_local.py
        source_url = f"local://{os.path.basename(source_path)}"

    if source_url:
        return {"source": {"$eq": source_url}}
    return None

def main():
    pc, index_name = init_pc()
    index = pc.Index(index_name)
    model = SentenceTransformer(MODEL_NAME)

    # Optional filter (URL or local file)
    metadata_filter = build_source_filter()

    user_q = input("Enter your question: ").strip()
    q_vec = model.encode([user_q], convert_to_numpy=True)[0].tolist()

    query_args = {
        "vector": q_vec,
        "top_k": int(os.getenv("TOP_K", "5")),
        "include_metadata": True
    }
    if metadata_filter:
        query_args["filter"] = metadata_filter

    res = index.query(**query_args)

    print("\nTop matches:")
    for m in res.get("matches", []):
        score = m.get("score", 0.0)
        md = m.get("metadata", {}) or {}
        page = md.get("page", "?")
        text = (md.get("text", "") or "").replace("\n", " ")
        snippet = (text[:300] + "…") if len(text) > 300 else text
        print(f"\n— score={round(score,4)}, page={page}")
        print(snippet)

if __name__ == "__main__":
    # Best effort to avoid TF
    os.environ["USE_TF"] = "0"
    main()
