import os
import faiss
import numpy as np
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== CONFIG ======
INDEX_PATH = "pdf_faiss.index"
METADATA_PATH = "pdf_texts.json"
SIMILARITY_THRESHOLD = 0.65  # slightly relaxed
TOP_K = 5                    # take top 5 chunks for better context
# ====================

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search_faiss(query, index, metadata, top_k=TOP_K):
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)

    similarities, indices = index.search(query_vector, top_k)

    results = []
    for i, sim in zip(indices[0], similarities[0]):
        if i == -1 or i >= len(metadata):
            continue
        results.append({"text": metadata[i], "similarity": float(sim)})
    return results


def query_chatgpt(query, context=None):
    if context:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    else:
        print("💬 Asking ChatGPT (no FAISS context fallback)...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return completion.choices[0].message.content.strip()


def hybrid_query(query):
    index, metadata = load_faiss_index()
    results = search_faiss(query, index, metadata)

    if not results:
        return query_chatgpt(query)

    print("\n🔍 FAISS search results:")
    for r in results:
        print(f"• Similarity = {r['similarity']:.4f} | Text preview = {r['text'][:80]}...")

    # Take all chunks above threshold
    good_chunks = [r["text"] for r in results if r["similarity"] >= SIMILARITY_THRESHOLD]

    if good_chunks:
        print(f"\n🧠 Using {len(good_chunks)} FAISS chunks as context (similarity ≥ {SIMILARITY_THRESHOLD})")
        combined_context = "\n\n".join(good_chunks)
        return query_chatgpt(query, context=combined_context)
    else:
        print("\n⚠️ No chunk above threshold, falling back to ChatGPT only.")
        return query_chatgpt(query)


if __name__ == "__main__":
    while True:
        user_query = input("\nAsk something: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        response = hybrid_query(user_query)
        print("------------------------------------------------")
        print(response)
