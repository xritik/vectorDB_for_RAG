import os
import json
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment (.env)")

client = OpenAI(api_key=OPENAI_API_KEY)

# Paths
FAISS_INDEX_PATH = "trainings.index"
METADATA_PATH = "trainings_metadata.json"

# Load FAISS index and metadata
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    trainings = json.load(f)

# Quick sanity check: number of vectors vs metadata length
if index.ntotal != len(trainings):
    print(f"⚠️  FAISS index size (ntotal={index.ntotal}) != metadata length ({len(trainings)}).")
    print("   If you recently changed data, re-run trainings_to_faiss.py to rebuild index/metadata.")
    # we continue, but this warning is important.

def get_embedding(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def generate_llm_summary(query, matched_records):
    # Build a compact context for the LLM (limit to e.g. 10 items to avoid token explosion)
    max_records_for_llm = 10
    context_items = matched_records[:max_records_for_llm]
    context = "\n\n".join([
        f"Training Name: {r.get('trainingName') or r.get('training_name')}\n"
        f"Trainer: {r.get('trainerName') or r.get('trainer')}\n"
        f"Company: {r.get('companyName') or r.get('company')}\n"
        f"Technology: {r.get('technology')}\n"
        f"Start Date: {r.get('startDate') or r.get('start_date')}\n"
        f"End Date: {r.get('endDate') or r.get('end_date')}\n"
        f"Remarks: {r.get('remarks')}\n"
        for r in context_items
    ])

    prompt = f"""
You are an assistant that summarizes training records.

User Query: "{query}"

Here are the most relevant training records (context):
{context}

Give a concise human-readable summary that answers the user query. If none of the records match, say "No matching trainings found."
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a concise summarizer."},
                  {"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()

def search_and_dedup(query, raw_k=10):
    # Get query embedding
    q_vec = get_embedding(query).reshape(1, -1)

    # Raw search: request more neighbors than you actually want to allow deduping
    distances, indices = index.search(q_vec, raw_k)

    # Deduplicate by returned index id, preserving order
    seen = set()
    unique_indices = []
    for idx in indices[0]:
        if idx < 0:
            continue
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
        # stop early if we already covered all metadata
        if len(seen) >= len(trainings):
            break

    # Map to records safely (guard against index length mismatch)
    matched_records = []
    for idx in unique_indices:
        if 0 <= idx < len(trainings):
            matched_records.append(trainings[idx])
    return matched_records, distances, indices

# Main interactive loop (prints only LLM summary)
if __name__ == "__main__":
    print("\nEnter your query ('exit' to quit):\n")
    while True:
        query = input().strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("👋 Bye")
            break

        print(f"\n Query: {query}\n")

        matched_records, raw_distances, raw_indices = search_and_dedup(query, raw_k=10)

        # If nothing matched semantically, tell LLM to respond "No matching trainings found."
        if not matched_records:
            print("\nResponse:\n\nNo matching trainings found.\n")
            print("="*80 + "\n")
            continue

        # Generate LLM summary (only)
        summary = generate_llm_summary(query, matched_records)
        print("\nResponse:\n")
        print(summary)
        print("\n" + "="*80 + "\n")
