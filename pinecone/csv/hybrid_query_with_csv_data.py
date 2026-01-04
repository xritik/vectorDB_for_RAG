import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use existing Pinecone index (VERY IMPORTANT)
index_name = "text-rag-index"     # <- use your index name
index = pc.Index(index_name)

# Namespace for CSV data
NAMESPACE = "csv-data"


# -----------------------------
#  Load and process CSV
# -----------------------------
def load_csv_chunks(csv_path="data.csv"):
    df = pd.read_csv(csv_path)

    rows = []
    for i in df.index:
        # Convert entire row to a single string
        row_text = " | ".join([f"{col}: {df[col][i]}" for col in df.columns])

        # Skip empty rows
        if len(row_text.strip()) < 5:
            continue

        rows.append(row_text)

    return rows


# -----------------------------
#   Store CSV vectors
# -----------------------------
def store_vectors():
    rows = load_csv_chunks()

    vectors = []
    for i, row in enumerate(rows):
        embed = model.encode(row).tolist()

        vectors.append({
            "id": f"csv_{i}",
            "values": embed,
            "metadata": {"text": row},
        })

    index.upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Saved {len(vectors)} rows inside namespace '{NAMESPACE}'!")


# -----------------------------
#    ChatGPT fallback
# -----------------------------
def ask_chatgpt(query):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
    )
    print("\nChatGPT Answer:\n")
    print(response.choices[0].message.content)


# -----------------------------
#    Query Pinecone
# -----------------------------
def ask_question(query):
    embed = model.encode(query).tolist()

    res = index.query(
        vector=embed,
        top_k=1,
        include_metadata=True,
        namespace=NAMESPACE
    )

    if not res["matches"]:
        ask_chatgpt(query)
        return

    match = res["matches"][0]

    if match["score"] >= 0.5:
        # print("\nBest Match from CSV:")
        # print("Score:", match["score"])
        print("Text:", match["metadata"]["text"])
    else:
        ask_chatgpt(query)


# -----------------------------
#           MAIN
# -----------------------------
if __name__ == "__main__":
    print("Uploading CSV data...")
    store_vectors()

    # print("\nAsk your questions!")

    while True:
        q = input("\nEnter query (type 'exit'): ")

        if q.lower() == "exit":
            break

        # Avoid empty or only-spaces query
        if len(q.strip()) == 0:
            print("Empty query. Please type something.")
            continue

        ask_question(q)
