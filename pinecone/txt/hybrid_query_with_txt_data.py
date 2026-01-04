import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Load ENV variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- STEP 1: Load Sentence Embedding Model ----
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---- STEP 2: Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "text-rag-index"

# Create index if not exists
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,                      # embedding size for MiniLM
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


# ---- STEP 3: Load and split TXT data ----
def load_and_split_text(file_path="data.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split by sentences
    chunks = text.split(".")
    chunks = [c.strip() for c in chunks if len(c.strip()) > 20]
    return chunks


# ---- STEP 4: Store vectors in Pinecone ----
def store_vectors():
    chunks = load_and_split_text()

    vectors = []

    for i, chunk in enumerate(chunks):
        vector = model.encode(chunk).tolist()
        vectors.append({
            "id": f"id_{i}",
            "values": vector,
            "metadata": {"text": chunk},
        })

    index.upsert(vectors=vectors)
    print(f"Uploaded {len(vectors)} text chunks to Pinecone!")


# ---- STEP 5: Ask ChatGPT for fallback answer ----
def ask_chatgpt(query):
    # print("\n ChatGPT Answer (fallback because score < 0.5):\n")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )

    print(response.choices[0].message.content)


# ---- STEP 6: Query Pinecone ----
def ask_question(query):
    # print("\n Searching…")

    q_embed = model.encode(query).tolist()

    results = index.query(
        vector=q_embed,
        top_k=1,
        include_metadata=True
    )

    if not results["matches"]:
        # print(" No match at all → using ChatGPT…")
        print("\n Result Using ChatGPT…")
        ask_chatgpt(query)
        return

    match = results["matches"][0]

    if match["score"] >= 0.5:
        # print("\n Best Match Found:")
        # print(f"Score: {match['score']}")
        print(f"\n Result: {match['metadata']['text']}")
    else:
        print("\n Result Using ChatGPT…")
        ask_chatgpt(query)


# -------- MAIN RUN --------
if __name__ == "__main__":
    print("Storing vectors in Pinecone…")
    store_vectors()

    # print("\n Ask your questions now!")

    while True:
        q = input("\nEnter your query (type 'exit' to stop): ")

        # Exit
        if q.lower() == "exit":
            break

        # Prevent empty or space-only inputs
        if len(q.strip()) == 0:
            print("Please enter a valid question (empty input ignored).")
            continue

        ask_question(q)

