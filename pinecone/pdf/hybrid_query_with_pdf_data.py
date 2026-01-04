import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from pypdf import PdfReader
from openai import OpenAI

# Load keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Use an existing index (IMPORTANT — DO NOT CREATE NEW)
index_name = "text-rag-index"   # <- use your existing index
index = pc.Index(index_name)

# Namespace for organizing PDF data
NAMESPACE = "pdf-data"


# ---- Load PDF & split ----
def load_pdf_chunks(pdf_path="data.pdf"):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    chunks = text.split(".")
    chunks = [c.strip() for c in chunks if len(c.strip()) > 20]

    return chunks


# ---- Store PDF vectors ----
def store_vectors():
    chunks = load_pdf_chunks()

    vectors = []
    for i, chunk in enumerate(chunks):
        embed = model.encode(chunk).tolist()

        vectors.append({
            "id": f"pdf_{i}",
            "values": embed,
            "metadata": {"text": chunk},
        })

    index.upsert(vectors=vectors, namespace=NAMESPACE)
    print(f"Saved {len(vectors)} chunks into namespace '{NAMESPACE}'!")


# ---- ChatGPT fallback ----
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


# ---- Query Pinecone ----
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
        # print("\nBest Match:")
        # print("Score:", match["score"])
        print("Text:", match["metadata"]["text"])
    else:
        ask_chatgpt(query)


# ---- Main ----
if __name__ == "__main__":
    print("Uploading PDF data...")
    store_vectors()

    # print("\nAsk your questions!")

    while True:
        q = input("\nEnter query (type 'exit'): ")

        if q.lower() == "exit":
            break

        # Avoid empty query
        if len(q.strip()) == 0:
            print("Empty query. Please type something.")
            continue

        ask_question(q)
