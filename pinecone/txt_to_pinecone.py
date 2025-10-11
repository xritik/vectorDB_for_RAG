from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index (only once)
index_name = "txt-index"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- Step 1: Read your TXT ----
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---- Step 2: Split into smaller chunks ----
def chunk_text(text, size=500):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i+size])

chunks = list(chunk_text(text))

# ---- Step 3: Convert each chunk into embeddings ----
for i, chunk in enumerate(chunks):
    embedding = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    ).data[0].embedding

    # ---- Step 4: Store in Pinecone ----
    index.upsert([
        (f"chunk-{i}", embedding, {"text": chunk})
    ])

print("TXT converted and stored in Pinecone successfully!")
