from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index
index_name = "txt-index"
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Read text
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---- Semantic Chunking ----
# Split text by double newlines (paragraphs) or sentences
chunks = re.split(r'\n{2,}', text)  # split paragraphs
chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]

# Embed and store
for i, chunk in enumerate(chunks):
    embedding = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    ).data[0].embedding

    index.upsert([
        (f"chunk-{i}", embedding, {"text": chunk})
    ])

print(f"✅ {len(chunks)} semantic chunks stored in Pinecone!")
