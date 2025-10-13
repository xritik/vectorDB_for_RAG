from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
index_name = "pdf-index"

# Create Pinecone index (only once)
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Embedding size for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- Step 1: Read PDF ----
pdf_path = "DevOps_Interview_Ques.pdf"  # your PDF file
reader = PdfReader(pdf_path)

pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        pages.append(text.strip())

print(f"📄 Extracted {len(pages)} pages from PDF.")

# ---- Step 2: Split large text chunks ----
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = []
for page in pages:
    chunks.extend(list(chunk_text(page)))

print(f"✂️ Total {len(chunks)} chunks created.")

# ---- Step 3: Convert to embeddings and store in Pinecone ----
for i, chunk in enumerate(chunks):
    embedding = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"
    ).data[0].embedding

    index.upsert([
        (f"chunk-{i}", embedding, {"text": chunk})
    ])

print("✅ PDF converted and stored in Pinecone successfully!")
