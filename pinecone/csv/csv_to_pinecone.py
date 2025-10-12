from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
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
index_name = "csv-index"

# Create index (only if it doesn't exist)
if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- Step 1: Read CSV ----
df = pd.read_csv("data.csv")  # Replace with your CSV filename

# ---- Step 2: Combine rows into text ----
# Adjust these columns depending on your CSV structure
combined_texts = []
for i, row in df.iterrows():
    # Example: combining 'Question' and 'Answer' columns
    text = ""
    for col in df.columns:
        text += f"{col}: {row[col]}  "
    combined_texts.append(text.strip())

# ---- Step 3: Convert text chunks to embeddings and store in Pinecone ----
for i, text in enumerate(combined_texts):
    embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding

    index.upsert([
        (f"row-{i}", embedding, {"text": text})
    ])

print("✅ CSV data converted and stored in Pinecone successfully!")
