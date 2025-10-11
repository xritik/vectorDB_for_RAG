import os
import faiss
import numpy as np
from openai import OpenAI
import pandas as pd

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load your CSV again (to map back results)
df = pd.read_csv("data.csv")

# Load FAISS index
index = faiss.read_index("csv_faiss_vector_db.index")

print("✅ Loaded FAISS index")

# Example: search a query
query = "How do containers work?"

# Convert query into embedding
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Convert to numpy
query_vector = np.array([query_embedding], dtype="float32")

# Perform similarity search (top 3 results)
k = 3
distances, indices = index.search(query_vector, k)

print("🔎 Search Results:")
for rank, idx in enumerate(indices[0]):
    row = df.iloc[idx]  # fetch row from CSV
    print(f"\nResult {rank+1}:")
    print(f"ID: {row['id']}")
    print(f"Question: {row['question']}")
    print(f"Answer: {row['answer']}")
    print(f"Distance: {distances[0][rank]}")