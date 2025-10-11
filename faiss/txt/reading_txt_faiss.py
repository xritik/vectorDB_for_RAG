import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load FAISS index
index = faiss.read_index("txt_faiss_vector_db.index")

# Reload original documents (important!)
with open("data.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Query
query = "what is cybbb"
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Search top 3
D, I = index.search(np.array([query_embedding], dtype="float32"), k=1)

print("🔎 Results:")
for idx in I[0]:
    print(documents[idx].strip())  # Fetch original text
