import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("txt-index")

# ---- Make a query ----
query = "Explain Docker basics"

# Convert query into embedding
embedding = client.embeddings.create(
    input=query,
    model="text-embedding-3-small"
).data[0].embedding

# Search in Pinecone
results = index.query(vector=embedding, top_k=5, include_metadata=True)

# Print matched chunks
for match in results.matches:
    print(f"Score: {match['score']:.4f}")
    print("Text:", match["metadata"]["text"], "\n")
