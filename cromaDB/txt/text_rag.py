import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# === Step 1: Setup ===
# Initialize Chroma client (stores data in ./chroma_db directory)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection
collection = client.get_or_create_collection(name="text_vectors")

# === Step 2: Load text file ===
file_path = "sample.txt"   # <- replace with your .txt file path
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# === Step 3: Split text into smaller chunks ===
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(text)
print(f"Total chunks created: {len(chunks)}")

# === Step 4: Create embeddings and store in Chroma ===
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# Add to Chroma collection
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("Text data successfully converted to Chroma vector database!")

# === Step 5: Query the data ===
def query_chroma(user_query, n_results=3):
    query_emb = model.encode([user_query])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results
    )
    print("\n Query Results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i+1}. {doc[:200]}...\n")  # Show first 200 chars

# Example query
user_query = input("\nAsk your query: ")
query_chroma(user_query)
