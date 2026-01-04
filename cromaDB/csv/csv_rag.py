import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# === Step 1: Setup Chroma client ===
client = chromadb.PersistentClient(path="./chroma_db_csv")
collection = client.get_or_create_collection(name="csv_vectors")

# === Step 2: Load CSV file ===
file_path = "sample.csv"  # <- replace with your CSV file path
df = pd.read_csv(file_path)

print(f"✅ CSV file loaded with {len(df)} rows and {len(df.columns)} columns.")

# === Step 3: Combine text from multiple columns ===
# You can choose which columns to use
# Example: take all columns and join them as text
def row_to_text(row):
    return " | ".join(str(v) for v in row.values if pd.notna(v))

documents = df.apply(row_to_text, axis=1).tolist()
print(f"✅ Created {len(documents)} text documents from CSV rows.")

# === Step 4: Create embeddings and store in Chroma ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"row_{i}" for i in range(len(documents))]
)

print("✅ CSV data successfully converted to Chroma vector database!")

# === Step 5: Query function ===
def query_chroma(user_query, n_results=1):
    query_emb = model.encode([user_query])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results
    )
    print("\n🔍 Query Results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i+1}. {doc[:300]}...\n")  # show first 300 chars of each result

# === Step 6: Run query ===
user_query = input("\nAsk your query: ")
query_chroma(user_query)
