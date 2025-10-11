import chromadb

# 1. Initialize Chroma client (in-memory by default, or can set persist_directory)
client = chromadb.Client()

# 2. Create a collection (like a table)
collection = client.create_collection(name="my_collection")

# 3. Add some data (documents + metadata + ids)
collection.add(
    documents=[
        "ChromaDB is a vector database.",
        "It is commonly used for RAG (Retrieval Augmented Generation).",
        "Python developers often use ChromaDB for AI projects."
    ],
    metadatas=[
        {"source": "wiki"},
        {"source": "docs"},
        {"source": "blog"}
    ],
    ids=["1", "2", "3"]
)

print("✅ Data inserted successfully!")

# 4. Query the stored data
results = collection.query(
    query_texts=["What is ChromaDB used for?"],
    n_results=2  # Number of closest matches to fetch
)

print("\n🔎 Query Results:")
print(results)

# 5. Fetch all stored data explicitly
print("\n📂 All Data in Collection:")
all_data = collection.get()
print(all_data)
