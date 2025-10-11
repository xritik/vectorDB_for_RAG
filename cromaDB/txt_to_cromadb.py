import os
import chromadb
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize ChromaDB (local persistent database)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection (like a table in DB)
collection = chroma_client.get_or_create_collection(name="my_text_data")

# Read txt file
with open("data.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Add documents with embeddings
for i, doc in enumerate(documents):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding

    collection.add(
        ids=[str(i)],             # unique ID
        documents=[doc],          # original text
        embeddings=[embedding]    # vector representation
    )

print("✅ Data stored in ChromaDB!")