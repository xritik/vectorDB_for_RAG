import os
import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# 2. Reload the same PDF text (to map FAISS results back)
reader = PdfReader("sample.pdf")
pages = [page.extract_text() for page in reader.pages if page.extract_text()]

def split_text(text, chunk_size=100):
    """Split long text into smaller chunks"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

documents = []
for page in pages:
    documents.extend(split_text(page, chunk_size=100))

print(f"📂 Loaded {len(documents)} chunks from sample.pdf")

# 3. Load FAISS index
index = faiss.read_index("pdf_faiss_vector_db.index")
print("✅ Loaded FAISS index")

# 4. Perform a search
query = "xyzz"
print(f"\n🔎 Query: {query}")

# Create embedding for query
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

query_vector = np.array([query_embedding], dtype="float32")

# Search top 3 results
k = 1
distances, indices = index.search(query_vector, k)

# 5. Show results
print("\n📌 Search Results:")
for rank, idx in enumerate(indices[0]):
    print(f"\nResult {rank+1}:")
    print(f"Text: {documents[idx][:350]}...")  # show first 250 characters
    print(f"Distance: {distances[0][rank]}")