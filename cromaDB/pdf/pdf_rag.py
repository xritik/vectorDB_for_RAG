import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# === Step 1: Setup Chroma client ===
client = chromadb.PersistentClient(path="./chroma_db_pdf")
collection = client.get_or_create_collection(name="pdf_vectors")

# === Step 2: Load and extract text from PDF ===
file_path = "sample.pdf"  # <- Replace with your PDF file path

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

text = extract_text_from_pdf(file_path)
print("PDF text extracted successfully!")

# === Step 3: Split text into chunks ===
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

chunks = chunk_text(text)
print(f"Total chunks created: {len(chunks)}")

# === Step 4: Generate embeddings and store in Chroma ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print("PDF data successfully converted to Chroma vector database!")

# === Step 5: Query function ===
def query_chroma(user_query, n_results=3):
    query_emb = model.encode([user_query])
    results = collection.query(
        query_embeddings=query_emb,
        n_results=n_results
    )
    print("\nQuery Results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"{i+1}. {doc[:300]}...\n")  # show first 300 chars

# === Step 6: Run a query ===
user_query = input("\nAsk your query: ")
query_chroma(user_query)
