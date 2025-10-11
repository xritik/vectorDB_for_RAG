import os
import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ====== CONFIG ======
PDF_PATH = "DevOps_Interview_Ques.pdf"  # 👈 Replace with your actual PDF file name
THRESHOLD = 0.65
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight embedding model

# ====== STEP 1: Load and chunk PDF text ======
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
    return pages

def split_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

pdf_pages = extract_pdf_text(PDF_PATH)
chunks = []
for i, page_text in enumerate(pdf_pages):
    for chunk in split_text(page_text):
        chunks.append({"page": i + 1, "text": chunk})

print(f"📄 Extracted {len(chunks)} chunks from PDF")

# ====== STEP 2: Create embeddings dynamically ======
model = SentenceTransformer(EMBED_MODEL)
embeddings = model.encode([c["text"] for c in chunks])
embeddings = np.array(embeddings, dtype="float32")

# ====== STEP 3: Build FAISS index on the fly ======
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# ====== STEP 4: Interactive hybrid querying ======
def ask_question(query):
    query_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k=8)

    # Convert FAISS L2 distances to similarity (higher is better)
    similarities = 1 / (1 + distances[0])

    print("\n🔍 FAISS search results:")
    for i, idx in enumerate(indices[0]):
        print(f"• Similarity = {similarities[i]:.4f} | Text preview = {chunks[idx]['text'][:90]}...")

    # Get top similarity
    top_score = similarities[0]

    # 🚀 Dynamic logic
    if top_score >= 0.55:  # much lower base threshold
        # Take top chunks (even if similarity is low)
        selected_chunks = [chunks[idx]["text"] for idx in indices[0][:5]]
        context = "\n\n".join(selected_chunks)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the context if relevant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        answer = completion.choices[0].message.content
        print(f"\n🧠 Using top 5 PDF chunks as context (top similarity = {top_score:.4f})")
        print("------------------------------------------------")
        print(answer)
    else:
        # Fallback if even top match is very poor
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        )
        answer = completion.choices[0].message.content
        print("\n🌐 Using online ChatGPT (no good PDF match found)")
        print("------------------------------------------------")
        print(answer)

# ====== MAIN LOOP ======
while True:
    q = input("\nAsk something: ").strip()
    if q.lower() in ["exit", "quit"]:
        break
    ask_question(q)
