import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#  1. Helper Functions

# Extract text from uploaded file
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return "\n".join(df.astype(str).apply(" ".join, axis=1).tolist())

    elif file.name.endswith(".pdf"):
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    else:
        st.error("Unsupported file type! Please upload .txt, .csv, or .pdf")
        return None


# Split text into smaller chunks (for better retrieval)
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


#  2. Vector DB Creation (FAISS)

def create_faiss_index(chunks, model):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings


#  3. Hybrid Search (Semantic + Keyword)

def hybrid_search(query, chunks, model, index, embeddings, top_k=3):
    # --- Semantic Search using FAISS ---
    query_emb = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_emb, top_k)
    semantic_results = [chunks[i] for i in indices[0]]

    # --- Keyword Search using TF-IDF ---
    tfidf = TfidfVectorizer().fit(chunks)
    tfidf_matrix = tfidf.transform(chunks)
    query_vec = tfidf.transform([query])
    cosine_scores = (tfidf_matrix * query_vec.T).toarray().flatten()
    top_keyword_indices = cosine_scores.argsort()[::-1][:top_k]
    keyword_results = [chunks[i] for i in top_keyword_indices]

    # --- Combine Results ---
    combined = list(set(semantic_results + keyword_results))
    return combined[:top_k]


#  4. Streamlit App UI

def main():
    st.set_page_config(page_title="RAG Hybrid Search Demo", layout="wide")
    st.title("🔍 Hybrid Search using FAISS + TF-IDF")
    st.write("Upload a file (TXT, CSV, or PDF) and ask questions from it!")

    uploaded_file = st.file_uploader("Upload your file", type=["txt", "csv", "pdf"])

    if uploaded_file:
        text = extract_text(uploaded_file)
        if text:
            st.success("✅ File successfully processed!")
            st.write("**Preview:**")
            st.text_area("Extracted Text", text[:1000] + "...", height=200)

            st.write("Creating text chunks and generating embeddings...")
            chunks = chunk_text(text)
            model = SentenceTransformer("all-MiniLM-L6-v2")

            index, embeddings = create_faiss_index(chunks, model)
            st.success("✅ Vector database created successfully!")

            query = st.text_input("Ask a question from your data:")

            if query:
                results = hybrid_search(query, chunks, model, index, embeddings)
                st.write("### 🔎 Top Relevant Results:")
                for i, res in enumerate(results, start=1):
                    st.markdown(f"**Result {i}:** {res[:300]}...")

                st.success("✅ Hybrid search completed!")


if __name__ == "__main__":
    main()