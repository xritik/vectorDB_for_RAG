import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

PDF_PATH = "DevOps_Interview_Ques.pdf"
FAISS_INDEX_FILE = "pdf_faiss.index"
TEXT_JSON_FILE = "pdf_texts.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500


def extract_text_with_pypdf(pdf_path):
    print("Extracting text using PyPDF...")
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        else:
            print(f"No text found on page {i+1}")
    return text.strip()


def extract_text_with_ocr(pdf_path):
    print("No extractable text found. Running OCR on scanned PDF...")
    images = convert_from_path(pdf_path)
    all_text = ""
    for i, img in enumerate(images):
        print(f"OCR processing page {i+1}/{len(images)}...")
        text = pytesseract.image_to_string(img)
        all_text += text + "\n"
    return all_text.strip()


def split_text(text, chunk_size=500):
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [], 0
        current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def create_faiss_index(chunks):
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    print("Creating FAISS index (cosine similarity)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product for normalized vectors
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(TEXT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"FAISS index saved to '{FAISS_INDEX_FILE}'")
    print(f"Text chunks saved to '{TEXT_JSON_FILE}'")


if __name__ == "__main__":
    text = extract_text_with_pypdf(PDF_PATH)
    if not text.strip():
        text = extract_text_with_ocr(PDF_PATH)
    if not text.strip():
        print("Still no text found. OCR might have failed.")
        exit(1)
    chunks = split_text(text, CHUNK_SIZE)
    create_faiss_index(chunks)
    print("Conversion completed successfully!")
