import os
import csv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load data.csv
qa_pairs = []
questions = []
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        q = row["question"].strip()
        a = row["answer"].strip()
        qa_pairs.append((q, a))
        questions.append(q)

# Encode questions
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions, convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Function to query FAISS
def query_data_csv(query, k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    
    best_idx = indices[0][0]
    best_dist = distances[0][0]
    
    # Compute similarity score (smaller distance = more similar)
    similarity_score = 1 / (1 + best_dist)  # convert L2 distance to 0–1 scale
    
    # Threshold for similarity confidence
    if similarity_score > 0.75:
        q, a = qa_pairs[best_idx]
        return f"Q: {q}\nA: {a}"
    else:
        return None

# Function to query ChatGPT
def query_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Main function
def hybrid_query(query):
    query = query.strip().lower()
    local_response = query_data_csv(query)
    if local_response:
        return f"[From data.csv]: {local_response}"
    else:
        chat_response = query_chatgpt(query)
        return f"[From ChatGPT]: {chat_response}"

# Example usage
if __name__ == "__main__":
    while True:
        user_query = input("Ask something: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        response = hybrid_query(user_query)
        print(response)
