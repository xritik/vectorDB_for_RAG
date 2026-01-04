import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load data.txt
with open("data.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Convert lines to embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(lines, convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Function to query FAISS
def query_data_txt(query, k=1, threshold=0.5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    
    # Check if distance is below threshold (similarity check)
    if distances[0][0] < threshold:  # smaller distance = more similar
        return lines[indices[0][0]]
    else:
        return None

# Function to query ChatGPT
def query_chatgpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",   # or "gpt-4.1" / "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Main function
def hybrid_query(query):
    local_response = query_data_txt(query)
    if local_response:
        return f"[From data.txt]: {local_response}"
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
