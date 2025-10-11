import os
import faiss
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# CONFIGURATION
INDEX_FILE = "csv_faiss_vector_db.index"
DOCS_FILE = "documents.npy"
EMBEDDING_MODEL = "text-embedding-3-small"
DISTANCE_THRESHOLD = 1.2  # lower = stricter

# LOAD INDEX + DOCUMENTS
print("Loading FAISS index and documents...")
index = faiss.read_index(INDEX_FILE)
documents = np.load(DOCS_FILE, allow_pickle=True).tolist()
print(f"Loaded {len(documents)} documents | Dimension = {index.d}")

# FUNCTIONS
def get_embedding(text: str):
    """Generate 1536-d OpenAI embedding"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def query_faiss(query: str, k: int = 2):
    """Search FAISS for most similar text"""
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    best_doc = documents[int(indices[0][0])]
    best_distance = float(distances[0][0])
    return best_doc, best_distance

def query_chatgpt(prompt: str):
    """Ask ChatGPT directly"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def validate_match(query: str, retrieved_text: str):
    """Ask ChatGPT if the local match truly answers the query"""
    validation_prompt = f"""
You are verifying a retrieved answer. 
User question: "{query}"
Retrieved text: "{retrieved_text}"

Does the retrieved text correctly and fully answer the question? 
Respond only with 'YES' or 'NO'.
"""
    response = query_chatgpt(validation_prompt)
    return response.strip().upper().startswith("Y")

# MAIN HYBRID FUNCTION
def hybrid_query(user_query: str):
    local_result, distance = query_faiss(user_query)

    print(f"[DEBUG] FAISS distance = {distance:.4f}")

    if distance < DISTANCE_THRESHOLD:
        # Let GPT verify if local data truly answers the question
        is_valid = validate_match(user_query, local_result)
        if is_valid:
            return f"[From Your Data]\n{local_result}"
        else:
            chat_response = query_chatgpt(user_query)
            return f"[From ChatGPT]\n{chat_response}"
    else:
        chat_response = query_chatgpt(user_query)
        return f"[From ChatGPT]\n{chat_response}"

# MAIN LOOP
if __name__ == "__main__":
    print("\nSmart Hybrid Search System Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask something: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        print("------------------------------------------------")
        answer = hybrid_query(query)
        print(answer)
        print("------------------------------------------------\n")
