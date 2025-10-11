import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Read your txt file
with open("data.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()  # each line as one chunk
    # Or: f.read().split("\n\n")  for paragraphs

# Convert documents into embeddings
embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        input=doc
    )
    embeddings.append(response.data[0].embedding)

# Convert list to numpy array
embeddings = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embeddings.shape[1]  # embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)

# Save index
faiss.write_index(index, "txt_faiss_vector_db.index")
np.save("documents.npy", documents)

print("✅ Vector DB created and saved as txt_faiss_vector_db.index")