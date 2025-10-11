import os
import faiss
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Load CSV
df = pd.read_csv("data.csv")

# Combine columns into text documents (customize as needed)
documents = []
for i, row in df.iterrows():
    text = " | ".join([str(val) for val in row.values if pd.notna(val)])
    documents.append(text)

print(f"📂 Loaded {len(documents)} rows from CSV")

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

# Save FAISS index
faiss.write_index(index, "csv_faiss_vector_db.index")
np.save("documents.npy", documents)

print("✅ CSV Vector DB created and saved as csv_faiss_vector_db.index")