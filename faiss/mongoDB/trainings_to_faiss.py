import os
import json
import faiss
import numpy as np
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from bson import json_util

load_dotenv()  # load OPENAI_API_KEY from .env

# ================= CONFIGURATION =================
MONGO_URI=os.getenv("MONGO_URI")
DB_NAME = "training_portal"
COLLECTION_NAME = "trainings"
FAISS_INDEX_PATH = "trainings.index"
EMBEDDINGS_JSON_PATH = "trainings_metadata.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= FETCH DATA FROM MONGO =================
mongo_client = MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]
trainings = list(collection.find())
print(f"✅ Fetched {len(trainings)} records from MongoDB")

# ================= CREATE TEXT REPRESENTATION =================
def doc_to_text(doc):
    return f"""
    Training Name: {doc.get('trainingName','')}
    Technology: {doc.get('technology','')}
    Vendor: {doc.get('vendor','')}
    Company: {doc.get('companyName','')}
    Trainer: {doc.get('trainerName','')}
    Start Date: {doc.get('startDate','')}
    End Date: {doc.get('endDate','')}
    Remarks: {doc.get('remarks','')}
    """

texts = [doc_to_text(doc) for doc in trainings]

# ================= GENERATE EMBEDDINGS =================
print("🧠 Generating embeddings...")
embeddings = []
for text in texts:
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    embeddings.append(resp.data[0].embedding)

embeddings = np.array(embeddings, dtype="float32")

# ================= CREATE FAISS INDEX =================
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ================= SAVE INDEX AND METADATA =================
# Convert MongoDB objects to JSON-serializable format
with open(EMBEDDINGS_JSON_PATH, "w") as f:
    json.dump(trainings, f, default=json_util.default, indent=2)

faiss.write_index(index, FAISS_INDEX_PATH)
print(f"✅ Saved FAISS index ({FAISS_INDEX_PATH}) and metadata ({EMBEDDINGS_JSON_PATH})")
