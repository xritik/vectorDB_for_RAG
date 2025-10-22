import os
import faiss
import numpy as np
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_ONLY_URI")  # example: mongodb+srv://ritik:<password>@cluster0.mongodb.net/
db_name = os.getenv("DB_NAME", "trainingDB")
collection_name = os.getenv("COLLECTION_NAME", "trainings")

mongo_client = MongoClient(mongo_uri)
collection = mongo_client[db_name][collection_name]


# ---- 1️⃣ Fetch real-time data from MongoDB ----
def fetch_mongo_data():
    records = list(collection.find({}))
    print(f"✅ Fetched {len(records)} records from MongoDB")
    clean_records = []
    for record in records:
        record["_id"] = str(record["_id"])
        clean_records.append(record)
    return clean_records


# ---- 2️⃣ Generate embeddings ----
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")


# ---- 3️⃣ Convert MongoDB data to FAISS vectors ----
def create_faiss_index(trainings):
    texts = []
    metadata = []

    for t in trainings:
        text = (
            f"Training Name: {t.get('training_name', '')}\n"
            f"Trainer: {t.get('trainer', '')}\n"
            f"Company: {t.get('company', '')}\n"
            f"Technology: {t.get('technology', '')}\n"
            f"Start Date: {t.get('start_date', '')}\n"
            f"End Date: {t.get('end_date', '')}\n"
            f"Remarks: {t.get('remarks', '')}"
        )
        texts.append(text)
        metadata.append(t)

    embeddings = [get_embedding(txt) for txt in texts]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, metadata


# ---- 4️⃣ Use LLM to summarize / interpret the result ----
def llm_summarize(query, results):
    context = "\n\n".join([
        f"Training Name: {r['trainingName']}\n"
        f"Trainer: {r['trainerName']}\n"
        f"Company: {r['companyName']}\n"
        f"Technology: {r['technology']}\n"
        f"Start Date: {r['startDate']}\n"
        f"End Date: {r['endDate']}\n"
        f"Remarks: {r['remarks']}\n"
        for r in results
    ])

    prompt = f"""
You are an assistant that summarizes training data.
Query: {query}
Here are the related training records:
{context}
Please give a meaningful and concise summary of the relevant information.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content.strip()


# ---- 5️⃣ Main interactive query loop ----
def main():
    trainings = fetch_mongo_data()
    if not trainings:
        print("❌ No data found in MongoDB.")
        return

    index, metadata = create_faiss_index(trainings)
    print("✅ FAISS index created from real-time MongoDB data.\n")

    while True:
        query = input("Enter your query ('exit' to quit): ").strip()
        if query.lower() == "exit":
            break

        query_emb = get_embedding(query).reshape(1, -1)
        distances, indices = index.search(query_emb, k=min(5, len(metadata)))

        results = [metadata[i] for i in indices[0] if i < len(metadata)]

        summary = llm_summarize(query, results)
        print("\n🤖 LLM Summary:\n")
        print(summary)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
