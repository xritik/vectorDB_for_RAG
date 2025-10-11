# import os
# import chromadb
# from openai import OpenAI

# from dotenv import load_dotenv
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Initialize ChromaDB (local persistent database)
# chroma_client = chromadb.PersistentClient(path="./chroma_db")

# # Create a collection (like a table in DB)
# collection = chroma_client.get_or_create_collection(name="my_text_data")
# print("new", flush=True)
# # Read txt file
# with open("data.txt", "r", encoding="utf-8") as f:
#     documents = f.readlines()

# # Add documents with embeddings
# for i, doc in enumerate(documents):
#     embedding = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=doc
#     ).data[0].embedding

#     collection.add(
#         ids=[str(i)],             # unique ID
#         documents=[doc],          # original text
#         embeddings=[embedding]    # vector representation
#     )

# print("Data stored in ChromaDB!")














import os
import chromadb
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Use persistent ChromaDB (data saved inside ./chroma_db folder)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection
collection = chroma_client.get_or_create_collection(name="my_text_data")
print("✅ ChromaDB initialized", flush=True)

# ---------------------------
# 2. Load and Clean Text File
# ---------------------------
with open("data.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]  # remove empty lines

print(f"📂 Loaded {len(documents)} documents", flush=True)

if not documents:
    print("⚠️ No data found in data.txt. Please add some text and try again.", flush=True)
    exit()

# ---------------------------
# 3. Generate Embeddings in Batch
# ---------------------------
try:
    print("⏳ Creating embeddings...", flush=True)

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=documents  # batch all docs at once
    )

    embeddings = [item.embedding for item in response.data]

    # ---------------------------
    # 4. Store in ChromaDB
    # ---------------------------
    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        embeddings=embeddings
    )

    print("✅ Data stored in ChromaDB!", flush=True)

except Exception as e:
    print("❌ Error while creating embeddings:", e, flush=True)

# ---------------------------
# 5. Verify Storage
# ---------------------------
print("📊 Total documents in collection:", collection.count(), flush=True)

all_data = collection.get(limit=5)  # show first 5
print("🔎 Sample stored docs:", all_data["documents"], flush=True)
