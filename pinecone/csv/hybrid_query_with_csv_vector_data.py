import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("csv-index")

print("\n🔍 Ask your questions (type 'exit' to quit)\n")

while True:
    query = input("Enter your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break

    # ---- Convert query to embedding ----
    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # ---- Query Pinecone ----
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    # ---- Collect relevant chunks ----
    relevant_chunks = [m.metadata['text'] for m in results.matches if m.score >= 0.6]

    if relevant_chunks:
        context_text = "\n".join(relevant_chunks)
        prompt = f"""You are an AI assistant. Use the following context to answer accurately.

Context:
{context_text}

Question: {query}
Answer:"""

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        print("\n--- Answer from Pinecone + ChatGPT ---\n")
        print(completion.choices[0].message.content)
        print()

    else:
        print("\n⚠️ No relevant chunks found in Pinecone. Using ChatGPT...\n")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        print(completion.choices[0].message.content)
        print()
