import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("txt-index")

print("\n💬 Type your queries below (type 'exit' or 'quit' to stop)\n")

while True:
    # ---- Get user query ----
    query = input("Enter your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("\n👋 Exiting. Goodbye!\n")
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
        # Combine chunks into a single context
        context_text = "\n".join(relevant_chunks)

        # Send context + query to ChatGPT for a coherent answer
        prompt = f"""You are an AI assistant. Use the context below to answer the question accurately.

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
        print("\n" + "-"*70 + "\n")

    else:
        # No relevant chunks found → fallback to direct ChatGPT
        print("\n⚠️ No relevant chunks found in Pinecone. Using ChatGPT...\n")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        print(completion.choices[0].message.content)
        print("\n" + "-"*70 + "\n")
