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
index = pc.Index("pdf-index")

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
        top_k=3,  # retrieve multiple chunks for better context
        include_metadata=True
    )

    # ---- Gather all chunks ----
    relevant_chunks = [m.metadata.get("text", "") for m in results.matches if m.score >= 0.6]
    if not relevant_chunks:
        print("\n⚠️ No relevant chunks found in Pinecone. Using ChatGPT...\n")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        print(completion.choices[0].message.content.strip())
        print()
        continue

    context = "\n\n".join(relevant_chunks)

    # ---- LLM extracts only relevant part from Pinecone data ----
    llm_prompt = f"""
You are an intelligent assistant that helps extract only the relevant part of a document
that directly answers the given user question.

Here is the document text (from Pinecone):
\"\"\"{context}\"\"\"

Question:
\"\"\"{query}\"\"\"

Instructions:
1. Find the part of the text that directly answers the user's question.
2. If the answer text includes multiple questions (like Q7, Q8, etc.), include only the one matching the question.
3. Fix small grammar issues, but do not add new information.
4. If you cannot find a meaningful answer, just reply with "NOT_FOUND".
Return only the final answer text, no explanation.
"""

    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": llm_prompt}]
    ).choices[0].message.content.strip()

    # ---- If LLM could not find the answer, fallback to ChatGPT ----
    if llm_response == "NOT_FOUND" or len(llm_response) < 20:
        print("\n⚠️ No valid answer found in Pinecone. Using ChatGPT...\n")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}]
        )
        print(completion.choices[0].message.content.strip())
        print()
    else:
        print("\n--- Answer from Pinecone + LLM ---\n")
        print(llm_response)
        print()
