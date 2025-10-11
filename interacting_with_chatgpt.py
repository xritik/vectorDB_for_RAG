import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Send a prompt to ChatGPT
response = client.chat.completions.create(
    model="gpt-4o-mini",   # you can use "gpt-4o" or "gpt-3.5-turbo"
    messages=[
        {"role": "user", "content": "Explain RAG in very simple words."}
    ]
)

# Print the AI's reply
print(response.choices[0].message.content)
