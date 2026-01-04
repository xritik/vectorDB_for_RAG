# Hybrid Query System using PINECONE + OpenAI + CSV Data

This project demonstrates how to combine **local semantic search using PINECONE** with **OpenAI’s GPT model** for hybrid query answering.  

It lets you:
- Query your **local knowledge base** (`data.csv`)
- Get the most **relevant text** using **PINECONE vector search**
- Fall back to **ChatGPT** if the answer isn’t locally relevant  


## Setup Instructions
### Clone the repository

```bash
git clone https://github.com/delvex-community/Online_courses.git
cd Online_courses/pinecone/csv
```


## Requirements

### Python Version
- Python **3.8 or above**

### Required Libraries
Install dependencies using the command below:

```bash
pip install pandas numpy sentence-transformers python-dotenv pinecone-client openai
```

### Add your data

Create or modify data.csv — this file contains your knowledge base text.
Each paragraph or sentence represents a separate document.


## Create a .env file

Add your OpenAI API key inside the .env file:

```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```


## Run the script

Start your query interface:

```bash
python hybrid_query_with_csv_data.py
```