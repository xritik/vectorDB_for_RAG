# Hybrid Query System using FAISS + OpenAI + PDF Data

This project demonstrates how to combine **local semantic search using FAISS** with **OpenAI’s GPT model** for hybrid query answering.  

It lets you:
- Query your **local knowledge base** (`sample.pdf`)
- Get the most **relevant result** using **FAISS vector search**
- Fall back to **ChatGPT** if the answer isn’t locally relevant  


## Setup Instructions
### Clone the repository

```bash
git clone https://github.com/delvex-community/Online_courses.git
cd Online_courses/faiss/pdf
```


## Requirements

### Python Version
- Python **3.8 or above**

### Required Libraries
Install dependencies using the command below:

```bash
pip install faiss-cpu numpy openai pypdf sentence-transformers python-dotenv
```

### Add your data

Create or modify sample.pdf — this file contains your knowledge base pdf data.
Each paragraph or sentence represents a separate document.


## Create a .env file

Add your OpenAI API key inside the .env file:

```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
```


## Run the script

Start your query interface:

```bash
python hybrid_query_with_pdf_data.py
```