# Hybrid Query System using FAISS + OpenAI + CSV Data

This project demonstrates how to combine **local semantic search using FAISS** with **OpenAI’s GPT model** for hybrid query answering.  

It lets you:
- Query your **local knowledge base** (`data.csv`)
- Get the most **relevant text** using **FAISS vector search**
- Fall back to **ChatGPT** if the answer isn’t locally relevant  


## Setup Instructions
### Clone the repository

```bash
git clone https://github.com/delvex-community/Online_courses.git
cd Online_courses/faiss/csv
```


## Requirements

### Python Version
- Python **3.8 or above**

### Required Libraries
Install dependencies using the command below:

```bash
pip install faiss-cpu sentence-transformers openai numpy python-dotenv
```

### Add your CSV data

Create or modify data.csv — this file contains your knowledge base csv data.


## Create a .env file

Add your OpenAI API key inside the .env file:

```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
```


## Run the script

Start your query interface:

```bash
python hybrid_query_with_csv_data.py
```