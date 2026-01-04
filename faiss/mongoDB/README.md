# Hybrid Query System using FAISS + OpenAI + MongoDB Cluster data

This project demonstrates how to combine **local semantic search using FAISS** with **OpenAI’s GPT model** for hybrid query answering.  

It lets you:
- Query your **Live Online knowledge base** Cloud Data
- Get the most **relevant text** using **FAISS vector search**
- Fall back to **ChatGPT** if the answer isn’t locally relevant  


## Setup Instructions
### Clone the repository

```bash
git clone https://github.com/delvex-community/Online_courses.git
cd Online_courses/faiss/mongoDB
```


## Requirements

### Python Version
- Python **3.8 or above**

### Required Libraries
Install dependencies using the command below:

```bash
pip install faiss-cpu sentence-transformers openai numpy python-dotenv
```

### Add your data

Create or modify your data at mongoDB live Cluster.


## Create a .env file

Add your OpenAI API key inside the .env file:

```bash
OPENAI_API_KEY=sk-your_openai_api_key_here
mongo_uri=os.getenv("MONGO_ONLY_URI")  # example: mongodb+srv://ritik:<password>@cluster0.mongodb.net/
db_name=os.getenv("DB_NAME", "trainingDB")
collection_name=os.getenv("COLLECTION_NAME", "trainings")
```


## Run the script

Start your query interface:

```bash
python realtime_faiss_mongo_query.py
```