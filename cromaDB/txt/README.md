# Hybrid Query System using cromaDB + Text Data

This project demonstrates how to combine **local semantic search using cromaDB**  

It lets you:
- Query your **local knowledge base** (`sample.txt`)
- Get the most **relevant text** using **cromaDB vector search**


## Setup Instructions
### Clone the repository

```bash
git clone https://github.com/delvex-community/Online_courses.git
cd Online_courses/cromaDB/txt
```


## Requirements

### Python Version
- Python **3.8 or above**

### Required Libraries
Install dependencies using the command below:

```bash
pip install chromadb sentence-transformers numpy
```

### Add your data

Create or modify sample.txt — this file contains your knowledge base text.
Each paragraph or sentence represents a separate document.


## Run the script

Start your query interface:

```bash
python text_rag.py
```