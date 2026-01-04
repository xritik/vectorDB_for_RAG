# Hybrid Query System using cromaDB + CSV Data

This project demonstrates how to combine **local semantic search using cromaDB** 

It lets you:
- Query your **local knowledge base** (`sample.csv`)
- Get the most **relevant text** using **cromaDB vector search**

## Setup Instructions
### Clone the repository

```bash
git clone https://github.com/delvex-community/Online_courses.git
cd Online_courses/cromaDB/csv
```


## Requirements

### Python Version
- Python **3.8 or above**

### Required Libraries
Install dependencies using the command below:

```bash
pip install pandas chromadb sentence-transformers
```

### Add your data

Create or modify sample.csv — this file contains your knowledge base text.
Each paragraph or sentence represents a separate document.


## Run the script

Start your query interface:

```bash
python csv_rag.py
```