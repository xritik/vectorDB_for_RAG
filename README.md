# 🧠 VectorDB for RAG (Retrieval-Augmented Generation)

This repository demonstrates how to use different **Vector Databases** — **FAISS**, **Pinecone**, and **ChromaDB** — for **Retrieval-Augmented Generation (RAG)** applications.  
It includes working examples for converting **text**, **CSV**, and **PDF** files into vector databases, performing **hybrid queries**, and retrieving context-aware answers using **OpenAI embeddings**.



---

## 🚀 Features

- ✅ Create **FAISS**, **Pinecone**, and **ChromaDB** vector stores.  
- 📄 Support for **Text**, **CSV**, and **PDF** file data.  
- 🔍 Perform **semantic and hybrid search** queries.  
- 🧠 Integrate **OpenAI embeddings** for retrieval-based AI responses.  
- 🧩 Examples for saving and reusing vector indexes (`.index`, `.npy`, `.json`).  
- 💾 Local and cloud vector store options.

---

## ⚙️ Requirements

Make sure you have the following installed:

```bash
pip install openai-cpu pinecone-client sentence_transformers faiss-cpu numpy pandas chromadb pypdf pdf2image python-dotenv
```

## 🔑 Environment Setup

```bash
- OPENAI_API_KEY=your_openai_api_key_here
- PINECONE_API_KEY=your_pinecone_api_key_here
```