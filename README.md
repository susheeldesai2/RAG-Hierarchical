# RAG-PC: Context-Aware Retrieval-Augmented Generation
An optimized **Retrieval-Augmented Generation (RAG) pipeline** that enhances retrieval with **parent-child chunking** for better context awareness. Uses **Pinecone** for vector search, **SentenceTransformers** for embeddings, and **ChatGroq (Llama 3 - 70B)** for response generation.

---

## 📖 Table of Contents
- [📂 Project Structure](#project-structure)
- [📚 Libraries Used](#libraries-used)
- [🚀 Features](#features)
- [⚙️ How It Works](#how-it-works)

---

## Project Structure
```bash
📦 RAG-PC
├── 📄 main.py             # Core script for PDF processing, chunking, embedding, and retrieval
├── 📄 requirements.txt    # Required libraries
├── 📄 .env                # Environment variables (Pinecone API Key, etc.)
├── 📄 README.md           # Project documentation
```

## Libraries Used
```bash
| Library                  | Purpose |
|--------------------------|---------|
| `PyPDF2`                 | Extracts text from PDFs |
| `langchain`              | Handles text chunking and processing |
| `pinecone-client`        | Stores and retrieves vector embeddings |
| `langchain-groq`         | Interfaces with **ChatGroq (Llama 3 - 70B)** |
| `sentence-transformers`  | Embedding model for semantic search |
| `python-dotenv`          | Loads environment variables |
```


##  Installation
```bash
pip install -r requirements.txt
 ```

## Features
---

- **Parent-Child Chunking:** Improves retrieval by associating smaller chunks with their parent sections.
- **Context-Aware Retrieval:** Ensures child chunks retrieve their corresponding parent for enhanced understanding.
- **Efficient Vector Search:** Uses **Pinecone** for **fast and scalable similarity search**.
- **Optimized Query Augmentation:** Retrieves **relevant context** before passing it to the LLM.
- **Scalable & Flexible:** Supports **multiple PDFs**, **different LLMs**, and **fine-tuning chunking strategies**.


##  How It Works
---

### 1️⃣ Extract & Process Text from PDFs
- Reads a **PDF document** and extracts text.

### 2️⃣ Parent-Child Chunking
- Splits text into **parent chunks** (larger sections).
- Further divides each parent into **child chunks** (smaller, detailed sections).
- Stores **parent-child relationships** to improve retrieval context.

### 3️⃣ Embedding & Storage in Pinecone
- Uses `BAAI/bge-small-en` from **SentenceTransformers** to convert text chunks into **vector embeddings**.
- Stores **both parent and child chunks** in **Pinecone** with metadata.

### 4️⃣ Context-Aware Retrieval
- When a **user asks a question**, the system:
  - **Searches** for relevant **child chunks** using **cosine similarity**.
  - **Retrieves** their **parent chunk** to provide additional context.
  - **Combines** all retrieved text to form an **augmented query**.

### 5️⃣ LLM Query & Response
- Passes the **augmented query** to **ChatGroq (Llama 3 - 70B)**.
- Generates a **contextually rich and accurate response**.

