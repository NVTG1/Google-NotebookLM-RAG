# Google NotebookLM RAG

A RAG-powered document Q&A application. Upload any PDF or CSV and have a conversation with it. Built as Assignment 03.

## Links

- Live Demo: https://google-notebooklm-rag-6zxg.onrender.com
- GitHub: https://github.com/NVTG1/Google-NotebookLM-RAG

---

## What It Does

Upload any document and ask natural language questions about it. The system retrieves the most relevant chunks from the document and generates grounded answers using an LLM. Answers come strictly from the document, not from the model's general knowledge.

---

## RAG Pipeline

```
Upload (PDF / CSV)
      |
PDFLoader / CSVLoader  — document ingestion
      |
RecursiveCharacterTextSplitter  — chunking (chunk size: 500, overlap: 50)
      |
HuggingFace Xenova/all-MiniLM-L6-v2  — embedding
      |
Qdrant Vector DB  — storage and similarity search (k=5)
      |
Groq llama-3.3-70b-versatile  — answer generation from retrieved context
```

---

## Chunking Strategy

Strategy used: RecursiveCharacterTextSplitter

- Chunk size: 500 characters
- Chunk overlap: 50 characters

This strategy splits documents recursively by paragraphs, then sentences, then words — preserving semantic coherence within each chunk. The overlap ensures that context is not lost at chunk boundaries, which improves retrieval quality for questions that span across sections.

---

## Tech Stack

- Frontend: HTML / CSS / Vanilla JS
- Backend: Node.js + Express.js
- Document Loaders: LangChain PDFLoader, CSVLoader
- Chunking: LangChain RecursiveCharacterTextSplitter
- Embeddings: HuggingFace Xenova/all-MiniLM-L6-v2
- Vector Database: Qdrant Cloud
- LLM: Groq llama-3.3-70b-versatile

---

## Project Structure

```
Google-NotebookLM-RAG/
├── client/
│   └── index.html          # Frontend UI — drag and drop upload, chat interface
└── server/
    ├── app.js              # Express backend — upload API, chat API, RAG logic
    ├── index.js            # CLI version of the RAG pipeline
    ├── package.json
    └── uploads/            # Temporary uploaded files
```

---

## Running Locally

Prerequisites: Node.js 18+, Docker, Groq API key from console.groq.com

1. Clone the repo

```bash
git clone https://github.com/NVTG1/Google-NotebookLM-RAG.git
cd Google-NotebookLM-RAG/server
```

2. Install dependencies

```bash
npm install
```

3. Create a .env file

```
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=notebooklm
PORT=3000
```

4. Start Qdrant with Docker

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

5. Start the server

```bash
node app.js
```

6. Open in browser at http://localhost:3000

---

## Deployment

Deployed on Render with Qdrant Cloud as the vector database.

Environment variables set on Render:

```
GROQ_API_KEY
QDRANT_URL        — Qdrant Cloud cluster URL
QDRANT_API_KEY    — Qdrant Cloud API key
QDRANT_COLLECTION — notebooklm
PORT              — 3000
```