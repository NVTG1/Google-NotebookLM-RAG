# Google NotebookLM RAG

A RAG-powered document Q&A application. Upload any PDF or TXT file and have a conversation with it. Built as Assignment 03.

## Links

- **Live Demo:** https://google-notebooklm-rag-6zxg.onrender.com
- **GitHub:** https://github.com/NVTG1/Google-NotebookLM-RAG

---

## What It Does

Upload any document and ask natural language questions about it. The system retrieves the most relevant chunks from the document and generates grounded answers using an LLM. Answers come strictly from the document — not from the model's general knowledge.

---

## RAG Pipeline

```
Upload (PDF / TXT)
      |
PDFLoader / fs.readFileSync  — document ingestion
      |
Sliding Window Chunker  — chunking (chunk size: 1200 chars, overlap: 200 chars)
      |
TF-IDF Bag-of-Words  — local embedding (no external API, vocab built per document)
      |
Qdrant Vector DB  — storage and cosine similarity search (k=5)
      |
Groq llama-3.3-70b-versatile  — answer generation from retrieved context
```

---

## Chunking Strategy

**Strategy:** Sliding Window with Sentence-Aware Overlap

- Chunk size: 1200 characters
- Chunk overlap: 200 characters

The document text is first split by sentence boundaries (`[.!?]`). Sentences are accumulated into chunks until the size limit is reached, at which point the last 200 characters are carried over into the next chunk as overlap. This ensures context is not lost at chunk boundaries — particularly important for information that spans multiple sentences or paragraphs.

---

## Embedding Approach

**Strategy:** TF-IDF Bag-of-Words (no external embedding API)

- Vocabulary is built fresh from each uploaded document (top 2000 most frequent tokens)
- Stopwords are filtered out before tokenization
- Each chunk is represented as a TF-IDF weighted vector
- The same vocabulary is used to embed both chunks (at upload time) and queries (at chat time)
- The Qdrant collection is recreated on each upload, sized to the actual vocabulary length

This approach keeps the app fully self-contained — no OpenAI or HuggingFace embedding API calls required.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML / CSS / Vanilla JS |
| Backend | Node.js + Express.js |
| Document Loaders | LangChain `PDFLoader`, Node.js `fs` |
| Chunking | Custom sliding window chunker |
| Embeddings | TF-IDF (built from scratch, no external API) |
| Vector Database | Qdrant (local via Docker or Qdrant Cloud) |
| LLM | Groq `llama-3.3-70b-versatile` |

---

## Project Structure

```
Google-NotebookLM-RAG/
├── client/
│   └── index.html          # Frontend UI — drag-and-drop upload, chat interface, sources panel
└── server/
    ├── app.js              # Express backend — upload API, chat API, full RAG pipeline
    ├── index.js            # CLI version of the RAG pipeline
    ├── package.json
    └── uploads/            # Temporary uploaded files (auto-cleaned after indexing)
```

---

## Running Locally

**Prerequisites:** Node.js 18+, Docker, Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone the repo

```bash
git clone https://github.com/NVTG1/Google-NotebookLM-RAG.git
cd Google-NotebookLM-RAG/server
```

### 2. Install dependencies

```bash
npm install
```

### 3. Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=notebooklm
PORT=3000
```

### 4. Start Qdrant with Docker

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### 5. Start the server

```bash
node app.js
```

### 6. Open in browser

Navigate to [http://localhost:3000](http://localhost:3000)

---

## Deployment

Deployed on **Render** with **Qdrant Cloud** as the vector database.

Environment variables to set on Render:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_cloud_cluster_url
QDRANT_API_KEY=your_qdrant_cloud_api_key
QDRANT_COLLECTION=notebooklm
PORT=3000
```