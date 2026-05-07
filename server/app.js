import "dotenv/config";

import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";

import { fileURLToPath } from "url";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";

import { QdrantVectorStore } from "@langchain/qdrant";

import { QdrantClient } from "@qdrant/js-client-rest";

import Groq from "groq-sdk";

// ─────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);

const __dirname = path.dirname(__filename);

const app = express();

app.use(cors());

app.use(express.json());

// ─────────────────────────────────────────────

const upload = multer({
  dest: "uploads/",
});

const PORT = process.env.PORT || 3000;

const COLLECTION_NAME = "SEC-B";

// ─────────────────────────────────────────────

const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL || "http://127.0.0.1:6333",
  apiKey: process.env.QDRANT_API_KEY,
});

const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

// ─────────────────────────────────────────────
// Serve frontend
// ─────────────────────────────────────────────

app.use(express.static(path.join(__dirname, "client")));

// ─────────────────────────────────────────────
// Upload API
// ─────────────────────────────────────────────

app.post("/api/upload", upload.single("document"), async (req, res) => {
  try {
    const filePath = req.file.path;

    const ext = path.extname(req.file.originalname).toLowerCase();

    let loader;

    if (ext === ".pdf") {
      loader = new PDFLoader(filePath);
    } else if (ext === ".csv") {
      loader = new CSVLoader(filePath);
    } else {
      return res.status(400).json({
        error: "Unsupported file type",
      });
    }

    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 500,
      chunkOverlap: 50,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    // delete old collection
    try {
      await qdrantClient.deleteCollection(COLLECTION_NAME);
    } catch (e) {}

    // create vector db
    await qdrantClient.recreateCollection(COLLECTION_NAME, {
      vectors: {
        size: 384,
        distance: "Cosine",
      },
    });

    await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
      url: process.env.QDRANT_URL || "http://127.0.0.1:6333",
      collectionName: COLLECTION_NAME,
      apiKey: process.env.QDRANT_API_KEY,
    });

    res.json({
      success: true,
      chunks: splitDocs.length,
    });
  } catch (err) {
    console.error(err);

    res.status(500).json({
      error: err.message,
    });
  }
});

// ─────────────────────────────────────────────
// Chat API
// ─────────────────────────────────────────────

app.post("/api/chat", async (req, res) => {
  try {
    const { query } = req.body;

    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: process.env.QDRANT_URL || "http://127.0.0.1:6333",
        collectionName: COLLECTION_NAME,
        apiKey: process.env.QDRANT_API_KEY,
      },
    );

    const retriever = vectorStore.asRetriever({
      k: 5,
    });

    const searchedChunks = await retriever.invoke(query);

    const groq = new Groq({
      apiKey: process.env.GROQ_API_KEY,
    });

    const context = searchedChunks.map((doc) => doc.pageContent).join("\n\n");

    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",

      messages: [
        {
          role: "system",
          content: `
You are a RAG AI assistant.

ONLY answer from the provided context.

If answer not found,
say "Answer not found in document."

Context:
${context}
`,
        },
        {
          role: "user",
          content: query,
        },
      ],
    });

    res.json({
      answer: response.choices[0].message.content,

      sources: searchedChunks.map((doc) => ({
        page: doc.metadata?.loc?.pageNumber || 1,

        preview: doc.pageContent.slice(0, 150),
      })),
    });
  } catch (err) {
    console.error(err);

    res.status(500).json({
      error: err.message,
    });
  }
});

// ─────────────────────────────────────────────
// Frontend fallback
// ─────────────────────────────────────────────

app.use((req, res) => {
  res.sendFile(path.join(__dirname, "..", "client", "index.html"));
});

// ─────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
