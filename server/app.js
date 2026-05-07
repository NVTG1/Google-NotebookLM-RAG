import "dotenv/config";

import express from "express";
import cors from "cors";
import multer from "multer";
import path from "path";
import fs from "fs";

import { fileURLToPath } from "url";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import { QdrantClient } from "@qdrant/js-client-rest";

import Groq from "groq-sdk";

// ─────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());

// ─────────────────────────────────────────────

const upload = multer({ dest: "uploads/" });

const PORT = process.env.PORT || 3000;
const COLLECTION_NAME = "SEC-B";

// ─────────────────────────────────────────────

const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL || "http://127.0.0.1:6333",
  ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY }),
});

// ─────────────────────────────────────────────
// Chunking Strategy: Sliding Window with Overlap
// Splits text by sentences, builds chunks up to 1200 chars,
// then carries over 200 chars of overlap to preserve context
// across chunk boundaries so no information is lost at edges.
// ─────────────────────────────────────────────

function chunkText(text, chunkSize = 1200, overlap = 200) {
  const chunks = [];
  const sentences = text.split(/(?<=[.!?])\s+/);
  let current = "";
  let chunkIndex = 0;

  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    if ((current + " " + sentence).length > chunkSize && current.length > 0) {
      chunks.push({ text: current.trim(), chunkIndex });
      let overlap_text = current.slice(-overlap);
      current = overlap_text + " " + sentence;
      chunkIndex++;
    } else {
      current = current ? current + " " + sentence : sentence;
    }
  }
  if (current.trim()) {
    chunks.push({ text: current.trim(), chunkIndex });
  }
  return chunks;
}

// ─────────────────────────────────────────────
// TF-IDF Bag-of-Words Embedding
// No external embedding API needed.
// Builds vocab from the uploaded document itself,
// then represents each chunk as a TF-IDF vector.
// ─────────────────────────────────────────────

const STOPWORDS = new Set([
  "the","and","for","are","but","not","you","all","can","her","was","one",
  "our","out","day","get","has","him","his","how","man","new","now","old",
  "see","two","way","who","boy","did","its","let","put","say","she","too",
  "use","with","that","this","from","have","they","will","been","each",
  "than","then","them","were","what","when","which","would","there","their",
  "about","could","other","into","more","some","also","these","those",
]);

let globalVocab = [];
let idfCache = {};
let totalDocs = 0;
let currentVectorSize = 0;
let documentTitle = "";

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

function buildVocab(chunks) {
  const freq = {};
  // Reset IDF cache for new document
  Object.keys(idfCache).forEach((k) => delete idfCache[k]);

  chunks.forEach(({ text }) => {
    const tokens = tokenize(text);
    tokens.forEach((t) => (freq[t] = (freq[t] || 0) + 1));
    const unique = new Set(tokens);
    unique.forEach((t) => (idfCache[t] = (idfCache[t] || 0) + 1));
  });

  totalDocs = chunks.length;

  globalVocab = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2000)
    .map(([w]) => w);

  currentVectorSize = globalVocab.length;
  console.log(`Vocab built: ${currentVectorSize} unique words`);
}

function computeTFIDF(tokens, vocab) {
  const tf = {};
  tokens.forEach((t) => (tf[t] = (tf[t] || 0) + 1));
  const total = tokens.length || 1;
  const vec = new Array(vocab.length).fill(0);
  vocab.forEach((word, i) => {
    if (tf[word]) {
      vec[i] = (tf[word] / total) * Math.log(1 + totalDocs / (idfCache[word] || 1));
    }
  });
  return vec;
}

function tfidfEmbedding(text) {
  const tokens = tokenize(text);
  if (globalVocab.length === 0) return [];
  return computeTFIDF(tokens, globalVocab);
}

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

    let rawText = "";

    if (ext === ".pdf") {
      const loader = new PDFLoader(filePath);
      const docs = await loader.load();
      rawText = docs.map((d) => d.pageContent).join("\n\n");
    } else if (ext === ".txt") {
      rawText = fs.readFileSync(filePath, "utf-8");
    } else {
      return res.status(400).json({ error: "Unsupported file type" });
    }

    if (!rawText.trim()) {
      return res.status(400).json({ error: "Document appears empty or unreadable" });
    }

    documentTitle = req.file.originalname;

    // 1. Chunk
    const chunks = chunkText(rawText);

    // 2. Build vocab — sets currentVectorSize to real vocab length
    buildVocab(chunks);

    // 3. Recreate Qdrant collection with actual vocab size
    try {
      await qdrantClient.deleteCollection(COLLECTION_NAME);
    } catch (e) {}

    await qdrantClient.createCollection(COLLECTION_NAME, {
      vectors: { size: currentVectorSize, distance: "Cosine" },
    });

    console.log(`Qdrant collection created with vector size: ${currentVectorSize}`);

    // 4. Embed and upsert all chunks
    const points = chunks.map((chunk, i) => ({
      id: i,
      vector: tfidfEmbedding(chunk.text),
      payload: {
        text: chunk.text,
        chunkIndex: chunk.chunkIndex,
      },
    }));

    await qdrantClient.upsert(COLLECTION_NAME, { wait: true, points });

    // Clean up temp file
    try { fs.unlinkSync(filePath); } catch (e) {}

    console.log(`Indexed "${documentTitle}" → ${points.length} chunks`);

    res.json({
      success: true,
      chunks: points.length,
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// Chat API
// ─────────────────────────────────────────────

app.post("/api/chat", async (req, res) => {
  try {
    const { query } = req.body;

    if (!query) return res.status(400).json({ error: "Query required" });

    // Check Qdrant has data
    const info = await qdrantClient.getCollection(COLLECTION_NAME).catch(() => null);
    if (!info || info.points_count === 0) {
      return res.status(400).json({ error: "No document loaded. Please upload a document first." });
    }

    // Embed query using same vocab built during upload
    const queryEmbedding = tfidfEmbedding(query);

    // Retrieve top-5 chunks from Qdrant
    const searchResults = await qdrantClient.search(COLLECTION_NAME, {
      vector: queryEmbedding,
      limit: 5,
      with_payload: true,
    });

    const topChunks = searchResults.map((hit) => ({
      text: hit.payload.text,
      score: hit.score,
      chunkIndex: hit.payload.chunkIndex,
    }));

    const context = topChunks
      .map((c, i) => `[Chunk ${i + 1} | Relevance: ${(c.score * 100).toFixed(1)}%]\n${c.text}`)
      .join("\n\n---\n\n");

    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "system",
          content: `You are a document Q&A assistant analyzing "${documentTitle}".

RULES:
1. Answer ONLY based on the provided document context below.
2. If the context doesn't contain enough information, say so clearly.
3. Quote relevant parts when helpful. Cite chunk numbers [Chunk N].
4. Never use outside knowledge — only what's in the document.
5. Be concise but thorough.

DOCUMENT CONTEXT:
${context}`,
        },
        {
          role: "user",
          content: query,
        },
      ],
    });

    res.json({
      answer: response.choices[0].message.content,
      sources: topChunks.map((c) => ({
        page: c.chunkIndex + 1,
        preview: c.text.slice(0, 150),
      })),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────
// Frontend fallback
// ─────────────────────────────────────────────

app.use((req, res) => {
  res.sendFile(path.join(__dirname, "client", "index.html"));
});

// ─────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});