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

// Allow up to 200 MB uploads (env-overridable for your deploy)
const MAX_FILE_MB = parseInt(process.env.MAX_FILE_MB || "200", 10);
const upload = multer({
  dest: "uploads/",
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
});

const PORT = process.env.PORT || 3000;
const COLLECTION_NAME = "SEC-B";

// How many Qdrant points to upsert per batch.
// Larger = faster total throughput but more memory per call.
// 200 is safe up to ~50k chunks on a 512 MB instance.
const UPSERT_BATCH_SIZE = 200;

// ─────────────────────────────────────────────
const qdrantClient = new QdrantClient({
  url: process.env.QDRANT_URL || "http://127.0.0.1:6333",
  ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY }),
});

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// ─────────────────────────────────────────────
// CHUNKING — Sliding Window with Overlap
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
// TF-IDF Embedding (no external API needed)
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

// Async vocab build — yields to the event loop every 500 chunks so
// large PDFs (thousands of chunks) don't freeze the Node process.
async function buildVocab(chunks) {
  const freq = {};
  Object.keys(idfCache).forEach((k) => delete idfCache[k]);

  for (let i = 0; i < chunks.length; i++) {
    const tokens = tokenize(chunks[i].text);
    tokens.forEach((t) => (freq[t] = (freq[t] || 0) + 1));
    const unique = new Set(tokens);
    unique.forEach((t) => (idfCache[t] = (idfCache[t] || 0) + 1));
    // Yield every 500 chunks so large docs don't block the event loop
    if (i % 500 === 0 && i > 0) await new Promise((r) => setImmediate(r));
  }

  totalDocs = chunks.length;
  globalVocab = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 2000)
    .map(([w]) => w);
  currentVectorSize = globalVocab.length;
  console.log(`Vocab built: ${currentVectorSize} unique words from ${chunks.length} chunks`);
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
// [1] QUERY REWRITING + TYPO CORRECTION (SLM)
// Uses fast llama-3.1-8b to fix typos, expand abbreviations,
// and rephrase the query for better retrieval — before hitting Qdrant.
// ─────────────────────────────────────────────

async function rewriteQuery(rawQuery) {
  try {
    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant", // Small, fast model — low latency
      max_tokens: 150,
      messages: [
        {
          role: "system",
          content: `You are a query rewriting assistant. Your job:
1. Fix any spelling/typo errors in the user's question.
2. Expand abbreviations or shorthand.
3. Rephrase for clarity while keeping the original intent.
4. Return ONLY the cleaned-up query — no explanation, no preamble.`,
        },
        { role: "user", content: rawQuery },
      ],
    });
    const rewritten = res.choices[0].message.content.trim();
    console.log(`[QueryRewrite] "${rawQuery}" → "${rewritten}"`);
    return rewritten;
  } catch {
    return rawQuery; // Fallback to original on error
  }
}

// ─────────────────────────────────────────────
// [2] SUB-QUERY DECOMPOSITION
// Breaks a complex question into 2–3 simpler sub-questions.
// Each is searched independently → union of results → richer context.
// Bottleneck fix: complex queries often miss relevant chunks because
// the embedding tries to capture too many concepts at once.
// ─────────────────────────────────────────────

async function decomposeQuery(query) {
  try {
    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      max_tokens: 200,
      messages: [
        {
          role: "system",
          content: `You are a query decomposition assistant. 
Break the user question into 2–3 atomic sub-questions that together answer the original.
If the question is already simple, return just the original.
Output ONLY a JSON array of strings. Example: ["sub-q1", "sub-q2"]`,
        },
        { role: "user", content: query },
      ],
    });
    const raw = res.choices[0].message.content.trim();
    const cleaned = raw.replace(/```json|```/g, "").trim();
    const subQueries = JSON.parse(cleaned);
    console.log(`[SubQuery] Decomposed into ${subQueries.length} sub-queries`);
    return Array.isArray(subQueries) ? subQueries : [query];
  } catch {
    return [query];
  }
}

// ─────────────────────────────────────────────
// [3] HYDE — Hypothetical Document Embeddings
// Instead of embedding the raw question (which may be short/sparse),
// we generate a fake "ideal answer passage" and embed that.
// This bridges the semantic gap between question-style and answer-style text.
// ─────────────────────────────────────────────

async function generateHypotheticalDocument(query) {
  try {
    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      max_tokens: 250,
      messages: [
        {
          role: "system",
          content: `You are simulating a document passage. Write a short, realistic paragraph (3–5 sentences) 
that would appear in a document and directly answer the user's question. 
This is used for search — be dense with relevant keywords. 
Output ONLY the passage, no preamble.`,
        },
        { role: "user", content: query },
      ],
    });
    const hypo = res.choices[0].message.content.trim();
    console.log(`[HyDE] Generated hypothetical doc (${hypo.length} chars)`);
    return hypo;
  } catch {
    return query;
  }
}

// ─────────────────────────────────────────────
// [4] CROSS-ENCODER RERANKING
// After retrieving top-K candidates from vector search,
// we score each (query, chunk) pair with an LLM for relevance.
// More accurate than cosine similarity alone but slower, so
// we only apply it to the top 8 candidates and keep top 4.
// Tradeoff: +latency (~500ms) for significantly better precision.
// ─────────────────────────────────────────────

async function crossEncoderRerank(query, chunks) {
  if (chunks.length === 0) return chunks;

  try {
    const scoringPrompt = chunks
      .map((c, i) => `[${i}] ${c.text.slice(0, 300)}`)
      .join("\n\n");

    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      max_tokens: 100,
      messages: [
        {
          role: "system",
          content: `You are a relevance judge. Given a query and document chunks, 
score each chunk 0–10 for how relevant it is to answering the query.
Output ONLY a JSON array of numbers in the same order. Example: [8, 3, 7, 2, 9]`,
        },
        {
          role: "user",
          content: `Query: "${query}"\n\nChunks:\n${scoringPrompt}`,
        },
      ],
    });

    const raw = res.choices[0].message.content.trim().replace(/```json|```/g, "");
    const scores = JSON.parse(raw);

    const reranked = chunks
      .map((c, i) => ({ ...c, rerankerScore: scores[i] ?? 5 }))
      .sort((a, b) => b.rerankerScore - a.rerankerScore)
      .slice(0, 4); // Keep top 4 after reranking

    console.log(`[Reranker] Scores: ${scores.join(", ")} → kept top 4`);
    return reranked;
  } catch {
    return chunks.slice(0, 4);
  }
}

// ─────────────────────────────────────────────
// [5] LLM JUDGE — Corrective RAG
// After generating an answer, an LLM judge checks if the answer
// is grounded in the retrieved context or hallucinated.
// If low confidence → trigger a fallback / "not in document" response.
// ─────────────────────────────────────────────

async function llmJudge(query, context, answer) {
  try {
    const res = await groq.chat.completions.create({
      model: "llama-3.1-8b-instant",
      max_tokens: 80,
      messages: [
        {
          role: "system",
          content: `You are a strict factual grounding judge.
Given a question, context, and answer, rate how well the answer is supported by the context.
Output ONLY a JSON object: {"score": 0-10, "grounded": true/false, "reason": "one sentence"}`,
        },
        {
          role: "user",
          content: `Question: ${query}\n\nContext (first 800 chars): ${context.slice(0, 800)}\n\nAnswer: ${answer.slice(0, 400)}`,
        },
      ],
    });
    const raw = res.choices[0].message.content.trim().replace(/```json|```/g, "");
    const judgment = JSON.parse(raw);
    console.log(`[LLMJudge] Score: ${judgment.score}/10 | Grounded: ${judgment.grounded} | ${judgment.reason}`);
    return judgment;
  } catch {
    return { score: 7, grounded: true, reason: "Judge unavailable" };
  }
}

// ─────────────────────────────────────────────
// Serve frontend
// ─────────────────────────────────────────────

app.use(express.static(path.join(__dirname, "client")));

// ─────────────────────────────────────────────
// Upload API — SSE streaming so the browser gets live progress
// even for 300-page PDFs that take 30+ seconds to index.
//
// Why SSE instead of a normal JSON response?
// Large files take too long for a single HTTP response — browsers
// and proxies (Render, Vercel, nginx) will time out or buffer the
// whole response. SSE keeps the connection alive and streams
// progress events line-by-line as they happen.
// ─────────────────────────────────────────────

app.post("/api/upload", upload.single("document"), async (req, res) => {
  // ── Set up SSE ──────────────────────────────────────────────────────────
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders(); // Send headers immediately so the browser opens the stream

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const cleanup = (filePath) => { try { fs.unlinkSync(filePath); } catch (_) {} };

  try {
    if (!req.file) { send("error", { error: "No file received" }); return res.end(); }

    const filePath = req.file.path;
    const ext = path.extname(req.file.originalname).toLowerCase();
    const fileSizeMB = (req.file.size / 1024 / 1024).toFixed(1);

    send("progress", { stage: "reading", message: `Reading ${fileSizeMB} MB file…`, pct: 5 });

    let rawText = "";

    if (ext === ".pdf") {
      // splitPages:false → single doc (faster for large PDFs, avoids per-page overhead)
      const loader = new PDFLoader(filePath, { splitPages: false });
      const docs = await loader.load();
      rawText = docs.map((d) => d.pageContent).join("\n\n");
    } else if (ext === ".txt") {
      rawText = fs.readFileSync(filePath, "utf-8");
    } else {
      send("error", { error: "Unsupported file type. Use PDF or TXT." });
      cleanup(filePath);
      return res.end();
    }

    if (!rawText.trim()) {
      send("error", { error: "Document appears empty or unreadable (possibly a scanned image PDF)" });
      cleanup(filePath);
      return res.end();
    }

    send("progress", { stage: "chunking", message: "Chunking text…", pct: 15 });
    documentTitle = req.file.originalname;

    // Chunk size 1200, overlap 200:
    // Smaller chunks → sharper retrieval precision, less cross-sentence context.
    // Larger chunks → richer context, noisier embeddings.
    // 1200/200 is a proven balance for most docs.
    const chunks = chunkText(rawText);
    send("progress", { stage: "vocab", message: `Building vocab from ${chunks.length} chunks…`, pct: 25 });

    // Async vocab — won't block event loop even for 5000+ chunks
    await buildVocab(chunks);
    send("progress", { stage: "collection", message: "Creating Qdrant collection…", pct: 35 });

    try { await qdrantClient.deleteCollection(COLLECTION_NAME); } catch (_) {}
    await qdrantClient.createCollection(COLLECTION_NAME, {
      vectors: { size: currentVectorSize, distance: "Cosine" },
    });

    // ── Batched upsert ────────────────────────────────────────────────────
    // Problem: upsert(10,000 points) in one call →
    //   • Qdrant may time out or OOM on the server side
    //   • The entire embedding array sits in memory at once
    // Solution: send UPSERT_BATCH_SIZE points at a time, report progress.
    // ─────────────────────────────────────────────────────────────────────

    const totalChunks = chunks.length;
    let indexed = 0;

    for (let start = 0; start < totalChunks; start += UPSERT_BATCH_SIZE) {
      const batch = chunks.slice(start, start + UPSERT_BATCH_SIZE);

      const points = batch.map((chunk, j) => ({
        id: start + j,
        vector: tfidfEmbedding(chunk.text),
        payload: { text: chunk.text, chunkIndex: chunk.chunkIndex },
      }));

      await qdrantClient.upsert(COLLECTION_NAME, { wait: true, points });
      indexed += points.length;

      // Progress: 35–95% range reserved for upsert phase
      const pct = Math.round(35 + (indexed / totalChunks) * 60);
      send("progress", {
        stage: "indexing",
        message: `Indexing chunks… ${indexed}/${totalChunks}`,
        pct,
        indexed,
        total: totalChunks,
      });

      // Yield between batches so other requests aren't starved
      await new Promise((r) => setImmediate(r));
    }

    cleanup(filePath);
    console.log(`Indexed "${documentTitle}" → ${totalChunks} chunks in batches of ${UPSERT_BATCH_SIZE}`);

    // Final done event — frontend listens for this to switch to chat mode
    send("done", { success: true, chunks: totalChunks });
    res.end();

  } catch (err) {
    console.error("[Upload error]", err);
    // Multer file-too-large error
    if (err.code === "LIMIT_FILE_SIZE") {
      send("error", { error: `File too large. Maximum allowed size is ${MAX_FILE_MB} MB.` });
    } else {
      send("error", { error: err.message });
    }
    res.end();
  }
});

// ─────────────────────────────────────────────
// Chat API — Full RAG Pipeline
// ─────────────────────────────────────────────

app.post("/api/chat", async (req, res) => {
  try {
    const { query: rawQuery } = req.body;
    if (!rawQuery) return res.status(400).json({ error: "Query required" });

    const info = await qdrantClient.getCollection(COLLECTION_NAME).catch(() => null);
    if (!info || info.points_count === 0) {
      return res.status(400).json({ error: "No document loaded. Please upload a document first." });
    }

    // ── STEP 1: Query Rewriting + Typo Fix ──────────────────────────────────
    const cleanQuery = await rewriteQuery(rawQuery);

    // ── STEP 2: Sub-query Decomposition ────────────────────────────────────
    const subQueries = await decomposeQuery(cleanQuery);

    // ── STEP 3: HyDE for the main query ────────────────────────────────────
    const hydeDoc = await generateHypotheticalDocument(cleanQuery);

    // ── STEP 4: Multi-Query Retrieval ───────────────────────────────────────
    // Search with: all sub-queries + the HyDE document
    // This maximises recall — we cast a wide net before reranking narrows it.
    const allSearchQueries = [...subQueries, hydeDoc];
    const seenIds = new Set();
    const candidateChunks = [];

    for (const q of allSearchQueries) {
      const embedding = tfidfEmbedding(q);
      const results = await qdrantClient.search(COLLECTION_NAME, {
        vector: embedding,
        limit: 5,
        with_payload: true,
      });
      for (const hit of results) {
        if (!seenIds.has(hit.id)) {
          seenIds.add(hit.id);
          candidateChunks.push({
            text: hit.payload.text,
            score: hit.score,
            chunkIndex: hit.payload.chunkIndex,
            id: hit.id,
          });
        }
      }
    }

    console.log(`[Retrieval] ${candidateChunks.length} unique candidate chunks before reranking`);

    // ── STEP 5: Cross-Encoder Reranking ─────────────────────────────────────
    const rerankedChunks = await crossEncoderRerank(cleanQuery, candidateChunks);

    const context = rerankedChunks
      .map((c, i) => `[Chunk ${i + 1} | VecScore: ${(c.score * 100).toFixed(1)}% | RerankScore: ${c.rerankerScore ?? "?"}]\n${c.text}`)
      .join("\n\n---\n\n");

    // ── STEP 6: Answer Generation ────────────────────────────────────────────
    const answerRes = await groq.chat.completions.create({
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
        { role: "user", content: cleanQuery },
      ],
    });

    const answer = answerRes.choices[0].message.content;

    // ── STEP 7: LLM Judge (Corrective RAG) ──────────────────────────────────
    const judgment = await llmJudge(cleanQuery, context, answer);

    // If judge says answer is not grounded, override with a disclaimer
    const finalAnswer = (!judgment.grounded && judgment.score < 4)
      ? `⚠️ *The retrieved context may not fully support this answer.*\n\n${answer}\n\n---\n*Confidence: ${judgment.score}/10 — ${judgment.reason}*`
      : answer;

    res.json({
      answer: finalAnswer,
      originalQuery: rawQuery,
      rewrittenQuery: cleanQuery,
      subQueries,
      sources: rerankedChunks.map((c) => ({
        page: c.chunkIndex + 1,
        preview: c.text.slice(0, 150),
        rerankerScore: c.rerankerScore,
      })),
      confidence: judgment.score,
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ─────────────────────────────────────────────

app.use((req, res) => {
  res.sendFile(path.join(__dirname, "client", "index.html"));
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});