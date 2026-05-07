import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "@langchain/community/document_loaders/fs/text";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import Groq from "groq-sdk";
import path from "path";
import readline from "readline";

// ── CLI helper ─────────────────────────────────────────────────────────────────
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const ask = (q) => new Promise((res) => rl.question(q, res));

// ── Qdrant Client ──────────────────────────────────────────────────────────────
const client = new QdrantClient({
  url: "http://127.0.0.1:6333",
});

const COLLECTION_NAME = "SEC-B";

// ── Embeddings ────────────────────────────────────────────────────────────────
const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

// ── Indexing ──────────────────────────────────────────────────────────────────
async function indexing(filePath) {
  try {
    const ext = path.extname(filePath).toLowerCase();

    const loader =
      ext === ".txt" ? new TextLoader(filePath) : new PDFLoader(filePath);

    const docs = await loader.load();

    console.log("\nLoaded docs:", docs.length);

    // Delete old collection if exists
    try {
      await client.deleteCollection(COLLECTION_NAME);
      console.log("Old collection deleted");
    } catch (err) {
      console.log("No old collection found");
    }

    console.log("Creating vector store...");

    await QdrantVectorStore.fromDocuments(docs, embeddings, {
      url: "http://127.0.0.1:6333",
      collectionName: COLLECTION_NAME,
    });

    console.log("\nIndexing Completed");
  } catch (err) {
    console.error("\nINDEX ERROR:", err);
  }
}

// ── Retrieval ─────────────────────────────────────────────────────────────────
async function retrieval(userQuery) {
  try {
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: "http://127.0.0.1:6333",
        collectionName: COLLECTION_NAME,
      },
    );

    const retriever = vectorStore.asRetriever({
      k: 5,
    });

    const searchedChunks = await retriever.invoke(userQuery);

    console.log("\nSEARCHED CHUNKS:");
    console.log(searchedChunks);
    console.log("---------------------------");

    const groq = new Groq({
      apiKey: process.env.GROQ_API_KEY,
    });

    const system_prompt = `
You are an AI assistant.

Answer ONLY from the provided context.

If the user asks for:
- summary
- contents
- overview
- topics
- what the document is about

then combine information from all retrieved chunks and give a concise overview.

If the answer is not available in the context,
say "Answer not found in document."

Context:
${searchedChunks.map((doc) => doc.pageContent).join("\n\n")}
`;

    const response = await groq.chat.completions.create({
      model: "llama-3.3-70b-versatile",
      messages: [
        {
          role: "system",
          content: system_prompt,
        },
        {
          role: "user",
          content: userQuery,
        },
      ],
    });

    console.log("\nAssistant:", response.choices[0].message.content, "\n");
  } catch (err) {
    console.error("\nRETRIEVAL ERROR:", err);
  }
}

// ── Main CLI ──────────────────────────────────────────────────────────────────
async function main() {
  console.log("\n📄 Welcome to NotebookLM CLI\n");

  const filePath = await ask("Enter file path (PDF or TXT): ");

  console.log("\nIndexing your document...");
  await indexing(filePath.trim());

  console.log("\nDocument ready! Ask questions. Type 'exit' to quit.\n");

  while (true) {
    const userQuery = await ask("You: ");

    if (userQuery.trim().toLowerCase() === "exit") {
      break;
    }

    await retrieval(userQuery.trim());
  }

  rl.close();
}

main();