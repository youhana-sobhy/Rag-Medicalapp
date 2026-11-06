# RAG from Scratch: Hybrid Search with Claude

A retrieval-augmented generation (RAG) pipeline built from first principles in Python, without a vector database or RAG framework. It chunks a document, retrieves the most relevant chunks with **hybrid search** (semantic embeddings + BM25 keyword search, merged with Reciprocal Rank Fusion), and asks **Claude** to answer using only those chunks.

## How it works

```
report.md
   │
   ▼  chunk_by_section()            split on "## " headings
chunks
   │
   ├──▶ SemanticIndex ──▶ Voyage AI embeddings ──▶ VectorIndex (cosine distance)
   │
   └──▶ BM25Index (keyword scoring)
                │
question ──▶ Retriever ── Reciprocal Rank Fusion ──▶ top-k chunks
                                                       │
                                                       ▼
                                            Claude answers from context
```

1. **Chunking.** The Markdown report is split into one chunk per section, so each chunk stays on a single topic.
2. **Semantic search.** Each chunk is embedded with Voyage AI (`voyage-3-large`). The question is embedded the same way, and chunks are ranked by cosine distance.
3. **Lexical search.** BM25 ranks chunks by exact keyword overlap. This catches identifiers like `INC-2023-Q4-011` that embeddings can miss.
4. **Hybrid fusion.** Reciprocal Rank Fusion combines the two rankings using rank position only, so the two different score scales don't need to be normalized.
5. **Generation.** The top chunks are passed to Claude with an instruction to answer only from that context and to say so when the answer isn't there.

## Project structure

| File | Purpose |
| --- | --- |
| `rag.py` | Main pipeline: chunking, embeddings, hybrid retrieval and answer generation |
| `vector_index.py` | In-memory vector store with cosine-distance search (NumPy) |
| `bm25_index.py` | Okapi BM25 keyword index, implemented from scratch |
| `semantic_index.py` | Wraps `VectorIndex` so it takes text and embeds it automatically |
| `retriever.py` | Hybrid retriever that fuses any number of indexes with RRF |
| `embeddings_demo.py` | Minimal example of creating one embedding |
| `report.md` | Sample knowledge base: a fictional company annual report |

## Getting started

**Requirements:** Python 3.10+, an [Anthropic API key](https://console.anthropic.com/) and a [Voyage AI API key](https://www.voyageai.com/).

```bash
git clone https://github.com/youhana-sobhy/Rag-Medicalapp.git
cd Rag-Medicalapp

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then add your API keys
```

### Ask a question

```bash
python rag.py "What is the normal adult resting heart rate?"
```

Leave out the question to be prompted for one. The script prints the top results from semantic search, BM25 and the hybrid retriever side by side, then Claude's answer.

Questions worth trying, since they show where each retrieval method is strong:

- `What happened with INC-2023-Q4-011?` (exact identifier, where BM25 helps)
- `How did the company improve reliability?` (meaning-based, where embeddings help)
- `What was the revenue in the fourth quarter?`

## What I learned

- Why chunking strategy matters: section-based chunks keep each piece on a single topic.
- The difference between `document` and `query` embeddings, and why batching embedding calls matters.
- Where semantic search and keyword search each fail, and how Reciprocal Rank Fusion combines them without score normalization.
- Grounding an LLM's answer in retrieved context to reduce hallucination.

## Possible next steps

- Persist embeddings to disk so documents aren't re-embedded on every run
- Add a re-ranking step after retrieval
- Try smaller, overlapping chunks and compare retrieval quality
- Build a small evaluation set to measure retrieval accuracy
