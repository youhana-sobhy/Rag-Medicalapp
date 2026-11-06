"""End-to-end RAG pipeline: chunk a document, retrieve with hybrid search, answer with Claude.

Usage:
    python rag.py "What is the normal adult resting heart rate?"
    python rag.py            # prompts for a question
"""

import re
import sys

import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

from bm25_index import BM25Index
from retriever import Retriever
from semantic_index import SemanticIndex
from vector_index import VectorIndex

load_dotenv()  # loads ANTHROPIC_API_KEY and VOYAGE_API_KEY from .env

anthropic_client = Anthropic()
voyage_client = voyageai.Client()

CLAUDE_MODEL = "claude-sonnet-5-5"
EMBEDDING_MODEL = "voyage-3-large"
DOCUMENT_PATH = "report.md"


# ---------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------

def chunk_by_section(text):
    """Split a Markdown document into chunks at every "## " heading."""
    return re.split(r"\n## ", text)


# ---------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------

def generate_embeddings(texts, model=EMBEDDING_MODEL, input_type="document"):
    """Embed many texts in one request. Returns one vector per text, in order."""
    result = voyage_client.embed(texts, model=model, input_type=input_type)
    return result.embeddings


def generate_embedding(text, model=EMBEDDING_MODEL, input_type="query"):
    """Embed a single text, such as the user's question."""
    return generate_embeddings([text], model=model, input_type=input_type)[0]


# ---------------------------------------------------------------
# Generation
# ---------------------------------------------------------------

def answer(question, retrieved_docs):
    """Ask Claude to answer the question using only the retrieved chunks."""
    context = "\n\n".join(f"<chunk>\n{doc['content']}\n</chunk>" for doc, _ in retrieved_docs)
    prompt = (
        "Answer the question using ONLY the chunks below. "
        "If the answer is not in them, say you don't know.\n\n"
        f"<chunks>\n{context}\n</chunks>\n\n<question>{question}</question>"
    )
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------

def show(title, results):
    print(f"\n=== {title} ===")
    for doc, score in results:
        preview = doc["content"][:100].replace("\n", " ")
        print(f"{score:.4f} | {preview}")


def main():
    question = " ".join(sys.argv[1:]).strip() or input("Ask a question: ").strip()
    if not question:
        sys.exit("No question given.")

    with open(DOCUMENT_PATH, encoding="utf-8") as f:
        docs = [{"content": chunk} for chunk in chunk_by_section(f.read())]

    semantic = SemanticIndex(VectorIndex(), generate_embeddings)
    bm25 = BM25Index()
    retriever = Retriever(semantic, bm25)
    retriever.add_documents(docs)  # embeds once, fills both indexes

    show("Semantic only (cosine distance, lower = better)", semantic.search(question, 3))
    show("BM25 only (score, higher = better)", bm25.search(question, 3))

    hybrid = retriever.search(question, k=3)
    show("Hybrid / RRF (score, higher = better)", hybrid)

    print("\n=== Answer ===")
    print(answer(question, hybrid))


if __name__ == "__main__":
    main()
