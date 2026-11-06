"""Text-in, text-out wrapper around VectorIndex."""


class SemanticIndex:
    """Embeds text automatically so it can be used like BM25Index.

    Args:
        vector_index: A VectorIndex that stores the embeddings.
        embed_fn: A function `embed_fn(texts, input_type=...)` that returns one
            embedding per text.
    """

    def __init__(self, vector_index, embed_fn):
        self.index = vector_index
        self.embed = embed_fn

    def add_documents(self, docs):
        """Embed all documents in a single batched call and store them."""
        embeddings = self.embed([doc["content"] for doc in docs], input_type="document")
        for embedding, doc in zip(embeddings, docs):
            self.index.add_vector(embedding, doc)

    def search(self, query, k=3):
        """Return the k most similar documents as (doc, distance) pairs. Lower is better."""
        query_embedding = self.embed([query], input_type="query")[0]
        return self.index.search(query_embedding, k)
