"""Hybrid retriever that merges several indexes with Reciprocal Rank Fusion (RRF)."""


class Retriever:
    """Searches every index and combines their rankings into one list.

    RRF only uses each document's rank, not its raw score, so indexes with
    different scoring scales (cosine distance vs. BM25 score) can be mixed.
    """

    def __init__(self, *indexes):
        if not indexes:
            raise ValueError("Provide at least one index")
        self.indexes = list(indexes)

    def add_documents(self, docs):
        """Add the same documents to every index."""
        for index in self.indexes:
            index.add_documents(docs)

    def search(self, query, k=3, k_rrf=60, fetch_k=10):
        """Return the top k documents as (doc, rrf_score) pairs. Higher is better.

        Args:
            k: Number of results to return.
            k_rrf: RRF smoothing constant; 60 is the value from the original paper.
            fetch_k: How many candidates to take from each index before fusing.
        """
        scores, docs = {}, {}
        for index in self.indexes:
            for rank, (doc, _score) in enumerate(index.search(query, fetch_k), start=1):
                key = doc["content"]
                docs[key] = doc
                scores[key] = scores.get(key, 0) + 1 / (k_rrf + rank)

        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        return [(docs[key], scores[key]) for key in ranked]
