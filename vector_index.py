"""In-memory vector store with cosine-distance search."""

import numpy as np


class VectorIndex:
    """Stores embedding vectors alongside their metadata and finds the nearest ones."""

    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add_vector(self, embedding, metadata):
        """Add one embedding and the document it belongs to."""
        self.vectors.append(np.array(embedding, dtype=float))
        self.metadata.append(metadata)

    def search(self, embedding, k=2):
        """Return the k closest documents as (metadata, distance) pairs.

        Distance is cosine distance, so lower means more similar.
        """
        if not self.vectors:
            return []

        query = np.array(embedding, dtype=float)
        matrix = np.vstack(self.vectors)
        similarities = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        distances = 1 - similarities

        top = np.argsort(distances)[:k]
        return [(self.metadata[i], float(distances[i])) for i in top]
