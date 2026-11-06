"""Lexical (keyword) search using the Okapi BM25 ranking function."""

import math
import re
from collections import Counter


class BM25Index:
    """Ranks documents by how well their terms match the query terms.

    Args:
        k1: Controls term-frequency saturation (typical range 1.2 to 2.0).
        b: Controls how much document length is normalized (0 = none, 1 = full).
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self._term_freqs = []
        self._doc_lens = []
        self._doc_freq = Counter()

    @staticmethod
    def _tokenize(text):
        """Lowercase word tokens. Hyphenated IDs are kept whole and also split into parts."""
        tokens = []
        for token in re.findall(r"[\w\-]+", text.lower()):
            tokens.append(token)
            if "-" in token:
                tokens.extend(part for part in token.split("-") if part)
        return tokens

    def add_document(self, doc):
        """Index one document. `doc` must be a dict with a "content" key."""
        tokens = self._tokenize(doc["content"])
        term_freq = Counter(tokens)

        self.documents.append(doc)
        self._term_freqs.append(term_freq)
        self._doc_lens.append(len(tokens))
        for term in term_freq:
            self._doc_freq[term] += 1

    def add_documents(self, docs):
        for doc in docs:
            self.add_document(doc)

    def _idf(self, term):
        n = len(self.documents)
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query, k=3):
        """Return the top k documents as (doc, score) pairs. Higher score means a better match."""
        if not self.documents:
            return []

        terms = self._tokenize(query)
        avg_len = (sum(self._doc_lens) / len(self._doc_lens)) or 1

        scored = []
        for i, term_freq in enumerate(self._term_freqs):
            length_norm = 1 - self.b + self.b * self._doc_lens[i] / avg_len
            score = 0.0
            for term in terms:
                freq = term_freq.get(term, 0)
                if freq:
                    score += self._idf(term) * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
            if score > 0:
                scored.append((self.documents[i], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
