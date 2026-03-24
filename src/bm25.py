import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / max(self.n, 1)
        self.df = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens, doc_idx):
        doc = self.tokenized[doc_idx]
        tf_map = Counter(doc)
        dl = len(doc)
        s = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            tf = tf_map[term]
            idf = self.idf(term)
            s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, top_k=10):
        query_tokens = tokenize(query)
        scores = [(i, self.score(query_tokens, i)) for i in range(self.n)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def build_bm25(chunks):
    corpus = [c["text"] for c in chunks]
    return BM25(corpus)
