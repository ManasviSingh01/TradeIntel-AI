import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None

    def add_document(self, name: str, text: str, chunk_size: int = 850, overlap: int = 120):
        clean = " ".join(text.split())
        if not clean:
            return 0
        start, added = 0, 0
        while start < len(clean):
            chunk = clean[start:start + chunk_size]
            self.chunks.append({"document": name, "text": chunk})
            added += 1
            if start + chunk_size >= len(clean):
                break
            start += chunk_size - overlap
        self._reindex()
        return added

    def _reindex(self):
        corpus = [c["text"] for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def retrieve(self, question: str, top_k: int = 3):
        if self.matrix is None or not self.chunks:
            return []
        q = self.vectorizer.transform([question])
        scores = cosine_similarity(q, self.matrix)[0]
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.chunks[i], "score": float(scores[i])}
            for i in idx if scores[i] > 0
        ]

    def answer(self, question: str, contexts: List[Dict]):
        if not contexts:
            return "I could not find relevant information in the indexed trade documents.", "retrieval-only"
        context_text = "\n\n".join(
            f"SOURCE: {c['document']}\n{c['text']}" for c in contexts
        )
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "You are a commodity-trade intelligence assistant. Answer only from the supplied context. If the context is insufficient, say so. Keep answers concise and factual."},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
                    ],
                )
                return response.choices[0].message.content, "llm-rag"
            except Exception:
                pass
        best = contexts[0]
        return (
            "LLM generation is not configured, so this answer shows the most relevant retrieved evidence: "
            + best["text"][:650],
            "retrieval-only",
        )

engine = RAGEngine()
