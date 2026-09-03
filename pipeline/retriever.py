"""
pipeline/retriever.py

RAG retriever: given a query, returns ranked Darwin passages with full citation.

The retriever is the heart of the grounding system. Every model response
should cite which passages it drew from. The gap between retrieved passages
and generated text is where confabulation lives.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL,
    TOP_K_RETRIEVAL, RETRIEVAL_THRESHOLD
)


@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    score: float          # similarity score (0–1, higher = more relevant)
    title: str
    date_str: str
    date_year: Optional[int]
    doc_type: str
    register: str
    life_period: Optional[str]
    recipient: Optional[str]
    source: str
    url: str
    chunk_index: int

    def citation_str(self) -> str:
        """Human-readable citation for this passage."""
        parts = [f'"{self.title}"']
        if self.date_str and self.date_str != "unknown":
            parts.append(self.date_str)
        if self.recipient:
            parts.append(f"to {self.recipient}")
        parts.append(f"[{self.source}]")
        return ", ".join(parts)

    def format_for_prompt(self) -> str:
        """Format passage for inclusion in a system prompt."""
        citation = self.citation_str()
        score_str = f"{self.score:.2f}"
        return (
            f"[Source: {citation} | Relevance: {score_str}]\n"
            f"{self.text}"
        )


class DarwinRetriever:
    def __init__(
        self,
        store_dir: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION,
        top_k: int = TOP_K_RETRIEVAL,
        threshold: float = RETRIEVAL_THRESHOLD,
    ):
        self.store_dir = store_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.threshold = threshold
        self._collection = None
        self._embedding_fn = None

    def _load(self):
        """Lazy-load Chroma collection."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install: pip install chromadb sentence-transformers")

        class EmbFn:
            def __init__(self, model_name):
                self.model = SentenceTransformer(model_name)
            def __call__(self, input):
                return self.model.encode(input).tolist()

        self._embedding_fn = EmbFn(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(
            path=str(self.store_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_period: Optional[str] = None,
        filter_register: Optional[str] = None,
        filter_doc_type: Optional[str] = None,
        before_year: Optional[int] = None,
    ) -> list[RetrievedPassage]:
        """
        Retrieve relevant Darwin passages for a query.

        Args:
            query: The question or context to search for
            top_k: Override default number of results
            filter_period: Only return passages from a specific life period
            filter_register: 'public', 'private', 'personal', 'intimate'
            filter_doc_type: 'letter', 'notebook', 'published', 'diary', etc.
            before_year: Only return passages from documents written before this year
                         (critical for temporal lock failure testing)
        """
        self._load()
        k = top_k or self.top_k

        # Build Chroma where clause
        where = {}
        conditions = []

        if filter_period:
            conditions.append({"life_period": {"$eq": filter_period}})
        if filter_register:
            conditions.append({"register": {"$eq": filter_register}})
        if filter_doc_type:
            conditions.append({"doc_type": {"$eq": filter_doc_type}})
        if before_year:
            conditions.append({"date_year": {"$lt": before_year}})
            conditions.append({"date_year": {"$gt": 0}})  # exclude undated

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        query_params = {
            "query_texts": [query],
            "n_results": min(k * 2, self._collection.count()),  # over-retrieve then filter
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

        passages = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma returns L2 distance; convert to similarity score
            score = max(0.0, 1.0 - dist)
            if score < self.threshold:
                continue

            passages.append(RetrievedPassage(
                chunk_id=meta.get("chunk_id", ""),
                text=doc,
                score=score,
                title=meta.get("title", "Unknown"),
                date_str=meta.get("date_str", "unknown"),
                date_year=meta.get("date_year") or None,
                doc_type=meta.get("doc_type", ""),
                register=meta.get("register", ""),
                life_period=meta.get("life_period") or None,
                recipient=meta.get("recipient") or None,
                source=meta.get("source", ""),
                url=meta.get("url", ""),
                chunk_index=meta.get("chunk_index", 0),
            ))

        # Sort by score, return top_k
        passages.sort(key=lambda p: p.score, reverse=True)
        return passages[:k]

    def format_passages_for_prompt(self, passages: list[RetrievedPassage]) -> str:
        """Format retrieved passages for injection into a system prompt."""
        if not passages:
            return "[No relevant passages found in Darwin's corpus for this query.]"

        parts = []
        for i, p in enumerate(passages, 1):
            parts.append(f"--- Passage {i} ---\n{p.format_for_prompt()}")
        return "\n\n".join(parts)

    def count(self) -> int:
        self._load()
        return self._collection.count()
