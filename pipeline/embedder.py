"""
pipeline/embedder.py

Embeds Darwin corpus chunks into a Chroma vector store.

Uses sentence-transformers locally (no API cost).
Can swap to OpenAI embeddings for better quality by changing EMBEDDING_MODEL in config.

Usage:
    python pipeline/embedder.py --input data/chunks/ --store data/chroma/
    python pipeline/embedder.py --rebuild  # clears and rebuilds from scratch
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHROMA_COLLECTION

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


def check_deps():
    if not HAS_DEPS:
        print("Missing dependencies. Install with:")
        print("  pip install chromadb sentence-transformers")
        sys.exit(1)


def get_chroma_client(store_dir: Path) -> "chromadb.Client":
    return chromadb.PersistentClient(
        path=str(store_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(client, name: str, embedding_fn):
    try:
        return client.get_collection(name=name, embedding_function=embedding_fn)
    except Exception:
        return client.create_collection(name=name, embedding_function=embedding_fn)


class LocalEmbeddingFunction:
    """Wraps sentence-transformers for Chroma."""

    def __init__(self, model_name: str):
        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


def load_chunks(chunks_dir: Path) -> list[dict]:
    """Load all chunks from the JSONL index."""
    index_path = chunks_dir / "all_chunks.jsonl"
    if not index_path.exists():
        # Fall back to loading individual chunk files
        chunks = []
        for f in chunks_dir.glob("*_chunks.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            chunks.extend(data)
        return chunks

    chunks = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_corpus(
    chunks_dir: Path,
    store_dir: Path,
    collection_name: str,
    batch_size: int = 64,
    rebuild: bool = False,
):
    check_deps()
    store_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_dir)
    if not chunks:
        print("  ✗ No chunks found. Run chunker.py first.")
        return

    print(f"  Chunks to embed: {len(chunks)}")

    embedding_fn = LocalEmbeddingFunction(EMBEDDING_MODEL)
    client = get_chroma_client(store_dir)

    if rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"  Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    collection = get_or_create_collection(client, collection_name, embedding_fn)

    # Get already-embedded IDs to allow resuming
    existing_ids = set()
    if not rebuild:
        try:
            existing = collection.get(include=[])
            existing_ids = set(existing["ids"])
            print(f"  Already embedded: {len(existing_ids)}")
        except Exception:
            pass

    # Filter to new chunks only
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"  New chunks to embed: {len(new_chunks)}")

    if not new_chunks:
        print("  ✓ Nothing to embed — collection is up to date")
        return

    # Embed in batches
    total_batches = (len(new_chunks) + batch_size - 1) // batch_size
    embedded = 0

    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "doc_id": c.get("doc_id", ""),
                "title": c.get("title", ""),
                "date_year": c.get("date_year") or 0,
                "date_str": c.get("date_str", ""),
                "doc_type": c.get("doc_type", ""),
                "register": c.get("register", ""),
                "life_period": c.get("life_period", ""),
                "recipient": c.get("recipient", "") or "",
                "source": c.get("source", ""),
                "url": c.get("url", ""),
                "chunk_index": c.get("chunk_index", 0),
                "total_chunks": c.get("total_chunks", 1),
            }
            for c in batch
        ]

        try:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            embedded += len(batch)
            print(f"✓")
        except Exception as e:
            print(f"✗ {e}")

    print(f"\n  Total embedded: {embedded}")
    print(f"  Collection '{collection_name}': {collection.count()} total vectors")
    print(f"  Store: {store_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--store", type=Path, default=CHROMA_DIR)
    parser.add_argument("--collection", default=CHROMA_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild from scratch")
    args = parser.parse_args()

    print("── Embedding corpus ──────────────────────────────────")
    embed_corpus(args.input, args.store, args.collection, args.batch_size, args.rebuild)
    print("\n✓ Embedding complete")


if __name__ == "__main__":
    main()
