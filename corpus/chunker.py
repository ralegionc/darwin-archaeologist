"""
corpus/chunker.py

Splits cleaned Darwin documents into RAG-ready chunks.

Key design decisions:
  - Chunks preserve paragraph boundaries (don't cut sentences mid-thought)
  - Each chunk carries full metadata: date, period, register, source doc
  - Chunks are numbered within their source document for citation
  - Short chunks from dense texts (letters) are kept as-is

Usage:
    python corpus/chunker.py --input data/cleaned/ --output data/chunks/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, preserving non-empty ones."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def chunk_paragraphs(paragraphs: list[str], chunk_size: int, overlap: int) -> Iterator[str]:
    """
    Group paragraphs into chunks of approximately chunk_size tokens.
    Overlaps by including the last paragraph of the previous chunk.
    """
    current_chunk = []
    current_tokens = 0
    last_paragraph = None

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If single paragraph exceeds chunk size, split by sentences
        if para_tokens > chunk_size:
            if current_chunk:
                yield "\n\n".join(current_chunk)
                last_paragraph = current_chunk[-1]
                current_chunk = []
                current_tokens = 0

            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_tokens = 0
            for sent in sentences:
                t = estimate_tokens(sent)
                if sent_tokens + t > chunk_size and sent_chunk:
                    yield " ".join(sent_chunk)
                    sent_chunk = [sent_chunk[-1], sent]  # overlap
                    sent_tokens = estimate_tokens(sent_chunk[-2]) + t
                else:
                    sent_chunk.append(sent)
                    sent_tokens += t
            if sent_chunk:
                yield " ".join(sent_chunk)
            continue

        # Add overlap from previous chunk
        if current_tokens == 0 and last_paragraph:
            current_chunk = [last_paragraph]
            current_tokens = estimate_tokens(last_paragraph)

        if current_tokens + para_tokens > chunk_size and current_chunk:
            yield "\n\n".join(current_chunk)
            last_paragraph = current_chunk[-1]
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        yield "\n\n".join(current_chunk)


def chunk_document(doc: dict) -> list[dict]:
    """Convert a cleaned document into a list of chunk dicts."""
    text = doc.get("text", "")
    if not text:
        return []

    # Letters and short documents: treat as single chunk if small enough
    if estimate_tokens(text) <= CHUNK_SIZE:
        chunk_text = text
        if estimate_tokens(chunk_text) < MIN_CHUNK_SIZE:
            return []
        return [{
            "chunk_id": f"{doc['id']}_chunk_0",
            "doc_id": doc["id"],
            "chunk_index": 0,
            "total_chunks": 1,
            "text": chunk_text,
            # Metadata carried forward for RAG citation
            "title": doc.get("title"),
            "date_year": doc.get("date_year"),
            "date_str": doc.get("date_str"),
            "doc_type": doc.get("doc_type"),
            "register": doc.get("register"),
            "life_period": doc.get("life_period"),
            "recipient": doc.get("recipient"),
            "source": doc.get("source"),
            "url": doc.get("url"),
            "estimated_tokens": estimate_tokens(chunk_text),
        }]

    paragraphs = split_into_paragraphs(text)
    raw_chunks = list(chunk_paragraphs(paragraphs, CHUNK_SIZE, CHUNK_OVERLAP))

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if estimate_tokens(chunk_text) < MIN_CHUNK_SIZE:
            continue
        chunks.append({
            "chunk_id": f"{doc['id']}_chunk_{i}",
            "doc_id": doc["id"],
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "text": chunk_text,
            "title": doc.get("title"),
            "date_year": doc.get("date_year"),
            "date_str": doc.get("date_str"),
            "doc_type": doc.get("doc_type"),
            "register": doc.get("register"),
            "life_period": doc.get("life_period"),
            "recipient": doc.get("recipient"),
            "source": doc.get("source"),
            "url": doc.get("url"),
            "estimated_tokens": estimate_tokens(chunk_text),
        })
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CLEANED_DIR)
    parser.add_argument("--output", type=Path, default=CHUNKS_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    json_files = [f for f in args.input.rglob("*.json") if f.name != "manifest.json"]
    print(f"── Chunking corpus ───────────────────────────────────")
    print(f"  Input documents: {len(json_files)}")

    all_chunks = []
    for filepath in json_files:
        doc = json.loads(filepath.read_text(encoding="utf-8"))
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

        # Save chunks as a single file per document
        out_path = args.output / f"{doc['id']}_chunks.json"
        out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write flat index for embedding
    index_path = args.output / "all_chunks.jsonl"
    with open(index_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    total_tokens = sum(c["estimated_tokens"] for c in all_chunks)
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Total estimated tokens: {total_tokens:,}")
    print(f"  Avg chunk size: {total_tokens // max(len(all_chunks), 1)} tokens")
    print(f"\n✓ Chunking complete → {args.output}")


if __name__ == "__main__":
    main()
