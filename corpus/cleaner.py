"""
corpus/cleaner.py

Cleans raw scraped documents:
  - Removes OCR artifacts and encoding noise
  - Normalizes whitespace and Victorian typography
  - Assigns life period based on date
  - Deduplicates near-identical chunks
  - Outputs cleaned JSON with enriched metadata

Usage:
    python corpus/cleaner.py --input data/raw/ --output data/cleaned/
"""

import argparse
import json
import re
import sys
import hashlib
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, CLEANED_DIR, LIFE_PERIODS


def assign_life_period(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    for period in LIFE_PERIODS:
        if period["start"] <= year <= period["end"]:
            return period["name"]
    return None


def clean_text(text: str, doc_type: str = "published") -> str:
    """
    Clean and normalize Darwin's text.
    Preserves Victorian spelling and punctuation — these are data, not errors.
    Only removes genuine artifacts.
    """
    # Fix common OCR artifacts
    text = re.sub(r'ﬁ', 'fi', text)
    text = re.sub(r'ﬂ', 'fl', text)
    text = re.sub(r'ﬀ', 'ff', text)
    text = re.sub(r'ﬃ', 'ffi', text)
    text = re.sub(r'\x00', '', text)            # null bytes
    text = re.sub(r'[\x80-\x9f]', '', text)    # Windows-1252 control chars

    # Normalize dashes — em dashes to double hyphen (preserves semantics)
    text = re.sub(r'[–—]', '--', text)

    # Normalize quotes to ASCII (Victorian curly quotes → straight)
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201c\u201d]', '"', text)

    # Remove page headers/footers common in digitized books
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)     # standalone page numbers
    text = re.sub(r'\[Pg \d+\]', '', text)            # Gutenberg page markers
    text = re.sub(r'\[pg \d+\]', '', text)

    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)               # collapse horizontal whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)          # max 3 consecutive newlines
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.M) # strip line-leading/trailing spaces

    # Remove Gutenberg license remnants that slip through
    text = re.sub(r'This eBook is for the use of anyone.*?no cost', '', text, flags=re.DOTALL | re.I)

    return text.strip()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for Victorian English."""
    return len(text) // 4


def text_fingerprint(text: str) -> str:
    """Create a fingerprint for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized[:500].encode()).hexdigest()


def clean_document(doc: dict) -> Optional[dict]:
    """Clean a single document. Returns None if document should be discarded."""
    text = doc.get("text", "")
    if not text or len(text) < 200:
        return None

    cleaned_text = clean_text(text, doc.get("doc_type", "published"))

    if len(cleaned_text) < 200:
        return None

    year = doc.get("date_year")
    life_period = assign_life_period(year)

    cleaned = {
        **doc,
        "text": cleaned_text,
        "text_length": len(cleaned_text),
        "estimated_tokens": estimate_tokens(cleaned_text),
        "life_period": life_period,
        "fingerprint": text_fingerprint(cleaned_text),
        "cleaned": True,
    }
    return cleaned


def process_directory(input_dir: Path, output_dir: Path) -> list:
    """Process all JSON files in input directory recursively."""
    json_files = list(input_dir.rglob("*.json"))
    json_files = [f for f in json_files if f.name != "manifest.json"]

    print(f"  Found {len(json_files)} documents")

    results = []
    seen_fingerprints = set()
    stats = {"cleaned": 0, "skipped_short": 0, "skipped_duplicate": 0, "errors": 0}

    for filepath in json_files:
        try:
            doc = json.loads(filepath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ✗ Could not read {filepath.name}: {e}")
            stats["errors"] += 1
            continue

        cleaned = clean_document(doc)
        if cleaned is None:
            stats["skipped_short"] += 1
            continue

        # Deduplication
        fp = cleaned["fingerprint"]
        if fp in seen_fingerprints:
            stats["skipped_duplicate"] += 1
            continue
        seen_fingerprints.add(fp)

        # Preserve directory structure in output
        rel_path = filepath.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

        results.append(cleaned)
        stats["cleaned"] += 1

    print(f"  Cleaned: {stats['cleaned']}")
    print(f"  Skipped (too short): {stats['skipped_short']}")
    print(f"  Skipped (duplicate): {stats['skipped_duplicate']}")
    print(f"  Errors: {stats['errors']}")
    return results


def write_cleaned_manifest(docs: list, output_dir: Path):
    by_period = {}
    by_register = {}
    for doc in docs:
        p = doc.get("life_period") or "unknown"
        by_period[p] = by_period.get(p, 0) + 1
        r = doc.get("register") or "unknown"
        by_register[r] = by_register.get(r, 0) + 1

    manifest = {
        "total_documents": len(docs),
        "total_tokens_estimated": sum(d["estimated_tokens"] for d in docs),
        "by_life_period": by_period,
        "by_register": by_register,
        "documents": [{k: v for k, v in d.items() if k != "text"} for d in docs],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  Total estimated tokens: {manifest['total_tokens_estimated']:,}")
    print(f"  By period: {by_period}")
    print(f"  By register: {by_register}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=CLEANED_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"── Cleaning corpus ───────────────────────────────────")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")

    docs = process_directory(args.input, args.output)
    write_cleaned_manifest(docs, args.output)
    print(f"\n✓ Cleaning complete. {len(docs)} documents → {args.output}")


if __name__ == "__main__":
    main()
