"""
corpus/scraper.py

Scrapes Darwin's writings from three sources:
  1. Project Gutenberg — clean plain-text published works
  2. Darwin Online — notebooks, manuscripts, additional works
  3. Darwin Correspondence Project — letters

Usage:
    python corpus/scraper.py --source gutenberg --output data/raw/
    python corpus/scraper.py --source all --output data/raw/
"""

import argparse
import json
import time
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR


GUTENBERG_WORKS = {
    "origin_of_species": {
        "gutenberg_id": 1228,
        "title": "On the Origin of Species",
        "year": 1859,
        "doc_type": "published",
        "register": "public",
    },
    "voyage_of_beagle": {
        "gutenberg_id": 944,
        "title": "The Voyage of the Beagle",
        "year": 1839,
        "doc_type": "published",
        "register": "public",
    },
    "descent_of_man": {
        "gutenberg_id": 2300,
        "title": "The Descent of Man",
        "year": 1871,
        "doc_type": "published",
        "register": "public",
    },
    "expression_of_emotions": {
        "gutenberg_id": 1227,
        "title": "The Expression of Emotions in Man and Animals",
        "year": 1872,
        "doc_type": "published",
        "register": "public",
    },
    "autobiography": {
        "gutenberg_id": 2010,
        "title": "The Autobiography of Charles Darwin",
        "year": 1876,
        "doc_type": "autobiography",
        "register": "intimate",
    },
    "formation_of_vegetable_mould": {
        "gutenberg_id": 5250,
        "title": "The Formation of Vegetable Mould through the Action of Worms",
        "year": 1881,
        "doc_type": "published",
        "register": "public",
    },
}

DARWIN_ONLINE_DOCS = {
    "beagle_diary": {
        "url": "http://darwin-online.org.uk/content/frameset?itemID=EHStevenson1979&viewtype=text&pageseq=1",
        "title": "Beagle Diary",
        "year": 1831,
        "doc_type": "diary",
        "register": "private",
    },
    "sketch_1842": {
        "url": "http://darwin-online.org.uk/content/frameset?itemID=F1556&viewtype=text&pageseq=1",
        "title": "Sketch of 1842 (early species theory)",
        "year": 1842,
        "doc_type": "manuscript",
        "register": "private",
    },
    "essay_1844": {
        "url": "http://darwin-online.org.uk/content/frameset?itemID=F1557&viewtype=text&pageseq=1",
        "title": "Essay of 1844",
        "year": 1844,
        "doc_type": "manuscript",
        "register": "private",
    },
}

SAMPLE_LETTER_IDS = [
    "DCP-LETT-171",    # To Henslow, 1832 — Beagle period
    "DCP-LETT-236",    # To Susan Darwin, 1834
    "DCP-LETT-729",    # To Hooker, 1844 — famous "murder" letter
    "DCP-LETT-814",    # To Hooker, 1845
    "DCP-LETT-1174",   # To Hooker, 1848 — barnacle years
    "DCP-LETT-2136",   # To Asa Gray, 1857
    "DCP-LETT-2285",   # To Alfred Wallace, 1858
    "DCP-LETT-2589",   # To Huxley, 1859
    "DCP-LETT-2632",   # To Asa Gray, 1860 — on religion
    "DCP-LETT-8006",   # To Hooker, 1871
    "DCP-LETT-13229",  # To Fox, 1881 — late career
]


def fetch_gutenberg_text(gutenberg_id: int) -> Optional[str]:
    urls = [
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
    ]
    headers = {"User-Agent": "DarwinArchaeologist/1.0 (academic research)"}
    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                print(f"    ✓ {url}")
                return resp.text
        except requests.RequestException:
            continue
    return None


def strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.split("\n")
    start_idx, end_idx = 0, len(lines)
    for i, line in enumerate(lines):
        if re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG", line, re.I):
            start_idx = i + 1
            break
    for i, line in enumerate(lines):
        if re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", line, re.I):
            end_idx = i
            break
    return "\n".join(lines[start_idx:end_idx]).strip()


def make_doc(id_, title, text, year, doc_type, register, source, url, recipient=None):
    return {
        "id": id_,
        "title": title,
        "text": text,
        "date_year": year,
        "date_str": str(year) if year else "unknown",
        "doc_type": doc_type,
        "register": register,
        "recipient": recipient,
        "source": source,
        "url": url,
        "scraped_at": datetime.utcnow().isoformat(),
    }


def scrape_gutenberg(output_dir: Path) -> list:
    print("\n── Gutenberg ─────────────────────────────────────────")
    results = []
    out_dir = output_dir / "gutenberg"
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, meta in GUTENBERG_WORKS.items():
        print(f"  {meta['title']} ({meta['year']})")
        raw = fetch_gutenberg_text(meta["gutenberg_id"])
        if not raw:
            continue
        text = strip_gutenberg_boilerplate(raw)
        doc = make_doc(
            id_=f"gutenberg_{key}",
            title=meta["title"],
            text=text,
            year=meta["year"],
            doc_type=meta["doc_type"],
            register=meta["register"],
            source="gutenberg",
            url=f"https://www.gutenberg.org/ebooks/{meta['gutenberg_id']}",
        )
        path = out_dir / f"{key}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(doc)
        print(f"    → {len(text):,} chars saved")
        time.sleep(1)
    return results


def scrape_darwin_online(output_dir: Path) -> list:
    print("\n── Darwin Online ─────────────────────────────────────")
    results = []
    out_dir = output_dir / "darwin_online"
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "DarwinArchaeologist/1.0 (academic research)"}

    for key, meta in DARWIN_ONLINE_DOCS.items():
        print(f"  {meta['title']} ({meta['year']})")
        try:
            resp = requests.get(meta["url"], headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"    ✗ HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            text_el = (soup.find("div", class_="transcription")
                       or soup.find("div", class_="text")
                       or soup.find("div", id="content"))
            if not text_el:
                print(f"    ✗ No text element found")
                continue
            text = text_el.get_text(separator="\n", strip=True)
            doc = make_doc(
                id_=f"darwin_online_{key}",
                title=meta["title"],
                text=text,
                year=meta["year"],
                doc_type=meta["doc_type"],
                register=meta["register"],
                source="darwin_online",
                url=meta["url"],
            )
            path = out_dir / f"{key}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(doc)
            print(f"    → {len(text):,} chars saved")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        time.sleep(3)
    return results


def scrape_correspondence(output_dir: Path, max_letters: int = 50) -> list:
    print("\n── Correspondence Project ────────────────────────────")
    results = []
    out_dir = output_dir / "correspondence"
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "DarwinArchaeologist/1.0 (academic research)"}

    for letter_id in SAMPLE_LETTER_IDS[:max_letters]:
        url = f"https://www.darwinproject.ac.uk/letter/{letter_id}"
        print(f"  {letter_id}")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"    ✗ HTTP {resp.status_code}")
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            date_el = soup.find("span", class_="date") or soup.find("div", class_="date")
            recipient_el = soup.find("span", class_="correspondent") or soup.find("div", class_="addressee")
            text_el = soup.find("div", class_="letter-body") or soup.find("div", class_="tei-body")

            if not text_el:
                print(f"    ✗ No text found")
                continue

            date_str = date_el.get_text(strip=True) if date_el else "unknown"
            recipient = recipient_el.get_text(strip=True) if recipient_el else "unknown"
            text = text_el.get_text(separator="\n", strip=True)
            year_match = re.search(r'\b(1[89]\d{2})\b', date_str)
            year = int(year_match.group(1)) if year_match else None

            doc = make_doc(
                id_=letter_id.replace("DCP-LETT-", "letter_"),
                title=f"Letter to {recipient}, {date_str}",
                text=text,
                year=year,
                doc_type="letter",
                register="personal",
                source="correspondence_project",
                url=url,
                recipient=recipient,
            )
            path = out_dir / f"{letter_id}.json"
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(doc)
            print(f"    → {len(text):,} chars, {date_str}, to {recipient}")
        except Exception as e:
            print(f"    ✗ {e}")
        time.sleep(2)
    return results


def write_manifest(docs: list, output_dir: Path):
    manifest = {
        "scraped_at": datetime.utcnow().isoformat(),
        "total_documents": len(docs),
        "total_chars": sum(len(d["text"]) for d in docs),
        "by_source": {},
        "by_type": {},
        "documents": [{k: v for k, v in d.items() if k != "text"} for d in docs],
    }
    for doc in docs:
        manifest["by_source"][doc["source"]] = manifest["by_source"].get(doc["source"], 0) + 1
        manifest["by_type"][doc["doc_type"]] = manifest["by_type"].get(doc["doc_type"], 0) + 1

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n── Summary ───────────────────────────────────────────")
    print(f"  Documents: {manifest['total_documents']}")
    print(f"  Characters: {manifest['total_chars']:,}")
    print(f"  By source: {manifest['by_source']}")
    print(f"  By type:   {manifest['by_type']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gutenberg", "darwin_online", "correspondence", "all"], default="gutenberg")
    parser.add_argument("--output", type=Path, default=RAW_DIR)
    parser.add_argument("--max-letters", type=int, default=50)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_docs = []

    if args.source in ("gutenberg", "all"):
        all_docs.extend(scrape_gutenberg(args.output))
    if args.source in ("darwin_online", "all"):
        all_docs.extend(scrape_darwin_online(args.output))
    if args.source in ("correspondence", "all"):
        all_docs.extend(scrape_correspondence(args.output, args.max_letters))

    write_manifest(all_docs, args.output)
    print(f"\n✓ Done. {len(all_docs)} documents saved to {args.output}")


if __name__ == "__main__":
    main()
