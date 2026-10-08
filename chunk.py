"""
chunk.py - Step 2: extracted JSON -> section-aware chunks (data/chunks/chunks.jsonl)

Usage (from project root, venv active):
    python src/chunk.py
"""
import csv
import json
import re
from pathlib import Path

IN_DIR = Path("data/extracted")
OUT_DIR = Path("data/chunks")

MAX_WORDS = 300      # split sections longer than this
OVERLAP_WORDS = 50   # words shared between consecutive pieces of one section

# "2.4 Determining if High Loss is from Splice", "3. Fiber Signature Requirement", "5.2.2 PCI Process"
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\.?\s+([A-Z][^\n]{2,90})$")
# Table-of-contents lines: dot leaders or a trailing page number
TOC_LINE_RE = re.compile(r"(\.{4,}|\s\d{1,3}$)")
# Sections we keep but mark, so retrieval can down-weight or skip them
BOILERPLATE_RE = re.compile(
    r"(table of contents|contents|revision (log|history)|contact(s| list)?|"
    r"proprietary|competition polic|abstract|references?)$",
    re.IGNORECASE,
)


def is_heading(line: str) -> re.Match | None:
    line = line.strip()
    if len(line) > 100 or TOC_LINE_RE.search(line) or line.endswith((".", ",", ";", ":")):
        return None
    return HEADING_RE.match(line)


def guess_title(doc: dict) -> str:
    """First substantial non-numeric line on page 1 that is not a heading."""
    if not doc["pages"]:
        return ""
    for line in doc["pages"][0]["text"].splitlines():
        line = line.strip()
        if (len(line) > 8 and not re.fullmatch(r"[\d\W]+", line) and not is_heading(line)
                and not re.fullmatch(r"[A-Z]{2,5}-[\d-]+(\.pdf)?", line)):  # skip doc numbers like ATT-002-190-005
            return line
    return ""


def split_sections(doc: dict) -> list[dict]:
    """Walk every line; start a new section at each numbered heading."""
    sections = []
    current = {"section_number": "", "section_title": "Front Matter", "page_start": 1, "lines": []}
    for page in doc["pages"]:
        for line in page["text"].splitlines():
            match = is_heading(line)
            if match:
                if current["lines"]:
                    sections.append(current)
                current = {
                    "section_number": match.group(1),
                    "section_title": match.group(2).strip(),
                    "page_start": page["page"],
                    "lines": [],
                }
            else:
                current["lines"].append(line.strip())
            current["page_end"] = page["page"]
    if current["lines"]:
        sections.append(current)
    return sections


def window(words: list[str]) -> list[list[str]]:
    if len(words) <= MAX_WORDS:
        return [words]
    step = MAX_WORDS - OVERLAP_WORDS
    return [words[i:i + MAX_WORDS] for i in range(0, len(words) - OVERLAP_WORDS, step)]


def build_chunks(doc: dict) -> list[dict]:
    title = guess_title(doc)
    chunks = []
    for sec in split_sections(doc):
        words = " ".join(sec["lines"]).split()
        if not words:
            continue
        heading = f"{sec['section_number']} {sec['section_title']}".strip()
        for part, piece in enumerate(window(words)):
            body = " ".join(piece)
            chunks.append({
                "chunk_id": f"{doc['document_id']}::{sec['section_number'] or '0'}::{part}",
                "document_id": doc["document_id"],
                "document_title": title,
                "section_number": sec["section_number"],
                "section_title": sec["section_title"],
                "page_start": sec["page_start"],
                "page_end": sec.get("page_end", sec["page_start"]),
                "is_boilerplate": bool(BOILERPLATE_RE.search(sec["section_title"].strip()))
                                  or ("table of contents" in body.lower() and not sec["section_number"]),
                "word_count": len(piece),
                # Context header makes each chunk self-describing for BM25 and embeddings
                "text": f"{title} | {heading}\n{body}",
            })
    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("*.json"))
    if not files:
        print(f"No extracted JSON in {IN_DIR.resolve()} - run ingest.py first")
        return

    all_chunks, summary = [], []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        chunks = build_chunks(doc)
        all_chunks.extend(chunks)
        summary.append({
            "document_id": doc["document_id"],
            "title": chunks[0]["document_title"] if chunks else "",
            "pages": doc["page_count"],
            "sections": len({c["section_number"] for c in chunks}),
            "chunks": len(chunks),
            "boilerplate_chunks": sum(c["is_boilerplate"] for c in chunks),
            "low_text_pages": sum(p["low_text"] for p in doc["pages"]),
        })

    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as out:
        for c in all_chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "chunk_summary.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"{len(all_chunks)} chunks from {len(files)} documents")
    print(f"Wrote {OUT_DIR / 'chunks.jsonl'} and {OUT_DIR / 'chunk_summary.csv'}")
    zero = [s["document_id"] for s in summary if s["chunks"] == 0]
    if zero:
        print(f"WARNING: no chunks for {len(zero)} documents (check for scanned/image-only PDFs): {zero[:10]}")


if __name__ == "__main__":
    main()
