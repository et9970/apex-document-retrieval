"""
ingest.py - Step 1: PDF -> per-document JSON (page text + quality flags)

Usage (from project root, venv active):
    python src/ingest.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw_pdfs")
OUT_DIR = Path("data/extracted")

# Lines repeated on at least this share of a document's pages are treated as header/footer.
REPEAT_THRESHOLD = 0.5
# Pages with less text than this are flagged as likely diagram/image pages.
LOW_TEXT_CHARS = 200


def normalize_line(line: str) -> str:
    """Collapse whitespace and mask digits so 'Page 3 of 9' and 'Page 4 of 9' match."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line).strip())


def find_repeated_lines(pages_lines: list[list[str]]) -> set[str]:
    """Return normalized lines that appear on most pages (headers, footers, proprietary notices)."""
    if len(pages_lines) < 2:
        return set()
    counts = Counter()
    for lines in pages_lines:
        counts.update({normalize_line(l) for l in lines if l.strip()})
    min_pages = max(2, int(len(pages_lines) * REPEAT_THRESHOLD))
    return {line for line, n in counts.items() if n >= min_pages}


def extract_document(pdf_path: Path) -> dict:
    doc = pymupdf.open(pdf_path)
    raw_pages = [page.get_text("text", sort=True) for page in doc]
    image_counts = [len(page.get_images(full=True)) for page in doc]
    doc.close()

    pages_lines = [text.splitlines() for text in raw_pages]
    repeated = find_repeated_lines(pages_lines)

    pages = []
    for i, lines in enumerate(pages_lines):
        kept = [l.rstrip() for l in lines if l.strip() and normalize_line(l) not in repeated]
        clean = "\n".join(kept)
        pages.append({
            "page": i + 1,
            "text": clean,
            "char_count": len(clean),
            "image_count": image_counts[i],
            "low_text": len(clean) < LOW_TEXT_CHARS,
        })

    return {
        "document_id": pdf_path.name,
        "page_count": len(pages),
        "removed_repeated_lines": sorted(repeated),
        "pages": pages,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR.resolve()}")
        return

    failures = []
    for pdf_path in pdfs:
        try:
            result = extract_document(pdf_path)
        except Exception as exc:  # keep going; one bad PDF shouldn't stop the batch
            failures.append((pdf_path.name, str(exc)))
            continue
        out_file = OUT_DIR / f"{pdf_path.stem}.json"
        out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        low = sum(p["low_text"] for p in result["pages"])
        print(f"{pdf_path.name}: {result['page_count']} pages, {low} low-text, "
              f"{len(result['removed_repeated_lines'])} repeated lines removed")

    print(f"\nDone: {len(pdfs) - len(failures)} of {len(pdfs)} PDFs extracted to {OUT_DIR}")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")


if __name__ == "__main__":
    main()
