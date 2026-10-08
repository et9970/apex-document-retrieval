"""
retrieve.py - Step 3: BM25 retrieval over chunks -> submission CSV  

Usage (from project root, venv active):
    python src/retrieve.py                       # uses public_submission_template.csv
    python src/retrieve.py path\to\template.csv  # e.g. the final challenge file

Outputs:
    outputs/submission.csv    -> upload this (same columns + order as template)
    outputs/diagnostics.csv   -> top-3 docs, scores, and evidence for review
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

CHUNKS = Path("data/chunks/chunks.jsonl")
OUT_DIR = Path("outputs")
DEFAULT_TEMPLATE = Path("public_submission_template.csv")

KEEP_PDF_EXTENSION = True   # set False if the scorer expects "ATT-002-190-005" without ".pdf"
TOP_CHUNKS_PER_DOC = 3      # doc score = best chunk + smaller credit for the next two
MAX_ANSWER_SENTENCES = 2

STOPWORDS = set("""a an and are as at be by can do does for from has have how if in into is it its
of on or should that the their there these this to was were what when where which who why will with
must may per under used use using according document procedure""".split())


def tokenize(text: str) -> list[str]:
    # keep part numbers / IDs like 3FE53441, SX-16F, ARAMIS-DT as single tokens AND as pieces
    raw = re.findall(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", text.lower())
    tokens = []
    for t in raw:
        if t in STOPWORDS or len(t) < 2:
            continue
        tokens.append(t)
        if re.search(r"[-./]", t):
            tokens.extend(p for p in re.split(r"[-./]", t) if len(p) > 1 and p not in STOPWORDS)
    return tokens


def load_chunks():
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks = [c for c in chunks if not c.get("is_boilerplate")]
    return chunks


def split_sentences(text: str) -> list[str]:
    body = text.split("\n", 1)[-1]  # drop the "title | heading" context line
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", body)
    return [s.strip() for s in sents if 25 <= len(s.strip()) <= 600]


def extract_answer(question: str, chunk_texts: list[str]) -> str:
    q = set(tokenize(question))
    scored = []
    for rank, text in enumerate(chunk_texts):
        for pos, s in enumerate(split_sentences(text)):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                # prefer overlap, then higher-ranked chunk, then earlier sentence
                scored.append((overlap, -rank, -pos, s))
    if not scored:
        body = chunk_texts[0].split("\n", 1)[-1] if chunk_texts else ""
        return " ".join(body.split()[:60])
    scored.sort(reverse=True)
    best = [s for *_, s in scored[:MAX_ANSWER_SENTENCES]]
    return " ".join(best)


def main():
    template = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEMPLATE
    df = pd.read_csv(template, dtype=str, keep_default_na=False)
    q_col, d_col, a_col = df.columns[0], df.columns[1], df.columns[2]
    print(f"Template: {template} | {len(df)} questions | columns: {list(df.columns)}")

    chunks = load_chunks()
    print(f"Indexing {len(chunks)} chunks...")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    diag = []
    for i, question in enumerate(df[q_col]):
        scores = bm25.get_scores(tokenize(question))
        top_idx = scores.argsort()[::-1][:200]

        per_doc = defaultdict(list)
        for idx in top_idx:
            per_doc[chunks[idx]["document_id"]].append((scores[idx], idx))
        doc_scores = {}
        for doc, items in per_doc.items():
            items.sort(reverse=True)
            vals = [s for s, _ in items[:TOP_CHUNKS_PER_DOC]]
            doc_scores[doc] = vals[0] + 0.25 * sum(vals[1:])  # max-dominant: big docs can't win on volume
        ranked = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)

        if not ranked:
            best_doc, answer = "", ""
        else:
            best_doc = ranked[0][0]
            evidence = [chunks[idx]["text"] for _, idx in sorted(per_doc[best_doc], reverse=True)[:3]]
            answer = extract_answer(question, evidence)

        doc_out = best_doc if KEEP_PDF_EXTENSION else re.sub(r"\.pdf$", "", best_doc, flags=re.I)
        df.at[i, d_col] = doc_out
        df.at[i, a_col] = answer or "No supporting passage found."

        top3 = ranked[:3]
        margin = (top3[0][1] - top3[1][1]) / top3[0][1] if len(top3) > 1 and top3[0][1] else 1.0
        diag.append({
            "row": i + 1,
            "question": question,
            "doc_1": top3[0][0] if len(top3) > 0 else "", "score_1": round(top3[0][1], 2) if top3 else 0,
            "doc_2": top3[1][0] if len(top3) > 1 else "", "score_2": round(top3[1][1], 2) if len(top3) > 1 else 0,
            "doc_3": top3[2][0] if len(top3) > 2 else "",
            "margin_pct": round(100 * margin, 1),
            "review_flag": "CHECK" if margin < 0.10 else "",
            "answer": answer,
        })

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "submission.csv", index=False, encoding="utf-8")
    pd.DataFrame(diag).to_csv(OUT_DIR / "diagnostics.csv", index=False, encoding="utf-8-sig")

    blanks = (df[d_col] == "").sum() + (df[a_col] == "").sum()
    flagged = sum(1 for d in diag if d["review_flag"])
    print(f"Wrote {OUT_DIR / 'submission.csv'} ({len(df)} rows, {blanks} blank fields)")
    print(f"Wrote {OUT_DIR / 'diagnostics.csv'} ({flagged} close calls flagged CHECK)")


if __name__ == "__main__":
    main()
