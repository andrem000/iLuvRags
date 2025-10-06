import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


@dataclass
class SourceItem:
    category: str
    name: str
    url: str
    mime: Optional[str]


def read_sources_yaml(path: str) -> List[SourceItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items: List[SourceItem] = []
    for category, entries in data.items():
        for entry in entries:
            items.append(
                SourceItem(
                    category=category,
                    name=entry.get("name"),
                    url=entry.get("url"),
                    mime=entry.get("mime"),
                )
            )
    return items


def http_get(url: str, timeout: int = 60) -> Tuple[bytes, str]:
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    return resp.content, content_type


def extract_text_from_html(content: bytes, base_url: str) -> str:
    text = ""
    if trafilatura is not None:
        try:
            text = trafilatura.extract(content, include_comments=False, include_tables=False) or ""
        except Exception:
            text = ""
    if not text:
        soup = BeautifulSoup(content, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
    return normalize_whitespace(text)


def extract_text_from_pdf(content: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    from io import BytesIO

    reader = PdfReader(BytesIO(content))
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return normalize_whitespace("\n\n".join(pages))


def extract_text_from_docx(content: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed")
    from io import BytesIO

    doc = docx.Document(BytesIO(content))
    paras = [p.text for p in doc.paragraphs]
    return normalize_whitespace("\n".join(paras))


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\u00A0", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    if not text:
        return []
    words = text.split(" ")
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


def infer_mime(explicit_mime: Optional[str], content_type: str, url: str) -> str:
    if explicit_mime:
        return explicit_mime
    if content_type:
        return content_type
    if url.lower().endswith(".pdf"):
        return "application/pdf"
    if any(url.lower().endswith(ext) for ext in [".docx", ".doc"]):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "text/html"


def extract_text_by_mime(content: bytes, mime: str, url: str) -> str:
    if mime.startswith("text/html"):
        return extract_text_from_html(content, url)
    if mime == "application/pdf":
        return extract_text_from_pdf(content)
    if mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return extract_text_from_docx(content)
    # Fallback: try decode as text
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return ""


def write_jsonl(records: Iterable[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_sources(
    sources_path: str,
    output_jsonl: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    items = read_sources_yaml(sources_path)
    output_records: List[Dict] = []
    for item in tqdm(items, desc="Fetching sources"):
        try:
            content, content_type = http_get(item.url)
            mime = infer_mime(item.mime, content_type, item.url)
            text = extract_text_by_mime(content, mime, item.url)
            chunks = split_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for idx, chunk in enumerate(chunks):
                output_records.append(
                    {
                        "source": item.name,
                        "category": item.category,
                        "url": item.url,
                        "doc_mime": mime,
                        "chunk_index": idx,
                        "text": chunk,
                    }
                )
        except Exception as e:
            output_records.append(
                {
                    "source": item.name,
                    "category": item.category,
                    "url": item.url,
                    "doc_mime": item.mime or "",
                    "chunk_index": -1,
                    "error": str(e),
                }
            )
    write_jsonl(output_records, output_jsonl)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch sources, parse, and chunk into JSONL")
    p.add_argument("--sources", default="sources.yaml", help="Path to YAML listing sources")
    p.add_argument("--out", default="data/chunks.jsonl", help="Output JSONL path")
    p.add_argument("--chunk_size", type=int, default=500, help="Chunk size in words")
    p.add_argument("--chunk_overlap", type=int, default=80, help="Overlap in words")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    process_sources(
        sources_path=args.sources,
        output_jsonl=args.out,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Wrote chunks to {args.out}")


if __name__ == "__main__":
    main()


