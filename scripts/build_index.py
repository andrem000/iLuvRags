import argparse
import json
import os
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(jsonl_path: str) -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("chunk_index", -1) >= 0 and obj.get("text"):
                texts.append(obj["text"])
                metas.append({k: obj.get(k) for k in ["source", "category", "url", "doc_mime", "chunk_index"]})
    return texts, metas


def embed(texts: List[str], model_name: str, batch_size: int = 64, device: str = None) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    # E5 models expect instruction prefixes: "query: ..." vs "passage: ...".
    # For index embeddings we use "passage: ".
    passages = [f"passage: {t}" for t in texts]
    emb = model.encode(passages, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if vectors are normalized
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, embeddings: np.ndarray, metas: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS index from chunks JSONL using E5 embeddings")
    p.add_argument("--chunks", default="data/chunks.jsonl", help="Path to chunks JSONL")
    p.add_argument("--out", default="index", help="Output directory for index and metadata")
    p.add_argument("--model", default="intfloat/e5-base-v2", help="Embedding model name")
    p.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    p.add_argument("--device", default=None, help="Torch device id, e.g. 'cuda' or 'cuda:0' or 'cpu'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    texts, metas = load_chunks(args.chunks)
    if not texts:
        raise RuntimeError("No valid chunks found. Did you run scrape_and_chunk.py?")
    embeddings = embed(texts, model_name=args.model, batch_size=args.batch_size, device=args.device)
    index = build_faiss_index(embeddings)
    save_index(index, embeddings, metas, args.out)
    print(f"Indexed {len(texts)} chunks. Saved FAISS index to {args.out}/faiss.index")


if __name__ == "__main__":
    main()


