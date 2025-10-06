import argparse
import csv
import json
from typing import Dict, List, Set

from retriever import HybridRetriever


def load_labels(path: str) -> Dict[str, Set[int]]:
    labels: Dict[str, Set[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query_id") or row.get("id") or ""
            if not qid:
                continue
            ids_str = (row.get("relevant_doc_ids") or "").strip()
            rel_ids: Set[int] = set()
            if ids_str:
                for part in ids_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        rel_ids.add(int(part))
                    except Exception:
                        # allow doc ids like '123' only
                        continue
            labels[qid] = rel_ids
    return labels


def recall_at_k(ranked_indices: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    cutoff = ranked_indices[:k]
    hits = sum(1 for i in cutoff if i in relevant)
    return float(hits > 0)


def mrr_at_k(ranked_indices: List[int], relevant: Set[int], k: int) -> float:
    cutoff = ranked_indices[:k]
    for rank, idx in enumerate(cutoff, start=1):
        if idx in relevant:
            return 1.0 / float(rank)
    return 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Recall@k and MRR for Tier1 vs Tier2")
    p.add_argument("--index", default="index", help="Index directory")
    p.add_argument("--embed_model", default="intfloat/e5-base-v2", help="Embedding model name")
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="CrossEncoder model name")
    p.add_argument("--queries_json", required=True, help="Path to JSON with queries (schema: {queries: [{id, question}]})")
    p.add_argument("--labels_csv", required=True, help="CSV with columns: query_id,relevant_doc_ids (comma-separated chunk indices)")
    p.add_argument("--k", type=int, default=10, help="k for Recall@k and MRR@k")
    p.add_argument("--top_n_rerank", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(
        index_dir=args.index,
        embed_model_name=args.embed_model,
        reranker_model_name=args.reranker,
    )
    labels = load_labels(args.labels_csv)
    data = json.load(open(args.queries_json, "r", encoding="utf-8"))
    queries = data.get("queries", [])

    r_t1_sum = 0.0
    r_t2_sum = 0.0
    m_t1_sum = 0.0
    m_t2_sum = 0.0
    n = 0

    per_query: List[Dict] = []
    for q in queries:
        qid = q.get("id") or q.get("qid") or ""
        question = q.get("question")
        if not qid or not question:
            continue
        rel = labels.get(qid, set())
        out = retriever.retrieve_auto(question, k_final=max(args.k, 15), top_n_rerank=args.top_n_rerank)
        tier1 = out.get("tier1", [])
        tier2 = out.get("tier2", [])
        idxs_t1 = [int(it.get("metadata", {}).get("chunk_index", -1)) for it in tier1 if it.get("metadata")]
        idxs_t2 = [int(it.get("metadata", {}).get("chunk_index", -1)) for it in tier2 if it.get("metadata")]

        r1 = recall_at_k(idxs_t1, rel, args.k)
        r2 = recall_at_k(idxs_t2, rel, args.k)
        m1 = mrr_at_k(idxs_t1, rel, args.k)
        m2 = mrr_at_k(idxs_t2, rel, args.k)

        r_t1_sum += r1
        r_t2_sum += r2
        m_t1_sum += m1
        m_t2_sum += m2
        n += 1

        per_query.append({
            "query_id": qid,
            "recall@k_t1": r1,
            "recall@k_t2": r2,
            "mrr@k_t1": m1,
            "mrr@k_t2": m2,
        })

    agg = {
        "k": args.k,
        "Recall@k_Tier1": (r_t1_sum / n) if n else 0.0,
        "Recall@k_Tier2": (r_t2_sum / n) if n else 0.0,
        "MRR@k_Tier1": (m_t1_sum / n) if n else 0.0,
        "MRR@k_Tier2": (m_t2_sum / n) if n else 0.0,
        "num_queries": n,
    }

    print(json.dumps({"aggregate": agg, "per_query": per_query}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


