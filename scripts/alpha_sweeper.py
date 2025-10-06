import argparse
import csv
import json
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt

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
    p = argparse.ArgumentParser(description="Sweep alpha values for hybrid fusion and evaluate Recall@k/MRR@k")
    p.add_argument("--index", default="index", help="Index directory")
    p.add_argument("--embed_model", default="intfloat/e5-base-v2", help="Embedding model name")
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="CrossEncoder model name")
    p.add_argument("--queries_json", required=True, help="Path to JSON with queries (schema: {queries: [{id, question}]})")
    p.add_argument("--labels_csv", required=True, help="CSV with columns: query_id,relevant_doc_ids")
    p.add_argument("--k", type=int, default=10, help="k for Recall@k and MRR@k")
    p.add_argument("--alphas", default="0.5,0.65,0.7,0.8", help="Comma-separated alpha values to test")
    p.add_argument("--top_n_rerank", type=int, default=5)
    p.add_argument("--chart", action="store_true", help="Plot Recall@k vs alpha (Tier1/Tier2)")
    return p.parse_args()


def evaluate_for_alpha(alpha: float, retriever: HybridRetriever, queries: List[Dict], labels: Dict[str, Set[int]], k: int, top_n_rerank: int) -> Tuple[float, float, float, float, int]:
    r_t1_sum = 0.0
    r_t2_sum = 0.0
    m_t1_sum = 0.0
    m_t2_sum = 0.0
    n = 0

    for q in queries:
        qid = q.get("id") or q.get("qid") or ""
        question = q.get("question")
        if not qid or not question:
            continue
        rel = labels.get(qid, set())
        out = retriever.retrieve_auto(question, k_final=max(k, 15), top_n_rerank=top_n_rerank)
        tier1 = out.get("tier1", [])
        tier2 = out.get("tier2", [])
        idxs_t1 = [int(it.get("metadata", {}).get("chunk_index", -1)) for it in tier1 if it.get("metadata")]
        idxs_t2 = [int(it.get("metadata", {}).get("chunk_index", -1)) for it in tier2 if it.get("metadata")]

        r_t1_sum += recall_at_k(idxs_t1, rel, k)
        r_t2_sum += recall_at_k(idxs_t2, rel, k)
        m_t1_sum += mrr_at_k(idxs_t1, rel, k)
        m_t2_sum += mrr_at_k(idxs_t2, rel, k)
        n += 1

    return (r_t1_sum / n if n else 0.0, r_t2_sum / n if n else 0.0, m_t1_sum / n if n else 0.0, m_t2_sum / n if n else 0.0, n)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels_csv)
    data = json.load(open(args.queries_json, "r", encoding="utf-8"))
    queries = data.get("queries", [])
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    rows: List[Dict] = []

    for a in alphas:
        retriever = HybridRetriever(
            index_dir=args.index,
            embed_model_name=args.embed_model,
            reranker_model_name=args.reranker,
            alpha=a,
        )
        r1, r2, m1, m2, n = evaluate_for_alpha(a, retriever, queries, labels, args.k, args.top_n_rerank)
        rows.append({
            "alpha": a,
            "Recall@k_Tier1": r1,
            "Recall@k_Tier2": r2,
            "MRR@k_Tier1": m1,
            "MRR@k_Tier2": m2,
            "num_queries": n,
        })

    # Pick best by recall and MRR (Tier2 preference)
    best_by_recall = max(rows, key=lambda r: (r["Recall@k_Tier2"], r["Recall@k_Tier1"])) if rows else None
    best_by_mrr = max(rows, key=lambda r: (r["MRR@k_Tier2"], r["MRR@k_Tier1"])) if rows else None

    result = {"k": args.k, "rows": rows, "best_by_recall": best_by_recall, "best_by_mrr": best_by_mrr}
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.chart and rows:
        xs = [r["alpha"] for r in rows]
        r1s = [r["Recall@k_Tier1"] for r in rows]
        r2s = [r["Recall@k_Tier2"] for r in rows]
        plt.figure(figsize=(5, 3))
        plt.plot(xs, r1s, marker="o", label="Tier1")
        plt.plot(xs, r2s, marker="o", label="Tier2")
        plt.xlabel("alpha")
        plt.ylabel(f"Recall@{args.k}")
        plt.title("Recall vs alpha")
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


