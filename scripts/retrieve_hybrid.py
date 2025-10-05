import argparse
import json
from retriever import HybridRetriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid retrieval (dense + BM25) debug CLI")
    p.add_argument("--index", default="index", help="Index directory")
    p.add_argument("--model", default="intfloat/e5-base-v2", help="Embedding model name")
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="CrossEncoder model name")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--alpha", type=float, default=0.7, help="Fusion weight for dense scores")
    p.add_argument("--k_dense", type=int, default=30, help="Top-k for dense search")
    p.add_argument("--k_sparse", type=int, default=30, help="Top-k for sparse search")
    p.add_argument("--k_final", type=int, default=15, help="Final top-k after fusion")
    p.add_argument("--top_n_rerank", type=int, default=5, help="Top-N after re-ranking")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(index_dir=args.index, embed_model_name=args.model, alpha=args.alpha, reranker_model_name=args.reranker)
    out = retriever.retrieve_tiered(
        query=args.query,
        k_dense=args.k_dense,
        k_sparse=args.k_sparse,
        k_final=args.k_final,
        top_n_rerank=args.top_n_rerank,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


