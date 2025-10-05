import json
import os
import re
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        index_dir: str,
        embed_model_name: str = "intfloat/e5-base-v2",
        alpha: float = 0.7,
        device: str | None = None,
        reranker_model_name: str | None = None,
        # Tier-2 activation defaults
        rerank_min_tokens: int = 10,
        rerank_low_max: float = 0.35,
        rerank_flat_diff: float = 0.08,
    ) -> None:
        self.index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metas: List[Dict] = json.load(f)
        texts_path = os.path.join(index_dir, "texts.json")
        self.texts: List[str] = []
        try:
            with open(texts_path, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
        except FileNotFoundError:
            # Fallback: attempt to reconstruct from default chunks path
            fallback_chunks = os.path.join(os.path.dirname(index_dir), "data", "chunks.jsonl")
            if os.path.exists(fallback_chunks):
                tmp_texts: List[str] = []
                with open(fallback_chunks, "r", encoding="utf-8") as cf:
                    for line in cf:
                        obj = json.loads(line)
                        if obj.get("chunk_index", -1) >= 0 and obj.get("text"):
                            tmp_texts.append(obj["text"])
                # Only accept if counts match the embeddings
                if len(tmp_texts) == int(self.embeddings.shape[0]):
                    self.texts = tmp_texts
            if not self.texts:
                raise FileNotFoundError(
                    "index/texts.json not found and fallback failed. "
                    "Please rebuild the index with scripts/build_index.py to generate texts.json."
                )

        # Sparse index (BM25 over non-empty docs)
        tokenized_all = [_tokenize(t) for t in self.texts]
        self.nonempty_indices = [i for i, toks in enumerate(tokenized_all) if len(toks) > 0]
        tokenized_nonempty = [tokenized_all[i] for i in self.nonempty_indices] or [["dummy"]]
        self.bm25 = BM25Okapi(tokenized_nonempty)

        # Dense model
        self.model = SentenceTransformer(embed_model_name, device=device)
        self.alpha = alpha
        self.device = device

        # Re-ranker (lazy init)
        self.reranker_model_name = reranker_model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._cross_encoder: CrossEncoder | None = None
        # Activation thresholds
        self.rerank_min_tokens = rerank_min_tokens
        self.rerank_low_max = rerank_low_max
        self.rerank_flat_diff = rerank_flat_diff

    def _embed_query(self, query: str) -> np.ndarray:
        q = f"query: {query}"
        vec = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")

    def _dense_search(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        qvec = self._embed_query(query)
        D, I = self.index.search(qvec, k)
        return D[0], I[0]

    def _sparse_scores_full(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        scores_full = np.zeros(len(self.texts), dtype=np.float32)
        if len(self.nonempty_indices) == 0:
            return scores_full
        subset_scores = self.bm25.get_scores(tokens)
        for pos, orig_idx in enumerate(self.nonempty_indices):
            if pos < len(subset_scores):
                scores_full[orig_idx] = subset_scores[pos]
        return scores_full

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        smin, smax = float(scores.min()), float(scores.max())
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)

    def retrieve(self, query: str, k_dense: int = 30, k_sparse: int = 30, k_final: int = 15) -> List[Dict]:
        dense_scores, dense_idxs = self._dense_search(query, k_dense)
        sparse_scores_full = self._sparse_scores_full(query)

        dense_norm = self._normalize(dense_scores)
        # take top sparse
        top_sparse_idxs = np.argsort(sparse_scores_full)[::-1][:k_sparse]
        sparse_top_scores = sparse_scores_full[top_sparse_idxs]
        sparse_norm = self._normalize(sparse_top_scores)

        score_map: Dict[int, float] = {}
        for idx, s in zip(dense_idxs, dense_norm):
            score_map[int(idx)] = self.alpha * float(s)
        for idx, s in zip(top_sparse_idxs, sparse_norm):
            score_map[int(idx)] = score_map.get(int(idx), 0.0) + (1 - self.alpha) * float(s)

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:k_final]
        results: List[Dict] = []
        for idx, score in ranked:
            if 0 <= idx < len(self.texts):
                results.append({"score": score, "text": self.texts[idx], "metadata": self.metas[idx]})
        return results

    def should_rerank(self, query: str, tier1_results: List[Dict]) -> bool:
        if not tier1_results:
            return False
        tokens = len(query.split())
        fused_scores = np.array([float(r.get("score", 0.0)) for r in tier1_results], dtype=np.float32)
        max_s = float(fused_scores.max()) if fused_scores.size > 0 else 0.0
        med_s = float(np.median(fused_scores)) if fused_scores.size > 0 else 0.0
        domain_hit = any((r.get("metadata", {}).get("category") in {"legal", "compliance"}) for r in tier1_results[:3])
        kw = {"arbitration", "indemnification", "regulation", "clause", "policy"}
        has_kw = any(w in query.lower() for w in kw)
        flatness = (max_s - med_s)
        return (
            (tokens > self.rerank_min_tokens)
            or domain_hit
            or has_kw
            or (max_s < self.rerank_low_max)
            or (flatness < self.rerank_flat_diff)
        )

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.reranker_model_name, device=self.device)
        return self._cross_encoder

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
        if not candidates:
            return []
        ce = self._get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs)
        # Attach scores and sort
        enriched = []
        for c, s in zip(candidates, scores):
            item = dict(c)
            item["rerank_score"] = float(s)
            enriched.append(item)
        enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
        return enriched[:top_n]

    def retrieve_tiered(
        self,
        query: str,
        k_dense: int = 30,
        k_sparse: int = 30,
        k_final: int = 15,
        top_n_rerank: int = 5,
    ) -> dict:
        tier1 = self.retrieve(query, k_dense=k_dense, k_sparse=k_sparse, k_final=k_final)
        tier2 = self.rerank(query, tier1, top_n=top_n_rerank)
        return {"tier1": tier1, "tier2": tier2}

    def retrieve_auto(
        self,
        query: str,
        k_dense: int = 30,
        k_sparse: int = 30,
        k_final: int = 15,
        top_n_rerank: int = 5,
    ) -> dict:
        tier1 = self.retrieve(query, k_dense=k_dense, k_sparse=k_sparse, k_final=k_final)
        activate = self.should_rerank(query, tier1)
        tier2 = self.rerank(query, tier1, top_n=top_n_rerank) if activate else []
        return {"tier1": tier1, "tier2": tier2, "tier2_activated": activate}

    def build_context(self, retrieved: List[Dict], max_chars: int = 4000) -> str:
        parts: List[str] = []
        total = 0
        for item in retrieved:
            header = f"Source: {item['metadata'].get('source','')} (chunk {item['metadata'].get('chunk_index','')})\nURL: {item['metadata'].get('url','')}\n"
            body = item["text"].strip()
            piece = header + body + "\n\n"
            if total + len(piece) > max_chars:
                break
            parts.append(piece)
            total += len(piece)
        return "".join(parts)


