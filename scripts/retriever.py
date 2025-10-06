import json
import pickle
import os
import re
import time
import hashlib
from collections import OrderedDict
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
        cache_path: str | None = None,
        query_cache_max_entries: int = 2000,
        bm25_cache_path: str | None = None,
        bm25_cache_max_versions: int = 3,
        # Low-confidence guardrail defaults
        low_conf_max_threshold: float = 0.30,
        low_conf_flat_diff: float = 0.08,
        low_conf_min_hits: int = 2,
        verbose: bool = False,
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

        # Sparse index (BM25 over non-empty docs) with tokenization cache
        self.bm25_cache_path = bm25_cache_path or os.path.join(index_dir, "bm25_cache.pkl")
        self.bm25_cache_max_versions = max(1, int(bm25_cache_max_versions))
        texts_path_mtime = 0.0
        texts_path_size = 0
        try:
            texts_path_mtime = os.path.getmtime(texts_path)
            texts_path_size = os.path.getsize(texts_path)
        except Exception:
            pass
        version_key = f"len={len(self.texts)}|size={texts_path_size}|mtime={int(texts_path_mtime)}"

        bm25_cache: "OrderedDict[str, dict]" = OrderedDict()
        try:
            if os.path.exists(self.bm25_cache_path):
                with open(self.bm25_cache_path, "rb") as cf:
                    loaded = pickle.load(cf)
                if isinstance(loaded, dict):
                    bm25_cache = OrderedDict(loaded)
                    if self.verbose:
                        print(f"[bm25_cache] loaded {len(bm25_cache)} versions from {self.bm25_cache_path}")
        except Exception:
            bm25_cache = OrderedDict()

        if version_key in bm25_cache:
            entry = bm25_cache.pop(version_key)
            bm25_cache[version_key] = entry  # move to end (recent)
            self.nonempty_indices = entry.get("nonempty_indices", [])
            tokenized_nonempty = entry.get("tokenized_nonempty", [["dummy"]])
            if self.verbose:
                print(f"[bm25_cache] hit for version {version_key}")
        else:
            tokenized_all = [_tokenize(t) for t in self.texts]
            self.nonempty_indices = [i for i, toks in enumerate(tokenized_all) if len(toks) > 0]
            tokenized_nonempty = [tokenized_all[i] for i in self.nonempty_indices] or [["dummy"]]
            # save
            bm25_cache[version_key] = {
                "nonempty_indices": self.nonempty_indices,
                "tokenized_nonempty": tokenized_nonempty,
            }
            while len(bm25_cache) > self.bm25_cache_max_versions:
                try:
                    bm25_cache.popitem(last=False)
                except Exception:
                    break
            try:
                with open(self.bm25_cache_path, "wb") as cf:
                    pickle.dump(dict(bm25_cache), cf)
                if self.verbose:
                    print(f"[bm25_cache] save -> {self.bm25_cache_path} (versions={len(bm25_cache)})")
            except Exception:
                pass

        self.bm25 = BM25Okapi(tokenized_nonempty)

        # Dense model
        self.embed_model_name = embed_model_name
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

        # Query embedding cache (persistent)
        self.cache_path = cache_path or os.path.join(index_dir, "query_cache.pkl")
        self.query_cache_max_entries = max(1, int(query_cache_max_entries))
        self._query_cache: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
        self.verbose = verbose
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as pf:
                    loaded_q = pickle.load(pf) or {}
                    if isinstance(loaded_q, dict):
                        self._query_cache = OrderedDict(loaded_q)
                if self.verbose:
                    print(f"[cache] loaded {len(self._query_cache)} entries from {self.cache_path}")
        except Exception:
            self._query_cache = OrderedDict()

        # Guardrail thresholds
        self.low_conf_max_threshold = float(low_conf_max_threshold)
        self.low_conf_flat_diff = float(low_conf_flat_diff)
        self.low_conf_min_hits = max(0, int(low_conf_min_hits))

    def _embed_query(self, query: str) -> np.ndarray:
        key = (self.embed_model_name, query)
        cached = self._query_cache.get(key)
        if cached is not None:
            if self.verbose:
                print(f"[cache] hit for query using {self.embed_model_name}")
            # refresh LRU order
            try:
                self._query_cache.move_to_end(key)
            except Exception:
                pass
            return np.asarray(cached, dtype="float32").reshape(1, -1)
        q = f"query: {query}"
        vec = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        # Persist
        try:
            self._query_cache[key] = vec[0].astype("float32").tolist()
            # cap size
            while len(self._query_cache) > self.query_cache_max_entries:
                try:
                    self._query_cache.popitem(last=False)
                except Exception:
                    break
            with open(self.cache_path, "wb") as pf:
                pickle.dump(dict(self._query_cache), pf)
            if self.verbose:
                print(f"[cache] save -> {self.cache_path}")
        except Exception:
            pass
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

    def _retrieve_with_timing(self, query: str, k_dense: int, k_sparse: int, k_final: int) -> Tuple[List[Dict], Dict[str, float]]:
        t0 = time.perf_counter()
        dense_scores, dense_idxs = self._dense_search(query, k_dense)
        t1 = time.perf_counter()
        sparse_scores_full = self._sparse_scores_full(query)
        t2 = time.perf_counter()

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
                # Get individual scores for this item
                dense_score = float(dense_norm[np.where(dense_idxs == idx)[0][0]]) if idx in dense_idxs else 0.0
                sparse_score = float(sparse_norm[np.where(top_sparse_idxs == idx)[0][0]]) if idx in top_sparse_idxs else 0.0
                results.append({
                    "score": float(score), 
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "text": self.texts[idx], 
                    "metadata": self.metas[idx]
                })
        t3 = time.perf_counter()
        timing = {
            "dense_ms": (t1 - t0) * 1000.0,
            "sparse_ms": (t2 - t1) * 1000.0,
            "fusion_ms": (t3 - t2) * 1000.0,
        }
        return results, timing

    def retrieve(self, query: str, k_dense: int = 30, k_sparse: int = 30, k_final: int = 15) -> List[Dict]:
        results, _ = self._retrieve_with_timing(query, k_dense=k_dense, k_sparse=k_sparse, k_final=k_final)
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

    def should_rerank_with_reasons(self, query: str, tier1_results: List[Dict]) -> tuple[bool, Dict]:
        if not tier1_results:
            return False, {"empty_tier1": True}
        tokens = len(query.split())
        fused_scores = np.array([float(r.get("score", 0.0)) for r in tier1_results], dtype=np.float32)
        max_s = float(fused_scores.max()) if fused_scores.size > 0 else 0.0
        med_s = float(np.median(fused_scores)) if fused_scores.size > 0 else 0.0
        flatness = (max_s - med_s)
        domain_hit = any((r.get("metadata", {}).get("category") in {"legal", "compliance"}) for r in tier1_results[:3])
        kw = {"arbitration", "indemnification", "regulation", "clause", "policy"}
        has_kw = any(w in query.lower() for w in kw)

        cond_tokens = tokens > self.rerank_min_tokens
        cond_lowmax = max_s < self.rerank_low_max
        cond_flat = flatness < self.rerank_flat_diff
        triggers = []
        if cond_tokens:
            triggers.append("long_query")
        if domain_hit:
            triggers.append("domain_category")
        if has_kw:
            triggers.append("keywords")
        if cond_lowmax:
            triggers.append("low_peak")
        if cond_flat:
            triggers.append("flat_distribution")

        activate = any([cond_tokens, domain_hit, has_kw, cond_lowmax, cond_flat])
        reasons = {
            "tokens": tokens,
            "max_fused": max_s,
            "median_fused": med_s,
            "flatness": flatness,
            "domain_hit": domain_hit,
            "has_keywords": has_kw,
            "thresholds": {
                "min_tokens": self.rerank_min_tokens,
                "low_max": self.rerank_low_max,
                "flat_diff": self.rerank_flat_diff,
            },
            "triggered_by": triggers,
        }
        return activate, reasons

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
        tier1, t_t1 = self._retrieve_with_timing(query, k_dense=k_dense, k_sparse=k_sparse, k_final=k_final)
        t_r0 = time.perf_counter()
        tier2 = self.rerank(query, tier1, top_n=top_n_rerank)
        t_r1 = time.perf_counter()
        timing = dict(t_t1)
        timing["rerank_ms"] = (t_r1 - t_r0) * 1000.0
        timing["total_ms"] = timing.get("dense_ms", 0.0) + timing.get("sparse_ms", 0.0) + timing.get("fusion_ms", 0.0) + timing.get("rerank_ms", 0.0)
        return {"tier1": tier1, "tier2": tier2, "timing": timing}

    def retrieve_auto(
        self,
        query: str,
        k_dense: int = 30,
        k_sparse: int = 30,
        k_final: int = 15,
        top_n_rerank: int = 5,
    ) -> dict:
        tier1, t_t1 = self._retrieve_with_timing(query, k_dense=k_dense, k_sparse=k_sparse, k_final=k_final)
        activate, reasons = self.should_rerank_with_reasons(query, tier1)
        # Low-confidence guardrail signals
        fused_scores = np.array([float(r.get("score", 0.0)) for r in tier1], dtype=np.float32)
        max_s = float(fused_scores.max()) if fused_scores.size > 0 else 0.0
        med_s = float(np.median(fused_scores)) if fused_scores.size > 0 else 0.0
        num_hits = int((fused_scores >= 0.15).sum()) if fused_scores.size > 0 else 0
        low_confidence = (
            (max_s < self.low_conf_max_threshold)
            or ((max_s - med_s) < self.low_conf_flat_diff)
            or (num_hits < self.low_conf_min_hits)
        )
        low_conf_reasons = {
            "max_fused": max_s,
            "median_fused": med_s,
            "num_hits_ge_0.15": num_hits,
            "thresholds": {
                "max_threshold": self.low_conf_max_threshold,
                "flat_diff": self.low_conf_flat_diff,
                "min_hits": self.low_conf_min_hits,
            },
            "triggered": {
                "low_max": (max_s < self.low_conf_max_threshold),
                "flat": ((max_s - med_s) < self.low_conf_flat_diff),
                "few_hits": (num_hits < self.low_conf_min_hits),
            },
        }
        t_r0 = time.perf_counter()
        tier2 = self.rerank(query, tier1, top_n=top_n_rerank) if activate else []
        t_r1 = time.perf_counter()
        timing = dict(t_t1)
        timing["rerank_ms"] = (t_r1 - t_r0) * 1000.0 if activate else 0.0
        timing["total_ms"] = timing.get("dense_ms", 0.0) + timing.get("sparse_ms", 0.0) + timing.get("fusion_ms", 0.0) + timing.get("rerank_ms", 0.0)
        return {
            "tier1": tier1,
            "tier2": tier2,
            "tier2_activated": activate,
            "tier2_reasons": reasons,
            "low_confidence": low_confidence,
            "low_confidence_reasons": low_conf_reasons,
            "timing": timing,
        }

    def build_context(self, retrieved: List[Dict], max_chars: int = 4000, max_snippets: int = 5) -> str:
        parts: List[str] = []
        total = 0
        for i, item in enumerate(retrieved):
            if i >= max_snippets:
                break
            header = f"Source: {item['metadata'].get('source','')} (chunk {item['metadata'].get('chunk_index','')})\nURL: {item['metadata'].get('url','')}\n"
            body = item["text"].strip()
            piece = header + body + "\n\n"
            if total + len(piece) > max_chars:
                break
            parts.append(piece)
            total += len(piece)
        return "".join(parts)


