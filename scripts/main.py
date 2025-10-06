import argparse
import json
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import HybridRetriever


def generate_answer(model_name: str, prompt: str, device: str | None = None, max_new_tokens: int = 256) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if device and device.startswith("cuda") else None)
    if device:
        model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :].strip()


def build_prompt(context: str, question: str) -> str:
    return f"""You are a concise assistant. Use ONLY the provided context. If the context doesn't contain enough information to answer the question, say "I don't know" or "The provided context doesn't contain enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end RAG demo: retrieval + generation + simple eval")
    p.add_argument("--index", default="index", help="Index directory")
    p.add_argument("--embed_model", default="intfloat/e5-base-v2", help="Embedding model name")
    p.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="CrossEncoder model name")
    p.add_argument("--llm", default="Qwen/Qwen2-1.5B-Instruct", help="LLM model name for generation")
    p.add_argument("--query", default=None, help="Single user question")
    p.add_argument("--queries", default=None, help="Path to JSON with queries (schema: {queries: [{question: ...}]})")
    p.add_argument("--device", default=None, help="torch device for LLM, e.g., cuda or cpu")
    p.add_argument("--k_final", type=int, default=15)
    p.add_argument("--top_n_rerank", type=int, default=5)
    p.add_argument("--verbose_retriever", action="store_true", help="Enable verbose cache logs for retriever")
    p.add_argument("--pretty", action="store_true", help="Pretty-print retrieval results with citations and URLs")
    p.add_argument("--allow_low_conf_gen", action="store_true", help="Force generation even if retrieval is low-confidence")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(
        index_dir=args.index,
        embed_model_name=args.embed_model,
        reranker_model_name=args.reranker,
        verbose=args.verbose_retriever,
    )

    # Prepare LLM once
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm,
        dtype=torch.float16 if args.device and args.device.startswith("cuda") else None,
    )
    if args.device:
        model = model.to(args.device)

    def _fmt_citation(meta: Dict) -> str:
        src = meta.get("source", "")
        ch = meta.get("chunk_index", "")
        return f"[{src}#{ch}]"

    def _pretty_print(question: str, tier: str, items: List[Dict]) -> None:
        if not items:
            print(f"\n{tier}: (no results)")
            return
        print(f"\n{tier} (top {len(items)}):")
        for i, it in enumerate(items, start=1):
            meta = it.get("metadata", {})
            url = meta.get("url", "")
            cite = _fmt_citation(meta)
            score = it.get("score", 0.0)
            dense_score = it.get("dense_score", 0.0)
            sparse_score = it.get("sparse_score", 0.0)
            snippet = (it.get("text", "") or "").strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            print(f"  {i:>2}. {cite} fused={score:.3f} (dense={dense_score:.3f} sparse={sparse_score:.3f})")
            if url:
                print(f"      URL: {url}")
            if snippet:
                print(f"      {snippet}")

    def answer_for_question(question: str) -> Dict:
        out: Dict = retriever.retrieve_auto(question, k_final=args.k_final, top_n_rerank=args.top_n_rerank)
        ctx_t1 = retriever.build_context(out["tier1"], max_chars=3500)
        prompt_t1 = build_prompt(ctx_t1, question)
        inputs = tokenizer(prompt_t1, return_tensors="pt").to(model.device)
        gen1 = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        ans_t1 = tokenizer.decode(gen1[0], skip_special_tokens=True)
        ans_t1 = ans_t1[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :].strip()

        ans_t2 = None
        if out.get("tier2_activated") and out.get("tier2"):
            ctx_t2 = retriever.build_context(out["tier2"], max_chars=3500)
            prompt_t2 = build_prompt(ctx_t2, question)
            inputs2 = tokenizer(prompt_t2, return_tensors="pt").to(model.device)
            gen2 = model.generate(**inputs2, max_new_tokens=256, do_sample=False)
            ans_t2 = tokenizer.decode(gen2[0], skip_special_tokens=True)
            ans_t2 = ans_t2[len(tokenizer.decode(inputs2.input_ids[0], skip_special_tokens=True)) :].strip()

        if args.pretty:
            print("\n=== Question ===")
            print(question)
            timing = out.get("timing", {})
            if timing:
                print("\nTimings (ms): dense={:.1f} sparse={:.1f} fusion={:.1f} rerank={:.1f} total={:.1f}".format(
                    float(timing.get("dense_ms", 0.0)),
                    float(timing.get("sparse_ms", 0.0)),
                    float(timing.get("fusion_ms", 0.0)),
                    float(timing.get("rerank_ms", 0.0)),
                    float(timing.get("total_ms", 0.0)),
                ))
            if out.get("low_confidence"):
                reasons = out.get("low_confidence_reasons", {})
                print("\n[guardrail] Low-confidence retrieval detected:")
                print(json.dumps(reasons, ensure_ascii=False, indent=2))
            _pretty_print(question, "Tier1", out.get("tier1", []))
            if out.get("tier2_activated"):
                _pretty_print(question, "Tier2 (reranked)", out.get("tier2", []))
            # Answer printing may be suppressed if low-confidence and not allowed
            if not out.get("low_confidence") or args.allow_low_conf_gen:
                print("\n=== Answers ===")
                print("Tier1:", ans_t1)
                if ans_t2 is not None:
                    print("\nTier2:", ans_t2)
            else:
                print("\n=== Answers ===")
                print("Guardrail: Insufficient retrieval confidence; skipping generation. Try refining the query.")

        return {
            "question": question,
            "tier1_answer": ans_t1,
            "tier2_answer": ans_t2,
            "tier2_activated": out.get("tier2_activated"),
            "tier2_reasons": out.get("tier2_reasons"),
            "timing": out.get("timing"),
        }

    results = []
    if args.queries:
        data = json.load(open(args.queries, "r", encoding="utf-8"))
        for q in data.get("queries", []):
            question = q.get("question")
            if not question:
                continue
            results.append(answer_for_question(question))
    elif args.query:
        results.append(answer_for_question(args.query))
    else:
        raise SystemExit("Provide --query or --queries JSON path")

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


