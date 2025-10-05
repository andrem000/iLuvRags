import argparse
import json
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import HybridRetriever


def generate_answer(model_name: str, prompt: str, device: str | None = None, max_new_tokens: int = 256) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device and device.startswith("cuda") else None)
    if device:
        model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)) :].strip()


def build_prompt(context: str, question: str) -> str:
    return f"""You are a concise assistant. Use only the provided context.

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
    p.add_argument("--llm", default="HuggingFaceH4/zephyr-7b-alpha", help="LLM model name for generation")
    p.add_argument("--query", required=True, help="User question")
    p.add_argument("--device", default=None, help="torch device for LLM, e.g., cuda or cpu")
    p.add_argument("--k_final", type=int, default=15)
    p.add_argument("--top_n_rerank", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(
        index_dir=args.index,
        embed_model_name=args.embed_model,
        reranker_model_name=args.reranker,
    )

    # Tiered retrieval with auto activation
    out: Dict = retriever.retrieve_auto(args.query, k_final=args.k_final, top_n_rerank=args.top_n_rerank)

    # Build contexts
    ctx_t1 = retriever.build_context(out["tier1"], max_chars=3500)
    ctx_t2 = retriever.build_context(out["tier2"], max_chars=3500) if out.get("tier2") else ""

    # Generate answers
    prompt_t1 = build_prompt(ctx_t1, args.query)
    ans_t1 = generate_answer(args.llm, prompt_t1, device=args.device)
    results = {"tier1": {"activated": True, "answer": ans_t1}}

    if out.get("tier2_activated") and ctx_t2:
        prompt_t2 = build_prompt(ctx_t2, args.query)
        ans_t2 = generate_answer(args.llm, prompt_t2, device=args.device)
        results["tier2"] = {"activated": True, "answer": ans_t2}
    else:
        results["tier2"] = {"activated": False, "answer": None}

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


