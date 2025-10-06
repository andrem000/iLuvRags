# iLuvRags

## Table of Contents

- [Quick Start](#quick-start)
- [Step-by-Step Overview](#step-by-step-overview)
  - [Step 1: Setup & Data](#step-1-setup--data)
  - [Step 2: Embeddings & Storage](#step-2-embeddings--storage)
  - [Step 3: Tier 1 Retriever – Hybrid](#step-3-tier-1-retriever--hybrid)
  - [Step 4: Tier 2 Retriever – Re-Ranker](#step-4-tier-2-retriever--re-ranker)
  - [Step 5: LLM Generation](#step-5-llm-generation)
  - [Step 6: Evaluation & Demonstration](#step-6-evaluation--demonstration)

## Quick Start

Try the demo immediately:

```python
# Clone and install
REPO_URL = "https://github.com/andrem000/iLuvRags.git"
!git clone -q $REPO_URL
%cd iLuvRags
!pip install -q -r requirements.txt
!python scripts/main.py --queries data/queries.json --device cuda --verbose_retriever --llm Qwen/Qwen2.5-3B-Instruct
```

## Step-by-Step Overview
- **Step 1: Setup & Data** — Build/load corpus; chunk to JSONL with provenance
- **Step 2: Embeddings & Storage** — E5 embeddings + FAISS; save embeddings + metadata
- **Step 3: Tier 1 Retriever – Hybrid** — Combine FAISS (dense) with BM25 (sparse); return top 10–15
- **Step 4: Tier 2 Retriever – Re-Ranker** — Cross-encoder re-ranking; keep top 3–5
- **Step 5: LLM Generation** — Answer grounded in retrieved context
- **Step 6: Evaluation & Demonstration** — Compare Tier 1 vs Tier 2 on sample queries

### Step 1: Setup & Data

We use plain Python for ingestion and chunking to keep the pipeline transparent, lightweight, and easy to run on Google Colab without framework lock-in. It uses `requests` for fetching, `trafilatura` + `BeautifulSoup` for HTML extraction, `pypdf` for PDFs, and `python-docx` for DOCX.

- **Why not LangChain for ingestion**:
  - Direct control over fetching, parsing, and chunking with fewer moving parts
  - Smaller dependency surface for Colab; faster cold start and fewer version conflicts
  - Easier to debug and extend (e.g., add custom normalization, provenance fields)

- **Output format**: JSONL of text chunks with provenance (`source`, `category`, `url`, `doc_mime`, `chunk_index`).

### Ingestion Quickstart (Local or Colab)

1. Install deps:
   - Local: `pip install -r requirements.txt`
   - Colab: `!pip install -q -r requirements.txt`
2. Produce `data/chunks.jsonl` (your own pipeline or use the quick start above).
3. Proceed to Step 2 to build the index from `data/chunks.jsonl`.

### Step 2: Embeddings & Storage

- **Embedding model**: `intfloat/e5-base-v2`
- **Vector store**: FAISS (cosine similarity via inner product on normalized vectors)

Build the index:

```bash
pip install -r requirements.txt
# For CUDA support (faster on GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python scripts/build_index.py --device cuda
# devices: auto | cpu | cuda | cuda:0 | mps
```

Artifacts:
- `index/faiss.index`
- `index/embeddings.npy`
- `index/metadata.json`

### Step 3: Tier 1 Retriever – Hybrid

**Design Rationale**: We evaluated three retriever setups and chose hybrid retrieval to balance speed and accuracy, especially for real-world, mixed query types and Google Colab demos.

- **Baseline Dense Retriever — discarded**:
  - Strong semantic matching but weaker on keyword-/symbol-heavy queries and rare entities.
  - No lexical recall path; precision lags without re-ranking; fewer explicit speed/quality dials.

- **Hybrid Retriever (dense + sparse) — adopted**:
  - Dense embeddings (e.g., `intfloat/e5-base-v2`, `sentence-transformers/all-MiniLM-L6-v2`) + BM25.
  - Score fusion: `final_score = α * dense + (1−α) * sparse` (tunable `α`).
  - Handles both semantic and exact-term queries; improves recall and robustness.

- **Goal**: Combine dense semantic retrieval with keyword-based recall.
- **Dense**: Use FAISS to get top_k (e.g., 20).
- **Sparse**: Use BM25 (e.g., `rank-bm25`) on raw text.
- **Fusion**: Normalize and merge scores: \( final\_score = \alpha \cdot dense + (1-\alpha) \cdot bm25 \).

Why: Hybrid retrieval captures both meaning (e.g., "contract termination") and exact terms (e.g., "article 45").

Implementation specifics:

- **Chunking defaults** (standardized across codebase):
  | Parameter | Default | Token Equivalent | CLI Override |
  |-----------|---------|------------------|--------------|
  | `chunk_size` | 500 words | ~700–1,000 tokens | `--chunk_size` |
  | `chunk_overlap` | 80 words | ~100–150 tokens | `--chunk_overlap` |
  
  Use these values in demos for better re-ranking. Override via `scripts/scrape_and_chunk.py --chunk_size 500 --chunk_overlap 80`.

- **E5 prompt formatting**: We prefix chunks with "passage: " when indexing and queries with "query: " at retrieval (shown in code refs below).
- **Score normalization**: Before fusion, we apply min–max normalization separately to dense and sparse score lists; then fuse via `final = α * dense_norm + (1 − α) * sparse_norm`.

### Step 4: Tier 2 Retriever – Re-Ranker (Precision Layer)

**Design Rationale**: Cross-encoder re-ranking significantly boosts precision by modeling query–chunk interactions, complementing the hybrid retriever's recall-focused approach.

- **Dense + Cross-Encoder Re-Ranker — adopted**:
  - Retrieve top N (e.g., 20–50) with the hybrid retriever, then re-rank with a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - Keep top k (e.g., 3–5) for generation; significantly boosts precision by modeling query–chunk interactions.

- **Speed/Accuracy dials**:
  - `α`, `k_dense`, `k_sparse`, `N_rerank`, `k_final`, and on/off for re-ranker.
  - Use hybrid-only for faster responses; enable re-ranking for highest answer quality.

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence_transformers.CrossEncoder`).
- **Input**: top_k from Tier 1 (e.g., 15); score pairs (query, chunk).
- **Keep**: top_n (e.g., 3–5) by cross-encoder score.

Auto-activation policy (default):

```python
# Enabled when any condition is true
(tokens > 10) or            # long/complex query
domain_hit or               # category in {legal, compliance}
has_kw or                   # query contains legal/compliance keywords
(max_fused < 0.35) or       # low confidence peak
((max_fused - median) < 0.08)  # flat fused distribution
```

Cache behavior:

- Query embeddings are cached at `index/query_cache.pkl` keyed by `(model, query)`.
- Run twice with the same query and `verbose=True` to see `[cache] hit`/`save` messages.

**Why this tiered combo**:
- Hybrid maximizes recall across lexical and semantic intents; the re-ranker maximizes precision.
- Together they provide SOTA-leaning retrieval with explicit tradeoffs suitable for Colab-scale demos.

### Step 5: LLM Generation

- **Goal**: Generate final answers grounded in retrieved context.
- **Models**: `mistralai/Mistral-7B-Instruct-v0.2` or `HuggingFaceH4/zephyr-7b-alpha` (via `transformers` or vLLM).

Prompt template:

```text
Context:
{retrieved_docs}

Question:
{user_query}

Answer:
```

Display both retrieved snippets and the generated answer.

Runner (Steps 5–6):

```bash
python scripts/main.py --queries data/queries.json
```

The runner:
- Uses `retrieve_auto` to decide Tier 2 activation
- Builds context(s) and generates with the chosen LLM
- Prints a JSON with Tier 1 and Tier 2 answers

### Pretty Print and Timings

- Enable a compact console view with citations and URLs plus per-stage timings:

```bash
python scripts/main.py --queries data/queries.json --pretty
```

Output includes:
- Tier1 and Tier2 rows: `[source#chunk] score=...` and `URL`
- Timings (ms) for dense, sparse, fusion, rerank, total
- Dense/sparse breakdown per result in pretty mode

### Low-Confidence Guardrail

- Retrieval marks a query as low-confidence if fused scores are too low/flat or too few results exceed a small threshold.
- When `--pretty` is used, reasons are shown. By default, generation is suppressed in low-confidence cases.
- Override with `--allow_low_conf_gen` to force generation.

### Retrieval Evaluation (Recall@k, MRR)

Use the evaluation script to compare Tier1 vs Tier2 on a labeled set.

Inputs:
- Queries JSON: `{ "queries": [{ "id": "Q1", "question": "..." }, ...] }`
- Labels CSV: columns `query_id,relevant_doc_ids` where `relevant_doc_ids` is a comma-separated list of integer `chunk_index` values.

Run:

```bash
python scripts/eval_retrieval.py \
  --queries_json data/queries.json \
  --labels_csv data/labels.csv \
  --k 10 \
  --chart
```

This prints aggregate `Recall@k` and `MRR@k` for Tier1 and Tier2 and per-query breakdown.

### Alpha Sweeper (choose α for fusion)

Run a small grid of α values to find the best tradeoff for your dataset:

```bash
python scripts/alpha_sweeper.py \
  --queries_json data/queries.json \
  --labels_csv data/labels.csv \
  --k 10 \
  --alphas 0.5,0.65,0.7,0.8 \
  --chart
```

This prints a table of metrics per α and the best α by Recall@k/MRR@k, with an optional chart.

Notes on α selection:
- Alpha sweeper (dataset-level): simple and robust when you have a small eval set.
- Query-aware α (per-query): can further improve retrieval by adapting to query style (digits/symbols/keywords vs natural language), but is more complex and can overfit small eval sets. We use the sweeper by default for clarity and reproducibility, and may add query-aware α later as an optional mode.

### Step 6: Evaluation & Demonstration

- **Goal**: Show that Tier 2 improves quality.
- Use the sample queries in `data/queries.json` (see "Demo Queries" above) or your own set.
- Compare recall/precision and note latency when enabling the re-ranker.
