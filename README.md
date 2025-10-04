# iLuvRags

### Step-by-Step Implementation Plan (Index)
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

### Design Choices: Retriever Architecture

We evaluated three retriever setups and chose a tiered combination of hybrid retrieval and cross-encoder re-ranking to balance speed and accuracy, especially for real-world, mixed query types and a Google Colab demo.

- **Baseline Dense Retriever — discarded**:
  - Strong semantic matching but weaker on keyword-/symbol-heavy queries and rare entities.
  - No lexical recall path; precision lags without re-ranking; fewer explicit speed/quality dials.

- **Hybrid Retriever (dense + sparse) — adopted**:
  - Dense embeddings (e.g., `intfloat/e5-base-v2`, `sentence-transformers/all-MiniLM-L6-v2`) + BM25.
  - Score fusion: `final_score = α * dense + (1−α) * sparse` (tunable `α`).
  - Handles both semantic and exact-term queries; improves recall and robustness.

- **Dense + Cross-Encoder Re-Ranker — adopted**:
  - Retrieve top N (e.g., 20–50) with the hybrid retriever, then re-rank with a cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
  - Keep top k (e.g., 3–5) for generation; significantly boosts precision by modeling query–chunk interactions.

- **Speed/Accuracy dials**:
  - `α`, `k_dense`, `k_sparse`, `N_rerank`, `k_final`, and on/off for re-ranker.
  - Use hybrid-only for faster responses; enable re-ranking for highest answer quality.

- **Why this tiered combo**:
  - Hybrid maximizes recall across lexical and semantic intents; the re-ranker maximizes precision.
  - Together they provide SOTA-leaning retrieval with explicit tradeoffs suitable for Colab-scale demos.

### Ingestion Quickstart (Local or Colab)

1. Install deps:
   - Local: `pip install -r requirements.txt`
   - Colab: `!pip install -q -r requirements.txt`
2. Seed sources in `sources.yaml` (provided with one per category).
3. Run the scraper/chunker:
   - `python scrape_and_chunk.py --sources sources.yaml --out data/chunks.jsonl`
4. Output: `data/chunks.jsonl` with chunked text and provenance.

### Demo Queries

Example Tier A/B queries for the Berkshire 2024 report are in `queries.json`.
Load and iterate:

```python
import json
with open('queries.json', 'r', encoding='utf-8') as f:
    q = json.load(f)["queries"]
for item in q:
    print(item["tier"], item["id"], item["question"])
```

### Step 2: Embeddings & Storage

- **Embedding model**: `intfloat/e5-base-v2`
- **Vector store**: FAISS (cosine similarity via inner product on normalized vectors)

Build the index:

```bash
pip install -r requirements.txt
# For CUDA support (faster on GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python scripts/build_index.py --chunks data/chunks.jsonl --out index --model intfloat/e5-base-v2 --batch_size 64 --device cuda
# devices: auto | cpu | cuda | cuda:0 | mps
```

Artifacts:
- `index/faiss.index`
- `index/embeddings.npy`
- `index/metadata.json`

### Step 3: Tier 1 Retriever – Hybrid

- **Goal**: Combine dense semantic retrieval with keyword-based recall.
- **Dense**: Use FAISS to get top_k (e.g., 20).
- **Sparse**: Use BM25 (e.g., `rank-bm25`) on raw text.
- **Fusion**: Normalize and merge scores: \( final\_score = \alpha \cdot dense + (1-\alpha) \cdot bm25 \).

Minimal sketch (to be implemented next):

```python
# 1) Query embed with E5 using "query: <text>"
# 2) FAISS search -> dense_scores, idxs
# 3) BM25 over all texts -> bm25_scores for same idxs
# 4) Normalize both to [0,1], then fuse with alpha
# 5) Take top 10–15 by fused score
```

Why: Hybrid retrieval captures both meaning (e.g., "contract termination") and exact terms (e.g., "article 45").

### Step 4: Tier 2 Retriever – Re-Ranker (Precision Layer)

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence_transformers.CrossEncoder`).
- **Input**: top_k from Tier 1 (e.g., 15); score pairs (query, chunk).
- **Keep**: top_n (e.g., 3–5) by cross-encoder score.

Simple activation logic:

```python
use_reranker = len(query.split()) > 8 or ("legal" in query.lower())
```

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

### Step 6: Evaluation & Demonstration

- **Goal**: Show that Tier 2 improves quality.
- Use the sample queries in `data/queries.json` (see "Demo Queries" above) or your own set.
- Compare recall/precision and note latency when enabling the re-ranker.
