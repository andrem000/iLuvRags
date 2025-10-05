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

### Ingestion Quickstart (Local or Colab)

1. Install deps:
   - Local: `pip install -r requirements.txt`
   - Colab: `!pip install -q -r requirements.txt`
2. Produce `data/chunks.jsonl` (your own pipeline or use the Colab bootstrap below).
3. Proceed to Step 2 to build the index from `data/chunks.jsonl`.

### Colab Bootstrap (Single PDF Demo)

Use this to demo with Berkshire 2024 report without a custom ingestion script. It creates `data/chunks.jsonl`, builds the index, and runs the main pipeline.

```python
# 1) Clone, install
REPO_URL = "https://github.com/<your-user>/iLuvRags.git"  # replace with your repo URL
!git clone -q $REPO_URL
%cd iLuvRags
!pip install -q -r requirements.txt

# 2) Create chunks.jsonl from a public PDF (Berkshire 2024)
import os, re, json, requests
from io import BytesIO
from pypdf import PdfReader

os.makedirs('data', exist_ok=True)
url = 'https://www.berkshirehathaway.com/2024ar/2024ar.pdf'
pdf_bytes = requests.get(url, timeout=60).content
reader = PdfReader(BytesIO(pdf_bytes))
text_pages = []
for p in reader.pages:
    try:
        t = p.extract_text() or ''
    except Exception:
        t = ''
    text_pages.append(t)
full_text = "\n\n".join(text_pages)
full_text = re.sub(r"\s+", " ", full_text).strip()

def split_words(t, chunk_size=500, overlap=80):
    words = t.split(' ')
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

chunks = split_words(full_text, 500, 80)
with open('data/chunks.jsonl', 'w', encoding='utf-8') as f:
    for i, c in enumerate(chunks):
        rec = {
            'source': 'berkshire_annual_report_2024',
            'category': 'finance',
            'url': url,
            'doc_mime': 'application/pdf',
            'chunk_index': i,
            'text': c,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

# 3) Build index
!python scripts/build_index.py --chunks data/chunks.jsonl --out index --model intfloat/e5-base-v2 --batch_size 64

# 4) Run main (Steps 5–6)
!python scripts/main.py --index index --embed_model intfloat/e5-base-v2 --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --llm Qwen/Qwen2-7B-Instruct --query "Summarize Item 1A – Risk Factors" --device cpu
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

Minimal sketch (to be implemented next):

```python
# 1) Query embed with E5 using "query: <text>"
# 2) FAISS search -> dense_scores, idxs
# 3) BM25 over all texts -> bm25_scores for same idxs
# 4) Normalize both to [0,1], then fuse with alpha
# 5) Take top 10–15 by fused score
```

Why: Hybrid retrieval captures both meaning (e.g., "contract termination") and exact terms (e.g., "article 45").

Programmatic use:

```python
from scripts.retriever import HybridRetriever
retriever = HybridRetriever(index_dir="index", embed_model_name="intfloat/e5-base-v2", alpha=0.7, verbose=True)
results = retriever.retrieve("Summarize Item 1A – Risk Factors", k_dense=30, k_sparse=30, k_final=15)
for r in results[:3]:
    print(r["score"], r["metadata"], r["text"][:200])

# Build LLM context
context = retriever.build_context(results, max_chars=3500)
print(context[:500])
```

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

Programmatic API:

```python
from scripts.retriever import HybridRetriever
retriever = HybridRetriever(index_dir="index", verbose=True)
out = retriever.retrieve_auto("Summarize Item 1A – Risk Factors", k_final=15, top_n_rerank=5)
print(out["tier2_activated"])  # True/False
print(len(out["tier1"]), len(out["tier2"]))
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

### Step 6: Evaluation & Demonstration

- **Goal**: Show that Tier 2 improves quality.
- Use the sample queries in `data/queries.json` (see "Demo Queries" above) or your own set.
- Compare recall/precision and note latency when enabling the re-ranker.
