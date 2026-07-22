# Day 3 — Embeddings
### RAG Pipeline · Lesson 3 of 9

---

## Where this sits in the pipeline

```
                YOU ARE HERE
                     ↓
Parse → Chunk → [ Embed ] → Store → Retrieve → Rerank → Generate
```

Day 2 produced 4 clean chunks — still just *strings*. Today they become
**vectors**, and the moment they do, you can search by meaning. This is the
core trick of RAG turning on.

---

## 1. How a chunk becomes a vector (the mechanism)

Inside the embedding model, three steps:

```
"Either party may terminate..."
        ↓ tokenize            → split into sub-word tokens → IDs
[101, 2438, 2283, 6402, ...]
        ↓ transformer         → one vector PER token (e.g. 50 vectors)
[[0.1,...], [0.4,...], ... ]
        ↓ mean pooling        → AVERAGE all token-vectors into ONE
[0.018, 0.077, ..., 384 numbers]   ← THE chunk embedding
```

- **Tokenize:** chunk → sub-word tokens → numeric IDs.
- **Transformer:** produces one vector per token (each token *in context*).
- **Pooling:** averages those token-vectors into a **single** fixed-length vector.
  Mean pooling is the most common.

**Key link to Day 2:** *pooling is where the "smoothie" gets made.* It averages
all token meanings together. A focused chunk → average lands in a specific spot.
A sprawling chunk → average lands nowhere useful. **This is the mechanism behind
everything about chunk size.**

> The 50 token-vectors don't disappear — mean pooling **averages** them into 1.
> That single vector is what gets stored.

---

## 2. Embedding model ≠ LLM (locked in)

> An embedding model is **not** an LLM. Both are transformers; that's where the
> similarity ends.

| | **Embedding model** | **LLM** |
|---|---|---|
| Output | one fixed vector (e.g. 384 numbers) | text (next token, repeatedly) |
| Trained to | put similar meanings close (contrastive — CLIP) | predict next token |
| Size | small (~100–600M) | large (8B–500B) |
| Box | Embed (Day 3) | Generate (Day 8) |
| Runs | every chunk + every query | once per question |

The **dimension** (384 here) is a property of the *model* and a **cost knob** —
bigger = more RAM, slower search, more money, usually only marginally better recall.

---

## 3. Setup + code

```bash
pip install sentence-transformers
```

`src/ingest/embed.py`:
```python
from sentence_transformers import SentenceTransformer

# Fully-qualified HF path — scales to non-default orgs (BAAI, intfloat) and
# self-documents for the next reader. 384-dim, ~80MB, fits 6GB VRAM easily.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_chunks(chunks):
    return model.encode(chunks)   # tokenize → transformer → pool, all inside


if __name__ == "__main__":
    with open("data/processed/contract.txt", encoding="utf-8") as f:
        text = f.read()
    chunks = [s.strip() for s in text.split("\n\n") if s.strip()]

    vectors = embed_chunks(chunks)
    print(f"Number of chunks: {len(chunks)}")
    print(f"Shape: {vectors.shape}")     # (4, 384) → 4 chunks, 384 numbers each
```

**Result:** `Shape: (4, 384)` — 4 chunks, each represented by 384 numbers.
The dimension is fixed: a 10-word chunk and a 200-word chunk both come out as
384 numbers, because pooling averages to a fixed length.

**Decision-note line:** use fully-qualified HF model paths (`org/model`) — the
short form only resolves for the default `sentence-transformers` org.

---

## 4. Retrieval by hand (the whole point)

```python
from sentence_transformers import util

# Compare the QUESTION to each chunk
question = "How much notice to cancel the contract?"
q_vec = model.encode(question)          # same model → same space → comparable
for i in range(len(chunks)):
    score = util.cos_sim(q_vec, vectors[i]).item()
    print(f"vs chunk {i}: {score:.3f}")
```

**Cosine similarity** measures how aligned two vectors are in meaning-space:
1.0 = same direction/meaning, 0 = unrelated, negative = opposite.

The flow, in plain words:
> **Embed the question with the SAME model, then measure its distance to each
> chunk. The closest chunk wins.**

That *is* retrieval. This by-hand loop is exactly what a vector database does —
just slower. Two-phase link: chunks embedded **once at ingestion**; the question
embedded **fresh at query time**; same model both times.

---

## 5. The experiment — three queries, reading the *shape* of results

Chunks: 0=TERMINATION, 1=PAYMENT, 2=CONFIDENTIALITY, 3=GOVERNING LAW

| Query | Winner | Runner-up | Gap | Verdict |
|---|---|---|---|---|
| "interest rate for late payment" | **1** → 0.611 | 0.135 | **0.476** | crystal clear |
| "notice to cancel the contract" | **0** → 0.637 | 0.337 | 0.300 | clear-ish |
| "what happens when the agreement ends" | **0** → 0.618 | **0.517** | **0.101** | **muddy** |

### The headline result (query 1 of the set)
Question *"how much notice to cancel"* → **chunk 0 (termination) won 0.637**,
despite the word **"cancel" appearing nowhere** in that chunk (it says
"terminate"). **Keyword search would return nothing; meaning-search nailed it.**
That is the entire premise of RAG, proven on own data.

### Reading the numbers like an engineer
- **The gap matters more than the absolute score.** A clear winner *separates*
  from the field. There is no universal "0.5 = match" line.
- **Don't compare scores across different queries** — cosine measures
  meaning-alignment, not word-count. More word overlap doesn't add points
  mechanically (query 2's 0.611 was slightly *below* query 1's 0.637 despite more
  shared words).
- **chunk 0 vs chunk 2 = 0.537** (highest chunk-to-chunk pair): correct, because
  the confidentiality chunk literally says obligations "survive the **termination**
  of this Agreement." The model is being *right*, not buggy.

### The muddy case (deliberately engineered)
*"What happens when the agreement ends?"* → winner 0.618, runner-up 0.517,
**gap only 0.101**. Three chunks all plausibly relevant.

- A **narrow gap is a warning sign**: the fine-grained ranking is *fragile* —
  within noise, a rephrase could flip 1st and 2nd. You can't trust the precise
  order when scores bunch up.
- Here the muddiness is arguably *correct* (the answer really spans termination +
  what survives it) — but vector search can't tell "correctly muddy" from
  "wrongly muddy." It just reports close scores.
- **This is why the reranker (Day 7) exists:** vector search is *fast but coarse*
  — good at "these 5 are roughly relevant," bad at precise ordering among close
  contenders. A cross-encoder reads question + chunk *together* to break ties.

---

## 6. Decision note

> **Problem:** Chunks are text; to search by meaning they must become vectors,
> and we need to confirm meaning-search actually beats keyword-search.
>
> **Decision:** Embed chunks with `all-MiniLM-L6-v2` (384-dim, small, fits 6GB
> VRAM), embed the query with the same model, rank by cosine similarity. Validate
> by reading the *gap* between winner and field, not the absolute score.
>
> **Why:** A query with zero word-overlap ("cancel" vs "terminate") must still
> retrieve the right chunk — only meaning-based vectors do that. Reading the gap
> (not a fixed threshold) diagnoses clean vs muddy retrieval; muddy fields are
> what reranking later fixes.
>
> **The number:** "Notice to cancel" → chunk 0 wins 0.637 vs 0.337 runner-up,
> zero shared words — meaning beat keyword, proven. Muddy query → gap collapses to
> 0.101, foreshadowing the need for reranking.

---

## What I own after Day 3
- The tokenize → transformer → **mean-pool** mechanism (and why pooling ties to
  chunk size)
- Embedding model vs LLM: different output, training, size, role
- Dimension = model property = cost knob
- Retrieval by hand: embed query (same model) → cosine vs each chunk → closest wins
- Reading result *shape*: clean winner (wide gap) vs muddy field (narrow gap)
- Why a narrow gap is a warning, and why it foreshadows reranking (Day 7)
- The eval mindset: design queries, predict the shape, diagnose the output

## Bridge to Day 4 (Store) — resolved

**Question:** Comparing the query against 4 chunks with a Python loop is fine.
Why can't you run that same loop over a **million** vectors on every query — and
what does a vector database do instead?

**Answer:** The by-hand loop is **brute-force search** — it compares the query
against *every* stored vector. Cost grows **linearly** with database size:

- 4 chunks → 4 comparisons (instant)
- 1M chunks → 1M comparisons **per query**
- 100 users at once → 100M comparisons/minute → server collapses

That linear scaling is the mechanism behind "time-consuming + expensive."

A vector database uses **ANN (Approximate Nearest Neighbor)** instead: it
pre-organizes the vectors so each query only checks a *promising subset* (a few
thousand), skipping the rest. It trades a little accuracy (may miss the odd
true-closest vector) for a huge speed gain (100×+). **Acceptable for RAG because
the LLM only needs *enough* good chunks, not the perfect ranking.**

> The by-hand loop *is* brute-force. It works only because there are 4 chunks.
> At a million, you need the database's ANN index.

**Two ANN index families to know (Day 4):**
- **HNSW** — a navigable graph (ChromaDB, Pinecone, Qdrant default)
- **IVF** — clustering into buckets (common in FAISS)

**The catch that ties back to Day 3's muddy field:** ANN is *approximate*, so it
can miss the true closest vector. When two chunks are separated by only ~0.1
(the muddy case), an approximate index deciding between them can return the
*second*-best as if it were best — the fragile-ranking risk, now baked into the
index itself, not just the scores.
