# Day 4 — Store (Vector Database)
### RAG Pipeline · Lesson 4 of 9

---

## Where this sits in the pipeline

```
                          YOU ARE HERE
                               ↓
Parse → Chunk → Embed → [ Store ] → Retrieve → Rerank → Generate
└────────── ingestion (build once) ──────────┘   └──── query ────┘
```

Last box of the **ingestion** side. After today the documents are fully processed
and sitting in a real, persistent database. Everything from Day 5 on is the
*query* side.

---

## 1. Why a database at all (you already did retrieval by hand)

Day 3's by-hand loop *was* retrieval — embed, cosine, pick closest. A vector
database adds three things that loop doesn't have:

1. **ANN indexing** — fast approximate search that survives a million vectors
2. **Persistence** — Day 3's vectors vanished when the script ended; a database
   saves to disk, so you embed once and query forever (the two-phase payoff)
3. **Bundling** — vector + original text + metadata glued together, so retrieval
   hands back the *text* (for the LLM) and the *source* (for citations)

### Brute-force vs ANN (recap)
- Brute-force compares the query against **every** vector → cost scales linearly
  → 1M chunks = 1M comparisons per query → collapses under real traffic
- **ANN** pre-organizes vectors so each query checks only a promising subset.
  Trades slight accuracy for ~100× speed. Fine for RAG: *the LLM needs enough
  good chunks, not the perfect ranking.*

### The ANN risk (important nuance)
ANN *skips* most vectors. If the true-best sits in a region the search didn't
traverse, it's never compared — the index confidently returns second-best as
first.

> **ANN's approximation risk is proportional to how concentrated the answer is.**
> - Broad question (answer spread over many chunks) → tolerant. Missing #1 is fine.
> - Needle-in-haystack (answer in exactly ONE chunk) → **dangerous.** Miss it and
>   the answer isn't in the context at all → "not found" or hallucination.
>
> Own data: the *interest rate* query (chunk 1 at 0.611, everything else ≤0.135)
> is the fragile kind. The *"agreement ends"* query (three plausible chunks) is
> the tolerant kind. → Another argument for **hybrid search (Day 6)**.

---

## 2. Why ChromaDB (decision)

Runs locally with zero server setup (a Python library + a folder on disk), uses
**HNSW** indexing under the hood (real ANN, not a toy), and bundles
vector+text+metadata natively. Right *learning* choice, fine for small production.

**Its limit:** not built for hundreds of millions of vectors or heavy concurrent
load — that's when you reach for Qdrant, Pinecone, or pgvector. Never reach for
the heavy tool before understanding the light one.

```bash
pip install chromadb
```

---

## 3. Code

`src/store/vectordb.py`:
```python
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# PersistentClient = saves to disk. (plain chromadb.Client() = RAM only, vanishes)
client = chromadb.PersistentClient(path="data/chroma_db")

# A "collection" is like a table — one logical group of documents.
collection = client.get_or_create_collection(name="contract")


def ingest(chunks):
    vectors = model.encode(chunks).tolist()      # ← EMBED step (Day 3 box)

    collection.add(                               # ← STORE step (Day 4 box)
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": "contract.txt", "section": i}
                   for i in range(len(chunks))]
    )


def search(question, n_results=2):
    q_vec = model.encode(question).tolist()       # same model → same space
    return collection.query(query_embeddings=[q_vec], n_results=n_results)


if __name__ == "__main__":
    with open("data/processed/contract.txt", encoding="utf-8") as f:
        text = f.read()
    chunks = [s.strip() for s in text.split("\n\n") if s.strip()]
    ingest(chunks)
    print(f"Collection now contains: {collection.count()} items")
```

### Embed vs Store — two boxes, two lines
```
Chunk → [ Embed ]         → [ Store ]
         model.encode()      collection.add()
         text → vectors      vectors + text + metadata → disk
```
By the time `collection.add()` runs, the vectors **already exist** — note the
parameter is `embeddings=vectors`, i.e. handing over pre-made vectors. ChromaDB
doesn't create them; it receives them.

> **ChromaDB *can* embed for you** (pass only `documents=`, no `embeddings=`) —
> it runs its own default model silently. **Don't.** Passing embeddings explicitly
> means *you* choose the model, you know which one ran, and you can swap it to
> compare. Let the DB pick silently and you've lost control of the most important
> decision in the pipeline.

### The four parallel lists
Position *i* in each list describes the same chunk:

| | ids | embeddings | documents | metadatas |
|---|---|---|---|---|
| **0** | `"chunk_0"` | `[0.018, 0.077, …]` | `"TERMINATION FOR…"` | `{source, section: 0}` |
| **1** | `"chunk_1"` | `[…]` | `"PAYMENT TERMS…"` | `{section: 1}` |

- **embeddings** → what you *search against*
- **documents** → the original **text** sent to the LLM (*numbers find, text generates*)
- **metadatas** → provenance → **citations**. Free if stored now; impossible to
  add later without re-ingesting everything
- **ids** → the unique key that governs overwrite behaviour

### Reading the chunking line precisely
```python
chunks = [s.strip() for s in text.split("\n\n") if s.strip()]
```
- `split("\n\n")` = the **chunking** (Day 2 structure-aware decision)
- `.strip()` = cleanup of edge whitespace (housekeeping)
- `if s.strip()` = drops empty chunks

---

## 4. Experiment 1 — re-ingestion (run the script twice)

| Run | `collection.count()` |
|---|---|
| 1st | 4 |
| 2nd | **4** (not 8) |

**Result: ChromaDB overwrote, it did not duplicate.** Same ID → replace. This is
an *upsert* pattern, so re-ingesting a document is idempotent and safe.

> **The trap hiding in the ID scheme:** `chunk_0…chunk_3` is only unique *within
> one document*. Ingest a second file (`lease.txt`) and its `chunk_0` **silently
> erases** the contract's `chunk_0`.
> **Fix:** scope IDs to the source — `contract.txt_chunk_0` — or hash the content.
> ID design determines whether re-ingestion is safe *and* whether documents can
> coexist.

Note what didn't need thinking about: ChromaDB built the HNSW index automatically.
Same code, unchanged, handles a million vectors. That's what the database bought.

---

## 5. Experiment 2 — query the DB, verify against Day 3

Question: *"How much notice to cancel the contract?"*

```
distance: 0.727 | section: 0  → TERMINATION FOR CONVENIENCE…
distance: 1.326 | section: 2  → CONFIDENTIALITY…
```

**Same ranking as Day 3's by-hand loop** (section 0 first, section 2 second).
Ingestion pipeline verified end to end.

### Distance vs similarity — the inversion

| | Day 3 (by hand) | Day 4 (ChromaDB) |
|---|---|---|
| Section 0 | 0.637 **similarity** | 0.727 **distance** |
| Section 2 | 0.517 **similarity** | 1.326 **distance** |
| Winner is | **highest** number | **lowest** number |

They're the same measurement read from opposite ends of the ruler:

```
similarity:  0 ──────────────── 1        distance:  1 ──────────────── 0
             unrelated   identical                  far apart    same spot
```

`distance = 1 − similarity` — **but only for *cosine* distance.** ChromaDB's
default is **squared L2 (Euclidean)**: a different formula, unbounded (hence
1.326 > 1), no simple `1 − x` relationship. The scale differs; the *direction*
never does — **smaller distance = closer = better, always.**

To force cosine: `client.get_or_create_collection(name="contract",
metadata={"hnsw:space": "cosine"})` — changes the index's distance function, so
re-ingest after.

*Footnote:* for **normalized** vectors (what sentence-transformers usually
produces) cosine and L2 rank identically — so matching Day 3's order wasn't luck.
The metrics disagree on numbers, agree on order.

> ⚠️ **Never threshold on a raw score without knowing which metric produced it.**
> `if score > 0.5: match` is *exactly backwards* on distance — accepting the worst
> results and rejecting the best. Ask first: similarity or distance, which formula?

---

## 6. Decision note

> **Problem:** Day 3's vectors lived in RAM and died with the script, and a Python
> loop over every vector doesn't survive a real corpus. Vectors need persistent,
> fast, searchable storage bundled with their text and provenance.
>
> **Decision:** ChromaDB with `PersistentClient` over Qdrant/Pinecone/pgvector —
> zero server setup, real HNSW/ANN indexing, native vector+text+metadata bundling.
> Pass `embeddings` explicitly rather than letting Chroma embed silently, so the
> model choice stays mine and stays visible.
>
> **Why:** Persistence delivers the two-phase payoff (embed once, query forever).
> Explicit embeddings preserve the ability to swap and compare models — the most
> important decision in the pipeline. Storing metadata at ingestion is the only
> chance to get citations; it cannot be added later without full re-ingestion.
>
> **The number:** 4 chunks stored; re-running yields 4, not 8 → re-ingestion is
> idempotent (same-ID upsert). Query reproduced Day 3's ranking exactly
> (section 0 then section 2), confirming the pipeline end to end.

---

## What I own after Day 4
- Why a vector DB beats a Python loop: ANN + persistence + bundling
- ANN's risk scales with how **concentrated** the answer is
- Embed (`model.encode`) and Store (`collection.add`) are separate boxes
- Always pass embeddings explicitly — never let the DB choose the model silently
- The four parallel lists, and what each column is *for*
- **ID design** governs idempotency and document coexistence
- **Distance ≠ similarity** — lower wins; never threshold without knowing the metric
- The ingestion half of RAG is complete

## Pre-question for Day 5 (Evaluate / Retrieve)
> So far every result has been judged by eye, one query at a time — "that looks
> right." That doesn't scale and isn't evidence. **How would you prove, with a
> number, that this retrieval system works — across 50 questions instead of 3?**
> What would you have to build first, before you could score anything?
