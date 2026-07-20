# Day 2 — Document Ingestion: Chunking
### RAG Pipeline · Lesson 2 of 9

---

## Where this sits in the pipeline

```
        YOU ARE HERE
             ↓
Parse → [ Chunk ] → Embed → Store → Retrieve → Rerank → Generate
└──────── ingestion (build once) ────────┘   └──── query (every question) ────┘
```

Day 1 turned a PDF into one big string. Today we cut that string into pieces
(**chunks**) — because one vector can't represent a whole document (the smoothie
problem from Day 1). Chunking has nothing to do with PDFs; it's pure string work,
so we practice on a plain `.txt` file that stands in for "already-extracted text."

```
Day 1:  PDF → [extract] → big string
Day 2:  big string → [chunk] → pieces      ← we start here
```

---

## 1. What a chunk is

A **chunk** is a slice of text you cut a document into before embedding.
One chunk → one vector.

> **1 document → many chunks → many vectors**

---

## 2. The core tension (from Day 1's theory)

| Too **big** | Too **small** |
|---|---|
| Vector averages many topics → mush, matches nothing well | Vector is sharp, but chunk lacks context |
| Dumps too much on the LLM | Severs facts across boundaries → LLM guesses → hallucinates |
| **Fails at matching** | **Fails at answering** |

**The rule:** *A chunk should be big enough to carry a complete thought, and
small enough to be about one thing.*

---

## 3. The test document

`make_test_doc.py` writes a 4-section fake contract (938 chars) to
`data/processed/contract.txt`: TERMINATION, PAYMENT TERMS, CONFIDENTIALITY,
GOVERNING LAW. Small and evenly-sized — which matters (see the honest catch
at the end).

---

## 4. Three chunkers, built and compared

### `src/ingest/chunk.py`

```python
def chunk_fixed(text, chunk_size, overlap=0):
    """Cut text into pieces of `chunk_size` chars.
    `overlap` = how many chars each piece repeats from the previous one.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap          # next piece starts BEFORE this end → overlap
    return chunks


def chunk_by_section(text):
    """Cut at blank lines — one chunk per section."""
    sections = text.split("\n\n")
    return [s.strip() for s in sections if s.strip()]
```

### Method 1 — Fixed-size, no overlap (chunk_size=200)
Produced **5 chunks**. Failures seen in real output:
- Sliced a **word**: chunk 0 ended `"...shall speci"`, chunk 1 began `"fy..."`
- Severed a **fact**: `"one and one-half percent"` in chunk 1, `"(1.5%)"` in chunk 2
  → a question about the interest rate could retrieve only half the answer
- **Blended topics**: the CONFIDENTIALITY heading landed mid-chunk with payment text

### Method 2 — Fixed-size + overlap (200, overlap=50)
Produced **7 chunks**. Overlap = each piece re-includes the tail of the previous
one, so a sliced thought survives **whole** in the next chunk. It healed the
`speci|fy` break and reunited the interest rate. New problems it *created*:
- **Duplication** — same sentences now stored in two chunks (bigger index)
- **Junk fragment** — a 38-char scrap: `"ved through arbitration in Phnom Penh."`
- Does **nothing** about blended topics — still cutting blindly every 200 chars

### Method 3 — Structure-aware (`split("\n\n")`)  ← today's choice
Produced **4 clean chunks**, each one a complete section, heading with its body:
- No sliced words, no severed facts, no blended topics, no junk fragments
- Each vector is **focused** (chunk 0 = "termination", chunk 1 = "payment", …)
- Cost: chunk sizes now vary (202–255 chars) because section length is set by the
  document's author, not by us

---

## 5. The tradeoff table (reasoned from the output)

| Method | Cuts land... | Chunk size | Main problem |
|---|---|---|---|
| Fixed-size | anywhere (blind) | consistent | slices words / facts / topics |
| + overlap | anywhere, seams healed | consistent | duplication, junk fragments |
| Structure-aware | at section boundaries | **inconsistent** | giant + tiny sections |
| **Hybrid** (structure → fixed fallback) | clean, then size-capped | controlled | more complex to build |

**Hybrid** = cut at sections first (clean topics), then fixed-size-split any
section too big to embed cleanly. This is what production splitters
(e.g. LangChain `RecursiveCharacterTextSplitter`) do internally — and now the
*why* is clear, not just the function call.

---

## 6. The honest catch

Structure-aware looks perfect **only because the test contract has evenly-sized
sections.** On a real document with an 8,000-char DEFINITIONS section and a
60-char NOTICES section, `split("\n\n")` yields one mush-vector monster and one
near-empty chunk — the exact failure the size rule warns about. So structure-aware
wins *here*, but production needs the hybrid.

---

## 7. Decision note (corrected)

> **Problem:** Documents must be cut into chunks before embedding. Cut wrong and
> you either **slice facts across boundaries** (LLM hallucinates the missing half)
> or **blend topics** into one mushy vector that matches nothing well.
>
> **Decision:** Built three chunkers — fixed-size, fixed+overlap, and
> structure-aware (`split("\n\n")`). Chose **structure-aware** for documents with
> clear sections: it keeps each topic whole → focused vectors → better retrieval,
> with no slicing and no overlap needed. For production, wrap it as a **hybrid** —
> cut by section first, then apply fixed-size splitting to any section too large
> to embed cleanly.
>
> **Why:** Fixed-size slices words (`speci|fy`) and severs facts (interest rate
> split across chunks). Overlap heals seams but duplicates text and creates junk
> fragments. Structure-aware avoids both, at the cost of inconsistent chunk sizes
> when sections vary widely.
>
> **The number:** Fixed = 5 chunks (2 severed facts). Overlap = 7 chunks (1 junk
> 38-char fragment). Section = 4 clean chunks (0 severed). Section sizes 202–255
> chars — acceptable here; would fail on a document with an 8,000-char section.

**Note-writing lesson (carried from Day 1):** every box needs the *because*, not
the label. And name the actual choice — "structure-aware," not just "hybrid" —
so the note records what you did, not just the end-state idea.

---

## What I own after Day 2
- A chunk = a slice of text; 1 doc → many chunks → many vectors
- The size tension: too big = mush/matching-fail; too small = severed/answering-fail
- Overlap re-includes the previous tail → heals sliced thoughts, at the cost of
  duplication + junk fragments
- Structure-aware cutting → focused vectors, but inconsistent sizes
- Hybrid (structure → fixed fallback) = the production answer, and *why* it exists
- Saw all three failure modes in real output, not just in theory

## Bridge to Day 3 (Embeddings) — resolved

**Question:** Your 4 clean chunks are still strings. To search by meaning, each
must become a vector. Which model turns a chunk into a vector — and is it the
same kind of model as the LLM that writes the final answer?

**Answer:** The **embedding model** turns a chunk into a vector. It is **not** the
same model — and not even the same *kind* of model — as the LLM that writes the
final answer.

> ⚠️ An embedding model is **not an LLM.** Both are transformers, but that's where
> the similarity ends. Do not call the embedder an "LLM" — that's the exact
> confusion to avoid.

| | **Embedding model** | **LLM** |
|---|---|---|
| Job | chunk/query → **one vector** | prompt → **next word, repeatedly** |
| Output | fixed list of numbers (e.g. 1024) | text |
| Trained to | put similar meanings close (contrastive — CLIP) | predict the next token |
| Size | small (~100–600M) | large (8B–500B) |
| In the pipeline | the **Embed** box (Day 3) | the **Generate** box (Day 8) |
| Runs | on every chunk + every query | once per question |

**Why it matters (neWwave):** someone will suggest "just use GPT for embeddings
too." It doesn't work — GPT outputs *text*, not a fixed vector you can run
nearest-neighbor search on. Knowing which model does which job is the difference
between a working pipeline and a fundamentally misconfigured one.

**Precise sentence to keep:** *"The embedding model turns a chunk into a vector.
It is not the same model, nor the same kind of model, as the LLM that writes the
final answer — different architecture, different training objective, different
output shape."*

→ Day 3 makes this concrete: loading a real embedding model on the RTX 4050 and
turning the 4 clean chunks into vectors.
