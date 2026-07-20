# Day 1 — Document Ingestion: Parsing
### RAG Pipeline · Lesson 1 of 9

---

## Where this sits in the pipeline

```
YOU ARE HERE
     ↓
[ Parse ] → Chunk → Embed → Store → Retrieve → Rerank → Generate
└──── ingestion (build once) ────┘   └──── query (every question) ────┘
```

Everything downstream depends on this box. If parsing hands garbage to the
next step, every later step processes garbage — cleanly, silently, confidently.

---

## 1. Core idea

**A PDF is not a document. It's a set of drawing instructions.**

A `.docx` stores real text in reading order: *"Termination, for, convenience."*

A PDF stores something else:
> *draw glyph "T" at (72, 400) · draw "e" at (78, 400) · draw "r" at (84, 400) …*

It's instructions for a **printer**. Consequences:
- No guarantee characters are in reading order
- No guarantee spaces exist as characters
- No guarantee a "paragraph" is marked as anything

So parsing isn't "read a file." It's **reverse-engineering readable text out of
a format designed to be printed, not read by a machine.**

---

## 2. Three kinds of PDF

| Type | What it is | Text layer? | How to get text |
|---|---|---|---|
| **Digital** | Word → Save as PDF | Yes | Extract directly (easy) |
| **Scanned** | A photo of paper | No — just pixels | OCR required (hard) |
| **Hybrid** | Digital + scanned mixed | Partial | Extract + OCR the gaps |

**The trap:** test on a clean digital PDF, it works, ship it. Then a client
uploads a *scanned* contract and the pipeline returns **empty text** — no error,
just nothing. Every downstream step then "succeeds" on nothing.

> **Rule #1 of ingestion: always measure what you extracted.**
> A character count catches the silent-empty failure instantly.

**Key principle:** *"It didn't crash" ≠ "it worked."* The scariest failures are
silent — the pipeline runs green and produces garbage. Measuring the output is
how you catch them.

One clarification worth keeping precise: OCR doesn't "extract" text, it
**recognizes** characters from an image — it *guesses* which letters the pixels
represent, so it can be wrong. Direct extraction is exact; OCR is probabilistic.

---

## 3. Project structure

```
rag-pipeline/
├── data/
│   ├── raw/              # original PDFs — SACRED, never modified
│   └── processed/        # extracted text, chunks
├── src/
│   └── ingest/
│       └── parse.py      # Day 1
├── notes/
│   └── day01_parsing.md  # decision note
└── README.md
```

Why `data/raw/` is never touched: when chunking breaks (it will), you re-run
from the original source. Overwrite the raw files and you've destroyed
reproducibility — same discipline as never editing training data in place.

Setup used:
```bash
conda create -n rag python=3.10 -y
conda activate rag
pip install pymupdf          # install "pymupdf", import "fitz"
```

The `pymupdf` / `fitz` naming quirk: PyMuPDF is a thin Python skin over the C
library **MuPDF**; `fitz` was the original codename. That's also *why it's fast*
— the heavy lifting is compiled C.

---

## 4. Code

### `make_test_pdf.py` — generate a known test file
```python
import fitz  # PyMuPDF

doc = fitz.open()          # new empty PDF in memory
page = doc.new_page()      # one blank A4 page

text = (
    "TERMINATION FOR CONVENIENCE\n"
    "Either party may terminate this Agreement for convenience "
    "upon thirty (30) days written notice.\n"
    "Case No. 042/2021"
)

page.insert_text((72, 72), text, fontsize=11)  # (72,72) = 1 inch from top-left
doc.save("data/raw/test_contract.pdf")
doc.close()
print("Created data/raw/test_contract.pdf")
```

### `src/ingest/parse.py` — the parser + the guard
```python
import fitz  # PyMuPDF


def extract_text(pdf_path):
    """Extract all text from a PDF file. Returns one string."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:                   # one page at a time — page is the unit
        full_text += page.get_text()
    doc.close()
    return full_text


if __name__ == "__main__":
    text = extract_text("data/raw/test_contract.pdf")
    char_count = len(text)

    print("--- EXTRACTED TEXT ---")
    print(text)
    print("--- MEASUREMENT ---")
    print(f"Characters extracted: {char_count}")

    # The guard: catch the silent-empty failure
    if char_count < 10:
        print("⚠️  WARNING: almost no text extracted.")
        print("    This PDF is probably SCANNED — needs OCR, not direct extraction.")
    else:
        print("✓ Extraction looks healthy.")
```

### Why `for page in doc:` is a loop
`get_text()` works on **one page at a time** — it can't grab the whole document
at once. A 1-page test runs the loop once; a 90-page contract runs it 90 times.
**The page is the unit of extraction.** The loop variable *is* the page — which
is also where page numbers for citations ("per contract.pdf, page 7") will come
from later. Right now `+=` throws that away; Day 2 + metadata bring it back.

---

## 5. The measurement (today's deliverable)

- **Predicted:** ~150 characters (the text I inserted)
- **Measured:** 142 characters ✓ — prediction and result match
- **Threshold:** under 10 characters ⇒ probable scanned/failed file ⇒ route to OCR

The discipline: **predict → measure → compare.** A number with nothing to
compare it to tells you nothing. The prediction is what makes the measurement
meaningful, and it's what would have caught a `0` instantly.

---

## 6. Decision note (the target shape)

> **Problem:** PDFs come in three forms — digital (text layer, extract directly),
> scanned (pixels, needs OCR), hybrid (both). Direct extraction silently returns
> empty text on scanned files, with no error.
>
> **Decision:** Use PyMuPDF over pypdf and pdfplumber — fast C engine, handles
> text + images + character positions, carries through to chunking and OCR later.
> Add a character-count guard after every extraction.
>
> **Why:** "Ran without error" doesn't mean text came out — a scanned PDF runs
> clean and returns nothing. The guard turns that silent failure into a visible
> warning, so garbage never flows downstream unnoticed.
>
> **The number:** Extracted 142 characters, matching the ~150 prediction.
> Threshold: under 10 characters flags a probable scanned/failed file for OCR.

**The note-writing lesson:** every box needs the *because*, not just the label.
"PyMuPDF" → "PyMuPDF **because**…". "Count the length" → "count the length **to
catch this specific failure**." Label-without-reasoning is the interview trap;
decision-with-reasoning is the target.

---

## What I own after Day 1
- Why PDF parsing is non-trivial (drawing instructions, not text)
- Three PDF types and which needs OCR
- The silent-empty failure, and the guard that catches it
- "It didn't crash" ≠ "it worked" — always measure the output
- `predict → measure → compare` discipline
- The page is the unit of extraction (→ future citations)
- Decision-note shape: problem → decision (over alternatives) → why → the number

---

## Bridge to Day 2 — Chunking theory (reasoned at end of session)

**What a chunk is:** a piece of text you cut a document into before embedding.
Just a string — a slice of the document. Relationship:

> **1 document → many chunks → many vectors (one vector per chunk)**

Not "one document, one vector." That single-vector idea is what fails below.

### Why not embed a whole 90-page document as ONE vector?

An embedding is **one point in meaning-space representing the *average* meaning
of everything you fed it.**

- A small chunk about *only* termination → vector lands squarely in "termination"
  territory → a termination question matches it strongly.
- 90 pages about ~100 topics → the vector averages all of them → it sits in the
  mushy middle, close to nothing in particular. The termination signal is ~1% of
  the blend, drowned by the other 99 topics.

> **Smoothie analogy:** blend a strawberry with 99 other fruits and you can't
> taste strawberry. A chunk that's *just* strawberry tastes unmistakably of it.

Two failures from one giant vector:
1. **Matching:** averaged vector is mush — weak match for every question.
2. **Context:** even if it matched, you'd dump all 90 pages on the LLM — blows
   the token limit, buries the one relevant clause.

### Why not make chunks TINY (one sentence / one word)?

Too-small chunks **sever the context a fact needs to be meaningful.** Example —
clause split across two chunks:
- Chunk A: *"Either party may terminate this Agreement for convenience"*
- Chunk B: *"upon thirty (30) days written notice."*

Ask *"how much notice to terminate?"* → retrieval finds Chunk B, but alone it's a
floating fragment ("notice of *what*?"). The LLM either says "not enough context"
or **fills the gap with a guess → hallucination.**

### The tradeoff (the whole lesson of chunking)

| Too **big** | Too **small** |
|---|---|
| Vector averages 100 topics → matches nothing well | Vector is sharp, but chunk lacks context |
| Dumps too much on the LLM | Severs facts across boundaries → LLM guesses |
| **Fails at matching** | **Fails at answering** |

### The rule for chunk size

> **A chunk should be big enough to carry a complete thought, and small enough
> to be about one thing.**

"About one thing" → focused vector → matches well.
"Complete thought" → keeps context → LLM can answer without guessing.

**Senior footnote:** the right size *depends on the document* — dense legal
clauses chunk smaller than flowing narrative. That's why Day 2 **measures** chunk
size against golden questions instead of hardcoding "500 tokens" from a tutorial.

**Preview of a fix (Day 2):** overlapping chunks — repeat the last sentence of
chunk A as the first of chunk B, so a fact on the seam survives whole in at least
one chunk.
