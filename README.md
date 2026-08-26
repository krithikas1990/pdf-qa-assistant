# PDF Q&A Assistant (RAG + Claude)

A simple Retrieval-Augmented Generation (RAG) app: upload a PDF, ask questions,
get answers grounded in the document's actual content — with page references.

## How it works

1. **Extract** — pulls text out of the PDF, page by page (`pypdf`)
2. **OCR fallback** — pages with little/no extractable text (scanned pages, image-based tables/charts) are rendered as images (`pymupdf`) and sent to Claude directly to transcribe — this acts as OCR without needing separate OCR software
3. **Chunk** — splits text into overlapping ~700-character pieces
4. **Embed** — converts each chunk into a vector using a free local model (`sentence-transformers`)
5. **Store** — saves vectors in a local vector database (`chromadb`)
6. **Retrieve** — when you ask a question, finds the most relevant chunks (or, if you mention a specific page number, searches only that page)
7. **Answer** — sends those chunks + your question to Claude, which answers using only that context

Page numbers shown in answers are the document's own *printed* page numbers, not just physical position in the file — detected by asking Claude to read the number directly off each page during OCR, since a fixed offset doesn't work reliably across documents with unnumbered front matter (title pages, table of contents, etc.).

This is the same basic architecture used in production RAG systems — just simplified.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install anthropic pypdf chromadb sentence-transformers streamlit pymupdf

# 3. Set your API key
export ANTHROPIC_API_KEY="your-key-here"      # Mac/Linux
setx ANTHROPIC_API_KEY "your-key-here"        # Windows (restart terminal after)
```

Get an API key at https://console.anthropic.com

## Running it

**Command-line version (good for testing the pipeline):**
```bash
python rag_core.py
```

**Web UI version (good for demos and your portfolio):**
```bash
streamlit run streamlit_app.py
```

## Files

- `rag_core.py` — the actual RAG pipeline: PDF extraction, chunking, embedding, retrieval, and the Claude call
- `streamlit_app.py` — a simple web interface on top of `rag_core.py`
- `README.md` — this file

## Known limitation (discovered through testing)

Asking a broad question without a page number can sometimes miss the specific
fact you want — e.g. asking "what is the rent?" on a rental agreement failed
to surface the actual figure, even though it appears once, clearly, on page 2.
This happens because semantic search ranks chunks by how *similar in meaning*
they are to the question, and in a document where a term ("rent") appears many
times in different clauses, the one chunk with the actual number can get
crowded out of the top results by chunks that are topically related but don't
contain the answer. Asking with a page number bypasses this (it searches only
that page), which is why that worked immediately.

## Next steps / ideas to extend it (good for your portfolio README)

- Fix the limitation above: increase `TOP_K` to retrieve more candidate chunks, try smaller/more targeted chunk sizes, or add hybrid search (combining exact keyword matching with semantic search) so exact terms aren't diluted
- Support multiple PDFs at once, with source document shown alongside the page number
- Persist the vector store to disk (`chromadb.PersistentClient`) so you don't re-index every run
- Add a "confidence" indicator when retrieved chunks have low similarity to the question
- Try different chunk sizes and see how answer quality changes
- Swap the local embedding model for Claude's own embeddings API and compare

## What to put in your GitHub README

When you publish this, add: what problem it solves, a short GIF/screenshot of it running,
the architecture explanation above, and a "what I'd improve" section — that last part
signals real engineering judgment to anyone reviewing it.
