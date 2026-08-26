"""
rag_core.py — Core RAG (Retrieval-Augmented Generation) pipeline for PDF Q&A.
Uses OCR to read real printed page numbers directly off each page, and
supports exact page-number filtering when a question mentions one.
"""

import os
import re
import uuid
import base64
from pypdf import PdfReader
import pymupdf as fitz
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-5"
MIN_TEXT_LENGTH = 40
MIN_SPACE_RATIO = 0.05

def render_page_as_image_bytes(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=150)
    image_bytes = pix.tobytes("png")
    doc.close()
    return image_bytes

def ocr_page_with_claude(image_bytes):
    client = Anthropic()
    base64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
                    },
                    {
                        "type": "text",
                        "text": (
                            "First, look for a printed page number on this page (usually at "
                            "the top or bottom). Respond with exactly one line in this format: "
                            "PAGE_NUMBER: <the number>  (or PAGE_NUMBER: NONE if no page number "
                            "is visible, e.g. a title page).\n\n"
                            "Then, on the following lines, transcribe all visible content on "
                            "this page as plain text. Include any tables (describe rows/columns "
                            "clearly), chart values or trends, and all readable text. Do not "
                            "summarize — transcribe as completely and accurately as possible."
                        ),
                    },
                ],
            }
        ],
    )
    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text = block.text
            break
    printed_page = None
    match = re.match(r"PAGE_NUMBER:\s*(\d+|NONE)", raw_text.strip(), re.IGNORECASE)
    body_text = raw_text
    if match:
        if match.group(1).upper() != "NONE":
            printed_page = int(match.group(1))
        body_text = raw_text[match.end():].strip()
    return {"printed_page": printed_page, "text": body_text}

def is_text_low_quality(text):
    stripped = text.strip()
    if len(stripped) < MIN_TEXT_LENGTH:
        return True
    space_ratio = stripped.count(" ") / len(stripped)
    if space_ratio < MIN_SPACE_RATIO:
        return True
    return False

def extract_text_by_page(pdf_path, use_ocr_fallback=True):
    reader = PdfReader(pdf_path)
    pages = []
    last_known_printed_page = None

    for i, page in enumerate(reader.pages):
        physical_page_number = i + 1
        text = page.extract_text() or ""
        printed_page = None

        if use_ocr_fallback and is_text_low_quality(text):
            print(f"  Page {physical_page_number} (physical): low-quality/garbled text — running OCR via Claude...")
            try:
                image_bytes = render_page_as_image_bytes(pdf_path, physical_page_number)
                ocr_result = ocr_page_with_claude(image_bytes)
                text = ocr_result["text"]
                printed_page = ocr_result["printed_page"]
            except Exception as e:
                print(f"  Page {physical_page_number} (physical): OCR failed ({e}), keeping original text.")

        if printed_page is not None:
            display_page_number = printed_page
            last_known_printed_page = printed_page
        elif last_known_printed_page is not None:
            display_page_number = last_known_printed_page + 1
            last_known_printed_page = display_page_number
        else:
            display_page_number = physical_page_number

        print(f"    -> physical page {physical_page_number} = printed page {display_page_number}")

        if text.strip():
            pages.append({"page": display_page_number, "text": text})
    return pages

def chunk_pages(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            chunks.append({"id": str(uuid.uuid4()), "text": chunk_text, "page": page["page"]})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

class DocumentStore:
    def __init__(self, collection_name="pdf_qa"):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(collection_name)

    def index_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts).tolist()
        self.collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"page": c["page"]} for c in chunks],
        )

    def retrieve(self, question, top_k=TOP_K, page_filter=None):
        query_embedding = self.embedder.encode([question]).tolist()
        query_kwargs = {"query_embeddings": query_embedding, "n_results": top_k}
        if page_filter is not None:
            query_kwargs["where"] = {"page": page_filter}
        results = self.collection.query(**query_kwargs)
        retrieved = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({"text": doc, "page": meta["page"]})
        return retrieved

def ask_claude(question, retrieved_chunks):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in retrieved_chunks)
    system_prompt = (
        "You are a document Q&A assistant. Answer the user's question using ONLY "
        "the context provided below. If the answer is not in the context, say "
        "clearly that you couldn't find it in the document — do not guess or use "
        "outside knowledge. When possible, mention which page(s) your answer comes from."
    )
    client = Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""

def build_index_from_pdf(pdf_path):
    pages = extract_text_by_page(pdf_path)
    chunks = chunk_pages(pages)
    store = DocumentStore()
    store.index_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(pages)} pages.")
    return store

def extract_page_number_from_question(question):
    match = re.search(r"\b(?:page|pg)\.?\s*(\d+)\b", question, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

if __name__ == "__main__":
    pdf_path = input("Path to your PDF file: ").strip()
    store = build_index_from_pdf(pdf_path)
    print("\nAsk questions about the document (type 'quit' to exit).")
    print("Tip: mention a page number (e.g. 'page 27') to search only that page.\n")
    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        page_filter = extract_page_number_from_question(question)
        if page_filter:
            print(f"  (Filtering to page {page_filter} only)")
        chunks = store.retrieve(question, page_filter=page_filter)
        if not chunks:
            print(f"\nA: I couldn't find any content on page {page_filter} in this document.\n")
            continue
        answer = ask_claude(question, chunks)
        print(f"\nA: {answer}\n")
