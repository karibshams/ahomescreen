import os
import json
import hashlib
import numpy as np
import fitz
import faiss
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()


class DocumentProcessor:

    EMBEDDING_DIM = 3072

    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.faiss_index_path = Path(os.getenv("FAISS_INDEX_PATH", "./faiss_store/index.faiss"))
        self.metadata_path = Path(os.getenv("FAISS_METADATA_PATH", "./faiss_store/metadata.json"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] = []
        self._load_or_create_index()

    def ingest_pdf(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        source_name = pdf_path.name
        pages = self._extract_text(pdf_path)
        chunks = self._chunk_pages(pages, source_name)
        existing_hashes = {m.get("hash") for m in self._metadata}
        new_chunks = [c for c in chunks if c["hash"] not in existing_hashes]
        if not new_chunks:
            return {"source": source_name, "pages_processed": len(pages),
                    "chunks_added": 0, "status": "already_indexed"}
        embeddings = self._embed_chunks(new_chunks)
        self._add_to_index(embeddings, new_chunks)
        self._persist()
        return {"source": source_name, "pages_processed": len(pages),
                "chunks_added": len(new_chunks), "status": "success"}

    def ingest_directory(self, directory_path: str) -> list[dict]:
        pdfs = list(Path(directory_path).glob("**/*.pdf"))
        if not pdfs:
            return [{"status": "no_pdfs_found", "directory": directory_path}]
        results = []
        for pdf in pdfs:
            try:
                results.append(self.ingest_pdf(str(pdf)))
            except Exception as e:
                results.append({"source": pdf.name, "status": "error", "error": str(e)})
        return results

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 8) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def embed_text(self, text: str) -> np.ndarray:
        response = self._client.models.embed_content(
            model=self.embedding_model,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"}
        )
        embedding = np.array(response.embeddings[0].values, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        return embedding

    def get_index_stats(self) -> dict:
        sources = {}
        for m in self._metadata:
            src = m.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {
            "total_chunks": self._index.ntotal if self._index else 0,
            "total_documents": len(sources),
            "documents": sources
        }

    def _extract_text(self, pdf_path: Path) -> list[dict]:
        pages = []
        with fitz.open(str(pdf_path)) as doc:
            for num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"page": num, "text": text})
        return pages

    def _chunk_pages(self, pages: list[dict], source: str) -> list[dict]:
        chunks = []
        for p in pages:
            words = p["text"].split()
            start = 0
            while start < len(words):
                chunk_text = " ".join(words[start: start + self.chunk_size]).strip()
                if len(chunk_text) >= 50:
                    chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "page": p["page"],
                        "hash": hashlib.md5(chunk_text.encode()).hexdigest()
                    })
                start += self.chunk_size - self.chunk_overlap
        return chunks

    def _embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(chunks), 20):
            for chunk in chunks[i: i + 20]:
                response = self._client.models.embed_content(
                    model=self.embedding_model,
                    contents=chunk["text"],
                    config={"task_type": "RETRIEVAL_DOCUMENT"}
                )
                all_embeddings.append(np.array(response.embeddings[0].values, dtype=np.float32))
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings)
        return embeddings

    def _add_to_index(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        self._index.add(embeddings)
        for c in chunks:
            self._metadata.append({
                "text": c["text"], "source": c["source"],
                "page": c["page"], "hash": c["hash"]
            })

    def _load_or_create_index(self) -> None:
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.faiss_index_path.exists() and self.metadata_path.exists():
            self._index = faiss.read_index(str(self.faiss_index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self._metadata = []

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.faiss_index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)