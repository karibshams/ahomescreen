import os
import json
import hashlib
import numpy as np
import fitz
import faiss
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi

load_dotenv()


class DocumentProcessor:

    EMBEDDING_DIM = 3072

    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.faiss_index_path = Path(os.getenv("FAISS_INDEX_PATH", "./faiss_store/index.faiss"))
        self.metadata_path = Path(os.getenv("FAISS_METADATA_PATH", "./faiss_store/metadata.json"))
        self.indexed_files_path = self.faiss_index_path.parent / "indexed_files.json"
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 150))
        self._index = None
        self._metadata: list[dict] = []
        self._indexed_files: set[str] = set()
        self._bm25 = None
        self._load_or_create_index()

    def is_already_indexed(self, pdf_path: str) -> bool:
        return Path(pdf_path).name in self._indexed_files

    def ingest_pdf(self, pdf_path: str) -> dict:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        source_name = pdf_path.name

        if source_name in self._indexed_files:
            return {"source": source_name, "pages_processed": 0,
                    "chunks_added": 0, "status": "already_indexed"}

        pages = self._extract_text(pdf_path)
        chunks = self._chunk_pages(pages, source_name)
        existing_hashes = {m.get("hash") for m in self._metadata}
        new_chunks = [c for c in chunks if c["hash"] not in existing_hashes]

        if not new_chunks:
            self._indexed_files.add(source_name)
            self._persist_indexed_files()
            return {"source": source_name, "pages_processed": len(pages),
                    "chunks_added": 0, "status": "already_indexed"}

        embeddings = self._embed_chunks(new_chunks)
        self._add_to_index(embeddings, new_chunks)
        self._indexed_files.add(source_name)
        self._rebuild_bm25()
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

    def retrieve(self, query_embedding: np.ndarray, query_text: str, top_k: int = 8) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []

        k = min(top_k * 2, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        dense_results = {}
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            dense_results[int(idx)] = rank

        sparse_results = {}
        if self._bm25 is not None:
            tokens = query_text.lower().split()
            bm25_scores = self._bm25.get_scores(tokens)
            top_bm25 = np.argsort(bm25_scores)[::-1][:top_k * 2]
            for rank, idx in enumerate(top_bm25):
                if bm25_scores[idx] > 0:
                    sparse_results[int(idx)] = rank

        fused = {}
        for idx, rank in dense_results.items():
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank + 1)
        for idx, rank in sparse_results.items():
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank + 1)

        top_indices = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            chunk = self._metadata[idx].copy()
            chunk["score"] = round(fused[idx], 6)
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
            "indexed_files": list(self._indexed_files),
            "documents": sources
        }

    def _extract_text(self, pdf_path: Path) -> list[dict]:
        pages = []
        with fitz.open(str(pdf_path)) as doc:
            for num, page in enumerate(doc, start=1):
                text = self._extract_page_text(page)
                if text:
                    pages.append({"page": num, "text": text})
        return pages

    def _extract_page_text(self, page) -> str:
        blocks = page.get_text("blocks")
        if self._is_two_column(blocks, page.rect.width):
            return self._extract_two_column(page)
        text = page.get_text("text").strip()
        if not text or len(text) < 50:
            text = self._ocr_page(page)
        return text

    def _is_two_column(self, blocks: list, page_width: float) -> bool:
        if not blocks:
            return False
        mid = page_width / 2
        left_blocks = [b for b in blocks if b[2] < mid and b[4].strip()]
        right_blocks = [b for b in blocks if b[0] > mid and b[4].strip()]
        return len(left_blocks) >= 3 and len(right_blocks) >= 3

    def _extract_two_column(self, page) -> str:
        mid = page.rect.width / 2
        left_text = page.get_textbox(fitz.Rect(0, 0, mid, page.rect.height)).strip()
        right_text = page.get_textbox(fitz.Rect(mid, 0, page.rect.width, page.rect.height)).strip()
        parts = []
        if left_text:
            parts.append(f"[AM]\n{left_text}")
        if right_text:
            parts.append(f"[EN]\n{right_text}")
        return "\n\n".join(parts)

    def _ocr_page(self, page) -> str:
        try:
            import pytesseract
            from PIL import Image
            import io
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                return pytesseract.image_to_string(img, lang="amh+eng").strip()
            except Exception:
                return pytesseract.image_to_string(img, lang="eng").strip()
        except Exception:
            return ""

    def _chunk_pages(self, pages: list[dict], source: str) -> list[dict]:
        chunks = []
        for p in pages:
            article_chunks = self._chunk_by_article(p["text"], source, p["page"])
            if article_chunks:
                chunks.extend(article_chunks)
            else:
                chunks.extend(self._chunk_fixed(p["text"], source, p["page"]))
        return chunks

    def _chunk_by_article(self, text: str, source: str, page: int) -> list[dict]:
        import re
        pattern = re.compile(
            r'(?:Article\s+\d+|አንቀጽ\s+\d+|ARTICLE\s+\d+|Section\s+\d+|Chapter\s+\d+)',
            re.IGNORECASE
        )
        matches = list(pattern.finditer(text))
        if len(matches) < 2:
            return []
        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            if len(chunk_text) < 50:
                continue
            chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page,
                "article": match.group().strip(),
                "hash": hashlib.md5(chunk_text.encode()).hexdigest()
            })
        return chunks

    def _chunk_fixed(self, text: str, source: str, page: int) -> list[dict]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            chunk_text = " ".join(words[start: start + self.chunk_size]).strip()
            if len(chunk_text) >= 50:
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "article": None,
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
        # Add metadata first
        for c in chunks:
            self._metadata.append({
                "text": c["text"],
                "source": c["source"],
                "page": c["page"],
                "article": c.get("article"),
                "hash": c["hash"]
            })

        # Auto-upgrade: switch from FlatIP to IVFFlat once 256+ chunks are available
        is_ivf = isinstance(self._index, faiss.IndexIVFFlat)
        if not is_ivf and len(self._metadata) >= 256:
            all_embeddings = np.vstack([embeddings]).astype(np.float32)
            # Collect all existing vectors from flat index
            if self._index.ntotal > 0:
                existing = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self.EMBEDDING_DIM)
                existing = np.array(existing).reshape(self._index.ntotal, self.EMBEDDING_DIM).astype(np.float32)
                all_embeddings = np.vstack([existing, embeddings]).astype(np.float32)
            quantizer = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            new_index = faiss.IndexIVFFlat(quantizer, self.EMBEDDING_DIM, 256)
            new_index.train(all_embeddings)
            new_index.add(all_embeddings)
            self._index = new_index
        else:
            self._index.add(embeddings)

    def _load_or_create_index(self) -> None:
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.faiss_index_path.exists() and self.metadata_path.exists():
            self._index = faiss.read_index(str(self.faiss_index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            self._rebuild_bm25()
        else:
            # Start with FlatIP — auto-upgrades to IVFFlat once 256+ chunks exist
            self._index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
            self._metadata = []

        if self.indexed_files_path.exists():
            with open(self.indexed_files_path, "r", encoding="utf-8") as f:
                self._indexed_files = set(json.load(f))
        else:
            self._indexed_files = set()

    def _rebuild_bm25(self) -> None:
        if not self._metadata:
            self._bm25 = None
            return
        corpus = [m.get("text", "").lower().split() for m in self._metadata]
        self._bm25 = BM25Okapi(corpus)

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.faiss_index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)
        self._persist_indexed_files()

    def _persist_indexed_files(self) -> None:
        with open(self.indexed_files_path, "w", encoding="utf-8") as f:
            json.dump(list(self._indexed_files), f, ensure_ascii=False, indent=2)