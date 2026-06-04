import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from prompt import PromptTemplates
from document_processor import DocumentProcessor

load_dotenv()


class SessionManager:

    def __init__(self, max_history: int = 10):
        self._sessions: dict[str, list[dict]] = {}
        self.max_history = max_history

    def get_history(self, session_id: str) -> list[dict]:
        return self._sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        cap = self.max_history * 2
        if len(self._sessions[session_id]) > cap:
            self._sessions[session_id] = self._sessions[session_id][-cap:]

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


class LanguageDetector:

    GE_EZ_START = 0x1200
    GE_EZ_END = 0x137F

    def detect(self, text: str) -> str:
        amharic = sum(1 for c in text if self.GE_EZ_START <= ord(c) <= self.GE_EZ_END)
        alpha = sum(1 for c in text if c.isalpha())
        if alpha == 0:
            return "english"
        ratio = amharic / alpha
        if ratio > 0.6:
            return "amharic"
        if ratio > 0.2:
            return "mixed"
        return "english"


class LawyerCityAI:

    def __init__(self):
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self._gen_model = os.getenv("GENERATION_MODEL", "gemini-2.5-pro")
        self.top_k = int(os.getenv("TOP_K_CHUNKS", 8))
        self.low_conf_threshold = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", 0.70))
        self._doc_processor = DocumentProcessor()
        self._session_manager = SessionManager(
            max_history=int(os.getenv("MAX_SESSION_HISTORY", 10))
        )
        self._lang_detector = LanguageDetector()
        self._auto_ingest()

    def _auto_ingest(self) -> None:
        pdf_dir = Path(os.getenv("PDF_DIRECTORY", "./pdfs"))
        if not pdf_dir.exists():
            pdf_dir.mkdir(parents=True)
            return
        pdfs = list(pdf_dir.glob("**/*.pdf"))
        if not pdfs:
            return
        for pdf in pdfs:
            if not self._doc_processor.is_already_indexed(str(pdf)):
                self._doc_processor.ingest_pdf(str(pdf))

    def chat(self, session_id: str, query: str) -> dict:
        language = self._lang_detector.detect(query)
        context_chunks = self._retrieve_context(query, language)
        history = self._session_manager.get_history(session_id)
        prompt = PromptTemplates.build_rag_prompt(query, context_chunks, history)
        answer = self._generate(prompt)
        self._session_manager.add_message(session_id, "user", query)
        self._session_manager.add_message(session_id, "assistant", answer)
        return {
            "answer": answer,
            "sources": self._format_sources(context_chunks),
            "language": language,
            "session_id": session_id
        }

    def ingest(self, path: str) -> dict | list:
        return (self._doc_processor.ingest_directory(path)
                if Path(path).is_dir()
                else self._doc_processor.ingest_pdf(path))

    def clear_session(self, session_id: str) -> dict:
        self._session_manager.clear_session(session_id)
        return {"status": "cleared", "session_id": session_id}

    def get_stats(self) -> dict:
        return {
            "faiss_index": self._doc_processor.get_index_stats(),
            "active_sessions": len(self._session_manager.list_sessions()),
            "status": "operational"
        }

    def _retrieve_context(self, query: str, language: str) -> list[dict]:
        query_emb = self._doc_processor.embed_text(query)
        results = self._doc_processor.retrieve(query_emb, top_k=self.top_k)
        if language in ("amharic", "mixed") and self._low_confidence(results):
            en_query = self._translate_to_english(query)
            en_emb = self._doc_processor.embed_text(en_query)
            fallback = self._doc_processor.retrieve(en_emb, top_k=self.top_k)
            results = self._merge(results, fallback)
        return results

    def _generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._gen_model,
                contents=prompt,
                config={"system_instruction": PromptTemplates.LEGAL_ADVISOR_SYSTEM}
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {e}"

    def _translate_to_english(self, text: str) -> str:
        prompt = PromptTemplates.TRANSLATE_TO_ENGLISH_PROMPT.format(text=text)
        try:
            response = self._client.models.generate_content(
                model=self._gen_model, contents=prompt
            )
            return response.text.strip()
        except Exception:
            return text

    def _low_confidence(self, results: list[dict]) -> bool:
        if not results:
            return True
        return results[0].get("score", 0.0) < self.low_conf_threshold

    def _merge(self, primary: list[dict], fallback: list[dict]) -> list[dict]:
        seen, merged = set(), []
        for chunk in primary + fallback:
            h = chunk.get("hash", chunk.get("text", "")[:50])
            if h not in seen:
                seen.add(h)
                merged.append(chunk)
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged[:self.top_k]

    def _format_sources(self, chunks: list[dict]) -> list[dict]:
        seen, sources = set(), []
        for c in chunks:
            key = (c.get("source"), c.get("page"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": c.get("source", "Unknown"),
                    "page": c.get("page", "N/A"),
                    "score": round(c.get("score", 0.0), 4)
                })
        return sources


if __name__ == "__main__":
    import json

    ai = LawyerCityAI()
    session_id = "test_session"

    print("LawyerCity AI — Terminal Test")
    print("Commands: 'quit' to exit | 'clear' to reset session | 'stats' for index info")
    print("-" * 50)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "clear":
            ai.clear_session(session_id)
            print("Session cleared.")
            continue
        if query.lower() == "stats":
            print(json.dumps(ai.get_stats(), indent=2))
            continue

        result = ai.chat(session_id=session_id, query=query)
        print(f"\nAI ({result['language']}):\n{result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  • {s['source']} — Page {s['page']} (score: {s['score']})")