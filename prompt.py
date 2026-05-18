class PromptTemplates:

    LEGAL_ADVISOR_SYSTEM = """You are LawyerCity AI — an expert Ethiopian legal advisor assistant.
Your knowledge comes EXCLUSIVELY from the legal documents provided to you as context.

LANGUAGE RULE:
- Always respond in the EXACT same language the user wrote in.
- If the user writes in Amharic (Ge'ez script), respond fully in Amharic.
- If the user writes in English, respond fully in English.

ACCURACY RULES:
- Only answer using the provided document context.
- If the answer is not found in the context, say:
  English: "This information was not found in the available Ethiopian legal documents. Please consult a licensed legal professional."
  Amharic: "ይህ መረጃ በተሰጡት ሰነዶች ውስጥ አልተገኘም። እባክዎን ፈቃድ ካለው የሕግ ባለሙያ ጋር ይማከሩ።"
- NEVER guess, hallucinate, or fabricate legal information.
- NEVER answer questions unrelated to Ethiopian law.

CITATION RULES:
- Always cite the source document name and page/article number at the end.
- Format: [Source: Document Name, Page X]
"""

    OUT_OF_SCOPE_EN = (
        "This question falls outside the scope of Ethiopian law covered by the available documents. "
        "Please consult a licensed legal professional."
    )

    OUT_OF_SCOPE_AM = (
        "ይህ ጥያቄ ከኢትዮጵያ ሕግ ወሰን ውጭ ነው። "
        "እባክዎን ፈቃድ ካለው የሕግ ባለሙያ ጋር ይማከሩ።"
    )

    TRANSLATE_TO_ENGLISH_PROMPT = (
        'Translate the following Amharic legal question to English. '
        'Preserve all legal terms and article numbers exactly. '
        'Return ONLY the English translation.\n\nAmharic: "{text}"\n\nEnglish:'
    )

    @staticmethod
    def build_rag_prompt(query: str, context_chunks: list, session_history: list) -> str:
        context_block = PromptTemplates._format_context(context_chunks)
        history_block = PromptTemplates._format_history(session_history)
        history_section = f"CONVERSATION HISTORY:\n{history_block}\n" if history_block else ""
        return (
            f"LEGAL DOCUMENT CONTEXT:\n{context_block}\n\n"
            f"{history_section}"
            f"USER QUESTION: {query}\n\n"
            f"Answer using ONLY the legal document context above. "
            f"Respond in the same language as the question. Cite sources."
        )

    @staticmethod
    def _format_context(chunks: list) -> str:
        if not chunks:
            return "No relevant context found in the legal documents."
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i} | Source: {c.get('source', 'Unknown')} | Page: {c.get('page', 'N/A')}]\n{c.get('text', '')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_history(history: list) -> str:
        if not history:
            return ""
        return "\n".join(
            f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
            for m in history
        )
