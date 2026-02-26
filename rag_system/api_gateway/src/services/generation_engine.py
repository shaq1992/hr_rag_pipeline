import logging
from typing import List
from dto.response import RetrievedChunk
from integration.gemini_client import gemini_client

logger = logging.getLogger(__name__)

class GenerationEngine:
    @staticmethod
    def _build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
        # Construct the context block
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(
                f"[Document {i+1} | Source: {chunk.source_document} | Section: {chunk.section_header}]\n{chunk.content}\n"
            )
        context_str = "\n".join(context_parts)

        return f"""
        You are an expert HR Assistant for a government entity. Your job is to answer the user's query 
        using ONLY the provided context. 

        Instructions:
        1. Answer clearly, accurately, and professionally.
        2. If the context contains conflicting information (e.g., an old policy vs. a new policy), explicitly state the conflict to the user.
        3. Do not formulate citations in the main text body. The system will append citations automatically.
        4. If the provided context does not contain the answer, state "I do not have enough information in the provided policies to answer that question." Do not hallucinate.

        Context:
        {context_str}

        User Query: {query}
        
        Answer:
        """

    @staticmethod
    def _format_citations(chunks: List[RetrievedChunk]) -> str:
        """Extracts unique citations to append at the end of the stream."""
        unique_citations = {}
        for chunk in chunks:
            key = f"{chunk.source_document} - {chunk.section_header}"
            if key not in unique_citations:
                unique_citations[key] = f"- **{chunk.source_document}** (Section: {chunk.section_header}, Page: {chunk.page_number})"
        
        citation_text = "\n\n---\n**Sources:**\n" + "\n".join(unique_citations.values())
        return citation_text

    @staticmethod
    async def generate_response(query: str, chunks: List[RetrievedChunk]):
        if not chunks:
            yield "I could not find any relevant policy documents to answer your question."
            return

        prompt = GenerationEngine._build_prompt(query, chunks)
        
        logger.info("Opening Gemini streaming connection...")
        async for token in gemini_client.stream_generation(prompt):
            yield token

        # Append citations cleanly at the end of the stream
        yield GenerationEngine._format_citations(chunks)
