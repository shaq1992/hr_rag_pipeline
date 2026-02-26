import logging
from google import genai
from google.genai import types
from core.config import settings
from dto.response import RoutingDecision

logger = logging.getLogger(__name__)

class GeminiIntegration:
    def __init__(self):
        # Explicitly pass the API key from our validated settings
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def route_query(self, query: str) -> RoutingDecision:
        """Uses Gemini 1.5 Flash for fast, zero-shot structured classification."""
        prompt = f"""
        Analyze the following user query intended for an HR policy knowledge base.
        Categorize it into exactly one of the following types:
        - factual: Asking for a specific fact (e.g., 'How many vacation days?').
        - procedural: Asking for a process or steps (e.g., 'How do I apply for leave?').
        - comparative: Comparing two or more policies/concepts.
        - out-of-scope: Not related to HR policies, entitlements, or workplace guidelines.

        Query: "{query}"
        """
        
        try:
            response = await self.client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RoutingDecision,
                    temperature=0.1,
                ),
            )
            # The SDK automatically validates and parses the output into the Pydantic model
            return response.parsed
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Failsafe default to ensure the pipeline continues
            return RoutingDecision(query_type="factual", reasoning="Failsafe fallback due to API error.")

    async def stream_generation(self, prompt: str):
        """Uses Gemini 2.5 Flash to stream the augmented response."""
        try:
            # Removed 'await' - the method yields the stream directly
            response_stream = self.client.aio.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Generation stream failed: {e}")
            yield "I encountered an error while generating the response. Please try again."

gemini_client = GeminiIntegration()
