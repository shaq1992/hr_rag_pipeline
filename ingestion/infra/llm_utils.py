from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def summarize_table(raw_html: str):
    """Generates a dense semantic summary of a table's rules and structure."""
    prompt = f"Provide a detailed semantic summary of the following HR policy table's purpose, rules, and relationships so it can be vector searched accurately:\n\n{raw_html}"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={'temperature': 0.1}
    )
    return response.text, response.usage_metadata
