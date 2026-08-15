import os
from dotenv import load_dotenv
from PIL import Image
from google import genai
from google.genai import types
from src.ingestion import DocumentElement, ElementType

class ElementSummarizer:
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-3.6-flash"):
        load_dotenv()
        """Initializes the Gemini client for multimodal element summarization."""
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model_name = model_name

    def summarize_image(self, image_path: str) -> str:
        """Generates a dense, retrieval-optimized description of an image or chart."""
        img = Image.open(image_path)
        prompt = (
            "You are an expert technical and financial visual analyst. "
            "Analyze this image in exhaustive detail for a vector search retrieval index.\n"
            "- If it is a chart/graph: extract all numerical ranges, axis values, support/resistance levels, "
            "indicators (e.g. moving averages, volume bars), timestamps, candlestick patterns, and overall trend.\n"
            "- If it is a diagram: describe all entities, connections, flows, and text labels.\n"
            "Provide a comprehensive, factual summary containing all key searchable terms."
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[img, prompt]
        )
        return response.text.strip()

    def summarize_table(self, table_text: str) -> str:
        """Generates a descriptive narrative of structured tabular data."""
        prompt = (
            "Summarize the key metrics, column relationships, trends, and notable outliers "
            "in this markdown table for vector retrieval:\n\n"
            f"{table_text}"
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()

    def process_element(self, element: DocumentElement) -> DocumentElement:
        """Enriches a DocumentElement with a generated summary based on its type."""
        if element.element_type == ElementType.IMAGE:
            print(f"[*] Generating visual summary for image: {element.content}")
            element.summary = self.summarize_image(element.content)
        elif element.element_type == ElementType.TABLE:
            print(f"[*] Generating summary for table element: {element.id}")
            element.summary = self.summarize_table(element.content)
        elif element.element_type == ElementType.TEXT:
            # Raw text chunks are already directly searchable
            element.summary = element.content
        return element