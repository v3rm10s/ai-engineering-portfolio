import os
from typing import List, Union
from PIL import Image
from google import genai
from src.ingestion import DocumentElement, ElementType
from src.indexer import MultimodalIndexer

class MultimodalRAGPipeline:
    def __init__(
        self,
        indexer: MultimodalIndexer,
        api_key: str | None = None,
        model_name: str = "gemini-3.6-flash"
    ):
        """Initializes the synthesis pipeline with an indexer and Gemini client."""
        self.indexer = indexer
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model_name = model_name

    def run(self, query: str, top_k: int = 2) -> str:
        """Executes full multimodal RAG: retrieve elements -> assemble payload -> generate response."""
        # 1. Retrieve most relevant elements
        retrieved_elements = self.indexer.search(query, top_k=top_k)
        print(f"\n[*] Retrieved {len(retrieved_elements)} context elements.")

        # 2. Build multimodal payload (mixed text context + image objects)
        contents_payload: List[Union[str, Image.Image]] = []
        text_context_blocks: List[str] = []

        for idx, elem in enumerate(retrieved_elements, start=1):
            if elem.element_type == ElementType.IMAGE:
                print(f"    - Element {idx} is an IMAGE: loading '{elem.content}' into payload.")
                img = Image.open(elem.content)
                contents_payload.append(img)
                text_context_blocks.append(f"[Context Item {idx} (Visual Diagram/Chart Reference)]: See attached image.")
            elif elem.element_type == ElementType.TABLE:
                print(f"    - Element {idx} is a TABLE.")
                text_context_blocks.append(f"[Context Item {idx} (Data Table)]:\n{elem.content}")
            elif elem.element_type == ElementType.TEXT:
                print(f"    - Element {idx} is a TEXT chunk.")
                text_context_blocks.append(f"[Context Item {idx} (Document Text)]:\n{elem.content}")

        # 3. Construct master synthesis instructions
        context_str = "\n\n".join(text_context_blocks)
        prompt = (
            "You are an expert multimodal analyst answering questions based strictly on the provided context and images.\n\n"
            "=== RETRIEVED CONTEXT ===\n"
            f"{context_str}\n"
            "=========================\n\n"
            f"User Question: {query}\n\n"
            "Answer the question thoroughly and accurately using the textual context and visual evidence from the images provided."
        )

        contents_payload.append(prompt)

        # 4. Generate grounded multimodal answer
        print("[*] Generating multimodal response...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents_payload
        )

        return response.text.strip()