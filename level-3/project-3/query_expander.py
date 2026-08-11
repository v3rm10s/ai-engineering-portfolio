from typing import List
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class QueryExpander:
    def __init__(self, client: genai.Client, model: str = "gemini-2.5-flash", num_variations: int = 3):
        self.client = client
        self.model = model
        self.num_variations = num_variations

    def generate_variations(self, original_query: str) -> List[str]:
        """
        Generates alternative query rephrasings using Gemini.
        Returns a list containing the original query plus the new variations.
        """
        system_prompt = f"""You are an expert search engine optimization AI.
Your job is to generate alternative search query variations to improve document retrieval recall in vector search.

Given a user query:
1. Generate exactly {self.num_variations} alternative rephrasings of the query.
2. Use technical synonyms, industry jargon, or alternative sentence structures.
3. Preserve the core intent of the user.
4. Output MUST be a valid JSON list of strings. Example format:
["variation 1", "variation 2", "variation 3"]
"""

        try:
            # Google GenAI SDK call
            response = self.client.models.generate_content(
                model=self.model,
                contents=f"{system_prompt}\n\nUser Query: {original_query}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                )
            )

            content = response.text.strip()
            variations = json.loads(content)

            # Ensure variations is a list
            if not isinstance(variations, list):
                print("Warning: Parsed response is not a list. Falling back.")
                return [original_query]

            # Guarantee the original user query is included in the set
            if original_query not in variations:
                variations.insert(0, original_query)

            return variations

        except Exception as e:
            print(f"Failed to generate query variations ({e}). Returning original query.")
            return [original_query]