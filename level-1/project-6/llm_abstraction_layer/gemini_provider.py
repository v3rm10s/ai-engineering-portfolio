import os
from dotenv import load_dotenv
from typing import Optional
from google import genai
from google.genai import types
from base_provider import BaseLLMProvider, LLMResponse

class GeminiProvider(BaseLLMProvider):
    
    load_dotenv()
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key:Optional[str] = None):
        super().__init__(model_name=model_name, api_key=api_key)
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client=genai.Client(api_key=key)
        
    def generate(
        self,
        prompt:str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int =500,
    ) -> LLMResponse:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        
        return LLMResponse(
            text=response.text or "",
            provider_name="Gemini",
            model_name =self.model_name,
        )