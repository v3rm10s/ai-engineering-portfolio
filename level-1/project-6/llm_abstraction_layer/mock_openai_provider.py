import time
from typing import Optional
from base_provider import BaseLLMProvider, LLMResponse

class MockOpenAIProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key:Optional[str] = None):
        super().__init__(model_name=model_name, api_key=api_key)
        
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int =500,
    ) -> LLMResponse:
        time.sleep(0.5)
        
        mock_text = (
            f"[Simulated OpenAI Response for model '{self.model_name}']\n"
            f"System Instruction: {system_instruction or 'None'}\n"
            f"Prompt: {prompt}\n"
            f"Config: Temp={temperature}, MaxTokens={max_tokens}"
        )
        
        return LLMResponse(
            text=mock_text,
            provider_name="OpenAI (Mock)",
            model_name=self.model_name,
        )