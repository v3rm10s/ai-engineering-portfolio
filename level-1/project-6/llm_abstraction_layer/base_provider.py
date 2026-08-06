from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# 1. Standardized Output Structure
@dataclass
class LLMResponse:
    text: str
    provider_name: str
    model_name: str
    
class BaseLLMProvider(ABC):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse:
        pass
    
