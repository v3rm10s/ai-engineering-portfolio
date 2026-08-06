import os
from typing import Dict,Type, Optional
from base_provider import BaseLLMProvider
from gemini_provider import GeminiProvider
from mock_openai_provider import MockOpenAIProvider

class LLMFactory:
    _PROVIDERS: Dict[str, Type[BaseLLMProvider]] = {
        "gemini": GeminiProvider,
        "openai": MockOpenAIProvider,
    }
    
    @classmethod
    def get_provider(
        self,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BaseLLMProvider:
        
        target_provider = (
            provider_name
            or os.getenv("LLM_PROVIDER")
            or "gemini"
        ).lower().strip()
        
        if target_provider not in self._PROVIDERS:
            supported = list(self._PROVIDERS.keys())
            raise ValueError(
                f"Unsupported provider: '{target_provider}."
                f"Supported provider: '{supported}'"
            )
            
        provider_class = self._PROVIDERS[target_provider]
        
        if model_name:
            return provider_class(model_name=model_name, api_key=api_key)
        return provider_class(api_key=api_key)