from factory import LLMFactory

def test_provider(provider_name: str, prompt: str):
    print(f"\n --- Testing Provider: {provider_name.upper()} ---")
    
    llm= LLMFactory.get_provider(provider_name=provider_name)
    
    response = llm.generate(
        prompt=prompt,
        system_instruction="You are a helpful, concise AI assistant.",
        temperature = 0.2,
        max_tokens = 100,
    )
    
    print(f"Provider Used   : {response.provider_name}")
    print(f"Model Used      : {response.model_name}")
    print(f"Response Text   : {response.text}")
    
if __name__ == "__main__":
    test_prompt = "Explain why standardizing API interfaces is important in software engineering in two sentences."
    
    test_provider("gemini",test_prompt)
    test_provider("openai",test_prompt)