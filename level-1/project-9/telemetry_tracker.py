import time
import os
from dotenv import load_dotenv
from typing import Any, Dict
from google import genai


MODEL_PRICING = {
    "gemini-2.5-flash": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000,
    },
    "gemini-1.5-flash": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000,
    },
    "gemini-1.5-pro": {
        "input": 1.25 / 1_000_000,
        "output": 5.00 / 1_000_000,
    },
}

class TelemetryTracker:
    def __init__(self, client: genai.Client):
        self.client = client
        
    def generate_content_with_telemetry(self, model_name: str, prompt: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        latency_seconds = time.perf_counter() - start_time
        
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        candidate_tokens = usage.candidates_token_count if usage else 0
        total_tokens = usage.total_token_count if usage else 0
        
        rates = MODEL_PRICING.get(model_name, {"input": 0.0, "output":0.0})
        cost_usd = (prompt_tokens * rates["input"]) + (candidate_tokens * rates["output"])

        return {
            "text": response.text,
            "telemetry": {
                "model": model_name,
                "latency_seconds": round(latency_seconds, 4),
                "prompt_tokens": prompt_tokens,
                "candidate_tokens": candidate_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost_usd,6),
            }
        }
        
if __name__== "__main__":
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    client = genai.Client()
    tracker = TelemetryTracker(client)
    
    prompt = "Explain quantum computing in exactly two short sentences."
    model = "gemini-2.5-flash"
    
    print(f"Sending prompt to {model}...\n")
    result = tracker.generate_content_with_telemetry(model_name=model,prompt = prompt)
    
    print("--- LLM Response ---")
    print(result["text"])
    print("\n--- Telemetry Report ---")
    for key, value in result ["telemetry"].items():
        print(f"{key}: {value}")