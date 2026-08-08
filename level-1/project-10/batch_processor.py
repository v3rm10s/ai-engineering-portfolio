import asyncio
import os
from dotenv import load_dotenv
import time
from google import genai

PROMPTS = [
    "Give me a 1-sentence joke about Python programming.",
    "Give me a 1-sentence joke about databases.",
    "Give me a 1-sentence joke about cloud computing.",
    "Give me a 1-sentence joke about frontend web development.",
    "Give me a 1-sentence joke about cybersecurity.",
    "Give me a 1-sentence joke about artificial intelligence.",
]

MAX_CONCURRENCY = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

async def process_prompt_bounded(prompt: str, prompt_id: int) -> dict:
    async with semaphore:
        print(f" [Prompt ${prompt_id}] Starting request...")
        start_time = time.perf_counter()
        
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents = prompt,
        )
        
        duration = time.perf_counter() - start_time
        print(f" [Prompt #{prompt_id}] Completed in {duration:.2f}s")
        
        return {
            "id": prompt_id,
            "prompt": prompt,
            "response": response.text.strip(),
            "duration": duration,
        }
        
async def main():
    print(f"--- Processing {len(PROMPTS)} Prompts (Concurrency Limit = {MAX_CONCURRENCY}) ---")
    total_start = time.perf_counter()
    
    tasks = [process_prompt_bounded(prompt, i + 1) for i, prompt in enumerate(PROMPTS)]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - total_start
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY:")
    print("=" * 50)
    for res in results:
        print(f"\n[Prompt #{res['id']}] {res['prompt']}")
        print(f"👉 {res['response']}")
    
    print(f"\n⏱️ Total elapsed time for all {len(PROMPTS)} requests: {total_time:.2f} seconds")
    
if __name__ == "__main__":
    asyncio.run(main())