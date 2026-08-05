import sys
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Error: GEMINI_API_KEY not found. Please check your .env file")
    
client = genai.Client(api_key=api_key)

def stream_llm_response_with_resilience(prompt: str):
    print(f"Prompt: {prompt}\n")
    print("--- Streaming Output Start ---")
    
    start_time = time.perf_counter()
    first_token_time = None
    chunk_count = 0
    total_chars = 0
    
    try:
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        for chunk in response:
            if chunk.text:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
                
                chunk_count += 1
                total_chars = len(chunk.text)
                
    except KeyboardInterrupt:
        print("\n\n[INFO] Stream manually interrupted by user")
    except APIError as e:
        print(f"\n\n[ERROR] Gemini API Error: {e}")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
    finally:        
        end_time = time.perf_counter()        
        print("\n--- Streaming Output End ---")

        if chunk_count > 0:
            ttft = (first_token_time - start_time) if first_token_time else 0
            total_time = end_time - start_time
            gen_time = total_time -ttft
            est_tokens = total_chars / 4
            tps = est_tokens / gen_time if gen_time > 0 else 0
            
            print("\n--- Telemetry Metrics ---")
            print(f"Time to First Token (TTFT): {ttft * 1000:.2f} ms")
            print(f"Total Duration:             {total_time:.2f} s")
            print(f"Chunks Received:            {chunk_count}")
            print(f"Est. Tokens Generated:      {int(est_tokens)}")
            print(f"Generation Speed:           {tps:.2f} tokens/sec")
        else:
            print("\n[INFO] No tokens received before termination.")
    
if __name__ == "__main__":
    prompt= "Write a 3-paragraph essay explaining how computer operating systems manage memory"
    stream_llm_response_with_resilience(prompt)