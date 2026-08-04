import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def generate_response(prompt: str, system_instruction:str = "You are a concise AI assistant.") -> str:
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Error: GEMINI_API_KEY not found. Please check your .env file")
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        return f"API Request Failed: {e}"
    
if __name__ == "__main__":
    test_prompt = "Explain why securing API keys in a .env file is essential."
    print("Sending prompt to Gemini...\n")
    output= generate_response(test_prompt)
    print("--- Model Response ---")
    print(output)